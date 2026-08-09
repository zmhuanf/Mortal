"""Akagi native_bot (4p) vs mortal 2v2 benchmark：N 场半庄，输出双方位次/顺位/pt/打点统计

Akagi 侧通过 N 个 Rust 子进程（akagi_bridge）驱动 native_bot，子进程按局分片并行推理
"""
import argparse
import json
import logging
import os
import secrets
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mgrpo.prelude  # noqa: E402  加载 libriichi 与旧模型模块

from engine import MortalEngine  # noqa: E402
from libriichi.arena import TwoVsTwo  # noqa: E402
from libriichi.stat import Stat  # noqa: E402
from model import Brain, DQN  # noqa: E402

log = logging.getLogger(__name__)

PTS = [90, 45, 0, -135]  # 半庄顺位赏
KEY = 0x2A44  # 每局固定 second seed，与 --seed 组合保证可复现
MORTAL_CKPT = ROOT / 'mortal' / 'mortal_v4_offline_baseline' / 'mortal.pth'
BRIDGE_BIN = (
    Path(__file__).resolve().parent / 'akagi_bridge' / 'target' / 'release'
    / ('akagi_mjai_bot.exe' if os.name == 'nt' else 'akagi_mjai_bot')
)
AKAGI_NAME = 'akagi-native'
MORTAL_NAME = 'mortal_v4'


def build_mortal_engine(device: torch.device, ckpt: Path, name: str) -> MortalEngine:
    """mortal.pth → Brain+DQN 旧架构引擎，action_source 按 checkpoint 是否含 policy_head 决定"""
    state = torch.load(ckpt, weights_only=True, map_location='cpu')
    cfg = state['config']
    version = cfg['control'].get('version', 4)
    brain = Brain(version=version, **cfg['resnet']).eval()
    dqn = DQN(version=version, num_heads=cfg.get('dqn', {}).get('num_heads', 1)).eval()
    brain.load_state_dict(state['mortal'], strict=False)
    dqn.load_state_dict(state['current_dqn'])
    return MortalEngine(
        brain,
        dqn,
        is_oracle=False,
        version=version,
        device=device,
        enable_amp=cfg['control'].get('enable_amp', False) and device.type == 'cuda',
        enable_rule_based_agari_guard=True,
        name=name,
        action_source='policy' if 'policy_head.weight' in state['mortal'] else 'q',
    )


class AkagiBot:
    """engine_type='mjai-log' 的 Python 引擎：经 N 个 Rust 子进程驱动 native_bot

    libriichi 每帧回调传入完整 mjai log（append-only），这里用字符串前缀比较
    切出增量事件透传给子进程，子进程维护各自游戏状态并回传动作
    """
    engine_type = 'mjai-log'

    def __init__(self, nprocs: int, games: int):
        self.name = AKAGI_NAME
        self._nprocs = max(1, min(nprocs, games))
        self._procs = [self._spawn() for _ in range(self._nprocs)]
        self._player_ids: list[int] = []
        self._game_proc: dict[int, int] = {}
        self._last_json: dict[int, str] = {}

    def _spawn(self) -> subprocess.Popen:
        if not BRIDGE_BIN.exists():
            raise FileNotFoundError(
                f'{BRIDGE_BIN} 不存在，请先构建：cd {BRIDGE_BIN.parent} && cargo build --release'
            )
        return subprocess.Popen(
            [str(BRIDGE_BIN)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            bufsize=1,
        )

    def set_player_ids(self, player_ids: list[int]) -> None:
        """完整座位序列，长度 = games，player_id_idx → 该局本方座位号"""
        self._player_ids = list(player_ids)

    def start_game(self, index: int) -> None:
        self._game_proc[index] = index % self._nprocs
        p = self._procs[self._game_proc[index]]
        seat = self._player_ids[index]
        cmd = json.dumps({'__cmd': 'new_game', 'game': index, 'seat': seat}, separators=(',', ':'))
        p.stdin.write(cmd + '\n')
        p.stdin.flush()

    def end_kyoku(self, index: int) -> None:
        # ctx.log 是小局级，通知子进程归零事件计数，同时清缓存让下一帧走全量路径
        p = self._procs[self._game_proc.get(index, 0)]
        cmd = json.dumps({'__cmd': 'reset', 'game': index}, separators=(',', ':'))
        p.stdin.write(cmd + '\n')
        p.stdin.flush()
        self._last_json.pop(index, None)

    def end_game(self, index: int, scores: list[int]) -> None:
        p = self._procs[self._game_proc.get(index, 0)]
        cmd = json.dumps({'__cmd': 'drop_game', 'game': index}, separators=(',', ':'))
        p.stdin.write(cmd + '\n')
        p.stdin.flush()
        self._last_json.pop(index, None)

    def react_batch(self, game_states: list) -> list[str]:
        """每帧返回一个 mjai 动作 JSON 字符串；各子进程按局分片，线程并行读写"""
        results: list[str] = [None] * len(game_states)
        groups: dict[int, list[int]] = {}
        for i, gs in enumerate(game_states):
            groups.setdefault(self._game_proc[gs.game_index], []).append(i)

        def worker(proc_idx: int, idxs: list[int]) -> None:
            proc = self._procs[proc_idx]
            for i in idxs:
                gs = game_states[i]
                events, full = self._incremental(gs.game_index, gs.events_json)
                cmd = {'__cmd': 'frame', 'game': gs.game_index, 'events': events}
                if full:
                    cmd['full'] = True
                proc.stdin.write(json.dumps(cmd, separators=(',', ':')) + '\n')
            proc.stdin.flush()
            for i in idxs:
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError(f'akagi_mjai_bot 子进程 {proc_idx} 意外退出')
                results[i] = line.rstrip('\n') or self._fallback(game_states[i])

        threads = [threading.Thread(target=worker, args=(p, idxs)) for p, idxs in groups.items()]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results

    def _incremental(self, game: int, events_json: str) -> tuple[str, bool]:
        """完整 log 数组 append-only，与上次帧做前缀比较切出增量；失败则全量交由子进程去重"""
        old = self._last_json.get(game)
        if old is None or not events_json.startswith(old):
            self._last_json[game] = events_json
            return events_json, True
        tail = events_json[len(old):]  # ",{...},{...}]"（追加多个事件时）
        inc = f'[{tail[1:-1]}]' if tail.startswith(',') else '[]'
        self._last_json[game] = events_json
        return inc, False

    def _fallback(self, gs) -> str:
        """子进程无动作（decide None/Pass/出错）时的兑底：打牌帧摸切，响应帧无反应"""
        seat = gs.state.player_id
        if gs.state.last_cans.can_discard:
            last = gs.state.last_self_tsumo()
            if last:
                return json.dumps(
                    {'type': 'dahai', 'actor': seat, 'pai': last, 'tsumogiri': True},
                    separators=(',', ':'),
                )
        return '{"type":"none"}'

    def close(self) -> None:
        for p in self._procs:
            try:
                p.stdin.close()
            except Exception:
                pass
            try:
                p.terminate()
            except Exception:
                pass
            try:
                p.wait(timeout=5)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
                try:
                    p.wait()
                except Exception:
                    pass
            for f in (p.stdout, p.stderr):
                try:
                    f.close()
                except Exception:
                    pass


def run_games(challenger, champion, seed: int, games: int, log_dir: Path, nprocs: int) -> None:
    """TwoVsTwo 执行 games 场，每 seed 分 a/b 两局轮换双方座位消除座位偏差"""
    env = TwoVsTwo(disable_progress_bar=False, log_dir=str(log_dir))
    env.py_vs_py(
        challenger=challenger,
        champion=champion,
        seed_start=(seed, KEY),
        seed_count=games // 2,
    )


def summarize(log_dir: Path, name: str) -> dict:
    """按玩家名聚合同名两席的统计量"""
    stat = Stat.from_dir(str(log_dir), name, disable_progress_bar=True)
    return {
        'games': stat.game,
        'ranks': [stat.rank_1, stat.rank_2, stat.rank_3, stat.rank_4],
        'avg_rank': stat.avg_rank,
        'total_pt': stat.total_pt(PTS),
        'avg_pt': stat.avg_pt(PTS),
        'avg_point_per_agari': stat.avg_point_per_agari,
    }


def report(log_dir: Path, keep_logs: bool, names: tuple[str, str]) -> None:
    """打印双方统计并清理对局日志"""
    try:
        for name in names:
            s = summarize(log_dir, name)
            ranks = ', '.join(str(x) for x in s['ranks'])
            print(
                f"{name:<12} 位次 [{ranks}]  "
                f"平均顺位 {s['avg_rank']:.3f}  "
                f"平均pt {s['avg_pt']:+.2f} (总 {s['total_pt']:+d})  "
                f"平均打点 {s['avg_point_per_agari']:.1f}  ({s['games']} 席次)"
            )
    finally:
        if not keep_logs:
            shutil.rmtree(log_dir, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description='Akagi native_bot vs mortal 2v2 benchmark')
    ap.add_argument('--games', type=int, default=2000, help='2v2 对局数，须为偶数')
    ap.add_argument('--device', default='cuda:0', help='Mortal 推理设备，如 cuda:0 / cpu')
    ap.add_argument('--seed', type=int, default=None, help='固定种子复现；默认随机')
    ap.add_argument('--log-dir', default=None, help='对局日志目录；显式指定则保留，否则统计后删除')
    ap.add_argument('--nprocs', type=int, default=os.cpu_count() or 4, help='Akagi 推理子进程数（按局分片并行）')
    ap.add_argument('--mortal-ckpt', type=Path, default=MORTAL_CKPT, help='Mortal checkpoint 路径')
    ap.add_argument('--swap', action='store_true', help='角色对调：Mortal 为 challenger，Akagi 为 champion')
    args = ap.parse_args()

    if args.games % 2 != 0:
        ap.error('--games 必须为偶数（每 seed 跑 a/b 两局）')
    device = torch.device(args.device)
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else ROOT / 'mgrpo' / '_bench_logs' / f'2v2_{datetime.now():%Y%m%d_%H%M%S}'
    )

    mortal = build_mortal_engine(device, args.mortal_ckpt, MORTAL_NAME)
    akagi = AkagiBot(nprocs=args.nprocs, games=args.games)
    challenger, champion = (mortal, akagi) if args.swap else (akagi, mortal)
    names = (challenger.name, champion.name)
    log.info('对局 %d 场 2v2（%s vs %s），seed=%d，akagi 子进程 %d 个', args.games, *names, seed, akagi._nprocs)
    try:
        run_games(challenger, champion, seed, args.games, log_dir, args.nprocs)
    finally:
        akagi.close()
    report(log_dir, keep_logs=args.log_dir is not None, names=names)


if __name__ == '__main__':
    main()
