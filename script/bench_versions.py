"""mortal_base 两个版本 vs baseline_v1 依次跑 2v2，输出对比表

用法: python script/bench_versions.py --ckpt-a out/backup/baseline_v2_100w.pth --ckpt-b out/mortal.pth --games 1000
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_bench(ckpt: Path, games: int, seed: int | None) -> str:
    """单版本跑一次 2v2 vs v1，捕获顺位行"""
    cmd = [sys.executable, str(ROOT / 'script' / 'bench_base_vs_baseline_v1.py'),
           '--games', str(games), '--base-ckpt', str(ckpt)]
    if seed is not None:
        cmd += ['--seed', str(seed)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    lines = [ln for ln in proc.stdout.splitlines() if '平均顺位' in ln]
    return lines[0].strip() if lines else f'[无输出] {proc.stderr[-200:]}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-a', type=Path, required=True, help='版本 A 路径')
    ap.add_argument('--ckpt-b', type=Path, required=True, help='版本 B 路径（通常为当前 mortal.pth）')
    ap.add_argument('--games', type=int, default=1000, help='每版本对局数')
    ap.add_argument('--seed', type=int, default=None, help='统一种子保证可比')
    args = ap.parse_args()

    print(f'版本 A: {args.ckpt_a}')
    print(f'版本 B: {args.ckpt_b}')
    print(f'每版本 {args.games} 局 2v2 vs baseline_v1，seed={args.seed or "随机"}')
    for name, ckpt in (('A', args.ckpt_a), ('B', args.ckpt_b)):
        print(f'--- 跑版本 {name} ---')
        print(run_bench(ckpt, args.games, args.seed))


if __name__ == '__main__':
    main()