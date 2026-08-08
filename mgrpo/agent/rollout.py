"""多进程流式 rollout：worker 分批回传，主进程先收摘要算 advantage 再逐批消费轨迹，obs 不驻留"""
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path

import torch

from mgrpo.agent.grpo import GRPO
from mgrpo.config import REWARD
from mgrpo.env.reward import game_reward


def _worker(task_q, summary_q, traj_q, version: int, device: str):
    """worker 主循环：收 (policy_state, opp_state, opp_name, seeds, log_dir, wid, batch_size) 任务"""
    torch.set_num_threads(1)  # 多 worker 各占核推理互相拖慢
    from mgrpo.config import MODEL
    from mgrpo.model.brain import PolicyNet
    from mgrpo.agent.engine import GRPOEngine, OpponentEngine
    from mgrpo.env.simulator import iter_play

    dev = torch.device(device)
    # 常驻实例，每任务只换权重，避免反复重建 CUDA 模型
    policy = PolicyNet(version=version, **asdict(MODEL)).to(dev)
    opp_net = PolicyNet(version=version, **asdict(MODEL)).to(dev)
    for task in iter(task_q.get, None):
        policy_state, opp_state, opp_name, seeds, log_dir, wid, batch_size = task
        policy.load_state_dict(policy_state)
        opp_net.load_state_dict(opp_state)
        engine = GRPOEngine(policy, device=device, name='trainee', version=version)
        opp_engine = OpponentEngine(opp_net, device=device, name=opp_name, version=version)
        for bidx, batch in enumerate(iter_play(engine, opp_engine, seeds, log_dir, version, batch_size)):
            summary_q.put((wid, bidx, [t.scores for t in batch]))
            traj_q.put((wid, bidx, batch))  # maxsize 有界，天然流控


class RolloutPool:
    def __init__(self, n_workers: int, version: int, device: str = 'cpu', batch_size: int = 8):
        ctx = mp.get_context('spawn')
        self.task_q = ctx.Queue()
        self.summary_q = ctx.Queue()
        self.traj_q = ctx.Queue(maxsize=1)  # 任一时刻只缓存一批轨迹，防内存堆积
        self.batch_size = batch_size
        self.procs = [
            ctx.Process(target=_worker, args=(self.task_q, self.summary_q, self.traj_q, version, device))
            for _ in range(n_workers)
        ]
        for p in self.procs:
            p.start()

    def rollout(self, policy_state, opp_state, opp_name: str, seeds: list[tuple[int, int]], log_dir: Path, progress=None):
        """生成器：首项为总批数；阶段 1 收摘要算 advantage（progress 逐批推进）；阶段 2 逐批 yield 轨迹"""
        per_worker = (len(seeds) + len(self.procs) - 1) // len(self.procs)
        chunks = [seeds[i:i + per_worker] for i in range(0, len(seeds), per_worker)]
        log_dir.mkdir(parents=True, exist_ok=True)
        total_batches = 0
        for i, chunk in enumerate(chunks):
            total_batches += (4 * len(chunk) + self.batch_size - 1) // self.batch_size
            self.task_q.put((policy_state, opp_state, opp_name, chunk, log_dir / f'w{i}', i, self.batch_size))
        yield total_batches

        # 阶段 1：边收摘要边消费轨迹（放行 worker），轨迹批驻留主进程
        keys, lens, all_scores = [], [], []
        traj_store = {}
        for _ in range(total_batches):
            wid, bidx, scores_list = self.summary_q.get()
            keys.append((wid, bidx))
            lens.append(len(scores_list))
            all_scores.extend(scores_list)
            twid, tbidx, batch = self.traj_q.get()  # 持续消费，避免 worker 阻塞在轨迹 put
            traj_store[(twid, tbidx)] = batch
            if progress:
                progress(1)
        rewards = torch.tensor([
            game_reward(s, 0, REWARD.pts, REWARD.score_weight, REWARD.init_score, REWARD.score_scale)
            for s in all_scores
        ])
        advantages = GRPO.compute_advantages(rewards)
        adv_map = {k: a for k, a in zip(keys, advantages.split(lens))}

        yield advantages
        # 阶段 2：按摘要顺序逐批喂训练
        for k in keys:
            yield traj_store[k], adv_map[k]

    def close(self):
        for _ in self.procs:
            self.task_q.put(None)
        for p in self.procs:
            p.join()
