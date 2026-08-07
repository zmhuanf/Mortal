"""多进程 rollout：worker 用当前权重采样对局，轨迹回传主进程"""
import multiprocessing as mp
from dataclasses import asdict
from pathlib import Path

import torch


def _worker(task_q, result_q, version: int):
    """worker 主循环：收 (policy_state, opp_state, opp_name, seeds, log_dir) 任务"""
    torch.set_num_threads(1)  # 多 worker 各占核推理互相拖慢
    from mgrpo.config import MODEL
    from mgrpo.model.brain import PolicyNet
    from mgrpo.agent.engine import GRPOEngine, OpponentEngine
    from mgrpo.env.simulator import play_one

    for task in iter(task_q.get, None):
        policy_state, opp_state, opp_name, seeds, log_dir = task
        policy = PolicyNet(version=version, **asdict(MODEL))
        policy.load_state_dict(policy_state)
        engine = GRPOEngine(policy, device='cpu', name='trainee', version=version)

        opp_net = PolicyNet(version=version, **asdict(MODEL))
        opp_net.load_state_dict(opp_state)
        opp_engine = OpponentEngine(opp_net, device='cpu', name=opp_name, version=version)

        trajs = []
        for i, seed in enumerate(seeds):
            trajs.append(play_one(engine, opp_engine, seed, log_dir / str(i), version))
        result_q.put(trajs)


class RolloutPool:
    def __init__(self, n_workers: int, version: int):
        ctx = mp.get_context('spawn')
        self.task_q = ctx.Queue()
        self.result_q = ctx.Queue()
        self.procs = [
            ctx.Process(target=_worker, args=(self.task_q, self.result_q, version))
            for _ in range(n_workers)
        ]
        for p in self.procs:
            p.start()

    def rollout(self, policy_state, opp_state, opp_name: str, seeds: list[tuple[int, int]], log_dir: Path) -> list:
        """seeds 均分给各 worker，按提交顺序回收轨迹"""
        per_worker = (len(seeds) + len(self.procs) - 1) // len(self.procs)
        chunks = [seeds[i:i + per_worker] for i in range(0, len(seeds), per_worker)]
        log_dir.mkdir(parents=True, exist_ok=True)
        for i, chunk in enumerate(chunks):
            self.task_q.put((policy_state, opp_state, opp_name, chunk, log_dir / f'w{i}'))
        results = []
        for _ in chunks:
            results.extend(self.result_q.get())
        return results

    def close(self):
        for _ in self.procs:
            self.task_q.put(None)
        for p in self.procs:
            p.join()
