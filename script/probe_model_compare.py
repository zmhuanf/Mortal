"""913k mortal_base vs baseline_v1 同一批真实观测下的模型输出对比

指标：policy 熵 / top1 动作重合 / 立直分歧 / Q 尺度 / Q-argmax 与 policy-argmax 一致性
观测取自训练数据真实牌谱（GameplayLoader 解析），两模型看同一批状态
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / 'mortal_base'
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

from libriichi.consts import obs_shape  # noqa: E402
from libriichi.dataset import GameplayLoader  # noqa: E402
from model import Brain, DQN  # noqa: E402
from evaluate import V4Brain, V4DQN  # noqa: E402

RIICHI = 37
AGARI = 43
ACTION_N = 46

BASE_CKPT = BASE_DIR / 'out' / 'mortal.pth'
V1_CKPT = ROOT / 'mortal' / 'baseline_v1' / 'mortal.pth'
INDEX = BASE_DIR / 'out' / 'file_index.pth'


def load_pair(device: torch.device):
    """分别加载 913k mortal_base 与 baseline_v1 的裸模型"""
    sb = torch.load(BASE_CKPT, weights_only=False, map_location='cpu')
    sv = torch.load(V1_CKPT, weights_only=True, map_location='cpu')
    brain_b = Brain(version=4, **sb['config']['model']).to(device).eval()
    brain_b.load_state_dict(sb['mortal'])
    dqn_b = DQN(version=4, **sb['config']['dqn']).to(device).eval()
    dqn_b.load_state_dict(sb['current_dqn'])
    brain_v = V4Brain(version=4, **sv['config']['resnet']).to(device).eval()
    brain_v.load_state_dict(sv['mortal'])
    dqn_v = V4DQN(num_heads=sv['config']['dqn']['num_heads']).to(device).eval()
    dqn_v.load_state_dict(sv['current_dqn'])
    return (brain_b, dqn_b), (brain_v, dqn_v)


def sample_obs(n_files: int, n_batch: int, batch_size: int, device: torch.device):
    """从训练数据文件解析出 n_batch×batch_size 条真实 transition（obs/masks）"""
    idx = torch.load(INDEX, weights_only=True)
    files = idx['file_list'][:n_files]
    loader = GameplayLoader(version=4, oracle=False, augmented=False)
    obs_all, mask_all = [], []
    for file in loader.load_gz_log_files(files):
        for game in file:
            obs_all.append(np.asarray(game.take_obs(), dtype=np.float32))
            mask_all.append(np.asarray(game.take_masks()))
    obs = np.concatenate(obs_all)[: n_batch * batch_size]
    masks = np.concatenate(mask_all)[: n_batch * batch_size]
    return (
        torch.from_numpy(obs).to(device),
        torch.from_numpy(masks).to(device),
    )


def masked_logprob(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~masks, -1e9).log_softmax(-1)


def masked_argmax(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~masks, -1e9).argmax(-1)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description='mortal_base(913k) vs baseline_v1 模型输出对比')
    ap.add_argument('--files', type=int, default=30, help='采样文件数')
    ap.add_argument('--batch', type=int, default=2, help='batch 数')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    device = torch.device(args.device)
    obs, masks = sample_obs(args.files, args.batch, args.batch_size, device)
    print(f'samples: {obs.shape[0]}')

    (brain_b, dqn_b), (brain_v, dqn_v) = load_pair(device)
    with torch.no_grad():
        phi_b = brain_b(obs)
        phi_v = brain_v(obs)
        lp_b = masked_logprob(brain_b.policy_logits(phi_b), masks)
        lp_v = masked_logprob(brain_v.policy_logits(phi_v), masks)

        p_b, p_v = lp_b.exp(), lp_v.exp()
        ent_b = -(p_b * lp_b).masked_fill(~masks, 0.).sum(-1).sum() / masks.sum(-1).sum()
        ent_v = -(p_v * lp_v).masked_fill(~masks, 0.).sum(-1).sum() / masks.sum(-1).sum()

        a_b = masked_argmax(brain_b.policy_logits(phi_b), masks)
        a_v = masked_argmax(brain_v.policy_logits(phi_v), masks)
        agree = (a_b == a_v)
        riichi_b, riichi_v = a_b == RIICHI, a_v == RIICHI
        agari_b, agari_v = a_b == AGARI, a_v == AGARI

        q_b = dqn_b(phi_b, masks).mean(1)  # (N, A) heads 平均
        q_v = dqn_v(phi_v, masks).mean(1)
        v_b = dqn_b.value(phi_b).mean(-1)
        v_v, _ = dqn_v.net(phi_v).split((dqn_v.num_heads, dqn_v.num_heads * ACTION_N), dim=-1)

        qa_b = masked_argmax(q_b, masks)
        qa_v = masked_argmax(q_v, masks)

    print(f'{"指标":<34}{"mortal_base(913k)":>16}{"baseline_v1":>14}')
    print(f'{"policy 熵":<32}{ent_b.item():>16.4f}{ent_v.item():>14.4f}')
    print(f'{"top1 动作重合率":<30}{(agree.float().mean().item() * 100):>16.1f}%')
    print(f'{"立直分歧率(一方立直)":<27}{(riichi_b ^ riichi_v).float().mean().item() * 100:>16.1f}%')
    print(f'{"其中 base 立直 baseline 不立":<25}{(riichi_b & ~riichi_v).sum().item():>16d}'
          f'{(riichi_v & ~riichi_b).sum().item():>14d}')
    print(f'{"和牌分歧率":<32}{(agari_b ^ agari_v).float().mean().item() * 100:>16.1f}%')
    print(f'{"Q-argmax==policy-argmax":<27}{(qa_b == a_b).float().mean().item() * 100:>16.1f}%'
          f'{(qa_v == a_v).float().mean().item() * 100:>13.1f}%')
    print(f'{"Q 均值/标准差":<30}{q_b.mean().item():>16.3f}/{q_b.std().item():.1f}'
          f'{q_v.mean().item():>13.3f}/{q_v.std().item():.1f}')
    print(f'{"Q min/max":<32}{q_b.min().item():>16.3f}/{q_b.max().item():.1f}'
          f'{q_v.min().item():>13.3f}/{q_v.max().item():.1f}')
    print(f'{"V 均值/标准差":<30}{v_b.mean().item():>16.3f}/{v_b.std().item():.1f}'
          f'{v_v.mean().item():>13.3f}/{v_v.std().item():.1f}')
    print(f'{"V min/max":<32}{v_b.min().item():>16.3f}/{v_b.max().item():.1f}'
          f'{v_v.min().item():>13.3f}/{v_v.max().item():.1f}')


if __name__ == '__main__':
    main()