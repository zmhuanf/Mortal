"""mortal_base 单文件配置

import config_base 后 sys.modules['config'] 指向本模块
model / dataset / train / evaluate 的 from config import config 均返回本配置
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'out')

config = {
    'control': {
        'version': 4,
        'device': 'cuda:0',
        'enable_cudnn_benchmark': True,
        'enable_amp': True,
        'amp_dtype': 'float16',  # 1660S(Turing sm_75) 不支持 bf16，统一 fp16
        'enable_compile': False,
        'batch_size': 512,
        'opt_step_every': 1,
        'save_every': 200,
        'state_file': os.path.join(OUT, 'mortal.pth'),
        'best_state_file': os.path.join(OUT, 'best.pth'),
        'tensorboard_dir': os.path.join(OUT, 'log'),
    },
    'env': {
        'pts': [10.0, 4.0, -1.0, -5.0],
        'gamma': 0.99,
        'n_step': 5,
    },
    'model': {
        'widths': (256, 384, 512),   # 三阶段通道递增
        'depths': (8, 10, 6),        # 各阶段 ConvNeXt 块数
        'kernel_sizes': (3, 9),      # 双尺度 DWConv：近邻搭子 + 花色内全段
        'layer_scale': 1e-6,
        'drop_rate': 0.0,
        'drop_path_rate': 0.05,      # 大模型 stochastic depth 正则，随深度递增
        'attn_layers': 2,            # 顶层全局关系层数
        'attn_heads': 8,
        'phi_dim': 1024,
        'scalar_dim': 128,           # 标量分离流注入维度
    },
    'dqn': {
        'num_heads': 5,
    },
    'aux': {
        'next_rank_weight': 0.2,
        'shanten_weight': 0.1,
        'fuuro_weight': 0.05,
        'riichi_turn_weight': 0.05,
    },
    'iql': {
        'tau': 0.7,
        'beta': 3.0,
        'clip': 20.0,
        'ema_decay': 0.995,
    },
    'reward': {
        'riichi': 0.15,
        'agari': 0.5,
        'houjuu': -0.15,
    },
    'grp': {
        'state_file': 'D:/Workspace/Mortal/mortal/grp_v2/grp_best.pth',
        'network': {'hidden_size': 128, 'num_layers': 2, 'nhead': 4},
    },
    'dataset': {
        'globs': ['D:/Data/**/*.mjson'],
        'file_index': os.path.join(OUT, 'file_index.pth'),
        'file_batch_size': 15,
        'reserve_ratio': 0.0,
        'num_workers': 2,
        'player_names_files': [],
        'prefetch_factor': 12,  # v7 工程参数：队列缓冲吸收解析方差
        'shuffle_seed': 42,  # v7 工程参数：确定性 shuffle，resume 顺序可复现
        'num_epochs': 1,
        'enable_augmentation': True,
        'augmented_first': False,
    },
    'optim': {
        'eps': 1e-8,
        'betas': [0.9, 0.999],
        'weight_decay': 0.1,
        'max_grad_norm': 1.0,
        'lr': 1,  # AdamW 初始 lr，实际由 scheduler 因子覆盖（同 baseline）
        'scheduler': {
            'peak': 1.5e-4,
            'final': 1.5e-4,
            'init': 1e-8,
            'warm_up_steps': 5000,
            'max_steps': 5000,
        },
    },
    'train': {
        'max_steps': 200000,
        'eval_every': 5000,   # 每 N 步 1v3 评估 vs baseline_v1
        'eval_games': 100,    # 单次评估局数，评估耗时约 2-3 分钟
    },
    'eval': {
        'games': 1000,
        'segment_seeds': 250,
        'log_dir': os.path.join(OUT, 'eval_play'),
        'opponents': [
            {'name': 'baseline_v1', 'state_file': 'D:/Workspace/Mortal/mortal/baseline_v1/mortal.pth'},
        ],
    },
}

os.makedirs(OUT, exist_ok=True)
sys.modules['config'] = sys.modules[__name__]
