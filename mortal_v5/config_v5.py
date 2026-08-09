"""mortal_v5 离线训练单文件配置

import config_v5 后 sys.modules['config'] 指向本模块
train.py / dataloader.py / evaluate.py 的 from config import config 均返回本配置
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

config = {
    'control': {
        'version': 4,  # obs 布局固定为 libriichi v4 (1012, 34)
        'device': 'cuda:0',
        'enable_cudnn_benchmark': True,
        'enable_amp': True,  # bf16
        'enable_compile': False,
        'batch_size': 64,
        'opt_step_every': 1,
        'save_every': 500,
        'state_file': os.path.join(ROOT, 'mortal.pth'),
        'best_state_file': os.path.join(ROOT, 'best.pth'),
        'tensorboard_dir': os.path.join(ROOT, 'log'),
    },
    'env': {
        'gamma': 0.99,
        'n_step': 5,
        'pts': [10.0, 4.0, -1.0, -5.0],
    },
    'model': {
        'widths': [192, 384, 768],
        'depths': [8, 12, 10],
        'kernel_size': 7,
        'layer_scale': 1e-6,
        'drop_rate': 0.0,
        'drop_path_rate': 0.1,
        'attn_layers': 6,
        'attn_heads': 12,
        'pos_embed': True,
        'phi_dim': 2048,
    },
    'dqn': {
        'num_heads': 5,
        'hidden': 1024,
    },
    'dataset': {
        'globs': ['D:/Data/**/*.mjson'],
        'file_index': os.path.join(ROOT, 'file_index.pth'),
        'file_batch_size': 8,
        'reserve_ratio': 0.0,
        'num_workers': 3,
        'prefetch_factor': 4,
        'persistent_workers': True,
        'num_epochs': 1,
        'enable_augmentation': True,
        'augmented_first': False,
    },
    'reward': {
        'riichi': 0.3,
        'agari': 1.5,
        'houjuu': -1.5,
        'score_scale': 1000.0,  # 局得分差分除以该值归一化
    },
    'iql': {
        'tau': 0.7,
        'beta': 4.0,
        'clip': 10.0,
        'ema_decay': 0.999,
    },
    'aux': {
        'next_rank_weight': 0.1,
        'shanten_weight': 0.03,
        'fuuro_weight': 0.02,
        'riichi_turn_weight': 0.02,
    },
    'train': {
        'stage': 'auto',  # auto 跟随 checkpoint 实际阶段，bc/iql 为强制指定
        'bc_steps': 60000,
        'iql_steps': 120000,
        'auto_proceed': True,
        'bc_peak': 1.5e-4,
        'bc_final': 1e-5,
        'iql_peak': 5e-5,
        'iql_final': 1e-5,
        'warm_up_steps': 5000,
        'weight_decay': 0.1,
        'max_grad_norm': 1.0,
        'betas': [0.9, 0.999],
        'eps': 1e-8,
    },
    'eval': {
        'games': 1000,
        'eval_every': 10000,
        'log_dir': os.path.join(ROOT, 'eval_play'),
        'opponents': [
            {'name': 'baseline_v1', 'state_file': 'D:/Workspace/Mortal/mortal/baseline_v1/mortal.pth'},
        ],
    },
}

sys.modules['config'] = sys.modules[__name__]
