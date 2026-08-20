"""mortal_base_v2 单文件配置：纯策略（BC）训练

import config_base_v2 后 sys.modules['config'] 指向本模块
只保留 policy 训练与评估所需，无任何 Q/价值/事件/辅助头配置
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
        'amp_dtype': 'bfloat16',
        'enable_compile': False,
        'batch_size': 256,
        'save_every': 500,
        # 初始从 mortal_base 复制参数，之后保存在本目录（只存 policy 权重）
        'init_from': 'D:/Workspace/Mortal/mortal_base/out/mortal.pth',
        'state_file': os.path.join(OUT, 'mortal.pth'),
        'best_state_file': os.path.join(OUT, 'best.pth'),
        'tensorboard_dir': os.path.join(OUT, 'log'),
    },
    'env': {
        'pts': [10.0, 4.0, -1.0, -5.0],  # 评估奖励口径
    },
    'model': {
        'widths': (256, 384, 512),
        'depths': (8, 10, 6),
        'kernel_sizes': (3, 9),
        'layer_scale': 1e-6,
        'drop_rate': 0.0,
        'drop_path_rate': 0.05,
        'attn_layers': 2,
        'attn_heads': 8,
        'phi_dim': 1024,
        'scalar_dim': 128,
    },
    'dataset': {
        'globs': ['D:/Data/**/*.mjson', 'D:/Data2/**/*.mjson'],
        'file_index': os.path.join(OUT, 'file_index.pth'),
        'file_batch_size': 1,
        'num_workers': 4,
        'prefetch_factor': 4,
        'num_epochs': 1,
        'enable_augmentation': True,
        'augmented_first': False,
        'shuffle_seed': 42,
    },
    'optim': {
        'eps': 3e-5,
        'betas': [0.9, 0.999],
        'weight_decay': 0.1,
        'max_grad_norm': 1.0,
        'lr': 3e-5,  # 恒定学习率（post_training 持续训练）
        'warm_up_steps': 5000,
    },
    'train': {
        'max_steps': 600000,
        'post_training': True,  # 随机读文件 + 无步数上限，Ctrl+C 手动停止
        'eval_every': 5000,
        'eval_games': 100,
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
