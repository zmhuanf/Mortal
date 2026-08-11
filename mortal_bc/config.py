"""mortal_bc 纯行为克隆配置：加权 BC + 辅助任务，超越 baseline_v1"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, 'out')
BASELINE = 'D:/Workspace/Mortal/mortal/baseline_v1/mortal.pth'

config = {
    'control': {
        'version': 4,
        'device': 'cuda:0',
        'enable_cudnn_benchmark': True,
        'enable_amp': True,
        'batch_size': 256,
        'opt_step_every': 2,  # 梯度累积，等效 batch 512，省显存
        'log_every': 100,   # TensorBoard loss 写入间隔
        'save_every': 1000,
        'eval_every': 20000,
        'state_file': os.path.join(OUT_DIR, 'mortal.pth'),
        'best_state_file': os.path.join(OUT_DIR, 'best.pth'),
        'tensorboard_dir': os.path.join(OUT_DIR, 'log'),
        'eval_log_dir': os.path.join(OUT_DIR, 'eval_play'),
    },
    'model': {
        'widths': [256, 512, 512],
        'depths': [8, 8, 8],
        'attn_layers': 6,
        'attn_heads': 8,
        'layer_scale': 1e-6,
        'drop_rate': 0.0,
    },
    'dataset': {
        'globs': ['D:/Data/**/*.mjson', 'D:/Data2/**/*.mjson'],
        'file_index': os.path.join(OUT_DIR, 'file_index.pth'),
        'file_batch_size': 8,
        'num_workers': 4,
        'prefetch_factor': 4,
        'persistent_workers': True,
        'num_epochs': 1,
        'enable_augmentation': True,
    },
    'bc': {
        # 按最终排名加权：赢牌局动作强信号，输牌局近乎丢弃
        'rank_weights': [2.0, 1.0, 0.4, 0.1],
    },
    'aux': {
        'next_rank_weight': 0.2,
        'shanten_weight': 0.1,
        'fuuro_weight': 0.05,
        'riichi_turn_weight': 0.05,
    },
    'optim': {
        'lr': 3e-5,  # 固定学习率，batch 512 对应 baseline 的 linear scaling
        'betas': [0.9, 0.999],
        'eps': 1e-8,
        'weight_decay': 0.1,
        'max_grad_norm': 1.0,
    },
    'eval': {
        'games': 2000,
        'opponent_state_file': BASELINE,
        'opponent_name': 'baseline_v1',
    },
}

os.makedirs(OUT_DIR, exist_ok=True)
