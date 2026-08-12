"""mortal_v7 单文件配置

import config_v7 后 sys.modules['config'] 指向本模块
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
        'amp_dtype': 'float16',  # 1660S(Turing sm_75) 不支持 bf16，统一 fp16：精度高于 bf16、显存占用相同
        'enable_compile': False,
        'batch_size': 8,
        'save_every': 1000,
        'state_file': os.path.join(OUT, 'mortal.pth'),
        'best_state_file': os.path.join(OUT, 'best.pth'),
        'tensorboard_dir': os.path.join(OUT, 'log'),
    },
    'env': {
        'pts': [6.0, 4.0, 2.0, 0.0],
    },
    'model': {
        'conv_channels': 256,
        'num_blocks': 32,
        'layer_scale': 1e-6,
        'drop_rate': 0.0,
        'd_model': 512,
        'num_layers': 10,
        'nhead': 8,
        'seq_len': 288,
    },
    'rtg': {
        'score_scale': 10000.0,  # 分数差分归一化尺度
        'target_score': 35000.0,  # 争一目标分
        'window': 64,  # 对齐 seg_len，评估窗口与训练段一致
        'target_ratio': 0.3,  # 训练混入目标语义 RTG 的比例，覆盖推理分布消解外推
    },
    'dataset': {
        'globs': ['D:/Data/**/*.mjson', 'D:/Data2/**/*.mjson'],
        'file_index': os.path.join(OUT, 'file_index.pth'),
        'file_batch_size': 1,  # 突发粒度=单文件解析，worker 交错产出段，GPU 平滑
        'reserve_ratio': 0.0,
        'num_workers': 4,  # 数据吞吐留富余，吸收解析方差避免 GPU 饥饿
        'prefetch_factor': 12,  # 队列缓冲 36 段，吸收解析方差形成的偶发间隙
        'persistent_workers': True,
        'num_epochs': 1,
        'enable_augmentation': False,  # 无增强：解析减半，且 baseline 成功配方即无增强
        'augmented_first': False,
        't_max': 96,
        'seg_len': 64,  # 长局切段长度，RTG 保持全局语义，控制 GPU 计算量
    },
    'optim': {
        'eps': 1e-8,
        'betas': [0.9, 0.999],
        'weight_decay': 0.1,
        'max_grad_norm': 1.0,
        'lr': 3e-4,  # 固定学习率，resume 不改变行为
    },
    'train': {
        'max_steps': 40000,
    },
    'eval': {
        'games': 1000,
        'segment_seeds': 250,  # 1000 局 seed_count=250，一次跑完；内存紧张时调小即分片
        'log_dir': os.path.join(OUT, 'eval_play'),
        'opponents': [
            {'name': 'baseline_v1', 'state_file': 'D:/Workspace/Mortal/mortal/baseline_v1/mortal.pth'},
        ],
    },
}

os.makedirs(OUT, exist_ok=True)
sys.modules['config'] = sys.modules[__name__]
