"""mortal_v6 单文件配置：BC 预训练 → XQL 精调 + 事件世界模型 + 想象搜索

import config_v6 后 sys.modules['config'] 指向本模块
train.py / dataloader.py / evaluate.py 的 from config import config 均返回本配置
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, 'out')  # 模型输出统一收进 out/，保持根目录干净

config = {
    'control': {
        'version': 4,  # obs 布局固定为 libriichi v4 (1012, 34)
        'device': 'cuda:0',
        'enable_cudnn_benchmark': True,
        'enable_amp': True,  # bf16
        'enable_compile': False,
        'batch_size': 64,
        'opt_step_every': 1,
        'save_every': 1000,
        'state_file': os.path.join(OUT_DIR, 'mortal.pth'),
        'best_state_file': os.path.join(OUT_DIR, 'best.pth'),
        'tensorboard_dir': os.path.join(OUT_DIR, 'log'),
    },
    'env': {
        'gamma': 0.99,
        'n_step': 8,  # 5→8：目标含更多真实 reward，减轻对自举的依赖
        'pts': [10.0, 4.0, -1.0, -5.0],
    },
    'model': {
        'widths': [192, 384, 640],  # 约 8000 万参数，8GB 显存余量优先
        'depths': [8, 10, 10],
        'kernel_size': 7,
        'layer_scale': 1e-6,
        'drop_rate': 0.0,
        'drop_path_rate': 0.1,
        'attn_layers': 6,
        'attn_heads': 8,
        'pos_embed': True,
        'phi_dim': 2048,
    },
    'q_head': {
        'hidden': 1024,
    },
    'event': {
        'horizon': 10,  # 事件轨迹预测长度（手）
        'n_types': 6,   # 无事/立直/和牌/放铳/流局/被自摸
        'dim': 256,
        'heads': 4,
        'layers': 2,
    },
    'event_loss': {
        'rewards': [0.0, 0.3, 1.5, -1.5, 0.0, -1.5],  # 与 reward 段一致，被自摸与放铳同为本家输分
        'weight': 0.1,  # BC 与 XQL 共用的事件监督权重
    },
    'xql': {
        'tau': 0.6,            # 保守化：远离 0.9 的高估区，0.5 为无偏
        'beta': 4.0,           # 策略优势温度
        'clip': 3.0,           # 优势 e 指数上限，防单样本权重失控
        'ema_decay': 0.999,
        'q_scale': 1.0,        # 原 100 会淹没策略/事件/aux 损失
        'q_delta': 10.0,       # Q 回归 Huber 阈值，抗 TD 离群
        'entropy_weight': 0.01,  # 策略熵正则，防 AWR 收窄分布
        'lr': 1e-5,            # XQL 固定学习率，不做退火
        'backbone_lr_ratio': 0.2,  # mortal 主干 lr 比例，保护 BC 表征
        'head_lr_ratio': 5.0,      # q_head lr 比例，随机初始化起步需更快
    },
    'aux': {
        'next_rank_weight': 0.1,
        'shanten_weight': 0.03,
        'fuuro_weight': 0.02,
        'riichi_turn_weight': 0.02,
    },
    'train': {
        'stage': 'xql',  # auto 跟随 checkpoint 实际阶段，bc/xql 为强制指定
        # BC 等价 batch256 约 8.3 万步，XQL 10 万步（各含事件世界模型监督）
        'bc_steps': 160000,
        'xql_steps': 120000,
        'auto_proceed': True,
        'bc_peak': 3.75e-5,  # linear scaling：batch 256→64 的 0.25 倍
        'bc_final': 2.5e-6,
        'xql_peak': 1.25e-5,
        'xql_final': 2.5e-6,
        'warm_up_steps': 5000,
        'weight_decay': 0.1,
        'max_grad_norm': 1.0,
        'betas': [0.9, 0.999],
        'eps': 1e-8,
    },
    'eval': {
        'games': 1000,
        'eval_every': 120000,
        'action_mode': 'policy',  # search=想象搜索 / greedy=直出精排 / policy=纯策略
        'search_k': 8,            # 搜索候选动作数
        'greedy_top_k': 3,        # 直出精排候选动作数
        'search_alpha': 0.5,      # Q 与 rollout 候选内标准化后的混合权重
        'log_dir': os.path.join(OUT_DIR, 'eval_play'),
        'opponents': [
            {'name': 'baseline_v1', 'state_file': 'D:/Workspace/Mortal/mortal/baseline_v1/mortal.pth'},
        ],
    },
    'dataset': {
        'globs': ['D:/Data/**/*.mjson'],
        'file_index': os.path.join(OUT_DIR, 'file_index.pth'),
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
        'clip': 5.0,            # 局奖励裁剪，防 Q 目标被 ±20+ 离群值拉大
    },
}

os.makedirs(OUT_DIR, exist_ok=True)
sys.modules['config'] = sys.modules[__name__]
