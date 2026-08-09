"""train_v2 蒸馏训练单文件配置

import config_v2 后 sys.modules['config'] 指向本模块
common / dataloader / player 的 from config import config 均返回本配置
train_v2.py / server.py / client.py 入口处 import config_v2 即可整体切换
"""

import sys

config = {
    'control': {
        'version': 4,
        'online': False,  # 离线训练从人类牌谱读取数据，不连接 server
        'is_baseline': False,
        'state_file': 'D:/Workspace/Mortal/mortal/mortal_v4_offline_baseline/mortal.pth',
        'best_state_file': 'D:/Workspace/Mortal/mortal/mortal_v4_offline_baseline/best.pth',
        'tensorboard_dir': 'D:/Workspace/Mortal/mortal/mortal_v4_offline_baseline/log',
        'device': 'cuda:0',
        'enable_cudnn_benchmark': True,
        'enable_amp': True,
        'enable_compile': False,
        'batch_size': 512,
        'opt_step_every': 1,
        'save_every': 200,
        'test_every': 10000,
        'submit_every': 400,
    },
    'env': {
        'gamma': 0.99,
        'n_step': 5,
        'pts': [10.0, 4.0, -1.0, -5.0],
    },
    'resnet': {
        'conv_channels': 192,
        'num_blocks': 40,
        'layer_scale': 1e-6,
        'drop_rate': 0.0,
    },
    'dqn': {
        'num_heads': 5,
        'uncertainty_scale': 2.0,
    },
    'dataset': {
        'globs': ['D:/Data2/**/*.mjson'],
        'file_index': 'D:/Workspace/Mortal/mortal/mortal_v4_offline_baseline/file_index.pth',
        'file_batch_size': 8,
        'reserve_ratio': 0.0,
        'num_workers': 3,
        'prefetch_factor': 4,
        'persistent_workers': True,
        'player_names_files': [],
        'num_epochs': 1,
        'enable_augmentation': True,
        'augmented_first': False,
        'online_human_ratio': 1.0,
    },
    'iql': {
        'tau': 0.7,
        'beta': 4.0,
        'clip': 10.0,
        'ema_decay': 0.999,
    },
    'distill': {
        'bc_weight': 0.0,  # 第一轮离线实验关闭粗粒度 BC 标签
        'top_k': 0,  # 关闭 BC
        'bc_mode': 'kyoku_plus',  # BC 标记来源：kyoku_plus=所在小局得点为正，top_k=最终前 top_k 名
        'bc_kyoku_threshold': 3000,  # kyoku_plus 模式净得分阈值
        'init_from': 'D:/Workspace/Mortal/mortal/baseline_v1/mortal.pth',  # state_file 不存在时热启动
    },
    'aux': {
        'next_rank_weight': 0.1,
        'shanten_weight': 0.03,
        'fuuro_weight': 0.02,
        'riichi_turn_weight': 0.02,
    },
    'reward': {
        'riichi': 0.3,
        'agari': 1.5,
        'houjuu': -1.5,
    },
    'optim': {
        'eps': 1e-8,
        'betas': [0.9, 0.999],
        'weight_decay': 0.1,
        'max_grad_norm': 1.0,
        'scheduler': {
            'peak': 8e-5,
            'final': 8e-5,  # 固定学习率
            'warm_up_steps': 0,
            'max_steps': 500000,
        },
    },
    'train_play': {
        'default': {
            'games': 800,
            'log_dir': 'D:/Workspace/Mortal/mortal/mortal_v4/train_play',
            'eps_max': 0.05,  # 探索率随 trainee 变强退火到 eps_min
            'eps_min': 0.01,
            'boltzmann_temp': 0.05,
            'top_p': 0.9,
            'temp_max': 1.0,
            'temp_min': 0.1,
            'target_pt': 10,  # 明显强于 champion 才退火，避免过早收敛探索
            'repeats': 1,
        },
    },
    'online': {
        'history_window': 50,
        'enable_compile': False,
        'remote': {
            'host': '192.168.1.239',
            'port': 5000,
        },
        'server': {
            'buffer_dir': 'D:/Workspace/Mortal/mortal/mortal_v4/buffer',
            'drain_dir': 'D:/Workspace/Mortal/mortal/mortal_v4/drain',
            'capacity': 2400,
            'force_sequential': False,
        },
        'pool': {
            'opponents_dir': 'D:/Workspace/Mortal/mortal/mortal_v4/opponents',
            'max_size': 8,  # 池子超限淘汰最弱非 baseline 对手
            'auto_promote': True,  # train_v2 评估相对上轮提升即晋级
        },
    },
    'baseline': {
        'train': {
            'device': 'cuda:0',
            'enable_compile': False,
            'state_file': 'D:/Workspace/Mortal/mortal/baseline_v1/mortal.pth',
        },
        'test': {
            'device': 'cuda:0',
            'enable_compile': False,
            'state_file': 'D:/Workspace/Mortal/mortal/baseline_v1/mortal.pth',
        },
    },
    'test_play': {
        'games': 2000,  # 1v3 评估噪声大，加局数稳定晋级判断
        'log_dir': 'D:/Workspace/Mortal/mortal/mortal_v4/test_play',
    },
    'grp': {
        'state_file': 'D:/Workspace/Mortal/mortal/grp_v2/grp_best.pth',
        'best_state_file': 'D:/Workspace/Mortal/mortal/grp_v2/grp_best.pth',
        'network': {
            'hidden_size': 128,
            'num_layers': 2,
            'nhead': 4,
        },
    },
    '1v3': {
        'seed_key': -1,
        'games_per_iter': 40,
        'iters': 1,
        'log_dir': 'D:/Workspace/Mortal/mortal/1v3/log',
        'one': {
            'type': 'py',
            'name': 'best',
            'dir': 'D:/Workspace/Mortal/akochan',
            'tactics': 'D:/Workspace/Mortal/akochan/tactics.json',
            'device': 'cuda:0',
            'state_file': 'D:/Workspace/Mortal/mortal/mortal_v4/best.pth',
            'enable_compile': False,
            'enable_amp': True,
            'enable_rule_based_agari_guard': True,
        },
        'three': {
            'type': 'py',
            'name': 'baseline_v1',
            'dir': 'D:/Workspace/Mortal/akochan',
            'tactics': 'D:/Workspace/Mortal/akochan/tactics.json',
            'device': 'cuda:0',
            'state_file': 'D:/Workspace/Mortal/mortal/baseline_v1/mortal.pth',
            'enable_compile': False,
            'enable_amp': True,
            'enable_rule_based_agari_guard': True,
        },
    },
}

sys.modules['config'] = sys.modules[__name__]
