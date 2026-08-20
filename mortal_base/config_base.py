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
        'enable_amp': True,  # 5050(Blackwell sm_120) 支持 bf16，统一 bfloat16
        # 换回 1660S(Turing sm_75) 仅需改 amp_dtype='float16'：bf16 不支持，
        # GradScaler 按 dtype 自动启用，batch/lr 与显卡无关无需动
        'amp_dtype': 'bfloat16',
        'enable_compile': False,
        'batch_size': 256,
        'save_every': 500,
        'state_file': os.path.join(OUT, 'mortal.pth'),
        'best_state_file': os.path.join(OUT, 'best.pth'),
        'tensorboard_dir': os.path.join(OUT, 'log'),
    },
    'env': {
        'pts': [10.0, 4.0, -1.0, -5.0],
        'gamma': 0.99,
        'n_step': 12,  # 窗口覆盖更多结算与事件，让 Q 目标含动作分辨率（原 3 几乎无信号）
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
        'a_reg_weight': 0.2,  # A 头事件监督权重（MC 目标高方差，下调防压扁 A 分支）
    },
    'iql': {
        'tau': 0.5,
        'beta': 1.5,  # AWR 温度：σ(adv)=0.53 的理论值过激进导致权重爆炸，提至 1.5 保稳
        'beta_end': 2.0,  # 局末锚定 turn = σ(adv_end)=1.93（理论值）
        'clip': 20.0,  # exp_adv 权重上限：基础版与 baseline_v1 对齐，撤销收窄实验
        'ema_decay': 0.995,
        'policy_mode': 'bc',  # bc=audit 期 AWR 权重固定 1（Q 隔离舱），awr=恢复 Q 注入策略
    },
    'event': {
        'horizon': 14,        # 事件轨迹预测长度（步），放宽视野覆盖远期和牌回报
        'n_types': 6,         # 无事/立直/和牌/放铳/流局/被自摸
        'dim': 256,
        'heads': 4,
        'layers': 1,
        'rewards': [0.0, 0.79, 2.26, -1.77, 0.0, -0.74],  # 与 reward 段完全同口径（统计均值，流局为 0）
        'weight': 0.3,        # 事件监督 CE 权重（搜索依赖事件头，0.1 喂不动）
    },
    'reward': {
        # 统计均值标定（100 文件/2952 局），与 event.rewards 完全同口径
        'riichi': 0.79,
        'agari': 2.26,
        'houjuu': -1.77,
        'ryukyoku': 0.0,   # 流局无排名变化（统计均值 -0.03≈0）
        'tsumogiri': -0.74,  # 被自摸（损失全场分摊，代价轻于放铳）
    },
    'grp': {
        'state_file': 'D:/Workspace/Mortal/mortal/grp_v2/grp_best.pth',
        'network': {'hidden_size': 128, 'num_layers': 2, 'nhead': 4},
    },
    'dataset': {
        'globs': ['D:/Data/**/*.mjson', 'D:/Data2/**/*.mjson'],
        'file_index': os.path.join(OUT, 'file_index.pth'),
        'file_batch_size': 1,  # v7 平滑参数：突发粒度=单文件，worker 交错产出，GPU 不饥饿
        'reserve_ratio': 0.0,
        'num_workers': 4,  # v7 平滑参数：吞吐留富余，吸收解析方差
        'player_names_files': [],
        'prefetch_factor': 4,  # 缓冲 = workers×2 = 8 batch，控制共享内存驻留防 Windows 页面文件 1455
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
        'scheduler': {
            'peak': 3e-5,
            'final': 3e-5,
            'init': 1e-8,
            'warm_up_steps': 5000,
            'max_steps': 5000,
        },
    },
    'train': {
        'max_steps': 600000,
        'post_training': True,  # 后训练：随机读文件 + 无步数上限，Ctrl+C 手动停止
        'bc_only': False,       # 纯 BC：跳过全部 Q/价值计算（含 target 前向），只跑 policy+event+aux 监督，提速约 1/3
        'eval_every': 5000,   # 每 N 步 1v3 评估 vs baseline_v1
        'eval_games': 100,    # 单次评估局数，评估耗时约 2-3 分钟
    },
    'eval': {
        'games': 1000,
        'segment_seeds': 250,
        'log_dir': os.path.join(OUT, 'eval_play'),
        # 推理动作源：policy=纯策略直出（audit 期默认）/ search=策略 top-k + 事件 rollout 评分
        #             vrisk=策略直出 + V 风险温度调节（领先压激进，落后抬激进）
        'action_mode': 'policy',
        'search': {
            'k': 8,           # 候选数（全展开后不用）
            'alpha': 0.0,     # Q 混合权重，audit 期 0 = 纯事件 rollout
            'gamma': 0.99,
        },
        'vrisk': {
            'gain': 2.5,      # 风险调节强度（拉大验证：信息冗余 vs 过弱）
            'v_mid': 1.6,     # V 基准线：高于此=领先（压制激进），低于此=落后（鼓励激进）
            'w_riichi': 1.5,  # 立直(37)改写系数
            'w_agari': 2.0,   # 和牌(43)改写系数
        },
        'opponents': [
            {'name': 'baseline_v1', 'state_file': 'D:/Workspace/Mortal/mortal/baseline_v1/mortal.pth'},
        ],
    },
}

os.makedirs(OUT, exist_ok=True)
sys.modules['config'] = sys.modules[__name__]
