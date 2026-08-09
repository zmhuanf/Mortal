# mortal_v5

日麻 AI 第五代离线训练方案：ConvNeXt 三阶段 + Transformer 顶层混合架构，约 1.1 亿参数。

## 架构

```
输入 (N, 1012, 34)                ← libriichi v4 观测（不可变）
├─ stem: Conv1d(1012→192, k=3) + GELU
├─ S1:  8 × ConvNeXtBlock(192)
├─ T1:  Conv1d(192→384, k=1)
├─ S2:  12 × ConvNeXtBlock(384)
├─ T2:  Conv1d(384→768, k=1)
├─ S3:  10 × ConvNeXtBlock(768)
├─ A:   6 × TransformerBlock(768, 12 heads, pre-norm, 可学习位置编码)
├─ neck: Conv1d(768→64) + Flatten + Linear(2176→2048)   ← phi
├─ policy_head: Linear(2048→46)
├─ DQN:  Linear(2048→1024) + GELU + Linear(1024→K×47)，K=5 集成 dueling
└─ AuxNet: Linear(2048→512) + GELU + Linear(512→25)
```

- 局部牌型模式靠卷积归纳偏置，全局局面交互（安全牌/立直读牌/点差攻防）靠自注意力
- 34 位置（牌型）不降采样，drop_path 0.1 贯穿防过拟合
- 奖励：局实际得分差分 ÷1000 + turn-level shaping（立直/和牌/放铳），无 GRP 模型依赖
- bf16 AMP 训练（Blackwell 原生优势）

## 训练

```bash
python train.py
```

两阶段自动衔接（config_v5.py → train 段）：

| 阶段 | 损失 | 步数 |
|---|---|---|
| BC | 策略 CE + 辅助任务 | 60 万 |
| IQL | expectile V + Huber Q + AWR 策略 + 辅助任务 | 40 万 |

- 中断后可续训（state_file 记录阶段与步数，`train.stage = 'auto'` 默认跟随 checkpoint 实际阶段自动续训）
- BC 阶段定期 1v3 评估，最优结果存 best.pth，BC 完成后从 best 自动切 IQL
- 手动跳过 BC：改 `train.stage = 'iql'`，将从 best.pth（或 BC checkpoint）起步
- 强制重训 BC：改 `train.stage = 'bc'`（checkpoint 为 iql 阶段时从 best 重训）

## 评估

```bash
python evaluate.py [--state-file mortal.pth] [--games 400]
```

challenger 为 v5 模型，champion 为 config 中 opponents 列表（支持 v4/v5 任意 checkpoint）。

## 配置要点（config_v5.py）

- `control.batch_size`：默认 256（8GB 显存预算），可配合 `opt_step_every` 梯度累积
- `model` 段：depths/widths/attn_layers 可整体缩放模型规模
- `train` 段：阶段、步数、学习率、warmup
- `eval` 段：评估局数、频率、对手列表
- `dataset.globs`：人类牌谱路径

## 目录产物

```
mortal_v5/
├── mortal.pth      训练进度 checkpoint（含阶段/步数，可续训）
├── best.pth        历次 1v3 评估最优
├── file_index.pth  牌谱文件列表缓存
├── log/            TensorBoard 日志
└── eval_play/      评估对局日志
```
