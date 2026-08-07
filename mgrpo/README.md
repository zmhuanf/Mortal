# mgrpo —— Mortal GRPO 日麻 AI

基于 GRPO（Group Relative Policy Optimization）的日麻强化学习训练项目，脱离老代码（`mortal/` 的 IQL+AWR 架构）重新设计。

## 架构

```
mgrpo/
├── prelude.py            # 公共导入（libriichi 路径、日志）
├── config.py             # 全局配置（训练超参待定）
├── env/
│   ├── reward.py         # 奖励原语：终局排名
│   └── simulator.py      # 单局 1v3 rollout → Trajectory
├── data/
│   └── mjson.py          # 人类牌谱加载（BC 预训练）
├── model/
│   └── brain.py        # PolicyNet：406 万参数宽而浅 CNN，无价值头
├── agent/                # 待设计（GRPO 核心、rollout 器、对手池）
├── train_bc.py         # BC 预训练入口（已完成）
├── train.py              # 待设计（训练循环）
└── evaluate.py           # 待设计（benchmark 评估）
```

复用：`libriichi`（Rust 引擎 + Python 绑定，`mortal/libriichi.pyd`）、人类牌谱（`D:/Data/**/*.mjson`）、GRP 权重（`mortal/grp_v2`，可选）。

## 已实现（公共设施）

- **`env/simulator.play_one`**：单局 1v3 对局 → `Trajectory`。关键机制：`py_vs_py(seed_count=1)` 为单线程顺序执行，引擎在 `react_batch` 中记录的 `rollout_log_probs` 与日志解析出的 obs 序列严格对齐，无并行竞态
- **`data/mjson.iter_human_games`**：流式加载人类牌谱，产出 BC 样本（obs/action/mask/rank）
- **`env/reward`**：终局排名奖励原语

## 待讨论的设计点

1. **模型结构（已定）**：输入 `obs_shape(4) = (1012, 34)`，输出 `ACTION_SPACE = 46`。宽而浅 1D CNN：Stem(1012→256) + 12×ConvNeXtBlock(256, k3, FFN×2) + GAP + 512 隐藏 + policy head，**406 万参数、FLOPs ≈ 老模型 1/3，无价值头**（GRPO 无需 critic）
2. **GRPO 组定义**：同局面 G 次采样（成本高）vs batch 内多局归一化（一局一组，排名即组内相对奖励）
3. **奖励（已定）**：`终局排名(90,45,0,-135) + λ×分数差shaping`，λ=1.0（<1.8 保证排名战略主导）。分数差与和牌得分挂钩，驱动立直追和风格；强进攻可后续加小局级 shaping
4. **训练流程**：BC 预训练初始化 → 纯 self-play GRPO；rollout 单机多进程（worker 池）还是跨机
5. **探索与 KL**：GRPO 对 reference 策略的 KL 约束系数、组内 advantage 归一化细节
