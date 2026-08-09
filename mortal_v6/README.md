# mortal_v6

日麻 AI 第六代：**BC 预训练 → XQL 精调 + 事件世界模型 + 想象搜索**。

## 方法概览

- **主干**：ConvNeXt 三阶段 + Transformer 顶层（约 8000 万参数，8GB 显存友好）
- **BC 阶段**：策略 CE（人类动作）+ 事件世界模型监督 + 辅助任务
- **XQL 阶段**：Gumbel 加权 Q 学习（免去 IQL 的 V+Q 两步）+ 优势策略提取 + 事件模型持续监督
- **事件世界模型**：给定 (局面, 动作) 预测未来 10 手的事件类别序列（无事/立直/和牌/放铳/流局/被自摸），把局末回报变成局内可见回报，缓解长链自举
- **想象搜索**：推理时策略 top-k 候选 → 事件 rollout 回报 + XQL Q 混合评分选动作

## 文件结构

```
mortal_v6/
├── config_v6.py      # 全部超参
├── model.py          # Brain + QHead(XQL) + EventModel + AuxNet
├── dataloader.py     # libriichi 数据 + n 步回报 + next 动作 + 事件轨迹标签
├── train.py          # BC + XQL 两阶段训练
├── search.py         # 想象搜索动作选择
├── evaluate.py       # 1v3 评估（search/policy 双模式）
└── README.md
```

## 训练

```bash
cd D:/Workspace/Mortal
python .\mortal_v6\train.py
```

- 阶段控制：`train.stage = 'auto'` 默认跟随 checkpoint 实际阶段，`'bc'`/`'xql'` 强制指定
- 输出统一收在 `out/`：每 500 步保存 checkpoint（out/mortal.pth），每 10000 步 1v3 评估并更新 out/best.pth
- BC 完成后 `auto_proceed` 自动切 XQL
- TensorBoard 按阶段分 run（out/log/bc、out/log/xql），评估回放写入 out/eval_play

## 评估

```bash
python .\mortal_v6\evaluate.py --games 1000                     # 默认 search 模式
python .\mortal_v6\evaluate.py --action-mode greedy --games 1000  # 直出精排（top-k policy + Q）
python .\mortal_v6\evaluate.py --action-mode policy --games 1000  # 纯策略直出（基线）
```

三种决策模式：`search`（想象搜索，最强）、`greedy`（policy top-k + Q 精排，零 rollout 成本）、`policy`（纯策略 argmax，对照）。

## 与 v5 的差异

| | v5 | v6 |
|---|---|---|
| 价值学习 | IQL（expectile V + Huber Q） | XQL（Gumbel 加权直接学 Q） |
| 策略更新 | AWR（Q−V） | 优势加权（Q − 合法动作 Q 均值） |
| 事件感知 | 无 | 事件世界模型（10 手轨迹监督） |
| 推理 | 单模型直出 | 想象搜索（K 候选 + rollout）可选 |
| 参数量 | 1.12 亿 | 约 0.88 亿（8GB 显存余量） |

## 关键超参

- `event.horizon=10`：事件轨迹长度；`event.rewards` 与 reward 段一致，被自摸与放铳同代价
- `xql.tau=0.9`：Gumbel 权重，>0.5 乐观（近似 max）；调低变保守
- `xql.beta=4.0`：策略优势温度，调小增强提升，调大变保守
- `xql.clip=3.0`：优势 e 指数上限，防单样本权重失控
- `eval.search_k=8`、`eval.search_alpha=0.5`：搜索候选数；Q 与 rollout 候选内标准化后按 alpha 混合
- `eval.greedy_top_k=3`：直出精排候选数（policy top-k + Q 选一）
