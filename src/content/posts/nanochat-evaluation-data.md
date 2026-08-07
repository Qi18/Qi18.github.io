---
title: "NanoChat 评测数据：Base、SFT 与 GSM8K RL 对比"
description: "记录 1.384B NanoChat 在 8×L20 上从预训练、SFT 到 GSM8K 强化学习的真实评测结果，并说明 BPB、CORE、ChatCORE 等指标分别衡量什么。"
publishedAt: 2026-08-07
tags:
  - NanoChat
  - LLM
  - 评测
  - 强化学习
series: "NanoChat 源码学习"
seriesOrder: 2
draft: false
featured: true
---

这篇文章记录一次 NanoChat d24 完整训练的评测数据。实验依次完成 Base Model 预训练、SFT、GSM8K 强化学习和全量任务评测。和只看训练 loss 相比，把多类指标放在一起，才能判断模型究竟提升了什么，又牺牲了什么。

上一篇：[读懂 NanoChat：先找到真实的训练主线](/posts/nanochat-source-reading/)

## 实验配置

| 项目 | 配置 |
| --- | --- |
| GPU | 8×NVIDIA L20，每卡约 46 GiB |
| 并行方式 | 8 卡 DDP |
| 计算精度 | BF16 |
| 模型 | d24，24 层，隐藏维度 1536，12 个 Attention Head |
| 参数量 | 1,384,122,122（约 1.384B） |
| 上下文长度 | 2048 |
| 词表大小 | 32,768 |
| 训练数据 | ClimbMix 171 个 shard，约 15 GB |

## NanoChat 中需要一起看的指标

| 指标 | 回答的问题 | 判断方向 |
| --- | --- | --- |
| Raw loss | 当前训练目标是否继续下降 | 越低越好 |
| BPB | 模型预测验证文本的能力 | 越低越好 |
| CORE | Base Model 的知识、推理和上下文学习能力 | 越高越好 |
| ChatCORE | SFT/RL 模型在问答、数学和代码等任务上的综合表现 | 越高越好 |
| 任务 Accuracy | 某项具体能力是否提升 | 越高越好 |
| Reward、Pass@k | RL rollout 获得正确答案的比例 | 越高越好 |
| token/s、MFU | 训练系统的吞吐和硬件利用率 | 用于工程诊断 |

BPB 是 **bits per byte（每字节比特数）**，计算方式为：

```text
bpb = 所有 Token 的负对数似然 / (文本的 UTF-8 字节数 × ln 2)
```

分子来自模型对目标 Token 的预测概率。除以 `ln 2`，是把自然对数计算的损失从 `nat` 转换成 `bit`；再除以 UTF-8 字节数，则把不同长度文本归一化到“每个字节平均需要多少 bit”。因此 `0.80 bpb` 优于 `0.90 bpb`。

BPB 比每 Token Loss 更少受到 Tokenizer 切分粒度影响，但它只衡量语言建模能力，不代表问答、数学或代码任务一定更强，所以还需要结合 CORE、ChatCORE 和具体任务 Accuracy。

## Base Model 结果

| 指标 | 结果 |
| --- | ---: |
| 训练步数 | 5,568 |
| 总训练时间 | 749.71 分钟（约 12 小时 30 分） |
| 最终 raw loss | 2.323202 |
| 末段吞吐 | 约 129,511 token/s |
| Train BPB | 0.715094 |
| Validation BPB | 0.713085 |
| CORE metric | 0.2541 |

这里不能直接比较 `raw loss = 2.323202` 和 `Validation BPB = 0.713085`：前者通常按 Token 统计交叉熵，后者按原始文本字节数归一化，统计口径不同。

## SFT 结果

SFT 从 Base checkpoint 继续训练，训练混合数据共 789,759 行，并对 MMLU 和 GSM8K 数据加权。

| 指标 | 结果 |
| --- | ---: |
| 训练步数 | 466 |
| 总训练时间 | 60.97 分钟 |
| 最低 Validation BPB | 0.2727 |
| 训练期最终 ChatCORE | 0.2386 |
| ChatCORE categorical | 0.3421 |

SFT 的 BPB 和 Base Model BPB 来自不同的数据分布与训练目标，不能用 `0.2727 < 0.713085` 直接推出 SFT 让所有通用能力都大幅提升。判断聊天模型仍然要看后面的全量任务评测。

## GSM8K RL 训练结果

强化学习从 SFT checkpoint 开始，只针对 GSM8K 数学任务训练，共完成 467 步。

| 指标 | 结果 |
| --- | ---: |
| 最终 step 平均 reward | 0.4141 |
| 最终平均序列长度 | 139.96 |
| step 240 Pass@1 | 16.75% |
| step 240 Pass@8 | 34.00% |
| step 420 Pass@1 | 16.50% |
| step 420 Pass@8 | 32.50% |

Pass@1 表示只生成一次时答对的比例；Pass@8 表示生成 8 个候选答案时，至少有一个正确答案的比例。两者差距较大，说明模型已经能偶尔找到正确推理路径，但单次生成还不稳定。

## SFT 与 RL 全量评测

| 指标 | SFT | RL | RL - SFT |
| --- | ---: | ---: | ---: |
| ARC-Easy | 64.27% | 64.27% | 0.00 pp |
| ARC-Challenge | 50.94% | 50.68% | -0.26 pp |
| MMLU | 37.07% | 36.62% | -0.45 pp |
| GSM8K | 1.90% | **18.65%** | **+16.75 pp** |
| HumanEval | 12.80% | 6.71% | **-6.09 pp** |
| ChatCORE | 0.2355 | **0.2549** | **+0.0194（约 +8.2%）** |

最明显的变化是 GSM8K 从 `1.90%` 提升到 `18.65%`，证明数学专项 RL 有效；但 HumanEval 从 `12.80%` 降到 `6.71%`，MMLU 和 ARC-Challenge 也略有下降。这说明总体 ChatCORE 上升并不代表所有能力都上升：模型在获得数学专项能力的同时，出现了能力迁移和遗忘。

## 如何解读这组数据

1. **BPB 用于持续观察语言建模质量**，适合比较预训练 checkpoint，但不能代替下游任务评测。
2. **CORE 更适合 Base Model**，用于观察预训练是否形成知识、推理和上下文学习能力。
3. **ChatCORE 是综合分**，必须展开到 ARC、MMLU、GSM8K、HumanEval 才能发现能力增减发生在哪里。
4. **单任务 RL 会改变能力分布**。本次 RL checkpoint 更适合作为数学专项模型，不能直接视为更强的通用聊天模型。
5. **一次回答正确不等于能力稳定**。最终 CLI 冒烟测试中，模型虽然能解释天空为什么是蓝色，也能生成正确的阶乘代码，却把 `37×48` 错答为 `192`；这与 GSM8K 只有 `18.65%` 的结果一致。

结论不是“RL 后模型整体更强”，而是：**数学任务显著改善，综合分有所上升，但代码和部分通用能力发生回退。** 评测的价值正是在一个总分之外，把这种变化拆出来。
