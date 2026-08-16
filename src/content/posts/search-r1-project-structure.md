---
title: "Search-R1 项目结构：从搜索 Agent 到 PPO/GRPO 更新"
description: "基于官方完整代码，梳理 Search-R1 的目录分层、多轮搜索拓扑、检索服务、奖励函数和 veRL 训练主线。"
publishedAt: 2026-08-16
tags:
  - Search-R1
  - Agentic RL
  - Reinforcement Learning
  - veRL
series: "源码阅读"
seriesPath:
  - LLM
  - Agent RL
  - Search-R1
  - 源码阅读
seriesOrder: 1
draft: false
featured: true
---

Search-R1 不是一个单独的模型实现，而是在 veRL 之上加入多轮搜索 Agent、独立检索服务和规则奖励，训练模型在推理过程中自主决定何时搜索、搜索什么以及何时回答。

本文基于 L20 上的官方完整仓库解释代码结构。当前实验基座对应官方 `main` 的 `598e61b`。

## 训练与搜索拓扑

![Search-R1 训练与搜索拓扑图](/images/search-r1-training-topology.svg)

把整张图压缩成一条主线：

```text
QA 数据
  → Parquet
  → RayPPOTrainer
  → LLMGenerationManager
  → vLLM 生成 <search>
  → HTTP 检索返回 <information>
  → 模型继续推理并生成 <answer>
  → Exact Match 奖励
  → GAE 或 GRPO Advantage
  → PPO 更新 Actor
```

## 顶层目录

```text
Search-R1/
├── train_ppo.sh
├── train_grpo.sh
├── retrieval_launch.sh
├── infer.py
│
├── scripts/
│   └── data_process/
│       ├── nq_search.py
│       ├── nq_rag.py
│       └── ...
│
├── search_r1/
│   ├── llm_agent/
│   │   ├── generation.py
│   │   └── tensor_helper.py
│   └── search/
│       ├── retrieval_server.py
│       ├── retrieval.py
│       ├── index_builder.py
│       ├── rerank_server.py
│       └── google_search_server.py
│
├── verl/
│   ├── trainer/
│   │   ├── main_ppo.py
│   │   ├── ppo/ray_trainer.py
│   │   └── config/ppo_trainer.yaml
│   ├── workers/
│   ├── models/
│   ├── single_controller/
│   └── utils/reward_score/
│
├── docs/
├── example/
└── public/
```

从职责上看，可以把仓库分成四层。

| 层次 | 主要路径 | 作用 |
| --- | --- | --- |
| 实验入口 | `train_ppo.sh`、`train_grpo.sh` | 选择模型、数据、算法、GPU、搜索轮数和检索地址。 |
| Agent 交互 | `search_r1/llm_agent/` | 解析 `<search>` / `<answer>`，执行多轮搜索并维护上下文。 |
| 检索服务 | `search_r1/search/` | 提供独立 `/retrieve` API，支持 Dense、BM25 和在线搜索。 |
| RL 底座 | `verl/` | 提供 Ray Worker、FSDP、vLLM rollout、PPO/GRPO、日志和 checkpoint。 |

## 实验入口

`train_ppo.sh` 和 `train_grpo.sh` 最终都执行：

```bash
python3 -m verl.trainer.main_ppo
```

区别主要在 Advantage 的计算方式：

| 模式 | `adv_estimator` | Critic | 每题轨迹 |
| --- | --- | --- | --- |
| PPO | `gae` | 需要 | 默认一条 |
| GRPO | `grpo` | 不需要 | 示例配置为 `n_agent=5` |

训练脚本通过 Hydra 覆盖 `ppo_trainer.yaml`。官方示例把检索地址设为 `http://127.0.0.1:8000/retrieve`、`topk=3`、`max_turns=2`。

当前官方 `train_grpo.sh` 定义了 `DATA_DIR`，但参数中使用 `TRAIN_DATA_DIR` 和 `TEST_DATA_DIR`。直接运行前需要导出这两个变量，或者改成 `$DATA_DIR/train.parquet` 与 `$DATA_DIR/test.parquet`。

## 数据如何进入训练

`scripts/data_process/nq_search.py` 将原始 NQ 数据转换为 Parquet。每条样本包含：

```python
{
    "data_source": "nq",
    "prompt": [{"role": "user", "content": question_with_instructions}],
    "ability": "fact-reasoning",
    "reward_model": {
        "style": "rule",
        "ground_truth": {"target": golden_answers}
    },
    "extra_info": {"split": split, "index": idx}
}
```

Prompt 规定模型通过两种标签行动：

```text
<search>检索问题</search>
<answer>最终答案</answer>
```

环境返回的信息统一包装为：

```text
<information>
Doc 1 ...
Doc 2 ...
Doc 3 ...
</information>
```

## Agent 循环

Search-R1 最关键的自定义类是 `search_r1/llm_agent/generation.py` 中的 `LLMGenerationManager`。

它执行以下循环：

1. 调用 ActorRollout Worker，让 vLLM 生成一段响应。
2. 在 `</search>` 或 `</answer>` 处截断。
3. 用正则解析第一个完整的 `search` 或 `answer` 动作。
4. 如果是搜索，批量调用 `/retrieve`，把 Top-K 文档拼回上下文。
5. 如果是回答，将轨迹标记为结束。
6. 如果格式非法，返回格式纠错提示并继续，但仍会消耗一个 turn。
7. 达到 `max_turns` 后再做一次最终生成，但不再执行新的搜索。

所以这里的环境动作空间非常小：

```text
search(query)
answer(answer)
```

它已经具备 Agent 的观察、动作和终止条件，但还不是面向任意工具的通用 Agent Runtime。

## 为什么需要 TensorHelper

一个 batch 中，不同问题可能在不同轮次结束。`tensor_helper.py` 负责：

- 对已结束样本补 padding；
- 重新生成 attention mask 和 position ids；
- 拼接 prompt、模型响应和搜索结果；
- 截断到 `max_prompt_length`；
- 生成 `info_mask`。

`info_mask` 很重要。检索结果是环境 observation，不是模型生成的动作，因此 Actor 更新时会屏蔽 `<information>` 中的 token：

```text
模型生成 token      → 参与策略损失
检索结果 token      → 不参与策略损失
```

## 独立检索服务

`retrieval_launch.sh` 启动 `search_r1/search/retrieval_server.py`，对外暴露：

```text
POST /retrieve
```

请求包含查询列表、`topk` 和是否返回分数。服务可以接入：

- E5、BGE 等 Dense Encoder；
- FAISS Flat GPU 或 ANN CPU 索引；
- BM25 / Pyserini；
- Google、Bing、Brave、SERP 等在线搜索；
- 可选 Reranker。

检索与 RL 训练通过 HTTP 解耦，所以替换搜索引擎时，主要保持 `/retrieve` 协议兼容即可。

## veRL 训练底座

`verl/trainer/main_ppo.py` 初始化 Ray，并创建不同角色的 Worker：

```text
ActorRollout  = FSDP Actor + vLLM rollout
RefPolicy     = 参考策略，用于 KL
Critic        = PPO/GAE 的 value 模型
RewardManager = 规则奖励
```

`RayPPOTrainer.fit()` 将 Agent 轨迹接回标准 RL 数据流：计算旧策略 log probability、参考策略 KL、value、reward 和 advantage，最后更新 Actor。

GRPO 模式不创建 Critic，而是给同一道题采样多条轨迹，通过组内相对奖励计算 Advantage。

## 奖励函数

奖励入口位于 `verl/trainer/main_ppo.py` 和 `verl/utils/reward_score/qa_em.py`。

`RewardManager` 解码完整轨迹，提取最后一个 `<answer>...</answer>`，经过大小写、标点、冠词和空格归一化后执行 Exact Match：

```text
答案正确 → 1
答案错误 → 0
```

奖励只放在响应最后一个有效 token 上。这是一种稀疏结果奖励：它能告诉模型整条轨迹最终是否正确，但不能直接指出哪次搜索最有价值。

源码要求轨迹里至少出现两个 `<answer>`。原因是 Prompt 示例本身包含 `<answer> Beijing </answer>`，模型的最终回答是最后一个 `<answer>`。

## 推荐阅读顺序

如果要继续做代码级实验，建议按真实执行链阅读：

1. `train_grpo.sh`：确认实验参数。
2. `scripts/data_process/nq_search.py`：理解训练样本。
3. `search_r1/llm_agent/generation.py`：理解 Agent 环境循环。
4. `verl/trainer/main_ppo.py`：理解 Worker 初始化和奖励入口。
5. `verl/trainer/ppo/ray_trainer.py`：理解 rollout 到更新的数据流。
6. `verl/utils/reward_score/qa_em.py`：理解最终奖励。
7. `search_r1/search/retrieval_server.py`：理解检索服务。

一句话总结：

> Search-R1 = veRL 分布式 PPO/GRPO 底座 + 两动作搜索 Agent 循环 + 独立 HTTP 检索服务 + 最终答案规则奖励。

## 项目链接

- [Search-R1 官方仓库](https://github.com/PeterGriffinJin/Search-R1)
- [我们的 Search-R1 实验仓库](https://github.com/Qi18/Search-R1/tree/experiment/l20-search-r1-20260816)
