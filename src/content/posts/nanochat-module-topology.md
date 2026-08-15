---
title: "NanoChat 源码阅读：一张图看懂 scripts / tasks / nanochat 的依赖拓扑"
description: "扫描仓库内所有 Python 文件的真实 import 语句，画出 scripts 入口、tasks 评测任务与 nanochat 核心库的三层依赖拓扑，理解整个项目的模块边界。"
publishedAt: "2026-08-15"
updatedAt: "2026-08-15"
tags:
  - "NanoChat"
  - "LLM"
  - "源码阅读"
series: "源码阅读"
seriesPath:
  - "LLM"
  - "实验"
  - "NanoChat"
  - "源码阅读"
seriesOrder: 5
draft: false
featured: false
---

> 源码版本：[`183f873`](https://github.com/Qi18/nanochat/tree/183f8738e0a502071c5b4ba31f641bb167d2a5c5)
> 数据来源：扫描仓库内全部 `.py` 文件的 `import` 语句（不含 tests 与 experiments）

读 NanoChat 源码时，单看某个文件容易迷路。把所有内部 `import` 关系画出来之后，会发现整个项目是一个非常干净的三层结构：

![nanochat 模块依赖拓扑图](/images/nanochat-import-topology.svg)

## 三层结构

**第 1 层：`scripts/` 入口脚本**。9 个脚本对应训练 pipeline 的各个阶段，全部可以直接运行：

- 训练链：`tok_train` → `base_train` → `chat_sft` → `chat_rl`
- 评测 / 推理链：`tok_eval`、`base_eval`、`chat_eval`、`chat_cli`、`infer_bench`

**第 2 层：`tasks/` 评测任务**。只被 chat 系列脚本使用。`mmlu` / `arc` / `gsm8k` / `smoltalk` / `humaneval` 全部基于 `tasks/common.py` 的 `Task` 基类；其中 `humaneval` 额外依赖 `nanochat/execution.py`，用于在受限环境里执行模型生成的代码。

**第 3 层：`nanochat/` 核心库**，内部又分两级：

- 上层组件：`engine`（推理引擎）、`checkpoint_manager`、`dataloader`、`loss_eval`、`core_eval`、`experiment_tracking`
- 底层基础：`gpt`（模型本体，依赖 `optim` 和 `flash_attention`）、`tokenizer`、`dataset`、`fp8`，以及全局汇聚点 `common.py`——几乎所有模块都依赖它（分布式初始化、路径管理、计算 dtype 等）

## 每个文件的职责

### `scripts/` 入口脚本（按 pipeline 顺序）

| 文件 | 作用 |
|---|---|
| `tok_train.py` | 用自研 rustbpe 库训练 GPT-4 风格的 BPE 分词器 |
| `tok_eval.py` | 评测分词器的压缩率（每 token 平均字节数），并与 GPT-2/GPT-4 分词器对比 |
| `base_train.py` | **预训练总控**：分布式初始化 → 构建 GPT → 缩放律推算规模 → 训练循环 → 中途评测 → 存 checkpoint，支持单卡和 `torchrun` 多卡 |
| `base_eval.py` | 基座模型统一评测，三种模式：`core`（DCLM CORE 指标）、`bpb`（验证集 bits-per-byte）、`sample`（采样看生成质量） |
| `chat_sft.py` | 监督微调：用 SmolTalk + MMLU + GSM8K 混合任务（`TaskMixture`）把基座模型调成对话模型 |
| `chat_eval.py` | 对话模型评测：跑 MMLU / ARC / GSM8K / HumanEval 四个任务 |
| `chat_rl.py` | 强化学习阶段：在 GSM8K 上做简化版 GRPO（实际接近 REINFORCE） |
| `chat_cli.py` | 命令行交互聊天，加载 checkpoint 后单卡推理 |
| `infer_bench.py` | 推理性能基准：扫不同 decode batch size，测延迟、吞吐、显存和带宽利用率 |

### `tasks/` 评测 / 训练任务

| 文件 | 作用 |
|---|---|
| `common.py` | `Task` 基类（任务 = 对话数据集 + 评估标准）+ `TaskMixture` 多任务混合器 + HuggingFace 数据集加载 |
| `mmlu.py` | MMLU 多选知识题（57 个学科） |
| `arc.py` | ARC 科学推理多选题（Easy/Challenge） |
| `gsm8k.py` | GSM8K 小学数学应用题，也是 RL 阶段的奖励来源 |
| `humaneval.py` | HumanEval 代码生成基准，生成的代码交给 `execution.py` 沙箱运行验证 |
| `smoltalk.py` | SmolTalk 通用对话数据集（smol 版），SFT 的主力语料 |

### `nanochat/` 核心库 —— 上层组件

| 文件 | 作用 |
|---|---|
| `engine.py` | 推理引擎：KV cache、批量生成、工具调用支持，一切围绕 token 序列 |
| `checkpoint_manager.py` | checkpoint 存取：模型权重、优化器状态、训练进度的保存与恢复 |
| `dataloader.py` | 分布式预训练数据加载器，BOS 对齐的 bestfit 打包（每行以 BOS 开头，减少截断浪费） |
| `loss_eval.py` | 基座模型 loss 评测，核心是 `evaluate_bpb`（bits-per-byte，与分词器无关的指标） |
| `core_eval.py` | DCLM 论文的 CORE 指标实现（一组 ICL 任务的加权准确率） |
| `experiment_tracking.py` | 可选的 SwanLab 实验记录，兼容 W&B 风格的日志调用 |
| `execution.py` | 沙箱执行 LLM 生成的 Python 代码（仿 OpenAI human-eval），供 HumanEval 用 |

### `nanochat/` 核心库 —— 底层基础

| 文件 | 作用 |
|---|---|
| `gpt.py` | GPT 模型本体：RoPE、QK norm、ReLU² MLP、无位置嵌入的简化 Transformer |
| `optim.py` | `MuonAdamW` 混合优化器：矩阵参数走 Muon，嵌入和标量走 AdamW，单卡/分布式同一套代码 |
| `flash_attention.py` | 统一注意力接口：兼容 FA3 API，在不支持的 GPU/MPS/CPU 上自动回退到 SDPA |
| `fp8.py` | 极简 FP8 训练（~150 行替代 torchao 的 Float8Linear），只做 tensorwise 动态缩放 |
| `tokenizer.py` | BPE 分词器封装：训练用 rustbpe，推理用 tiktoken，管理特殊 token 和对话模板 |
| `dataset.py` | 预训练数据集（parquet 文件）的下载、列举和文档迭代 |
| `common.py` | 全局工具箱：分布式初始化/清理、设备探测、路径管理、计算 dtype、日志——被几乎所有模块依赖 |

一句话总结：`common.py` 是地基，`gpt.py + optim.py + dataloader.py` 支撑训练，`engine.py + checkpoint_manager.py` 支撑推理，`tasks/` 提供评测语料，`scripts/` 把它们按 tok → base → sft → rl 的顺序串成完整 pipeline。

## 几个值得注意的细节

**脚本之间也有复用**。图里第一行内部有两条弧线：`base_train` 导入 `base_eval.evaluate_core`，`chat_sft` 导入 `chat_eval.run_chat_eval`——训练脚本在训练中途直接调用评测脚本的函数跑 CORE / chat 评测，而不是另起进程。

**依赖方向严格单向**：`scripts → tasks → nanochat`，`nanochat` 内部也没有循环依赖。唯一一处 `tasks/common → tasks/mmlu` 是函数内的延迟导入（`TaskMixture` 按名字实例化任务），不构成真正的循环。

**`base_train` 是依赖最重的入口**（11 个内部模块），因为它要串起模型构建、数据加载、优化器、评测、checkpoint、实验记录的完整闭环；`tok_eval` 最轻，只用到 `tokenizer` 和 `dataset`。

**推理侧的核心链路**是 `engine → checkpoint_manager → gpt / tokenizer`：所有需要加载模型做推理的脚本（`chat_eval`、`chat_cli`、`chat_rl`、`infer_bench`）都走这一条链，这也是为什么改 `checkpoint_manager` 的接口会牵动半个仓库。

读新模块之前先看一眼这张图，就能知道它在整个依赖链里的位置，以及改动它会影响到谁。
