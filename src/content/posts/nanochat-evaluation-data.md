---
title: "NanoChat 评测指标说明"
description: "从 Tokenizer、Base Model、Chat Model 到 GSM8K RL，解释 NanoChat 每个评测指标衡量什么、如何计算、应该怎么看。"
publishedAt: 2026-08-07
updatedAt: 2026-08-07
tags:
  - NanoChat
  - LLM
  - 评测
series: "源码阅读"
seriesPath:
  - "LLM"
  - "Train"
  - "NanoChat"
  - "源码阅读"
seriesOrder: 2
draft: false
featured: true
---

NanoChat 不只有一个“模型分数”。Tokenizer、Base Model、Chat Model 和 RL 使用的评测对象不同，输出的指标也回答不同问题。理解这些指标，关键不是背数值，而是先确认：**它在测什么、如何判分、基线是什么，以及数值能不能跨配置比较。**

## 指标总览

| 阶段 | 指标 | 主要回答的问题 | 方向 |
| --- | --- | --- | --- |
| Tokenizer | Compression Ratio | 一个 Token 平均覆盖多少原始字节 | 越高越好 |
| Base Model | Train / Validation BPB | 模型预测文本的能力如何 | 越低越好 |
| Base Model | Accuracy / Centered Score | 单项 ICL 任务表现如何 | 越高越好 |
| Base Model | CORE | Base Model 的综合 ICL 能力如何 | 越高越好 |
| Chat Model | ARC / MMLU Accuracy | 知识和选择题能力如何 | 越高越好 |
| Chat Model | GSM8K Accuracy | 数学推理最终答案是否正确 | 越高越好 |
| Chat Model | HumanEval Accuracy | 生成代码能否通过测试 | 越高越好 |
| Chat Model | SpellingBee Accuracy | 字母拼写和计数是否正确 | 越高越好 |
| Chat Model | ChatCORE | Chat Model 的综合任务能力如何 | 越高越好 |
| GSM8K RL | Average Reward | 当前 rollout 中答对了多少 | 越高越好 |
| GSM8K RL | Pass@k | 生成 k 次时至少答对一次的概率 | 越高越好 |

## Tokenizer 评测

### Vocabulary Size

Vocabulary Size 是 Tokenizer 中可使用的 Token 总数。词表更大通常能把常见片段合并成更长的 Token，但会增加 embedding 和输出层的参数量。

它不是“越大越好”的分数。比较 Tokenizer 时，需要同时考虑压缩率、模型参数开销、多语言表现和实际训练数据分布。

### Compression Ratio

NanoChat 的 `tok_eval.py` 用下面的方式衡量 Tokenizer 压缩率：

```text
compression ratio = UTF-8 字节数 / Token 数
```

它表示一个 Token 平均覆盖多少原始字节，因此数值越高，说明同一段文本需要的 Token 越少。例如相同文本有 1,000 bytes：

- 编码成 500 Tokens，压缩率为 `2.0 bytes/token`；
- 编码成 400 Tokens，压缩率为 `2.5 bytes/token`，压缩效果更好。

NanoChat 会分别在新闻、韩文、代码、数学、科学文本和训练/验证语料上比较自己的 Tokenizer 与 GPT-2、GPT-4 Tokenizer。不能只看一种英文文本，因为不同语言和代码的 UTF-8 分布差异很大。

### Relative Diff

相对差异按 Token 数计算：

```text
relative diff = (基线 Token 数 - 当前 Token 数) / 基线 Token 数
```

结果为正，表示当前 Tokenizer 使用的 Token 更少；结果为负，表示比基线需要更多 Token。比较时必须使用完全相同的原始文本。

## Base Model 评测

`base_eval.py` 默认包含三部分：BPB、CORE 和文本采样。BPB 衡量语言建模质量，CORE 衡量任务能力，采样用于人工观察生成质量。

### Train BPB 与 Validation BPB

BPB 是 **bits per byte（每字节比特数）**：

```text
bpb = 有效目标 Token 的负对数似然总和 / (有效目标的 UTF-8 字节数 × ln 2)
```

其中：

- 特殊 Token，例如 `<|bos|>`，不计入；
- 被 `ignore_index` mask 的目标不计入；
- 除以 `ln 2`，将自然对数单位 `nat` 转换成 `bit`；
- 按字节归一化，降低不同词表和 Tokenizer 切分粒度带来的影响。

BPB 越低越好。`Train BPB` 衡量训练分布上的拟合程度；`Validation BPB` 使用未参与训练的验证文本，更能反映泛化能力。通常优先关注 Validation BPB，同时观察 Train 与 Validation 的差距是否持续扩大。

BPB 不是问答正确率。BPB 下降说明模型更善于预测文本，但不保证数学、知识问答或代码能力同步提升。

### Accuracy

CORE 中每个子任务首先计算 Accuracy：

```text
accuracy = 答对样本数 / 总样本数
```

但不同任务的随机猜测基线不同。例如四选一任务随机猜测约为 25%，二选一约为 50%。直接平均原始 Accuracy，会让不同任务无法公平汇总。

### Centered Score

NanoChat 会先扣除任务的随机基线，再归一化：

```text
centered score = (accuracy - random baseline) / (1 - random baseline)
```

这样：

- `0` 表示与随机猜测相当；
- `1` 表示全部答对；
- 小于 `0` 表示低于随机基线。

Centered Score 的作用是让不同选择数量、不同随机基线的任务可以放在一起平均。

### CORE

CORE 来自 DCLM 的 Base Model 评测协议，覆盖常识、阅读理解、知识问答、符号操作和上下文学习等多类任务。NanoChat 的 CORE 为所有子任务 Centered Score 的平均值：

```text
CORE = 所有 CORE 子任务 centered score 的平均值
```

CORE 越高，说明 Base Model 的综合 ICL（In-Context Learning）能力越强。比较 CORE 时必须保持评测任务版本、few-shot 设置和 `max_per_task` 一致；只抽少量样本得到的是快速近似值，波动会明显大于全量评测。

### Sample

`base_eval.py` 还会输出有提示和无条件生成样本。Sample 不是自动化数值指标，主要用于人工检查：

- 文本是否连贯；
- 是否出现严重重复；
- 是否能遵循简单提示；
- 是否存在明显格式异常。

它直观但主观，不能代替 BPB 和 CORE。

## Chat Model 评测

`chat_eval.py` 同时支持选择题评测和生成式评测。默认任务包括 ARC-Easy、ARC-Challenge、MMLU、GSM8K、HumanEval 和 SpellingBee。

### ARC-Easy 与 ARC-Challenge

ARC 是科学问答选择题：

- `ARC-Easy` 题目相对基础；
- `ARC-Challenge` 包含更难、通常不能仅靠简单词面匹配回答的问题。

NanoChat 不让模型自由生成整段回答，而是在答案位置只比较可选字母的 logits，选择概率最高的字母。指标是 Accuracy，四选一随机基线按 25% 计算。

### MMLU

MMLU 是覆盖数学、物理、计算机、医学、法律、人文社科等 57 个学科的四选一知识测试。NanoChat 同样比较答案字母的 logits，并计算 Accuracy。

MMLU 主要反映知识覆盖和部分推理能力，但分数会受到训练数据污染、学科分布以及提示格式影响。四选一随机基线为 25%。

### GSM8K

GSM8K 是小学数学文字题。NanoChat 让模型生成完整解题过程，再从 `####` 后提取最终数字，与标准答案比较。

```text
GSM8K accuracy = 最终数字正确的题数 / 总题数
```

它只根据最终答案判分：推理文字写得漂亮但数字错误仍算错；中间推理不完整但最终数字正确仍会通过。因此它适合程序化奖励，却不能单独评价推理过程质量。

### HumanEval

HumanEval 衡量 Python 代码生成能力。NanoChat 提取模型生成的程序，与题目测试代码组合并实际执行；只有通过测试才算成功。

默认单次采样时，它相当于 `pass@1`。这个指标关心功能正确性，不关心代码是否与参考答案文字相似。结果会受到采样次数、temperature、最大生成长度以及执行超时设置影响。

### SpellingBee

SpellingBee 测试模型能否拼出单词并统计指定字母出现次数，例如“`strawberry` 中有几个 `r`”。评测从 `####` 后提取数字，与程序生成的正确计数比较。

它专门检查 Tokenizer 容易掩盖的字符级能力：模型内部处理的是 Token，并不天然看到单词由哪些字母组成。

### Generative Accuracy 与 num_samples

GSM8K、HumanEval 和 SpellingBee 属于生成式任务。对每道题，NanoChat 会生成 `num_samples` 个回答，只要其中任意一个通过就把该题计为成功。

因此：

- `num_samples=1` 时，结果相当于 Pass@1；
- `num_samples>1` 时，日志虽然仍叫 Accuracy，实际含义更接近 Pass@num_samples。

比较两个模型时必须保持 `num_samples`、temperature、top-k 和最大生成长度相同。

### ChatCORE

ChatCORE 把六个 Chat 任务的 Accuracy 汇总成一个综合指标。其计算流程与 CORE 类似：先扣除随机基线，再求平均。

```text
centered accuracy = (accuracy - baseline) / (1 - baseline)
ChatCORE = 所有 Chat 任务 centered accuracy 的平均值
```

ARC-Easy、ARC-Challenge 和 MMLU 的基线是 25%；GSM8K、HumanEval 和 SpellingBee 的基线是 0%。`ChatCORE=0` 表示整体约等于随机基线，`1` 表示所有任务满分，低于随机水平时也可能出现负数。

ChatCORE 适合看总体趋势，但会隐藏能力迁移：总分上涨时，数学可能明显提升，而代码或知识能力正在下降。因此报告 ChatCORE 时，也应该同时展示六个子任务分数。

## GSM8K RL 评测

NanoChat 的 `chat_rl.py` 使用 GSM8K 的可验证答案作为强化学习信号。

### Average Reward

当前实现直接复用 GSM8K 判分：最终数字正确奖励为 `1`，错误为 `0`。Average Reward 是一个 rollout batch 中二值奖励的平均值：

```text
average reward = 正确 completion 数 / completion 总数
```

因此它本质上是训练 rollout 上的正确率，不是通用“回答质量分”。它受训练题目分布、采样温度和每题生成数量影响，也不能直接替代独立测试集上的 GSM8K Accuracy。

### Pass@k

Pass@k 表示对同一道题生成 k 个候选答案时，至少有一个正确的比例：

```text
pass@k = 至少一个候选正确的题数 / 总题数
```

通常 `Pass@8 ≥ Pass@1`。两者差距较大，说明模型可能找到正确路径，但单次生成不稳定。比较 Pass@k 时必须使用相同的 `k`、temperature、top-k、评测题数和随机种子。

### Average Sequence Length

Average Sequence Length 是 rollout 平均生成长度。它不是能力分数，而是行为和效率指标，可用于发现：

- 模型是否越来越啰嗦；
- 是否过早停止；
- 是否因为输出变长而增加训练成本；
- reward 上升是否只是伴随更长的搜索过程。

需要和 Reward、Pass@k 一起观察，不能单独判断好坏。

## 容易混淆但不属于能力评测的指标

### Training Loss

Training Loss 是优化器直接最小化的 Token 级交叉熵，用于判断训练是否正常收敛。它和按字节归一化的 BPB 统计口径不同，数值不能直接比较。

### token/s

`token/s` 表示单位时间处理的 Token 数，是吞吐指标。它受 GPU、batch size、序列长度、梯度累积、通信和编译状态影响，不表示模型能力。

### MFU

MFU（Model FLOPs Utilization）估计模型实际计算量占硬件理论峰值的比例，用于判断训练系统是否充分利用 GPU。MFU 高说明工程效率较好，不代表模型答案更准确；如果硬件峰值 FLOPs 未配置，日志中的 `0` 也可能只是无法计算。

## 如何正确报告 NanoChat 评测结果

一份可比较的报告至少应同时记录：

1. checkpoint、模型深度和参数量；
2. 评测数据集、split 和样本数量；
3. categorical 还是 generative 评测；
4. `num_samples`、temperature、top-k、最大生成长度；
5. CORE 的 few-shot 与 `max_per_task` 设置；
6. 每个子任务分数，而不只报 CORE 或 ChatCORE；
7. 运行代码版本和随机种子。

一句话总结：**BPB 看语言建模，CORE 看 Base Model 的综合 ICL，任务 Accuracy 看具体能力，ChatCORE 看聊天模型总体趋势，Reward 和 Pass@k 看数学 RL 的训练与采样效果；token/s 和 MFU 只看工程效率。**
