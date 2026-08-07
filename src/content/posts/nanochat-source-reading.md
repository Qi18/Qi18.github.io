---
title: "读懂 NanoChat：先找到真实的训练主线"
description: "面对一个完整训练仓库，最有效的入口往往不是模型文件，而是实际运行的训练脚本。"
publishedAt: 2026-08-02
tags:
  - NanoChat
  - LLM
  - 源码阅读
series: "源码阅读"
seriesPath:
  - "LLM"
  - "实验"
  - "Nanochat"
  - "源码阅读"
seriesOrder: 1
draft: false
featured: true
---

阅读训练项目时，我们很容易直接打开模型定义文件，从 Transformer 结构开始逐行分析。但这通常不能回答一个更重要的问题：**这个项目究竟是怎样跑起来的？**

## 从执行顺序开始

真实训练脚本提供了一张天然的路线图。一个典型流程可能包含：

```text
数据准备 → 分词器训练 → 基座训练 → 评估 → SFT → 对话评估
```

沿着这个顺序阅读，每一步都有明确的输入和输出。前一个阶段生成的工件，会成为后一个阶段的依赖。

## 每一步回答四个问题

### 输入是什么

是原始文本、token 序列，还是上一步保存的 checkpoint？先确认输入，才能理解当前代码所在的层次。

### 输出是什么

关注真正落盘的文件、指标和日志。它们是阶段完成的证据，也是排查问题时最可靠的坐标。

### 张量如何变化

记录关键张量的形状，例如：

```python
tokens.shape      # [batch_size, sequence_length]
logits.shape      # [batch_size, sequence_length, vocab_size]
targets.shape     # [batch_size, sequence_length]
```

### 参数控制什么

不要只抄参数名。把它翻译成训练行为：改变批大小、上下文长度或梯度累积，分别会对显存、吞吐和优化过程产生什么影响？

## 形成可以复用的笔记

好的源码笔记不需要覆盖每一行代码，但应该能够让未来的自己回答：程序从哪里进入，数据经过哪些关键函数，最终生成什么，以及失败时应该先检查哪里。
