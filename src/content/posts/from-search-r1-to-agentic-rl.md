---
title: "从 Search-R1 到 Agentic RL：智能体强化学习发生了什么变化"
description: "Search-R1 让模型学会搜索并回答，而新的 Agentic RL 开始训练模型在有状态环境中持续规划、操作、观察和纠错。"
publishedAt: 2026-08-14
tags:
  - Agentic RL
  - Search-R1
  - Reinforcement Learning
  - Agent
series: "Agentic RL"
seriesPath:
  - LLM
  - 实验
  - Agentic RL
seriesOrder: 1
draft: false
featured: true
---

Search-R1 是理解 Agentic RL 很好的入口。它已经不再把语言模型限制在一次性生成答案，而是允许模型在推理过程中主动搜索、读取结果，再决定下一步动作。

但从 2025 年下半年到 2026 年出现的新实验，正在把问题从“如何用搜索工具回答问题”推进到“如何在真实、有状态的环境中完成长期任务”。两者属于同一条技术路线，但训练对象、系统复杂度和研究难点已经明显不同。

## Search-R1 训练的是什么

Search-R1 的典型交互循环可以概括为：

```text
思考 → 搜索 → 获得文本 → 再思考 → 最终回答
```

与只训练模型输出思维链相比，它增加了外部环境和工具反馈，因此可以看作一种搜索场景下的 Agentic RL。它的优点是动作空间清晰、环境容易复现、答案能够自动验证，很适合研究 GRPO、工具调用格式、搜索策略和结果奖励。

不过，这类实验通常仍然具有几个明显边界：任务主要是问答，工具主要负责获取信息，轨迹相对较短，最终答案的 EM、F1 或规则奖励仍是核心训练信号。

## Search-R1 生态项目地图

Search-R1 官方 README 的 **Awesome work powered or inspired by Search-R1** 清单目前收录了 23 个代表项目。它们大多发表于 2025 年，可以看作搜索型 Agent RL 从单一检索闭环向多模态、多工具和规模化训练扩展的过程。

### 大致时间线

- **2025 年 4 月前后：建立搜索 RL 基线。** DeepResearcher、Multimodal-Search-R1、OTC 和 ZeroSearch 分别把方向扩展到真实网页研究、多模态搜索、工具调用效率以及不连接真实搜索引擎的训练。
- **2025 年 5 月前后：细化搜索过程。** IKEA、Scent of Knowledge、AutoRefine、O^2-Searcher、MaskSearch、VRAG-RL、R1-Code-Interpreter 和 StepSearch 开始研究内外知识选择、信息觅食、边搜索边修正、开放式问答、搜索预训练、视觉检索、代码执行和逐步奖励。
- **2025 年 6 月前后：走向稳定多轮和多工具。** R-Search、SimpleTIR、Router-R1 与 SkyRL 分别探索多奖励、稳定端到端训练、多轮路由聚合和模块化训练基础设施。
- **2025 年 7—8 月：扩大动作空间和训练规模。** AutoTIR、ASearcher 与 ParallelSearch 将重点推进到自主选择工具、大规模搜索 Agent RL，以及查询分解后的并行检索。
- **2025 年 9—10 月：增强探索和证据组织。** verl-tool、Tree-GRPO、EviNote-RAG 与 GlobalRAG 进一步覆盖多样化工具、树搜索、证据笔记和全局多跳推理。

以上月份按论文编号及项目公开顺序粗略划分，重点是观察研究主题的迁移，不代表严格的首发日期。

### 按研究问题分类

| 方向 | 代表项目 | 主要推进 |
| --- | --- | --- |
| 深度搜索与开放问答 | [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher)、[O^2-Searcher](https://arxiv.org/pdf/2505.16582)、[ASearcher](https://arxiv.org/abs/2508.07976)、[GlobalRAG](https://arxiv.org/pdf/2510.20548v1) | 从单跳问答扩展到真实网页、开放式问题、大规模训练和全局多跳推理。 |
| 搜索策略、奖励与效率 | [OTC](https://arxiv.org/pdf/2504.14870)、[ZeroSearch](https://github.com/Alibaba-NLP/ZeroSearch)、[IKEA](https://github.com/hzy312/knowledge-r1)、[Scent of Knowledge](https://arxiv.org/abs/2505.09316)、[AutoRefine](https://www.arxiv.org/pdf/2505.11277)、[R-Search](https://arxiv.org/abs/2506.04185)、[StepSearch](https://arxiv.org/pdf/2505.15107)、[SimpleTIR](https://simpletir.notion.site/report)、[ParallelSearch](https://www.arxiv.org/abs/2508.09303)、[Tree-GRPO](https://arxiv.org/abs/2509.21240) | 减少无效工具调用，在训练中模拟搜索，学习何时检索、如何修正、如何分步奖励，并引入并行或树形探索。 |
| 多模态与证据增强 RAG | [Multimodal-Search-R1](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)、[VRAG-RL](https://arxiv.org/abs/2505.22019)、[EviNote-RAG](https://arxiv.org/abs/2509.00877) | 将搜索对象从纯文本扩展到图像和视觉丰富文档，并显式整理支持答案的证据。 |
| 通用工具与路由 | [R1-Code-Interpreter](https://arxiv.org/abs/2505.21668)、[Router-R1](https://arxiv.org/pdf/2506.09033)、[AutoTIR](https://arxiv.org/pdf/2507.21836) | 从固定搜索动作扩展到代码执行、多个工具之间的路由和自主工具集成推理。 |
| 预训练与训练基础设施 | [MaskSearch](https://arxiv.org/pdf/2505.20285)、[SkyRL](https://skyrl.readthedocs.io/en/latest/)、[verl-tool](https://arxiv.org/pdf/2509.01055) | 将搜索能力前置到预训练阶段，并提供模块化、支持多种工具的 RL 训练栈。 |

这些工作并不都等同于完整的环境型 Agentic RL。多数仍围绕“检索或工具增强的推理”优化；其中 DeepResearcher、Router-R1、AutoTIR、R1-Code-Interpreter 等由于环境更开放、工具更丰富，更接近下一阶段的 Agentic RL。

## 新 Agentic RL 的变化

一句话概括：

> Search-R1 优化的是“带搜索工具的回答策略”，新的 Agentic RL 优化的是“在环境中持续行动的决策系统”。

| 维度 | Search-R1 等搜索 RL | 新 Agentic RL |
| --- | --- | --- |
| 环境 | QA 数据集和搜索引擎 | GUI、操作系统、代码仓库、多应用 API、游戏等有状态环境 |
| 动作 | 搜索查询和最终回答 | 点击、输入、文件操作、终端、API、代码修改、测试和多 Agent 协作 |
| 轨迹 | 通常是若干轮搜索 | 数十到上百轮，部分研究讨论十万乃至百万 token 的轨迹 |
| 奖励 | 最终答案正确性和搜索过程奖励 | 最终环境状态、测试结果、任务完成度，以及 turn/token 级信用分配 |
| 训练系统 | rollout 与训练紧耦合，通常同步执行 | 环境集群、异步 rollout、策略版本管理，Agent Runtime 与训练解耦 |
| 数据 | 静态 QA 数据集 | 动态任务、课程学习、成功与失败轨迹回流、数据飞轮 |
| 评测 | EM、F1、检索命中率 | 软件是否修好、界面状态是否正确、业务状态是否改变，以及成本与安全性 |

## 从获取信息到改变环境

搜索工具主要改变模型拥有的信息，并不会直接改变任务世界。新的智能体动作却会持续改变环境状态：修改文件、提交表单、调用业务 API、运行测试，甚至让另一个智能体继续执行子任务。

[SWE-Gym](https://arxiv.org/abs/2412.21139) 让智能体在真实代码仓库中修改代码，并通过单元测试判断结果；[LOOP/AppWorld](https://arxiv.org/abs/2502.01600) 研究智能体在多个有状态应用中调用 API；[UI-TARS-2](https://arxiv.org/abs/2509.02544) 则把 GUI、文件系统和终端放入统一沙箱。

这时，智能体面对的不只是“下一次应该搜什么”，还要处理部分可观测状态、工具失败、前置条件、不可逆操作和错误恢复。

## 长轨迹让信用分配成为核心问题

如果最终任务失败，奖励只告诉模型“整条轨迹不好”，却没有指出问题究竟发生在哪里：可能是早期计划错误，也可能是中途忽略了一次工具异常，或者最后没有验证环境状态。

轨迹越长，最终奖励越难有效指导前面的动作。2026 年的 [Agentic RL 信用分配综述](https://arxiv.org/abs/2604.09459) 将 turn-level MDP、反事实分析和 privileged critic 等方法归纳为这一方向的关键技术。更新的 [ADRS](https://arxiv.org/abs/2608.03223) 尝试把稀疏的轨迹奖励转化成更细粒度的 token 级训练信号。

因此，新 Agentic RL 不再只判断一条轨迹是否成功，还要判断哪些观察、计划和动作真正促成了结果。

## 训练对象从模型扩展到完整 Agent 系统

Search-R1 通常由训练代码直接生成搜索轨迹。新的框架开始尝试训练已经存在的 Agent 工作流。

[Agent Lightning](https://arxiv.org/abs/2508.03680) 将 Agent 的执行过程与强化学习训练解耦，把不同框架产生的运行轨迹转换成统一的训练数据。这使训练对象可以包含多种工具、动态分支、多个智能体，或者已经投入使用的复杂 Agent Runtime。

与此同时，[DART](https://arxiv.org/abs/2509.23866) 等 GUI Agent 工作把环境集群、rollout 服务、数据管理和模型训练拆开，通过异步方式提高昂贵环境的利用率。这里的主要创新已经不只是 PPO 或 GRPO 的公式，而是如何稳定地运行、记录和学习大量交互轨迹。

## 课程学习和数据飞轮变得更重要

长任务的随机探索成功率很低。直接把交互长度从几轮增加到一百轮，通常只会得到大量没有训练价值的失败轨迹。

[AgentGym-RL](https://arxiv.org/abs/2509.08755) 采用逐渐增加交互长度的方式，让模型先掌握短任务，再进入更长的任务。[RAGEN](https://arxiv.org/abs/2504.20073) 则观察到多轮训练中的奖励方差突变、梯度尖峰和 Echo Trap：模型生成了看似合理的思考，但实际策略仍然十分浅层。

因此，任务难度分层、初始状态多样性、成功轨迹回流、失败类型标注和按难度分配 rollout，正在成为 Agentic RL 实验的重要组成部分。

## 它们是不是同一类研究

可以把它们放在一条连续的演进路径上：

```text
静态推理 RL
    ↓
Search-R1：单一搜索工具
    ↓
多工具推理 Agent
    ↓
有状态、长时程环境 Agent
    ↓
GUI / Code / API / 多 Agent 的持续学习系统
```

所以，Search-R1 确实属于 Agentic RL，但它是一个边界清楚、反馈稳定的子问题。最近的新实验并不是简单地给 Search-R1 增加更多搜索轮次，而是在环境真实性、动作空间、轨迹长度、信用分配和训练基础设施上同时扩展。

## 一条适合实践的实验路线

对个人或小团队来说，没有必要一开始就复现完整的 GUI Agent 集群。更稳妥的路线是：

1. 先复现官方 Search-R1，确认搜索轨迹、奖励和 GRPO 训练闭环。
2. 把搜索过程抽象成统一的 `reset`、`step`、`observation` 和 `reward` 接口。
3. 增加 turn-level 奖励、失败原因和轨迹可视化，研究长程信用分配。
4. 接入第二类工具，例如代码执行或结构化 API，让动作能够改变环境状态。
5. 将 rollout 与 trainer 解耦，在单环境稳定后再尝试并发和异步训练。

这条路线保留了 Search-R1 容易验证的优点，同时会逐步触及真正 Agentic RL 的核心问题：规划、行动、状态变化、延迟奖励和失败恢复。

本文讨论的是截至 2026 年 8 月的研究趋势，其中部分工作仍是较新的预印本，实验结论还需要更多复现与比较。
