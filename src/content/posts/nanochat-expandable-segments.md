---
title: "NanoChat base_train.py：一行显存配置 expandable_segments"
description: "解读 base_train.py 顶部的 PYTORCH_ALLOC_CONF=expandable_segments:True：PyTorch CUDA 缓存分配器的默认段模式为什么会碎片化，可扩展段如何避免训练中后期 OOM。"
publishedAt: 2026-08-07
updatedAt: 2026-08-07
tags:
  - NanoChat
  - LLM
  - PyTorch
  - 显存
series: "源码阅读"
seriesPath:
  - "LLM"
  - "Train"
  - "NanoChat"
  - "源码阅读"
seriesOrder: 1
draft: false
featured: false
---

`scripts/base_train.py` 在 import torch 之前有一行不起眼的配置：

```python
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
```

它让 PyTorch 的 CUDA 缓存分配器启用**可扩展段（expandable segments）**模式。这一行不影响任何训练逻辑，却直接决定长时间训练会不会因为显存碎片而 OOM。

## 背景：训练时显存请求一直在变

训练过程中张量形状不断变化：不同 batch、激活值、梯度、优化器状态都会反复申请和释放显存。PyTorch 不会每次都调用 `cudaMalloc`（太慢），而是维护一个**缓存分配器**：向 CUDA 申请大块"段"（segment），再从段里切小块分给张量；张量释放后小块回到缓存，等待复用。

问题出在"段"的管理方式上。

## 默认模式：固定大小的段会碎片化

默认情况下，每个段由一次 `cudaMalloc` 创建，**大小固定，之后不能伸缩**，空闲块也无法跨段合并。于是会出现这种局面：

```text
段 A (20MB)          段 B (30MB)
┌──────────────┐     ┌──────────────────┐
│  空闲 20MB    │     │   空闲 30MB       │
└──────────────┘     └──────────────────┘

新请求：45MB 的张量
  → 空闲总量 20 + 30 = 50MB > 45MB
  → 但两块不连续，哪个段都放不下
  → 只能再 cudaMalloc；显存不够就 OOM
```

这就是显存碎片化：`reserved`（保留）远大于 `allocated`（实际使用），`nvidia-smi` 看着还有余量，训练却报 OOM。张量形状越多变（可变序列长、`torch.compile` 的编译缓存、优化器状态的延迟创建），碎片积累越快。

## expandable_segments：让段可以"生长"

开启后，分配器改用 CUDA 虚拟内存 API（`cuMemAddressReserve` + `cuMemMap`）管理段：

1. 先**预留一大段连续的虚拟地址空间**（不占物理显存）；
2. 需要多少，就把物理页**按需映射**到这段地址的尾部；
3. 释放时可以解映射物理页，但虚拟地址保持连续。

```text
一个可扩展的段（虚拟地址预留区）
┌──────────┬──────────────┬┄┄┄┄┄┄┄┄┄┄┄┄┄┐
│ 已用 20MB │  已用 30MB    │  未映射（预留） │
└──────────┴──────────────┴┄┄┄┄┄┄┄┄┄┄┄┄┄┘

新请求：45MB
  → 段直接向后映射 45MB 新物理页
┌──────────────────────────┬───────────┬┄┄┄┐
│   原有 50MB（可释放复用）    │ +45MB 新映射│    │
└──────────────────────────┴───────────┴┄┄┄┘
```

所有空闲显存都在同一个段里、地址连续，任何大小的请求都能复用，碎片几乎消失。代价是虚拟内存 API 带来的少量映射开销，对训练场景可以忽略。

## 两种模式对比

| | 默认模式 | expandable_segments:True |
| --- | --- | --- |
| 段的创建 | `cudaMalloc`，大小固定 | 虚拟地址预留 + `cuMemMap` 按需映射 |
| 空闲显存 | 散落在多个段里，不连续 | 集中在一个段里，始终连续 |
| 碎片表现 | reserved 远大于 allocated | reserved ≈ allocated |
| 典型后果 | 训练中后期 OOM | 有效可用显存更多 |

## 一个关键细节：必须在 import torch 之前

这行环境变量必须放在文件顶部、**任何 `torch` import 之前**。CUDA 缓存分配器只在初始化时读取一次 `PYTORCH_ALLOC_CONF`，如果 `torch` 已经被 import（包括被其他模块间接 import），配置就不会生效。这也是它出现在 `base_train.py` 第 15 行、先于所有 torch 相关 import 的原因。

等价的做法是在启动命令里设置：

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True torchrun ... base_train.py
```

NanoChat 直接写进脚本，省去每次手动设置，也避免忘记设置导致的"偶发 OOM"难以排查。
