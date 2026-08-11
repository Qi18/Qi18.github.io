---
title: "NanoChat flash_attention.py：FA3、SDPA 与 KV Cache 三条路径"
description: "沿着 NanoChat 的真实调用链，解释 flash_attention.py 如何在 FA3 与 PyTorch SDPA 之间切换，以及完整 Prefill、单 Token 解码和 Chunk Prefill 三条路径的形状、Mask 与 KV Cache。"
publishedAt: 2026-08-11
updatedAt: 2026-08-11
tags:
  - NanoChat
  - LLM
  - PyTorch
  - FlashAttention
  - KV Cache
series: "源码阅读"
seriesPath:
  - "LLM"
  - "实验"
  - "NanoChat"
  - "源码阅读"
seriesOrder: 3
draft: false
featured: false
---

`nanochat/flash_attention.py` 的名字容易让人误以为它实现了一套 FlashAttention CUDA Kernel。实际上，它做的是更重要的工程工作：**为模型提供统一的注意力接口，在 FlashAttention 3 和 PyTorch SDPA 之间自动切换，同时统一训练与 KV Cache 推理的输入输出形状。**

从 `gpt.py` 看，整个调用链可以压缩成：

```text
CausalSelfAttention
  │
  ├─ 生成 q、k、v，形状为 (B, T, H, D)
  │
  ├─ 训练 / 无 KV Cache
  │    └─ flash_attn_func(q, k, v)
  │
  └─ 推理 / 有 KV Cache
       └─ flash_attn_with_kvcache(q, k_cache, v_cache, k, v)
              │
              ├─ Hopper SM90 + BF16 → FlashAttention 3
              └─ 其他设备           → PyTorch SDPA
```

这意味着上层模型不关心当前是 H100、L20、CPU 还是 MPS，只调用同一套 `flash_attn` API。

## 先认识 Q、K、V 的形状

NanoChat 在 `gpt.py` 中把 Q、K、V 整理为 FA3 原生布局：

```text
q: (B, Tq, H,    D)
k: (B, Tk, H_kv, D)
v: (B, Tk, H_kv, D)
```

各维度含义如下：

| 维度 | 含义 |
| --- | --- |
| `B` | Batch Size |
| `Tq` | Query 的 Token 数量 |
| `Tk` | Key/Value 中已有的 Token 数量 |
| `H` | Query Head 数量 |
| `H_kv` | Key/Value Head 数量 |
| `D` | 每个 Head 的维度 |

训练时通常有 `Tq == Tk`。使用 KV Cache 推理时，Query 只包含新 Token，而 Key/Value 还包含历史缓存，因此常见 `Tq < Tk`。

## FA3 和 SDPA 是怎样选择的

模块加载时，`_load_flash_attention_3()` 会检查：

1. CUDA 是否可用；
2. GPU Compute Capability 的主版本是否为 `9`，也就是 Hopper 的 SM90；
3. `varunneal/flash-attention-3` Kernel 能否成功加载。

加载成功只代表 `HAS_FA3=True`。真正执行时还会检查计算类型，当前代码只在 `COMPUTE_DTYPE == torch.bfloat16` 时设置 `USE_FA3=True`。

因此在 NVIDIA L20 上：

```text
L20 = Ada Lovelace = SM89
                    ↓
major != 9
                    ↓
HAS_FA3 = False
                    ↓
走 PyTorch SDPA
```

这里需要注意：**走 SDPA 不等于一定走普通 Math Attention。** `torch.nn.functional.scaled_dot_product_attention` 是统一接口，PyTorch 会根据 GPU、dtype、形状和 Mask 自动选择 FlashAttention-2、Memory-Efficient Attention 或 Math 后端。

可以把两层选择理解为：

```text
NanoChat 第一层：外部 FA3 还是 PyTorch SDPA？
PyTorch 第二层：FlashAttention-2、Memory-Efficient 还是 Math？
```

所以 L20 没有使用这个文件加载的 FA3，但仍可能通过 PyTorch SDPA 使用内置的 FlashAttention-2 Kernel。

## 训练接口 flash_attn_func

没有 KV Cache 时，`gpt.py` 调用：

```python
y = flash_attn.flash_attn_func(
    q, k, v,
    causal=True,
    window_size=window_size,
)
```

如果 FA3 可用，Q、K、V 的 `(B, T, H, D)` 布局可以直接交给 FA3。

PyTorch SDPA 使用的是 `(B, H, T, D)`，所以 fallback 会交换时间维和 Head 维：

```text
(B, T, H, D)
      ↓ transpose(1, 2)
(B, H, T, D)
      ↓ SDPA
(B, H, T, D)
      ↓ transpose(1, 2)
(B, T, H, D)
```

上层模型最终收到的形状没有变化，可以继续把多个 Head 合并回 Residual Stream。

## SDPA 实际计算什么

SDPA 是 Scaled Dot-Product Attention 的缩写。对每个 Batch、每个 Query Head，它计算：

```text
q_h: (Tq, D) @ k_hᵀ: (D, Tk)
                  ↓
          scores: (Tq, Tk)
                  ↓ 加 Mask / 除以 √D
             softmax
                  ↓
probs: (Tq, Tk) @ v_h: (Tk, D)
                  ↓
             out: (Tq, D)
```

汇总所有 Head 后，输出为：

```text
(B, H, Tq, D)
```

输出的时间长度跟 Query 的 `Tq` 走。`Tk` 只决定注意力得分矩阵的宽度，不改变输出长度。

## 路径一：完整 Prefill

假设一次输入 5 个 Token，并且这一层使用完整上下文：

```text
Tq = Tk = 5

q: (B, H,    5, D)
k: (B, H_kv, 5, D)
v: (B, H_kv, 5, D)
```

此时 Query 与 Key 从位置 0 开始一一对齐，可以直接调用：

```python
F.scaled_dot_product_attention(
    q, k, v,
    is_causal=True,
    enable_gqa=enable_gqa,
)
```

`is_causal=True` 会生成标准下三角约束：

```text
        k0 k1 k2 k3 k4
q0      ✓  ✗  ✗  ✗  ✗
q1      ✓  ✓  ✗  ✗  ✗
q2      ✓  ✓  ✓  ✗  ✗
q3      ✓  ✓  ✓  ✓  ✗
q4      ✓  ✓  ✓  ✓  ✓
```

逻辑上的注意力得分矩阵为 `(B, H, 5, 5)`。融合 Kernel 不一定真的在显存中保存完整矩阵，但数学关系不变。

## 路径二：单 Token 解码

自回归生成时，每轮通常只有一个新 Query。假设当前来到全局位置 9：

```text
q: (B, H,    1,  D)    [q9]
k: (B, H_kv, 10, D)    [k0 k1 ... k9]
v: (B, H_kv, 10, D)    [v0 v1 ... v9]
```

若 `window=3`，表示当前 Token 最多向左看 3 个历史位置，加上当前位置一共保留 4 个 Key：

```python
start = max(0, Tk - (window + 1))
      = max(0, 10 - 4)
      = 6

k = k[:, :, 6:, :]  # k6 k7 k8 k9
v = v[:, :, 6:, :]  # v6 v7 v8 v9
```

得分矩阵从 `(B, H, 1, 10)` 缩小为 `(B, H, 1, 4)`：

```text
scores = [q9·k6, q9·k7, q9·k8, q9·k9] / √D
probs  = softmax(scores)
out9   = p6·v6 + p7·v7 + p8·v8 + p9·v9
```

这里使用 `is_causal=False`，但并不会泄漏未来信息。原因是 KV Cache 只保存到当前位置 `k9`，切片后更只剩 `k6～k9`，传给 SDPA 的所有 Key 都是合法位置。

反而不能直接使用普通的 `is_causal=True`：SDPA 只看到局部形状 `Tq=1、Tk=4`，并不知道唯一的 Query 是全局位置 9，局部下三角可能只允许它看到第一个 Key。

## 路径三：Chunk Prefill

Chunk Prefill 是一次处理多个新 Token，同时复用之前的 KV Cache。假设缓存中已有位置 `0～4`，这次加入 `5～7`：

```text
旧 KV Cache：[k0 k1 k2 k3 k4]
新 Chunk：   [k5 k6 k7]

q: [q5 q6 q7]        Tq = 3
k: [k0 ... k7]       Tk = 8
```

正确的因果关系应该是：

```text
        k0 k1 k2 k3 k4 k5 k6 k7
q5      ✓  ✓  ✓  ✓  ✓  ✓  ✗  ✗
q6      ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✗
q7      ✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓
```

如果直接按照 Query 的局部位置 `0、1、2` 生成下三角，`q5` 会被误认为只能看 `k0`。因此代码需要先计算 Query 的全局位置：

```python
row_idx = (Tk - Tq) + torch.arange(Tq).unsqueeze(1)
col_idx = torch.arange(Tk).unsqueeze(0)
mask = col_idx <= row_idx
```

代入 `Tq=3、Tk=8`：

```text
Tk - Tq = 5

row_idx = [[5],       shape = (3, 1)
           [6],
           [7]]

col_idx = [[0,1,2,3,4,5,6,7]]
                          shape = (1, 8)
```

通过广播比较，得到 `(3, 8)` 的二维 Bool Mask。这个 Mask 会继续广播到所有 Batch 和 Head，不需要显式创建 `(B, H, 3, 8)` 的四维 Mask。

如果还有有限滑动窗口，会再叠加一个条件：

```python
mask = mask & ((row_idx - col_idx) <= window)
```

例如 `window=3`：

```text
        k0 k1 k2 k3 k4 k5 k6 k7
q5      ✗  ✗  ✓  ✓  ✓  ✓  ✗  ✗
q6      ✗  ✗  ✗  ✓  ✓  ✓  ✓  ✗
q7      ✗  ✗  ✗  ✗  ✓  ✓  ✓  ✓
```

因果条件负责屏蔽未来，窗口条件负责丢掉太远的历史，两个条件必须同时满足。

## 推理接口如何管理 KV Cache

`flash_attn_with_kvcache()` 接收预分配的：

```text
k_cache: (B, T_max, H_kv, D)
v_cache: (B, T_max, H_kv, D)
```

SDPA fallback 会先把本轮新 K/V 原地写入缓存：

```python
k_cache[:, pos:pos+T_new, :, :] = k
v_cache[:, pos:pos+T_new, :, :] = v
```

然后只取已经有效的部分：

```python
end_pos = pos + T_new
k_full = k_cache[:, :end_pos, :, :]
v_full = v_cache[:, :end_pos, :, :]
```

这样历史 Token 的 K/V 不需要重复计算。每轮生成只计算新 Token 的 Query、Key 和 Value，再把新 K/V 追加到缓存末尾。

### 为什么只缓存 K 和 V

生成全局位置 `t` 时，当前层需要：

```text
当前 Query：q_t
历史 Key：  k_0 ... k_t
历史 Value：v_0 ... v_t
```

注意力计算为：

```text
scores_t = q_t @ [k_0 ... k_t]ᵀ / √D
out_t    = softmax(scores_t) @ [v_0 ... v_t]
```

历史 Token 的 K/V 已经计算完成，并且后续不会变化，因此可以重复使用。Query 则只在自己的位置使用一次：生成位置 5 使用 `q5`，下一轮生成位置 6 使用新的 `q6`，旧的 `q5` 不再参与计算，所以不需要缓存 Q。

Attention Scores 也不适合缓存。每个新 Query 都必须与历史 Key 重新点积，旧的 `q5 @ Kᵀ` 无法用于 `q6 @ Kᵀ`；保存完整 Scores 还会产生随序列长度平方增长的 `T×T` 空间。因此只保存稳定、可复用的 K/V，是计算量和显存之间的折中。

### 整体缓存为什么是五个维度

`flash_attn_with_kvcache()` 收到的是某一层的缓存视图：

```text
(B, T_max, H_kv, D)
```

但 `Engine` 中完整的 KV Cache 还要包含所有 Transformer 层，因此整体形状是：

```text
k_cache: (L, B, T_max, H_kv, D)
v_cache: (L, B, T_max, H_kv, D)
```

每个维度都来自推理时必须保留的一类信息：

| 维度 | 为什么必须存在 |
| --- | --- |
| `L` | 每一层的输入和 `Wk/Wv` 不同，同一个 Token 在不同层产生的 K/V 不能共享 |
| `B` | Batch 中每条序列的 Prompt 不同，需要独立缓存 |
| `T_max` | 完整因果注意力可能访问每一个历史 Token |
| `H_kv` | 每个 KV Head 都有自己的一组 K/V；GQA 通过减少它来降低缓存大小 |
| `D` | 每个 Head 的 Key 和 Value 都是长度为 `D` 的向量 |

例如同一个 Token 在第 0 层和第 1 层会分别计算：

```text
k_t^(0) = hidden_t^(0) @ Wk^(0)
k_t^(1) = hidden_t^(1) @ Wk^(1)
```

两层的 Hidden State 和投影矩阵都不同，所以 KV Cache 必须带有 `L` 这一维。代码通过 `get_layer_cache(layer_idx)` 取出当前层的 `(B,T_max,H_kv,D)` 视图，再交给 Attention。

### 为什么预分配 T_max

如果每生成一个 Token 都重新创建更大的张量，过程会变成：

```text
长度 5 的 Cache
  ↓ 申请长度 6 的新显存，并复制旧数据
长度 6 的 Cache
  ↓ 申请长度 7 的新显存，并复制旧数据
长度 7 的 Cache
```

这会造成频繁的显存申请、旧数据复制和显存碎片。预先创建 `T_max` 容量后，每轮只需原地写入下一个位置：

```text
[k0 k1 k2 k3 k4 _  _  _]
                   ↑
                 写入 k5
```

它还有两个工程收益：Tensor 地址保持稳定，方便 CUDA Graph 等优化；FA3 的 `flash_attn_with_kvcache` 也可以直接原地更新缓存。

需要区分的是：NanoChat 的 `KVCache` 对象一旦创建，就会按指定形状完整分配张量；但这个对象是在生成请求到达后创建的，不是模型部署时永久创建一份最大缓存。当前生成流程先为 Prompt 创建 Batch 1 的 Prefill Cache，再按 `num_samples` 和预计总长度创建 Decode Cache，复制有效前缀后释放 Prefill Cache。

### 固定缓存与 Paged KV Cache

固定连续张量实现简单、访问直接，但请求很短时也会预留 `T_max` 空间；多个请求长度差异很大时，显存利用率会下降。

生产推理系统通常在模型部署时创建全局 KV Cache 内存池，再按固定大小的 Block 分给请求：

```text
全局 KV Cache 内存池
├─ 请求 A：Block 0、1、2
├─ 请求 B：Block 3～8
└─ 空闲 Block：等待后续请求
```

这就是 Paged KV Cache 的基本思路：缓存内容仍然是各层、各 Token 的 K/V，只是物理存储不要求每条序列占据一整块连续的最大长度空间。

对于纯滑动窗口层，理论上还可以只保留最近 `window+1` 个位置，并用环形缓存覆盖更早的数据。NanoChat 当前仍为各层统一分配 `T_max`：一方面实现更简单，另一方面 `SSSL` 模式中仍有完整上下文层，最后一层也固定使用完整上下文。

KV Cache 的显存占用可以估算为：

```text
2 × 层数 × Batch × 最大长度 × KV Head 数 × Head Dim × dtype 字节数
↑
K 和 V
```

NanoChat d24 使用 BF16、24 层、12 个 KV Head、Head Dim 128，因此每个缓存 Token 大约占用：

```text
2 × 24 × 12 × 128 × 2 bytes
= 147456 bytes
= 144 KiB / token
```

单条 2048 Token 序列约需要 288 MiB KV Cache；8 条并行序列约需要 2.25 GiB。

## GQA 为什么不需要手动复制 K/V

当 `H_kv < H` 时，模型使用 Grouped-Query Attention。代码通过：

```python
enable_gqa = q.size(1) != k.size(1)
```

告诉 SDPA 让多个 Query Head 共享同一组 KV Head。逻辑上，每个 KV Head 服务 `H / H_kv` 个 Query Head，但调用方不用先把 K/V 真正复制成 `(B, H, Tk, D)`，因此节省了 KV Cache 显存。

## 两个值得注意的实现边界

第一，SDPA fallback 的 `_sdpa_attention()` 没有接收公开接口中的 `causal` 参数，它内部始终按照 NanoChat 的因果自注意力场景处理。当前 `gpt.py` 所有调用都传入 `causal=True`，因此没有问题；但不能把这层 fallback 原样当作通用的双向 Attention 接口。

第二，fallback 使用：

```python
pos = cache_seqlens[0].item()
```

它假设同一个 Batch 中所有序列的缓存位置一致。NanoChat 当前的批量生成流程满足这一约束；如果以后支持每条请求长度不同的连续批处理，就需要按样本分别管理位置，或者改用分页式 KV Cache。

## 相关源码

- [nanochat/flash_attention.py](https://github.com/Qi18/nanochat/blob/experiment/l20-d24-swanlab-20260806/nanochat/flash_attention.py)
- [nanochat/gpt.py](https://github.com/Qi18/nanochat/blob/experiment/l20-d24-swanlab-20260806/nanochat/gpt.py)
- [nanochat/engine.py](https://github.com/Qi18/nanochat/blob/experiment/l20-d24-swanlab-20260806/nanochat/engine.py)
- [PyTorch scaled_dot_product_attention 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention)

## 总结

`flash_attention.py` 的价值不在于重新发明 Attention，而在于隔离硬件与推理状态的复杂性：

1. 在 Hopper BF16 上调用 FA3，其他设备回退到 PyTorch SDPA；
2. 在 FA3 的 `(B,T,H,D)` 和 SDPA 的 `(B,H,T,D)` 之间转换；
3. 用 `enable_gqa` 统一处理 MHA 与 GQA；
4. 分别处理完整 Prefill、单 Token 解码和 Chunk Prefill；
5. 在推理时原地更新 KV Cache，并正确对齐全局因果位置；
6. 让 `gpt.py` 始终只面对一套稳定的注意力 API。

读懂这个文件后，NanoChat 中“训练为什么不需要 KV Cache”“解码为什么可以只算一个 Query”“Chunk Prefill 为什么不能直接使用普通下三角”就串成了同一条执行链。
