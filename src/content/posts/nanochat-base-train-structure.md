---
title: "NanoChat 源码阅读：base_train.py 如何串起一次完整预训练"
description: "沿着 scripts/base_train.py 的真实执行顺序，拆解 NanoChat 如何完成分布式初始化、模型构建、缩放律、优化器、训练循环、评测与 checkpoint 管理。"
publishedAt: "2026-08-12"
updatedAt: "2026-08-12"
tags:
  - "NanoChat"
  - "LLM"
  - "PyTorch"
  - "预训练"
series: "源码阅读"
seriesPath:
  - "LLM"
  - "Train"
  - "NanoChat"
  - "源码阅读"
seriesOrder: 4
draft: false
featured: false
---

<!-- Generated from Obsidian. Do not edit directly. -->

> 实验：`l20-d24-swanlab-20260806`
> 源码版本：[`f5e012f`](https://github.com/Qi18/nanochat/blob/f5e012fc47dbc5b5da6e4b8765b66de8000a54b7/scripts/base_train.py)
> 实验运行版本：`3126420e9662c87e299b6cf4c65398b5b8936696`
> SwanLab：[Base Pretrain 7malxqoi](https://swanlab.cn/@richliu0153/nanochat-lab/runs/7malxqoi)

第一次打开 NanoChat 的 [`scripts/base_train.py`](https://github.com/Qi18/nanochat/blob/f5e012fc47dbc5b5da6e4b8765b66de8000a54b7/scripts/base_train.py)，很容易把它当成一个普通的 PyTorch 训练脚本：创建模型、读取数据、反向传播、保存权重。

但它真正承担的是“预训练总控”的角色。模型内部结构在 `nanochat/gpt.py`，数据打包在 `nanochat/dataloader.py`，优化器在 `nanochat/optim.py`，checkpoint 逻辑在 `nanochat/checkpoint_manager.py`；`base_train.py` 的任务，是把这些模块按正确顺序拼成一条可训练、可评测、可恢复的流水线。

整份脚本可以压缩成下面这张控制流图：

```text
参数解析
  ↓
设备 / 多卡 / 实验记录初始化
  ↓
Tokenizer → Meta Device 构建 GPT → 初始化权重 / 恢复权重
  ↓
可选 FP8 转换 → torch.compile
  ↓
缩放律计算训练 token、batch、LR、weight decay
  ↓
MuonAdamW + DataLoader + Scheduler
  ↓
┌──────────────────── Training Loop ────────────────────┐
│ validation BPB / CORE / sample / checkpoint（按周期） │
│                         ↓                              │
│ gradient accumulation → optimizer.step → logging      │
└────────────────────────────────────────────────────────┘
  ↓
最终统计 → 结束实验记录 → 销毁分布式进程组
```

## 1. 参数层：把训练问题拆成六组开关

脚本首先定义 CLI 参数（[L39-L81](https://github.com/Qi18/nanochat/blob/f5e012fc47dbc5b5da6e4b8765b66de8000a54b7/scripts/base_train.py#L39-L81)）。这些参数不是平铺的，而是对应六类决策：

| 参数组 | 解决的问题 | 代表参数 |
| --- | --- | --- |
| 运行环境 | 在什么设备上跑 | `device-type` |
| 精度 | 是否使用 FP8 | `fp8`、`fp8-recipe` |
| 模型结构 | 模型有多深、多宽、上下文多长 | `depth`、`aspect-ratio`、`head-dim`、`max-seq-len` |
| 训练长度 | 到底训练多少步或多少 FLOPs | `num-iterations`、`target-flops`、`target-param-data-ratio` |
| 优化 | 每步喂多少 token、不同参数用什么 LR | `device-batch-size`、`total-batch-size`、各类 LR、weight decay |
| 评测与产物 | 多久评测、采样和保存 | `eval-every`、`core-metric-every`、`sample-every`、`save-every` |

其中训练长度有明确优先级：

```text
num_iterations > target_flops > target_param_data_ratio
```

只要显式传入 `--num-iterations`，后两种自动计算就不会生效。

## 2. 运行时初始化：先建立“每个进程是谁”

[`compute_init`](https://github.com/Qi18/nanochat/blob/f5e012fc47dbc5b5da6e4b8765b66de8000a54b7/scripts/base_train.py#L85-L96) 返回：

```python
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device
```

随后脚本把 `rank == 0` 定义为 master process。它负责主要日志、采样和模型文件保存；其他 rank 仍要参加前向、反向、BPB/CORE 评测，以及各自 optimizer state 的保存。

这里有一个容易误解的地方：NanoChat 没有在本文件中写出典型的 `DistributedDataParallel(model)`。多卡参数同步封装在组合优化器的通信路径里，所以 NCCL 仍然是训练核心依赖，不是一个只负责启动进程的外围组件。

初始化阶段还会完成三件事：

- 识别 `COMPUTE_DTYPE`，决定 BF16、FP16 或 FP32；
- 初始化 SwanLab/W&B 兼容的实验记录器；
- 检查实际使用 FA3 还是 PyTorch SDPA，并对不支持 sliding window 的 SDPA 配置报警。

本次实验使用 8×L20、BF16 和 FA3。脚本无法识别 L20 的理论峰值 FLOPS，所以日志中的 MFU 为 0；本实验用 step 时间、吞吐和显存判断性能。

## 3. 模型构建：为什么先放在 Meta Device

模型初始化位于 [L125-L167](https://github.com/Qi18/nanochat/blob/f5e012fc47dbc5b5da6e4b8765b66de8000a54b7/scripts/base_train.py#L125-L167)，分三步：

```python
model = build_model_meta(args.depth)
model.to_empty(device=device)
model.init_weights()
```

第一步只在 `meta` device 上建立参数的 shape 和 dtype，不分配真实存储；第二步在目标 GPU 上分配未初始化空间；第三步才真正初始化权重。这样可以避免先在 CPU 创建一整份大模型，再复制到 GPU 时产生额外内存峰值。

宽度不是直接写死的，而是由深度推导：

```text
base_dim = depth × aspect_ratio
model_dim = base_dim 向上取整到 head_dim 的整数倍
num_heads = model_dim / head_dim
```

本次 d24 实验对应：

```text
depth=24, model_dim=1536, num_heads=12, sequence_len=2048
```

如果指定 `--resume-from-step`，脚本会在初始化完成后加载模型、当前 rank 的 optimizer shard 和 metadata。恢复的不只是模型权重，还包括 DataLoader 游标、平滑 loss、最低 validation BPB 和累计训练时间。

## 4. FP8 与 `torch.compile`：顺序不能交换

可选 FP8 转换发生在 `torch.compile` 之前（[L170-L251](https://github.com/Qi18/nanochat/blob/f5e012fc47dbc5b5da6e4b8765b66de8000a54b7/scripts/base_train.py#L170-L251)）。脚本只转换尺寸满足硬件约束的 Linear，并用 `disable_fp8` context manager 在评测时临时恢复为 BF16 Linear。

这体现了训练精度与评测精度的分离：可以用 FP8 加速训练，但 validation BPB、CORE 和采样仍尽量在 BF16 下执行，减少评测口径漂移。

然后脚本保留两份引用：

```python
orig_model = model
model = torch.compile(model, dynamic=False)
```

- `model`：固定 shape 的编译训练路径；
- `orig_model`：保存、CORE 和生成所使用的原始模型。

CORE 和生成的输入 shape 会变化，强行走固定 shape 编译图容易触发重复编译，因此这里主动绕开 compiled model。

## 5. 缩放律：`depth` 为什么能成为主要复杂度旋钮

NanoChat 并不要求用户手动填写所有超参数。[L254-L310](https://github.com/Qi18/nanochat/blob/f5e012fc47dbc5b5da6e4b8765b66de8000a54b7/scripts/base_train.py#L254-L310) 会根据模型规模推导训练预算和优化参数。

用于训练 horizon 的不是总参数量，而是：

```text
scaling_params = transformer_matrices + lm_head
target_tokens = target_param_data_ratio × scaling_params
```

本次模型总参数是 1,384,122,122，但 scaling parameters 是 729,810,624。设置 ratio=8 后：

```text
target_tokens ≈ 8 × 729,810,624
num_iterations = target_tokens // total_batch_size
               = 5,838,471,168 // 524,288
               = 11,136
```

脚本还以 d12 为参考点：

- 用 `Bopt ∝ D^0.383` 预测合适的 total batch；
- 根据 `sqrt(B/B_ref)` 缩放学习率；
- 根据训练 horizon 和 batch 修正 weight decay。

因此 `depth` 不只是“增加几层”，它会继续影响宽度、参数量、token horizon、batch、学习率和 weight decay。这正是 NanoChat 把模型深度称为主要复杂度旋钮的原因。

## 6. 优化器与数据：Muon、AdamW 和 best-fit packing 在这里汇合

[`model.setup_optimizer`](https://github.com/Qi18/nanochat/blob/f5e012fc47dbc5b5da6e4b8765b66de8000a54b7/scripts/base_train.py#L312-L324) 返回组合优化器：

- Transformer matrix 参数走 Muon；
- embedding、lm_head、标量等参数走 AdamW；
- FP16 才启用 GradScaler，BF16/FP32 不需要。

DataLoader 分成两个入口（[L334-L338](https://github.com/Qi18/nanochat/blob/f5e012fc47dbc5b5da6e4b8765b66de8000a54b7/scripts/base_train.py#L334-L338)）：

- train loader 带 state，可以从 checkpoint 恢复 parquet、row group 等游标；
- validation loader 每次评测时重新创建，保证评测从固定起点开始。

训练 batch 的三个层次必须分清：

```text
单 rank 单 micro-batch token = device_batch_size × sequence_len
全局单 micro-batch token      = 上式 × world_size
gradient accumulation         = total_batch_size / 全局单 micro-batch token
```

本次实验：

```text
2 × 2048 × 8 = 32,768 token / 全局 micro-batch
524,288 / 32,768 = 16 次梯度累积
```

## 7. 三条调度曲线：LR、Muon momentum、weight decay

训练开始前定义三类 scheduler（[L341-L391](https://github.com/Qi18/nanochat/blob/f5e012fc47dbc5b5da6e4b8765b66de8000a54b7/scripts/base_train.py#L341-L391)）：

1. LR：线性 warmup → 恒定 → 线性 warmdown；
2. Muon momentum：从 0.85 升到 0.97，warmdown 阶段降到 0.90；
3. Muon weight decay：余弦衰减到 0。

这里的 scheduler 不直接调用 PyTorch 的 `lr_scheduler`，而是在每个 step 前手动修改 optimizer param group。好处是 Muon 特有的 momentum 和 weight decay 可以与 LR 一起统一调度。

## 8. 训练循环：为什么先评测，再训练一步

主循环位于 [L394-L589](https://github.com/Qi18/nanochat/blob/f5e012fc47dbc5b5da6e4b8765b66de8000a54b7/scripts/base_train.py#L394-L589)。它的顺序是：

```text
判断当前 step 是否为 last_step
  ↓
按周期执行 validation / CORE / sample / checkpoint
  ↓
如果是 last_step，退出
  ↓
否则完成一次 forward + backward + optimizer step
  ↓
记录指标，step += 1
```

因此循环会访问 `0...num_iterations` 共 `num_iterations + 1` 个 step 状态，但只执行 `num_iterations` 次优化。最后一次进入循环是为了在“训练已经完成”的模型上做最终评测和保存，不会多训练一步。

### 8.1 Validation BPB

所有 rank 一起计算 BPB。BPB 按 token 对应的原始字节数归一化，比普通 cross entropy 更适合比较不同 tokenizer。

### 8.2 CORE

CORE 使用 `orig_model`，并在 FP8 训练时临时切到 BF16。训练内默认每个任务最多取 500 条，因此本次训练末尾 CORE `0.27082` 与后续完整数据集评测 `0.259960` 的样本口径不同。

### 8.3 Sample

只有 master process 生成固定提示词，用于快速发现模型是否仍在输出乱码、重复文本或明显事实错误。它是定性健康检查，不替代正式 benchmark。

### 8.4 Checkpoint

保存内容包括：

```text
model_<step>.pt
meta_<step>.json
optim_<step>_rank0.pt ... optim_<step>_rank7.pt
```

metadata 还保存 DataLoader state 和 loop state，所以完整恢复需要模型、metadata，以及所有 rank 的 optimizer shard。实验结束后为节省 CPFS，本项目保留了最终模型和 metadata，删除了 optimizer shard；这意味着仍能推理和复评，但不能无损续训。

## 9. 一次真正的训练 step 做了什么

训练部分位于 [L497-L548](https://github.com/Qi18/nanochat/blob/f5e012fc47dbc5b5da6e4b8765b66de8000a54b7/scripts/base_train.py#L497-L548)：

```python
for micro_step in range(grad_accum_steps):
    loss = model(x, y)
    loss = loss / grad_accum_steps
    loss.backward()
    x, y, dataloader_state_dict = next(train_loader)

optimizer.step()
model.zero_grad(set_to_none=True)
```

关键点有四个：

1. 每个 micro-step 的 loss 除以累积次数，确保累计梯度等价于大 batch 的平均梯度；
2. 下一个 batch 在当前 GPU 工作期间触发预取；
3. optimizer step 前写入当步 LR、Muon momentum 和 weight decay；
4. FP16 下所有 rank 会通过 all-reduce 统一是否跳过包含 Inf/NaN 的 step，避免不同 rank 状态分叉。

## 10. 日志和 GC：性能问题不只在 CUDA kernel

脚本记录平滑 loss、step 时间、token/s、MFU、DataLoader 游标、累计时间和 ETA。前 10 step 包含编译和 warmup，不计入正式训练时间。

另一个很有 NanoChat 风格的优化是手动管理 Python GC（[L579-L587](https://github.com/Qi18/nanochat/blob/f5e012fc47dbc5b5da6e4b8765b66de8000a54b7/scripts/base_train.py#L579-L587)）：

- 第一次训练 step 后主动 `gc.collect()`；
- 冻结已有对象并关闭自动 GC；
- 每 5000 step 手动收集一次。

原因是自动 GC 会周期性扫描大量长期存活对象，可能带来约数百毫秒的抖动。训练系统的吞吐不仅取决于矩阵乘法，也取决于 Python runtime 是否制造尾延迟。

## 11. 用本次 d24 实验验证这套结构

| 指标 | 结果 |
| --- | ---: |
| 模型参数 | 1,384,122,122 |
| Scaling parameters | 729,810,624 |
| 优化 step | 11,136 |
| 训练 token | 5,838,471,168 |
| 梯度累积 | 16 |
| 总训练时间 | 15 小时 57 分 44.67 秒 |
| 最终 loss | 2.35472 |
| 最低 validation BPB | 0.699602 |
| 训练末尾 CORE（每任务最多 500 条） | 0.27082 |
| 最终吞吐 | 约 101,500 token/s |
| 峰值显存 | 17,890.36 MiB/rank |

11,136 次优化全部完成，没有出现 NaN、OOM 或 NCCL 错误，最终模型与 metadata 已通过 SHA256 校验。实验结果说明，这个脚本的价值不只是能执行 `loss.backward()`，而是把模型规模推导、分布式训练、自动训练预算、在线评测和断点恢复放进了同一条可复现路径。

## 12. 阅读这份脚本时最值得带走的三个认识

第一，`base_train.py` 是 orchestration layer。想理解某个算法细节，应继续进入它调用的 `gpt.py`、`optim.py`、`dataloader.py` 和 `checkpoint_manager.py`。

第二，训练配置之间不是独立的。depth 改变后，宽度、参数量、token horizon、batch、学习率和 weight decay 会沿缩放律连锁变化。

第三，一次可靠预训练不只有训练 loss。validation BPB、CORE、样本、吞吐、显存、checkpoint 完整性和 DataLoader 恢复状态，都是训练是否真正完成的组成部分。

---

文中的代码结构来自固定 commit `f5e012f`；实验指标来自 `config/base-run.json`、`logs/base-final.md` 和 SwanLab run `7malxqoi`。源码事实、实验观察和解释性判断已分别注明，未将 L20 训练时间作为官方 8×H100 Time-to-GPT-2 排名。
