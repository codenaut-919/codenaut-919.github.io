---
title: DeepInfer KV Cache 管理机制梳理
date: 2026-11-03 10:00:00 +0800
categories: [deepseek系列, deepinfer拆解]
tags: [deepinfer, kv-cache, 推理框架]
---

这篇文章梳理 DeepInfer 在 GPU 侧管理 KV cache 的完整机制：显存怎么规划、物理上怎么布局、块怎么分配与记账、调度时如何分配与抢占释放、fork 的写时复制怎么做，以及对外落盘接口约定了什么。目标是把这套设计讲清楚，而不是逐行导读源码，所以下文会尽量用机制和数据流来叙述。

## 速览

- **两类资源，两个分配器**。paged KV 按 block 分配，每个 block 表示 128 个原始 token；每个 seq 另有一个固定大小的 state slot，存 SWA 环形窗口和压缩 state。两者各由一个 C++ `BlockManager` 管理。
- **分配器很简单**：双向链表 free list + 引用计数，block 0 永不分配（0 当空指针用）。没有 prefix cache、没有 LRU、没有 CPU swap。
- **显存规划**：profile 一次峰值显存，扣掉 2 倍 state 区，剩余全部切成 block，全 world all-reduce 取平均后向下取偶。
- **记账在 Sequence 上**：`num_tokens`（KV 已就绪的 token 数）采用"先乐观推进、再按实际接受数提交"，被拒绝的 draft token 不占 KV。
- **调度**：prefill 受 0.9 水位约束，新 seq 一次性分配全部 prompt block + 1 个 state slot；decode 每步每 seq 最多追加 1 个 block，不够就抢占 output 最长的 seq。
- **抢占**：默认重算（释放全部资源，放回 waiting 队首）；COMPLETION 引擎或开启结构化输出时直接以 `FINISHED_PREEMPTED` 结束。
- **fork / COW**：子 seq 先 pin 共享父 seq 的 block 和 state，真正调度时才 clone state 与最后一个半满 block，GPU 拷贝完成后再延迟 unpin。
- **落盘接口**：只落 SWA state（compression state 不落），落盘点必须是 128 token 的整数倍；保存/加载钩子在开源代码里是空实现。

## 两类资源与总体结构

### 资源划分

| 资源 | 粒度 | 存什么 | 物理张量 |
|---|---|---|---|
| paged KV block | 每 block 表示 `block_size=128` 个原始 token | 各层压缩 KV（`ca_cache`）、indexer K（`ica_cache`） | `gpu_blocks [num_gpu_blocks, bytes_per_block]` uint8 |
| state slot | 每 seq 一个 | 各层 SWA 环形窗口（`swa_cache`）、压缩 state（`ca_state` / `ica_state`，fp32） | `gpu_states [num_state_slots, state_bytes]` uint8 |

paged KV 由 `block_manager = BlockManager(num_gpu_blocks)` 管理，state slot 由 `state_manager = BlockManager(num_state_seqs)` 管理，后者在 `bytes_of_state_per_seqs == 0` 时为 None。

两个池都是"一维 uint8 大块 + 每层 view"。一个 block 内部把所有层拼在一起（layer 在 block 内，不是 block 在 layer 内），因此一个 block id 同时对应所有层的那 128 个 token 的 KV。

block_size 固定为 128。压缩率为 `r` 的层，一个 block 只存 `128 / r` 个压缩 token。

### 总体流程

```
LLMEngine.__init__
  └─ profile_num_available_blocks()
       ├─ model_runner.profile_run()          # 跑一次最大 batch，量峰值显存
       └─ num_gpu_blocks = (可用显存 - 2×state) / block_bytes → world all-reduce 平均
  └─ BlockManager(num_gpu_blocks) / BlockManager(num_state_seqs)
  └─ init_cache_engine() → CacheEngine → GPUCacheInterface（每层 KVCache view）
  └─ warm_up_model()（CUDA graph 捕获，使用真实 cache）

每个 step
  scheduler._schedule()                       # 持 BlockManagerMutexAcquire
    ├─ prefill：allocate_seq / fork 克隆
    └─ decode ：_append_slot / _preempt
  → 若有 clone_states/clone_blocks/zero_states：worker 上做 GPU 拷贝
  → prepare_tensor_input：block_table → slot_mapping / state_block_idx
  → forward：store_kv_cache 写入、load_paged_kv_cache 读取
  → process_outputs：commit_processed_tokens，free 结束的 seq
  → 下一步 pre_sampler_sync：DeferredRelease 解 pin 克隆源
```

## 显存规划：block 数是怎么算出来的

初始化阶段，引擎在每个 worker 上执行一次显存 profile，用结果反推 KV 池能切出多少 block。

### profile_run：用最大负载量峰值

profile 过程会用最大负载跑一遍，量出峰值显存：

1. 构造 `max_num_seqs` 条 dummy seq。PREFILL/NORMAL 引擎按 `max_num_batched_tokens` 分配 q/k 长度；COMPLETION 引擎每条 `seqlen_q = seqlen_k = 1 + nextn`。
2. 每条 seq 的 `block_table` 是连续递增的 block id，`state_id = seq_id % num_state_slots`。
3. 用这批 seq 实际需要的 block 数创建一个**临时真实 CacheEngine** 并清零。分配走 `cuMemAlloc`，不进 torch 显存统计，函数返回即释放，因此不干扰后面的记账；清零是因为 chunked-prefill 的全量 gather 会读到未写区域，需要保证读出有限值。
4. 运行期间把 `DEEPINFER_FORCE_IMBALANCE_RATIO` 设为 1.8，模拟 MoE 负载不均的最坏情况，跑完恢复为 0。

### 可用显存与 block 数

```python
peak_allocated   = max_memory_allocated() + max(non_torch_allocated, 0)
available_kv     = total_gpu_memory * gpu_memory_utilization - peak_allocated
state_size       = prod(get_state_shape(cfg, max_num_seqs)) * 2   # state 本体 + copy buffer
available_kv    -= state_size
num_gpu_blocks   = max(int(available_kv // cache_block_size), 0)
```

- `non_torch_allocated` = 设备已用显存 − torch 已分配，覆盖 NCCL、userbuffer、CUDA graph 池等非 torch 分配。
- `cache_block_size` = 一个 block 叠加所有层后的字节数。
- state 区按 `2×` 预留，注释写的是"state store & copy buffer"。copy buffer 对应引擎侧的 `copy_state`，开源代码里它始终是 None，但这份显存仍然被扣掉了。

随后在 **world group** 上做 all-reduce SUM，再除以 world size 并向下取偶：

```python
num_gpu_blocks = allreduce_sum(num_gpu_blocks) // world_size // 2 * 2
```

所有 rank 用同一个 block 数（取平均而非取最小）。向下取偶数的原因代码里没有注释。

### 启动校验与初始化顺序

- `num_gpu_blocks <= 0` 直接报错，提示调大 `gpu_memory_utilization`。
- `max_seq_len = block_size × num_gpu_blocks × 0.8` 必须 ≥ `runtime_max_model_len`，即单条最长请求最多只能吃掉 80% 的 KV 池。
- 之后依次执行 `init_cache_engine` → `init_pd_split`（非 NORMAL 引擎，开源为空实现）→ `warm_up_model`（CUDA graph 捕获）。

### state slot 数量

```python
num_state_slots = int(max_num_seqs * DEEPINFER_NUM_STATE_SLOTS_RATIO) + 1   # 默认 ratio = 1.2
```

`+1` 是因为 `BlockManager` 永远不分配 0 号，所以真正可用的 slot 是 `int(max_num_seqs × 1.2)` 个。代码没有注释多出 20% 的用途；从调度逻辑看，fork 出的子 seq 在 clone 前与父 seq 共享 slot，clone 时才需要新 slot，比例余量可以让这类场景不被 `max_num_seqs` 卡死（推断）。

同理，paged KV 池可用的 block 数是 `num_gpu_blocks − 1`。

## 物理布局

布局全部由 cache engine 侧计算。核心入口是 `GPUCacheInterface._kvcache_layout(cfg, block_size, max_num_seqs)`，它用 `KVCacheManager` 逐层 `process()`，同时返回：

- `layout`：每层 `(layer_id, kv_offset, kv_length)`，只包含占用 paged KV 的层；
- `state_size_to_disk`：需要落盘的 state 字节数；
- kv shape `(bytes_per_block,)`；
- state shape `(num_state_slots, state_bytes)`；
- `cache_list`：每层一个 `KVCache` view。

`block_size = -1` 是"只要 state shape"的约定，profile 阶段估算 state 大小时就这样调用。

### 每层的 cache 视图

| 字段 | 所在池 | shape | 含义 |
|---|---|---|---|
| `ca_cache` | paged KV | `[num_blocks, 128 // r, 1, bytes_per_token]` uint8 | 主注意力的压缩 KV，压缩率 `r = kv_compress_ratio` |
| `ica_cache` | paged KV | `[num_blocks, 128 // r, 1, index_bytes]` uint8 | 稀疏层 indexer 的 K |
| `swa_cache` | state slot | `[num_state_slots, swa_block_size, 1, 528]` uint8 | 滑窗注意力的环形窗口 |
| `ca_state` | state slot | `[num_state_slots, ring, 2, dim]` fp32 | 主 KV 压缩器尚未凑满 `r` 个 token 的中间状态 |
| `ica_state` | state slot | `[num_state_slots, ring, 2, index_dim]` fp32 | indexer 压缩器的中间状态 |

每 token 字节格式：

| 用途 | 格式 | 字节/token |
|---|---|---|
| SWA（`V41_FORMAT`） | 512 × e4m3 + 每 32 通道 1 个 ue8m0 scale | 528 |
| 主 KV 默认（`V41_FP4_FORMAT`） | 512 × e2m1 + 每 16 通道 1 个 e4m3 scale | 288 |
| indexer K（FP4） | `index_head_dim / 2 + index_head_dim / 32`，128 维时 | 68 |

### paged KV block 的内部排布

一个 block 是一段连续字节，按 runtime 层序把各层依次排进去：

```
block i: [L0.ica | L0.ca | L1.ica | L1.ca | ... | Ln.ca | pad]
          └ 每段按 32B 对齐            └ 总长对齐到 lcm(align_to, bytes_per_token)
```

- 每段长度 = `(128 // r) × 每 token 字节数`，再向上对齐到 32B，offset 累加。
- 只有 SWA 的层不占 paged KV，`kv_length = 0`，不进入 `layout`。
- 总长对齐到 `lcm(align_to, bytes_per_token)`：DeepGEMM 的 index kernel 要求 block stride 是 128B 的倍数，主注意力要求 stride 是 `bytes_per_token` 的倍数。
- 每层 view 用 `as_strided`，stride(0) 是整个 block 的字节数，所以 `ca_cache[block_id]` 直接定位到该 block 中这一层的那一段。
- full attention 层（无压缩）在这里会触发 assert，即 `KVCacheManager` 只支持 SWA + 压缩注意力的 V4 系列结构。

**共享层**：`layer_info_shared_cfg` 中的 `layer_kv_map` / `layer_index_module_map` / `layer_index_k_map` 把某层映射到前面某层（必须 `mapped <= layer_id`）。被映射的层直接复用 `cache_list[mapped]` 的 `ca_cache` / `ica_cache`，不推进 offset，也不分配 state。

### state slot 的内部排布

```
state slot s: [ stored region: 需落盘的 SWA 层 ... ][ dropped region: YoCo consumer SWA 层 ... | compression state ... ]
               └ state_size_to_disk 只取这一段（inference 布局）
```

- **SWA 分组**：`_is_dropped_swa_layer(cfg, layer_id)` 在 `n_prefill_layer < num_hidden_layers` 时把 `layer_id >= n_prefill_layer` 的层划入 dropped region。这些是 YoCo consumer 层，恢复时可以靠 prefill 重算，所以不落盘。分组只由模型 config 决定，与 YoCo 截断模式无关。
- **SWA 占位**：`_has_SWA` 只看 `sink_mode == "swa"`，有意不看 `use_swa`。`use_swa=False` 的 consumer 层也占一个永不读写的 SWA 槽，以便所有层共用同一套 slot mapping。
- **每层 SWA 大小**：`swa_block_size × 528`，再对齐到 32B。
- **compression state**：只给 `ratio > 1` 且"拥有"该 cache（未被共享映射）的层分配，大小为 `ring × 2 × dim × 4` 字节（fp32）。
- **整体对齐**：存在 SWA 时总长对齐到 `lcm(32, 528)`，因为 `swa_cache.stride(0)` 继承整个 slot 的长度。
- 两套尺寸：`*_kvstore` 是 GPU 上实际分配的尺寸，`*_inference` 是推理/传输需要的尺寸。不开 `DEEPINFER_PREFILL_STORE_EXTRA_STATE` 时两者相等。

### 环形窗口尺寸

| 名称 | 公式 | 默认值（window=128，nextn=5） |
|---|---|---|
| `DEEPINFER_MAX_BACKTRACKING_TOKENS` | `max(env, nextn + 1)` | 6 |
| SWA inference ring | `window + MAX_BACKTRACKING` | 134 |
| SWA kvstore ring | inference ring + `(PREFILL_STORE_EXTRA_SIZE + YoCo KEEP)`，仅在开启 EXTRA_STATE 时加 | 134（未开启）/ 262（开启，EXTRA_SIZE=128，YoCo 关） |
| compression ring | `r + max(0, MAX_BACKTRACKING − 1)` | `r + 5` |

多出的 `MAX_BACKTRACKING` 是给投机解码回退用的：一步最多写入 `1 + nextn` 个 token，若 draft 被拒，需要回到更早的窗口位置，环里必须还保留着被覆盖前的数据。

以 V4.1 为例：前 20 层 SWA 需要落盘，`state_size_to_disk = 20 × 134 × 528 = 1,415,040 B`。

### CacheEngine 实际分配

```python
blocks     = alloc_tensor(num_gpu_blocks × bytes_per_block)        # uint8，cuda_utils.alloc_tensor
gpu_blocks = blocks.view(num_gpu_blocks, -1)
gpu_states = alloc_tensor(num_state_slots × state_bytes)           # state_bytes == 0 时退化为空 tensor
gpu_cache  = GPUCacheInterface(cache_config, cfg, gpu_blocks, gpu_states)   # 内部再跑一次布局切 view
```

分配完成后把 `bytes_of_state_per_seqs`、`num_state_seqs`、`bytes_per_block` 回写到 `cache_config`。引擎据此决定是否创建 `state_manager`，PD 分离的传输层也用这几个值。日志会打印 block 内 padding 字节数及占比。

## BlockManager：块分配器

paged KV 和 state slot 共用同一套 `BlockManager` 代码。

### 数据结构

- `std::vector<block>`，每个元素含 `ref_count`、`prev`、`next`。
- free list 是嵌在数组里的双向链表。**0 号永不分配**，链表指针用 0 表示空；构造时链表为 `1 → 2 → … → n-1`，`_size = n - 1`。
- `allocate()`：从表头 `_pop_front`，`ref_count += 1`。
- `pin_block(id)`：要求 `ref_count > 0`（不能 pin 空闲块），`ref_count += 1`。
- `free(id)`：`ref_count -= 1`，降到 0 时 `_push_back` 挂回表尾。
- `reset()`：只做检查，要求所有块都已空闲。

分配从表头取，释放挂到表尾，因此空闲块大致按"最早释放最先复用"的顺序循环使用。释放一个 block_table 时逆序释放，所以同一条 seq 的块挂回链表时是倒序的。

### 对外接口

| 接口 | 语义 |
|---|---|
| `can_allocate_nolock(n)` | `n >= num_blocks` → `NEVER`；空闲 ≥ n → `OK`；否则 `LATER` |
| `allocate_nolock(n)` / `allocate_single_nolock()` | 分配 n 个 / 1 个 |
| `can_allocate_single_nolock()` | 空闲 ≥ 1 |
| `free_blocks(_nolock)` | 逐个 `ref_count - 1` |
| `pin_blocks(_nolock)` / `pin_block` | 逐个 `ref_count + 1` |
| `get_num_free_blocks()` | 空闲数 |

pybind 层把 `unpin_blocks` 绑定为 `free_blocks` 的别名。"unpin"和"free"是同一个操作，区别只在调用方的意图。

实际可用块数是 `num_blocks - 1`（0 号保留），`n >= num_blocks` 即 `n > num_blocks - 1` 时判 `NEVER`，边界是对的。`get_gpu_cache_usage()` 则用 `num_blocks` 作分母，0 号块被计入"已用"，所以 prefill 水位判断里的使用率比真实值略高（多 1/num_blocks）；打日志时用 `num_gpu_blocks - 1`，两处口径不同。

### 锁

`BlockManager` 继承 `deepinfer::Lockable`。`BlockManagerMutexAcquire` 是 RAII 锁守卫：先锁 `state_manager` 再锁 `block_manager`，析构时逆序解锁。`_schedule()` 整个调度过程都持有它，内部一律调用 `*_nolock` 接口；Python 侧的 `DeferredRelease` 调用带锁的 `unpin_blocks`，因此和调度互斥。

它还提供两个组合操作：

- `allocate_seq(seq)`：`block_table = allocate_nolock(seq.get_num_blocks())`，并分配 1 个 state slot 写入 `state_id`。
- `try_allocate_seq(seq)`：要求 seq 处于 WAITING、`block_table` 为空、`state_id == -1`；资源不足返回 false，`NEVER` 直接 `TORCH_CHECK` 失败。供外部（PD 分离等）预先占位使用。

## Sequence 上的 KV 记账

每条 seq 用一个 `KvCacheState` 记录自己的 KV 账：

| 字段 | 含义 |
|---|---|
| `block_size` | 128 |
| `num_tokens` | KV 已经就绪（已写入 cache）的 token 数，对外叫 `num_cached_tokens` |
| `previous_num_tokens` | 本步乐观推进前的值，用于回滚 |
| `hit_tokens` | 外部命中的 token 数（fork 时继承） |
| `swa_valid_start` | zero-state 重算时 SWA state 的有效下界，正常路径为 0 |
| `block_table` | paged KV 的 block id 列表 |
| `state_id` | state slot id，-1 表示未分配 |
| `last_copy_state_token_idx` | 上次快照 state 时的 token 位置（copy_state 路径用） |

### 需要多少 block

```cpp
get_num_blocks() = ceil((get_len() + total_inflight_tokens + draft_tokens.size()) / block_size)
```

- `get_len()`：prompt + 已输出 token 总长。
- `total_inflight_tokens`：ZOS（零开销调度）下已经发射、结果尚未回来的 step 占用的 token。
- `draft_tokens.size()`：本步投机 draft 数（nextn）。

也就是说，block 按"这一步最多可能写到的位置"预留。decode 时 block_size=128 远大于单步写入量 `1 + nextn`，所以 `_append_slot` 每步最多只需追加 1 个块。

`_block_table_capacity = ceil((len + max_tokens) / 128) + 1` 只是给 `std::vector::reserve` 用的容量，避免 decode 过程中反复扩容，**不会**提前占用 block。

### 乐观推进与提交

```cpp
advance_optimistically(n):  previous_num_tokens = num_tokens; num_tokens += n;
commit_processed_tokens(n): num_tokens = previous_num_tokens + n;
set_num_tokens(v):          num_tokens = previous_num_tokens = v;   // 外部命中注入
reset_for_recompute():      num_tokens = previous_num_tokens = swa_valid_start = 0;
```

`process_outputs` 中，每个参与本步的 seq 先 `advance_optimistically(seqlen_q)`；decode 步的 `seqlen_q = 1 + draft 数`，推进后 `num_tokens` 落在 `[len, len + nextn]`。随后把接受的 draft 和新采样 token 追加进序列，再 `commit_processed_tokens(token_cnt)` 回填为实际接受数（`token_cnt` = 接受的 draft 数 + 1）。被拒绝的 draft 已经写进了 cache，但 `num_tokens` 不再覆盖它们，下一步会被覆写。提交后检查 `num_tokens <= len`；decode 稳态下 `num_tokens = len - 1`，因为最新采样出的那个 token 要到下一步作为输入才写 KV。

paged KV 的写入位置由 `num_tokens` 推导，SWA 和压缩 state 是环形的，靠环里多留的 `MAX_BACKTRACKING` 个 token 保证回退后数据仍在。

## 调度中的分配

`Scheduler::_schedule()` 开头构造 `BlockManagerMutexAcquire`，并清空 `clone_states` / `clone_blocks` / `zero_states` 三个缓冲。这三个缓冲以引用形式借给 `SchedulerOutputs`，Python 侧必须在下一次 schedule 前用完。

### prefill 何时允许

NORMAL 引擎在投票点上判断本 rank 是否"想做 prefill"：

```cpp
want = get_gpu_cache_usage() <= GPU_CACHE_PREFILL_WATERMARK /* 0.9 */ && _has_prefill_waiting();
```

KV 池使用率超过 90% 时本 rank 不再发起 prefill，把余量留给正在 decode 的 seq 逐步追加 block。投票机制（ALL/ANY 点、二次确认）属于步型决策，这里不展开。

### prefill 调度

从 `waiting` 队头依次处理，每条 seq 按 block_table 的状态分三类：

**(a) 新 seq（`block_table` 为空）**

1. `len > prompt_limit` → `FINISHED_IGNORED`。
2. `_can_allocate_with_state(get_num_blocks())`：block 不够或 state slot 不够都算 `LATER`，直接 `break`（队头阻塞，后面的 seq 也不调度）；`NEVER`（超过整个池）→ `FINISHED_IGNORED`。
3. 通过其余约束（`max_num_prompts`、`prefill_max_sum_seqlen_k`、chunk 切分）后，在"确定调度"段调用 `acquire.allocate_seq(*seq)`。

注意 `get_num_blocks()` 按整条 prompt 计算。**即使这一步只跑第一个 chunk，也会一次性分配整条 prompt 需要的全部 block 和 1 个 state slot**。外部命中（`set_num_tokens` 注入）的 token 同样占 block，`num_prompt_tokens = len - num_tokens` 只决定计算量。

**(b) chunked prefill 的续跑 seq（`block_table` 非空、非 fork）**

资源在第一个 chunk 时已经分配完毕，这里不再分配，只计算本步的 `seqlen_q`。

**(c) fork 出的 seq（`branch_id` 以 `_` 开头）**，见后文 fork 一节。

代码里变量名叫 `prefix_caching_hit`，但它在 (a)、(c) 两种情况下置 true，实际含义是"首次进入 prefill"，与 prefix cache 无关。

chunk 切分规则：受 `max_num_batched_tokens` 剩余量约束并按 `cp_align` 对齐，遇到 `chunk_checkpoints` 时截断到 checkpoint，多模态不能切在图片中间。NORMAL 引擎一步最多只有一条 chunked seq。未跑完的 chunked seq 在 `process_outputs` 中被放回 `waiting` 队首。

### decode 调度

1. `free_finished_seqs()`：把 `running` 中已结束的 seq 移出并释放 MTP slot。**此处不释放 block**，block 在本步 `pre_sampler_sync` 中由 `finalize_seq → scheduler.free_seq` 释放。因此一条 seq 结束后，它的 block 要晚一步才能被复用。
2. 清理 `waiting` 中已终结的 seq（abort、VE 失败等）。
3. 把 `prompted` 队列（prefill 完成待 decode 的 seq）转入 `running`，fork seq 先做 state/block 克隆。
4. `running` 按 `output_len` 降序排序，从尾部（output 最短）开始逐个调度：

```cpp
l = 0, r = running.size();
while (l < r && scheduled.size() < max_seqs) {
    seq = running[r - 1];                                   // 当前候选：output 最短的
    while (!can_allocate_single() && r - l > 1)
        _preempt(running[l++]);                             // 抢占 output 最长的
    if (can_allocate_single()) { _append_slot(seq); scheduled.push(seq); }
    else _preempt(seq);
    r -= 1;
}
```

5. `_append_slot`：`block_table.size() < get_num_blocks()` 时追加 1 个块；大于时（例如上一步多留的 draft 位置不再需要）释放多余的尾部块。最后 `TORCH_CHECK` 两者相等，所以单步增长超过 1 个块会直接报错。
6. `scheduled` 按 `len` 降序重排，交给 `_mtp_varlen.schedule` 决定每条 seq 的 `seqlen_q`。

关于抢占对象：函数注释写的是"LCFS，抢占 output 最短的 seq"，但代码的实际行为是从 `running[l]`（output_len 最大）开始抢占，优先保住 output 最短的 seq。阅读时以代码为准。

## 抢占与释放

### 抢占

| 条件 | 行为 |
|---|---|
| `stop_preempted_request == true`，或 seq 带结构化输出（`structured_output.status != 0`） | 状态置 `FINISHED_PREEMPTED`（finish reason `insufficient_system_resource`），`free_seq_nolock`，放入 `ignored_seqs` 返回给用户 |
| 其它 | `reset_for_recompute()`，放回 `waiting` **队首**，状态置 WAITING，`free_seq_nolock` |

重算模式会丢掉全部 KV，重新 prefill 整条 prompt 和已生成的 token。结构化输出的 grammar 状态无法回滚，所以不支持重算。COMPLETION 引擎（PD 分离的 decode 侧）没有 prompt 可重跑，构造函数里强制要求 `stop_preempted_request`。

### 释放

```cpp
release_mtp_info_slot(seq);
if (state_id != -1) state_manager->free_blocks_nolock({state_id});
block_manager->free_blocks_nolock(block_table);
block_table.clear();
```

这里只做 `ref_count - 1`。fork 共享的块要等所有持有者都释放后才真正回到 free list。`state_id` 释放后没有复位为 -1：重算的 seq 再次走 `allocate_seq` 时会被直接覆盖，没有问题；但如果对它调用 `try_allocate_seq`，会因 `state_id == -1` 的检查失败而报错。

### ZOS 下在途 seq 被抢占

开启 `DEEPINFER_ZERO_OVERHEAD_SCHEDULING` 后，调度下一步时上一步的结果还没回来。若某个在途 seq 在这次调度中被抢占（重算），`process_outputs` 检测到它在 decode 步中处于 WAITING，直接作废本步输出，不推进 `num_tokens`，避免重复入队和计数错乱。在途 token 数由 `start_inflight_step` / `finish_inflight_step` 维护，并计入 `get_num_blocks()`。

## fork 与写时复制

fork 让一条新 seq 复用另一条 seq 已计算好的前缀 KV。开源仓库里唯一的调用方是 benchmark：先跑一条源请求算出共享前缀，再 fork 出 `max(N, K)` 条请求各自计算非缓存后缀。

### fork_seq

1. 前置检查：父 seq `num_tokens > 0`；不支持 `prompt_logprobs`。
2. 复制 token（可用 `replace_last_token_id` 替换最后一个），`hit_tokens` 继承父 seq。
3. `block_table`：`blocks` 参数为 None 时直接复制父 seq 的 block_table，否则使用调用方给的 block 列表。
4. `set_num_tokens(min(parent.num_tokens, len - 1))`：至少留最后 1 个 token 重新计算，以便产生 logits。
5. **只 pin，不拷贝**：`block_manager->pin_blocks(block_table)`，子 seq 的 `state_id` 直接设为父 seq 的 `state_id` 并 `pin_block`。此时父子共享全部 block 和同一个 state slot。
6. `branch_id` 由调用方传入，约定以 `_` 开头作为"待克隆"标记。prompt 还没算完（`num_tokens < len - 1`）的子 seq 进 `waiting` 队首，否则进 `prompted` 并带 `forked = true`。

### prefill 侧的克隆

```
num_reserve = num_tokens / 128                      # 完整块数，继续共享
to_alloc    = get_num_blocks() - num_reserve
_clone_state(seq)                                   # 新 state slot，(旧, 新) → clone_states
new_blocks  = allocate(to_alloc)
if num_tokens % 128:  clone_blocks += (table[num_reserve], new_blocks[0])   # 半满的尾块需要拷贝
for 其余父块:         clone_blocks += (table[i], 0)                         # dst = 0 表示只解 pin
table = table[:num_reserve] + new_blocks
```

- 完整块只读共享：父 seq 不会再写 `num_tokens` 之前的完整块，共享是安全的。
- 半满尾块需要复制：父子都会往这个块的后半段写各自的 token。
- 克隆时 `branch_id` 去掉前缀 `_`，之后这条 seq 按普通 chunked 续跑处理。
- 克隆在后续约束检查（`max_num_prompts` 等）**之前**提交。如果后面因为约束 `break`，seq 留在 waiting 中，但资源已经换成自己的了，下次按普通续跑 seq 处理。
- fork seq 的 `NEVER` 是硬错误（`TORCH_CHECK`），超长 fork 会先 `free_seq_nolock` 释放 pin 再 IGNORED。

### decode 侧的克隆

prompt 已算完的 fork seq 在 `prompted → running` 时克隆，要求 state 和 block 各有 1 个空闲：

```
_clone_state(seq)
n = ceil(num_tokens / 128)
clone_blocks += (table[n-1], new_block)      # 最后一个已用块总是拷贝（不判断是否半满）
for i in [n, len(table)): clone_blocks += (table[i], 0)
table[n-1] = new_block; table.resize(n)
```

### GPU 拷贝与延迟解 pin

1. 调度结束后，引擎检查三个缓冲是否非空，非空则在所有 worker 上执行 `clone_and_zero_states_and_blocks_on_default_stream`。
2. C++ 实现在当前 stream 上：
   - 对每个 `(src, dst)` state 对，`cuda_copy_async` 整个 state slot；
   - 对每个 `(src, dst)` block 对，`dst != 0` 时拷贝整个 block（所有层），`dst == 0` 时跳过；
   - 对 `zero_states` 中的每个 slot 执行 `zero_()`；
   - 返回所有 src 列表，即需要解 pin 的 state 和 block。
3. 引擎调用 `deferred_release.collect_clone_unpins(...)`，同时在当前 stream 上 record `clone_done_evt`。
4. 同一步的 `pre_sampler_sync` 开头调用 `DeferredRelease.release()`：先 `clone_done_evt.synchronize()` 等拷贝完成，再 `state_manager.unpin_blocks(states)`、`block_manager.unpin_blocks(blocks)`。

延迟到这里解 pin，是为了保证 src 在 GPU 拷贝读完之前不会被重新分配、覆写。解 pin 和本步 forward 在时间上重叠，不额外阻塞 CPU 调度。

`zero_states` 和 `swa_valid_start` 的接口都已打通（kernel 中 SWA 可见窗口会 clamp 到 `swa_valid_start`），但开源 C++ 调度器里没有任何地方往 `zero_states` 里填数据，也没有把 `swa_valid_start` 设成非 0。这条"清零 state 后重算"的路径只留了接口。

## 从 block_table 到 kernel：slot_mapping

每步 prepare 阶段把 Sequence 上的 `block_table` / `state_id` 展平成 kernel 需要的索引张量。

### 写入：paged KV（压缩 KV / indexer K）

对每个压缩率 `n`，`compressed_block_size = 128 / n`，第 `j` 个压缩 token 的物理 slot：

```cpp
slot = block_table[j / compressed_block_size] * compressed_block_size + j % compressed_block_size;
```

- prefill：为本 chunk 覆盖的每个完整压缩 token 生成一个 slot；本 chunk 一个完整压缩 token 都没产生的 seq 补一个 `-1` 占位。
- decode：输出 shape 为 `[total_q]`，只有 `(pos + 1) % n == 0`（凑满 n 个原始 token）的 query 位置有 slot，其余为 `-1`。
- 没凑满 n 个的原始 token 不进 paged KV，而是暂存在 state slot 的 `ca_state` / `ica_state` 环里，凑满后由压缩器输出一个压缩 token 写入 paged KV。

`store_kv_cache` kernel 遇到 `slot == -1` 跳过。

### 写入：SWA 环

```cpp
if (seqlen_q - j <= swa_block_size)
    slot = state_id * swa_block_size + (seqlen_k - seqlen_q + j) % swa_block_size;
else
    slot = -1;
```

只有每条 seq 本步最后 `swa_block_size` 个 token 会写入 SWA 环，更早的 token 反正会被覆盖，直接跳过。环下标就是 `绝对位置 % ring_size`。YoCo compact 模式下，被截断的 clip 占位 token 的 SWA slot 也是 -1。

`state_block_idx`：prefill 每 seq 一个 `state_id`，decode 每个 query token 一个，供压缩 state 和 SWA 读取定位 slot。

### 读取

- `get_kv_index`：`block_table[pos // tpb] × tpb + pos % tpb`，`tpb = 128 / ratio`。稀疏注意力先由 indexer 选出 top-k 压缩 token 位置，再用这个公式翻译成物理 slot。
- `load_paged_kv_cache` 按 `BlockLoadPlan` 批量 gather。
- ZOS decode 把各 seq 的 block_table 拼成扁平数组加 offsets。每条 seq 截取的块数是 `ceil(max_k / ratio / tpb)`，其中 `max_k = seqlen_k - seqlen_q + max_seqlen_q`，覆盖 GPU 侧 varlen 修正后 q 可能增长到的长度；`block_table_compress_ratio` 参数决定按哪个压缩率计算。

## 外部 KV 接口与落盘

开源仓库没有自带 KV store 或 PD 分离的实现，但 cache 侧留好了数据格式和函数，引擎侧留了虚函数钩子。本节说明这些接口约定了什么。

### 落盘粒度与落盘内容

- `KVSTORE_GRAIN = 128`，与 block_size 相同。落盘只发生在 128 的整数倍位置。
- paged KV（`ca_cache` / `ica_cache`）按 block 落盘即可，因为一个 block 正好是 128 个原始 token 对应的压缩 token。
- 压缩 state（`ca_state` / `ica_state`）不落盘。理由是：压缩窗口在 128 边界处正好闭合，没有残留的未压缩 token。
- 只有 stored SWA 需要落盘。`state_size_to_disk` 只覆盖 StateBlock 开头的 stored SWA 区域。以 V4.1 为例：20 层 × 134 token × 528 B = 1,415,040 B。
- dropped region（YoCo consumer SWA 和压缩 state）不落盘，只在 PE→CE 传输（TO_CE）时整体发送。

### EXTRA_STATE：为什么 kvstore 环更大

落盘点是 128 的倍数，但真正执行落盘时，seq 可能已经越过边界最多 127 个 token。推理用的 SWA 环只有 `window + MAX_BACKTRACKING` 个槽位，边界前的数据可能已经被覆盖。

开启 `DEEPINFER_PREFILL_STORE_EXTRA_STATE` 后，SWA 环额外多留 `PREFILL_STORE_EXTRA_SIZE`（默认 128）个槽位，YoCo 打开时再加 `YOCO_PREFILL_KEEP_TOKENS`。这样在落盘时，边界前的完整窗口仍然在环里。代价是两个环的大小不同，读写时需要重排：

| 环 | 大小（V4.1 默认） | 用途 |
|---|---|---|
| inference 环 `bs_inf` | `window + MB` = 134 | 落盘 / 传输的数据格式 |
| kvstore 环 `bs_kvstore` | `bs_inf + 128 (+ KEEP)` | 显存里实际的 state slot |

两者都按 `绝对位置 % 环大小` 寻址，所以重排是一次环到环的 gather/scatter。PyTorch 参考实现（正式路径是同逻辑的 tilelang kernel）：

```python
index = torch.arange(last_hit - bs_inf, last_hit)
tensor[:, index % bs_inf] = cache[:, index % bs_kvstore]   # select_state_slice（读出）
cache[:, index % bs_kvstore] = tensor[:, index % bs_inf]   # load_state_to_cache（写回）
```

测试用例验证 kernel 与参考实现一致，其中命中位置取 `last_process_position // 128 * 128`。

`_carve_state_views` 把 state slot 和目标 buffer 切成 `[层数, 环 token 数, 每 token 字节数]` 的视图，每个 token record 内包含主数据和 scale，一次拷贝完整搬走。

### 读出与写回函数

**`select_state_slice_to_store(state_id, last_hit_position, state_cache, state_tensor=None, TO_CE=False)`**

1. 非 TO_CE（落盘）时断言 `last_hit_position % 128 == 0`；TO_CE 无对齐要求。
2. 未开启 EXTRA_STATE：两个环大小相同，不需要重排。直接取 `state_cache[state_id, :size]`，其中落盘取 `state_size_to_disk`，TO_CE 取 `total_bytes_inference`。没传 `state_tensor` 时返回的是 state slot 的切片视图，不是拷贝，调用方必须在 slot 被复用前用完。
3. 开启 EXTRA_STATE：切出视图后对 stored SWA（TO_CE 时还有 consumer SWA）逐个调用 `select_state_slice` 重排。TO_CE 时，dropped region 里的压缩 state 部分不按环重排，直接按字节拷贝。

**`load_state_from_store_to_prefill_cache(state_id, last_hit_position, state_tensor, state_cache)`**

1. 模型有 dropped SWA 层时，断言 `YOCO_PREFILL_TRUNCATION_MODE != "off"`。落盘数据不含 dropped 层，推断是因为只有开启 YoCo 截断时，续算的 prefill 才不需要这些层的历史 state。
2. 未开启 EXTRA_STATE：直接写入 `state_cache[state_id, :state_size_to_disk]`。
3. 开启 EXTRA_STATE：调用 `load_state_to_cache`，按环重排写回。

### 外部命中的注入方式

`KvCacheState::set_num_tokens(value)` 同时设置 `num_tokens` 和 `previous_num_tokens`，用途是"外部注入命中数时同步建立增量基线"。结合 `try_allocate_seq`，可以推断外部 KV store 的接入顺序是：

1. 查询外部存储，得到命中长度（128 的倍数）。
2. `set_num_tokens(hit)`，让调度器认为前 `hit` 个 token 已有 KV。
3. `try_allocate_seq(seq)` 一次性分配整条 prompt 的 block 和一个 state slot；资源不够时返回 false，`NEVER` 时直接报错。
4. 把外部 KV 写入对应的 block，用 `load_state_from_store_to_prefill_cache` 恢复 SWA。
5. 调度器之后只对 `[hit, len)` 做 prefill。

开源代码里唯一调用 `try_allocate_seq` 的地方是 benchmark：用它直接给源请求分配资源，再用 `fork_seq` 复制共享前缀，不走外部存储。上述第 1、4 步没有开源实现。

### 引擎侧钩子

引擎里有一组虚函数，基类 `LLMEngine` 里大多是空实现，留给 PD 分离 / 落盘的子类覆盖：

| 钩子 | 调用时机 | 基类行为 |
|---|---|---|
| `init_pd_split()` | 初始化 | 空 |
| `post_schedule(outputs)` | 每步调度后 | 空 |
| `should_copy_state(seq)` | decode 步 forward 后，决定是否快照 state | 返回 False |
| `do_copy_state(seqs)` | 同上，`copy_state` 非 None 时 | 把 `gpu_states[state_id]` 拷到 `copy_state[state_id]`，记 `last_copy_state_token_idx` |
| `do_save_kvcache(scheduled, finished)` | decode 步的 `pre_sampler_sync`，先等 `copy_state_event` | 空 |
| `pe_prepare_save_kvcache` / `pe_do_save_kvcache` | prefill 步 | 空 |
| `send_processed_prompts()` | PE 把 prefill 完的 seq 发给 CE 并从调度器删除 | 空 |
| `finalize_seq(seq)` | 结束的 seq（COMPLETION / NORMAL 引擎） | `scheduler.free_seq(seq)` |

快照路径的用意：decode 时 state 是滚动环，每步都会被覆盖；落盘需要的是某个 128 边界时刻的 state，所以先拷进 `copy_state` 这块"影子"缓冲，再在下一步 `pre_sampler_sync` 里异步落盘。

基类中 `copy_state` 始终为 None，这条路径不会执行。但 profile 阶段仍然按 2 倍 state 预留了显存（注释为 "state store & copy buffer"）。不用落盘时，这部分显存是空置的。

## 限制、边界与需要注意的点

### 机制层面没有的东西

- **没有 prefix cache**：block 只按 seq 私有分配，没有基于 token hash 的跨请求复用。块共享只发生在 `fork_seq`（同一请求派生出的 n 个样本或 benchmark 显式 fork）。
- **没有 LRU 回收、没有 swap**：`SequenceStatus::SWAPPED` 定义了但没有任何代码路径进入该状态。显存不够时只有抢占（重算或直接结束）。
- **不支持 full attention 层**：`KVCacheManager` 只处理压缩注意力和 SWA 两类层，遇到无压缩的 full attention 层会触发 assert。

### 行为上值得注意的细节

| 细节 | 影响 |
|---|---|
| 新 seq 一次性分配整条 prompt 的 block，即使 chunked prefill 分多步算 | 长 prompt 早早占满显存，并发数受 prompt 总长限制 |
| 结束的 seq 在下一步 `pre_sampler_sync` 才释放 | 释放比结束晚一步，满载时有一步的资源空档 |
| 水位用的使用率把 0 号块算作已用，日志里的使用率不算 | 两处数字有 1/num_blocks 的差，block 少时才明显 |
| 抢占注释写"最短"，代码选 `running[l]`，即输出最长的 seq | 被抢占的往往是已投入最多算力的请求，重算代价最大 |
| `free_seq_nolock` 不把 `state_id` 复位为 -1 | 重算走 `allocate_seq` 会覆盖，没问题；对被抢占过的 seq 调 `try_allocate_seq` 会因 `state_id == -1` 检查失败 |
| `zero_states`、`swa_valid_start` 只有接口 | 清零 state 后部分重算的路径没有调用方 |

### 配置上的坑

- **YoCo KEEP 的默认值**：`DEEPINFER_YOCO_PREFILL_KEEP_TOKENS` 没有默认值（等于 0），而官方推荐 V4.1 用 140，benchmark 和示例脚本的默认值也是 140。开启 `YOCO_PREFILL_TRUNCATION_MODE` 时必须显式设置，否则会断言失败；它还必须不小于 `window + MAX_BACKTRACKING`（V4.1 为 134）。
- **KEEP 会放大 state slot**：开启 EXTRA_STATE 且 YoCo 打开时，kvstore 环长度是 `134 + 128 + KEEP`，KEEP=140 时为 402，每个 state slot 的 SWA 部分约是不开时的 3 倍，可用 block 数相应减少。
- **nextn 影响环长度**：`MAX_BACKTRACKING = max(env, nextn + 1)`，调大 `DEEPINFER_USE_NEXTN_DECODE` 会同时增大 SWA 环和压缩 state 环。
- **`NUM_STATE_SLOTS_RATIO`**（默认 1.2）决定 state slot 数 `int(max_num_seqs × 1.2) + 1`。state slot 不够时 prefill 会被判 `LATER`，即使 block 还有空闲。
