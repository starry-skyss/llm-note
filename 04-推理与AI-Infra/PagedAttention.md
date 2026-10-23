---
type: concept
domain: 推理与AI-Infra
status: active
aliases:
  - 分页注意力
  - vLLM PagedAttention
---

# PagedAttention

> [!abstract] 核心结论
> PagedAttention 把每条序列逻辑上连续增长的 KV Cache 映射到固定大小、物理上可不连续的块，类似操作系统分页。它减少预留浪费和外部碎片，支持按需增长、共享与回收，从而让有限显存容纳更多并发请求。

## 连续缓存的问题

在线请求的 prompt 长度、生成长度和结束时间各不相同。如果为每条序列按最大长度预留连续显存，会出现：

- 实际只生成少量 token，却占用整块保留容量；
- 序列增长时缺少相邻空间，需要搬迁或重新分配；
- 长短请求反复进入和退出，形成难以利用的碎片；
- 为避免 OOM 只能保守降低 batch 和并发。

问题的核心不是 Attention 公式，而是动态增长的 [[KV Cache]] 如何映射到物理显存。

## 逻辑块与物理块

PagedAttention 把每条序列按 token 划分为固定容量的逻辑块，并维护块表：

```text
请求 A 的逻辑块:  [0] [1] [2]
块表映射:          7 → 2 → 11
物理显存块:       ... [2] ... [7] ... [11] ...
```

逻辑序列仍保持顺序，但对应物理块无需连续。Attention kernel 根据块表定位历史 K/V，并把多个块视为一条逻辑序列。

最后一个块可能存在少量内部碎片，但不再需要为未知最大长度预留整段空间；块可从全局空闲池按需分配，序列结束时立即归还。

## 请求生命周期

```text
创建请求
→ 为 prompt 分配若干块
→ Prefill 写入 K/V
→ Decode 写满当前块
→ 需要时追加新块
→ 请求结束/取消
→ 释放其物理块
```

这使调度器可以在每轮 Decode 重新组织活跃请求，即 continuous batching。新请求不必等待旧 batch 全部结束，已完成序列也能及时释放块和计算槽位。

## 共享与 Copy-on-Write

多个候选序列或请求若拥有完全相同的前缀，可以让逻辑块表暂时指向同一物理块，并维护引用计数。后续某条序列需要修改共享块时，再复制该块（copy-on-write）。

安全共享要求前缀 token、模型权重、adapter、位置处理和相关推理配置一致。只比较文本看似相同不足以保证缓存可复用。

## 块大小的取舍

- 块大：块表短、调度和寻址开销较低，但最后一块内部浪费更大。
- 块小：按需分配更精细，但块表、元数据和 kernel 寻址开销增加。

实际选择取决于 head Shape、数据类型、kernel、典型长度和硬件页粒度，不存在只由公式决定的通用最优块大小。

## 与相关技术的边界

| 技术 | 主要解决的问题 | 是否改变 Attention 数学结果 |
| --- | --- | --- |
| KV Cache | 避免重算历史隐藏状态 | 否 |
| PagedAttention | 管理动态 KV 内存与批处理 | 否 |
| FlashAttention | 降低 Attention 计算中的 HBM IO | 否，精度差异除外 |
| MQA/GQA | 减少 K/V heads 和缓存读取 | 改变模型参数化 |

PagedAttention 不会自动减少每个 token 的逻辑 K/V 元素，也不会把对长历史的读取变为 $O(1)$。它提升的是显存利用率和 serving 调度空间。

## 易错点

> [!warning] “分页”不等于把 GPU 内存换到磁盘
> 这里借用了虚拟内存的块表思想，核心是在 GPU KV 内存中做非连续映射和按需管理。是否发生 CPU offload、跨 GPU 传输或磁盘换入是额外系统策略，不是 PagedAttention 定义的一部分。

## 掌握要求

- [ ] 能解释连续预留为何产生浪费和碎片。
- [ ] 能画出逻辑块表到物理块的映射。
- [ ] 能说明分页怎样支持按需增长、回收和 continuous batching。
- [ ] 能解释共享前缀与 copy-on-write 的必要条件。
- [ ] 能区分 PagedAttention、FlashAttention、KV Cache 和 MQA/GQA。

## 关联

- 属于：[[KV Cache]]

## 来源

- [[精炼-06-推理优化与模型服务]]：迁移分页缓存、碎片与连续批处理内容。
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)：PagedAttention、vLLM、块表和共享机制。

