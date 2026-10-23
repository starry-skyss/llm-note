---
type: concept
domain: 推理与AI-Infra
status: active
aliases:
  - KV缓存
  - Key-Value Cache
---

# KV Cache

> [!abstract] 核心结论
> KV Cache 保存自回归生成中每一层、每个历史 token 的 Key 和 Value，使新 token 无需重复计算整段历史隐藏状态。它用显存换计算，但当前 Query 仍要读取全部可见历史 K/V，因此上下文增长后 Decode 不会变成常数成本。

## 为什么缓存 K/V

每层 Attention 都由隐藏状态生成 Q/K/V。对刚生成的新 token：

- 新 Q 只用于当前查询，通常不需要长期保存；
- 历史 K 会与后续每个新 Q 计算匹配分数；
- 历史 V 会继续被后续注意力权重聚合。

```text
Prefill
prompt hidden states → 各层 K/V → 写入缓存

Decode 第 t 步
新 token → 新 Q/K/V
新 Q × [历史 K; 新 K] → 当前注意力权重
权重 × [历史 V; 新 V] → 当前输出
把新 K/V 追加到各层缓存
```

缓存的是每一层 Attention 投影后的状态，不是输入 Embedding，也不是只保存最后一层输出。

## Shape 与增长

常见缓存逻辑 Shape：

```text
K_cache, V_cache [B, L, H_kv, T, d_h]
```

实现中通常按层保存为列表，因此单层更常见：

```text
K_l, V_l [B, H_kv, T, d_h]
```

Decode 产生 `new_k/new_v [B,H_kv,1,d_h]`，沿序列维追加。动态 `torch.cat` 适合教学但会反复分配和复制；推理引擎通常预分配或采用分页块管理。

## 显存估算

设并发序列数 $B$、层数 $L$、缓存长度 $T$、K/V head 数 $H_{kv}$、单头维度 $d_h$、每元素字节数 $s$：

$$
M_{KV}\approx 2BLTH_{kv}d_hs
$$

系数 2 来自 K 和 V。这个公式尚未包含块对齐、元数据、保留容量、跨设备副本和其他运行时开销。

示例：$B=1,L=32,T=4096,H_{kv}=32,d_h=128,s=2$（FP16/BF16），得到约：

$$
2\times1\times32\times4096\times32\times128\times2
=2^{31}\text{ bytes}\approx2\text{ GiB}
$$

若同一模型改用 $H_{kv}=8$ 的 GQA，理论 K/V 元素数约降为四分之一。详细头共享关系见 [[MHA、MQA与GQA]]。

## 节省了什么，没节省什么

KV Cache 避免：

- 重算历史 token 在各层的 K/V 投影；
- 重算历史 token 的 Attention、FFN 和层间隐藏状态。

它仍然需要：

- 为新 token 计算完整一遍所有层；
- 让当前 Q 读取可见历史 K/V；
- 计算当前 token 对历史位置的 Attention；
- 保存新增 K/V，并维护每条序列的位置和有效长度。

因此单步 Decode 的历史读取随 $T$ 增长；长上下文、高并发和大层数会同时放大显存与带宽压力。

## 生命周期与批处理

服务系统需要处理不同长度、不同结束时间的请求：

```text
分配缓存 → Prefill 写入 → Decode 逐步增长
→ EOS/取消/超时 → 回收
```

请求被重新排序、batch 合并或 beam 扩展时，逻辑序列与物理缓存位置必须保持一致。共享前缀可以避免重复保存相同 K/V，但只有模型、位置编码、adapter 和前缀 token 完全兼容时才能安全复用。

## 与其他优化的边界

- [[MHA、MQA与GQA]] 减少 $H_{kv}$，直接降低缓存元素数和读取量。
- [[PagedAttention]] 改善缓存的分配、增长、共享和回收，不改变每个逻辑 token 需要的 K/V 内容。
- FlashAttention 优化 Attention 计算中的 IO 和分块方式，不等同于 KV Cache 或 PagedAttention。
- KV Cache 量化降低 $s$，但会引入量化误差、尺度元数据和 kernel 支持要求。

## 易错点

> [!warning] Cache 正确性依赖位置和 Mask
> 增量推理不仅要拼接 K/V，还必须为新 token 使用正确的位置 ID、旋转位置偏移、有效长度和因果/滑窗 mask。Cache Shape 正确但偏移错误时，模型仍能输出文本，却会悄然降低质量。

## 掌握要求

- [ ] 能解释为何缓存 K/V 而通常不缓存 Q。
- [ ] 能追踪 Prefill 后和每次 Decode 后的缓存 Shape。
- [ ] 能用模型配置估算单请求与并发 KV 显存。
- [ ] 能说清 KV Cache 节省与未节省的计算。
- [ ] 能区分 MQA/GQA、KV Cache、PagedAttention 和 FlashAttention。

## 关联

- 属于：[[自回归推理]]
- 子机制：[[PagedAttention]]
- 头结构影响：[[MHA、MQA与GQA]]

## 来源

- [[精炼-06-推理优化与模型服务]]：迁移缓存机制、复杂度、显存公式和常见误区。
- [[精炼-02-Transformer与模型架构]]：Q/K/V 与多头 Shape 背景。
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)：增量解码的 K/V 带宽问题与 MQA。
- [GQA paper](https://arxiv.org/abs/2305.13245)：K/V head 数与解码效率的折中。
