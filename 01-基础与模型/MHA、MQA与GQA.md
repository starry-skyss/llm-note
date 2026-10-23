---
type: concept
domain: 基础与模型
status: active
aliases:
  - MHA MQA GQA
  - Grouped-Query Attention
  - Multi-Query Attention
---

# MHA、MQA 与 GQA

> [!abstract] 核心结论
> MHA、MQA 和 GQA 都保留多个 Query head，差别在于 K/V head 是否共享。共享越强，需要保存和读取的 K/V 越少，Decode 通常越高效；与此同时，K/V 表达容量减少，模型质量与推理成本之间需要权衡。

## 从多头注意力开始

设模型维度为 $D$，Query head 数为 $H_q$，单头维度为 $d_h$，通常满足：

$$
D=H_qd_h
$$

标准 MHA 为每个 Query head 配置独立的 Key 和 Value head：

```text
Q [B, H_q, T, d_h]
K [B, H_q, T, d_h]
V [B, H_q, T, d_h]
```

每个 head 独立执行缩放点积 [[Attention]]，再把输出拼接并用 $W_O$ 映射回 `[B,T,D]`。多个 head 允许模型在不同投影子空间中学习不同的匹配和信息聚合方式。

## 三种头组织

| 结构 | Query heads | K/V heads | 一个 K/V head 服务多少 Q head | 主要特点 |
| --- | ---: | ---: | ---: | --- |
| MHA | $H_q$ | $H_q$ | 1 | 表达最独立，KV 最大 |
| MQA | $H_q$ | 1 | $H_q$ | K/V 全共享，缓存和带宽需求最小 |
| GQA | $H_q$ | $H_{kv}$ | $H_q/H_{kv}$ | 在质量与推理效率间折中 |

GQA 要求 Query heads 能均匀分组，常见实现中 $H_q$ 可被 $H_{kv}$ 整除。第 $h$ 个 Query head 使用的 KV group 可写为：

$$
g(h)=\left\lfloor\frac{h}{H_q/H_{kv}}\right\rfloor
$$

例如 $H_q=8,H_{kv}=2$ 时，Q heads 0～3 共享第 0 组 K/V，Q heads 4～7 共享第 1 组。

## Shape 与参数量

三者的 Q Shape 相同，K/V 的头维不同：

```text
Q                 [B, H_q,  T, d_h]
K, V              [B, H_kv, T, d_h]
重复/广播到 Q 组    [B, H_q,  T, d_h]
Attention 输出     [B, H_q,  T, d_h]
合并并输出投影       [B, T, D]
```

只比较 Q/K/V 投影矩阵、忽略 bias：

$$
\#W_Q=D(H_qd_h)=D^2
$$

$$
\#W_K+\#W_V=2D(H_{kv}d_h)
$$

因此从 MHA 减少到 GQA/MQA，不仅会减小 K/V 投影参数，更关键的是降低推理时每层、每 token 的 K/V 存储和读取量。

## 为什么主要影响 Decode

自回归 Decode 每生成一个 token，都要为每层读取历史 [[KV Cache]]。忽略对齐和元数据，缓存大小近似：

$$
M_{KV}\approx 2BLTH_{kv}d_hs
$$

其中 2 表示 K 和 V，$B$ 为并发序列数，$L$ 为层数，$T$ 为缓存长度，$s$ 为每个元素的字节数。其他条件相同时，缓存与 $H_{kv}$ 近似成正比：

```text
MHA: H_kv = H_q
GQA: 1 < H_kv < H_q
MQA: H_kv = 1
```

减少 K/V heads 也减少 Decode 反复读取历史 K/V 的带宽压力。它不会消除 Attention，也不会让当前 Query 不再访问全部历史位置。

## 实现要点

概念性实现可以在计算 Attention 前，把每组 K/V 映射到对应的 Query heads：

```python
def repeat_kv(x, n_rep):
    # x: [B, H_kv, T, d_h]
    if n_rep == 1:
        return x
    b, h_kv, t, d_h = x.shape
    x = x[:, :, None, :, :].expand(b, h_kv, n_rep, t, d_h)
    return x.reshape(b, h_kv * n_rep, t, d_h)
```

逻辑上的 repeat 不一定意味着真正复制整块内存；高效 kernel 可以利用广播或分组布局直接计算。阅读实现时要区分“张量语义上的扩展”和“物理内存复制”。

## 对比与边界

- MQA/GQA 改变头的组织方式；[[KV Cache]] 是保存历史 K/V 的推理机制，两者不是同一个概念。
- GQA 的推理收益取决于硬件、kernel、batch、上下文和 serving 系统，不能只凭头数直接推断端到端加速倍数。
- MQA 不是单头注意力：Query 仍然有多个 head，只是它们共享 K/V。
- 从已有 MHA checkpoint 转成 GQA/MQA 不能只删除权重；原论文讨论了 uptraining 方案，生产模型还需重新训练或适配。

## 易错点

> [!warning] 不要混淆 $H_q$、$H_{kv}$ 和 $D$
> $D=H_qd_h$ 描述 Query 输出总维度；K/V 总维度是 $H_{kv}d_h$。用 $2LTD$ 估算 KV Cache 默认隐含 $H_{kv}d_h=D$，只适合标准 MHA，直接套到 GQA/MQA 会高估缓存。

## 掌握要求

- [ ] 能画出 MHA、MQA 和 GQA 的 Q/K/V head 对应关系。
- [ ] 能从 $H_q,H_{kv},d_h$ 推导投影和 KV Cache Shape。
- [ ] 能解释减少 K/V heads 为什么缓解 Decode 带宽和显存压力。
- [ ] 能实现或验证 `repeat_kv` 的分组映射。
- [ ] 能说明效率收益、表达共享和模型质量之间的边界。

## 关联

- 属于：[[Attention]]
- 直接影响：[[KV Cache]]

## 来源

- [[精炼-02-Transformer与模型架构]]：迁移多头结构、Shape 和三类注意力的比较。
- [[精炼-06-推理优化与模型服务]]：迁移 KV Cache 容量与 Decode 带宽关系。
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)：MQA 的动机、结构和解码收益。
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)：GQA 结构、uptraining 与质量/速度折中。

