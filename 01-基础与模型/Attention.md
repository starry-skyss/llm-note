---
type: concept
domain: 基础与模型
status: active
aliases:
  - 注意力机制
  - Self-Attention
---

# Attention

> [!abstract] 核心结论
> Attention 用 Query 与 Key 的匹配程度生成权重，再对 Value 加权汇总，使每个位置按当前需求读取其他位置的信息。Self-Attention 的 Q、K、V 来自同一序列，是 Transformer 中跨 token 通信的核心机制。

## 解决的问题

- 根据当前 token 的查询动态选择上下文，而不是使用固定窗口或固定权重。
- 让远距离位置之间建立直接信息路径。
- 在多头结构中并行学习不同的匹配与聚合子空间。

## 原理与机制

输入 $X$ 分别经过线性投影得到 Q、K、V：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

Query 表示当前位置“要找什么”，Key 表示每个位置“可用什么匹配”，Value 表示匹配后实际汇总的内容。缩放点积 Attention 为：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}+M\right)V
$$

$M$ 是加到分数上的掩码。Decoder 自回归生成通常使用上三角因果掩码，把未来位置对应的分数设为 $-\infty$；掩码必须在 Softmax 前应用，使这些位置的概率变为 0。Padding mask 则屏蔽无效 token，两类 mask 的目的和广播 Shape 不同。

多头注意力会把隐藏维度拆成若干 head，在各 head 内独立计算，再拼接并投影回模型维度。MHA、MQA 与 GQA 的 K/V 共享方式见 [[MHA、MQA与GQA]]。

## 公式与 Shape

设输入为 $X\in\mathbb{R}^{B\times T\times D}$，共有 $H$ 个头，单头维度 $d_h=D/H$：

```text
X                    [B, T, D]
Q, K, V 投影          [B, T, D]
拆分并转置             [B, H, T, d_h]
Q @ K^T              [B, H, T, T]
softmax 后 @ V        [B, H, T, d_h]
合并各头               [B, T, D]
输出投影               [B, T, D]
```

除以 $\sqrt{d_k}$ 是为了抑制维度增大导致点积幅度和 softmax 饱和程度上升。

若 Query 长度为 $T_q$、Key/Value 长度为 $T_k$，更一般的分数 Shape 是 `[B,H,T_q,T_k]`；只有 Self-Attention 中二者相等时才是 `[B,H,T,T]`。Softmax 应沿最后的 Key 维归一化，使每个 Query 对可见 Key 的权重和为 1。

## 对比与边界

- Self-Attention：Q、K、V 来自同一序列；Cross-Attention：Query 与 Key/Value 来自不同表示。
- 因果掩码限制“能看到哪些位置”；[[位置编码]] 表示“位置之间是什么关系”，职责不同。
- Attention 负责信息路由，不等于完整的 [[Transformer]]。
- 权重高表示当前计算中的相对贡献更大，不应直接解释为因果关系或人类意义上的理解。

## 实现或实验

```python
scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
scores = scores.masked_fill(causal_mask == 0, float("-inf"))
weights = torch.softmax(scores, dim=-1)
context = weights @ v  # [B, H, T, d_h]
```

实验时至少检查：权重是否沿 Key 维归一化、被屏蔽位置是否恰为 0、拆头前后元素数量是否一致。若一整行都被屏蔽，Softmax 可能产生 NaN，因此 mask 设计必须保证每个有效 Query 至少能看见一个 Key。

## 易错点

> [!warning] Shape 比符号更可靠
> $QK^{\mathsf T}$ 中的转置发生在最后两个维度；多头实现还包含 batch 和 head 维。掩码广播错误、softmax 维度错误或把 $T$ 与 $d_h$ 混淆，都可能让代码可运行但语义错误。

## 掌握要求

- [ ] 能用自己的话解释 Q、K、V 的职责。
- [ ] 能从 $[B,T,D]$ 追踪到注意力矩阵 $[B,H,T,T]$ 再回到输出。
- [ ] 能推导缩放点积 Attention 并说明缩放项的作用。
- [ ] 能区分 Self-Attention、Cross-Attention 和因果 Self-Attention。
- [ ] 能用小张量验证掩码与 softmax 维度。

## 关联

- 属于：[[Transformer]]
- 子概念：[[MHA、MQA与GQA]]
- 配合机制：[[位置编码]]

## 来源

- [[大模型概念-重构前]]：重组其中 Attention、Transformer、Embedding 等既有条目。
- [[精炼-02-Transformer与模型架构]]：迁移 Q/K/V、因果掩码、多头 Shape 与常见误区。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：缩放点积注意力和 Multi-Head Attention 的原始定义。
