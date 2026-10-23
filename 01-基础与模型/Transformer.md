---
type: concept
domain: 基础与模型
status: active
aliases:
  - Transformer 架构
---

# Transformer

> [!abstract] 核心结论
> Transformer 是以 Attention 做 token 间信息混合、以前馈网络做逐位置特征变换，并用残差连接和归一化稳定深层训练的序列模型架构。它解决了序列中远距离依赖难以并行建模的问题，也是现代大语言模型的主要结构基础。

## 解决的问题

- 让序列中任意位置能够直接交换信息，缩短信号传播路径。
- 训练时同时处理多个位置，减少循环网络的串行依赖。
- 用重复 Block 构成可扩展的深层网络，并允许按任务选择 Encoder、Decoder 或两者组合。

## 原理与机制

以常见的 Pre-LayerNorm Decoder Block 为例：

```text
输入 x
├─ LayerNorm → Causal Self-Attention → 与 x 残差相加
└─ LayerNorm → Feed-Forward Network → 再次残差相加
```

一个语言模型通常还包含：

```text
Token IDs
→ Token Embedding + 位置信息
→ N × Transformer Block
→ Final Norm
→ Linear / LM Head
→ 下一 token 的 logits
```

Attention 决定“从哪些位置取信息”，FFN 决定“每个位置如何变换已有特征”；残差连接保留信息通路，归一化控制各层数值尺度。

### Embedding 与 LM Head

Token ID 只是词表中的整数索引。`nn.Embedding(V,D)` 是 `[V,D]` 的可学习查找表：

```text
token ids       [B,T]
token embedding [B,T,D]
hidden states   [B,T,D]
LM Head logits  [B,T,V]
```

相同 token 的初始 embedding 相同，经过多层 Attention 后会因上下文不同得到不同隐藏表示。LM Head 把隐藏维映射回词表维；它与多头 Attention 内部的输出投影 $W_O$ 职责不同。部分模型会让输入 Embedding 与 LM Head 共享权重，但这是一种参数化选择，不改变两端的功能。

### FFN、归一化与残差

Attention 沿序列维混合 token 信息；FFN 对每个位置独立使用同一组参数：

$$
\operatorname{FFN}(x)=W_2\,\phi(W_1x+b_1)+b_2
$$

```text
[B,T,D] → [B,T,D_ff] → [B,T,D]
```

现代模型常使用门控 FFN，例如：

$$
\operatorname{SwiGLU}(x)=W_{down}\left(\operatorname{SiLU}(W_{gate}x)\odot W_{up}x\right)
$$

残差 $x\leftarrow x+F(x)$ 保留恒等路径，使子层学习对现有表示的修正。LayerNorm 对每个 `[D]` token 表示独立归一化，不在 token 之间传递信息，也不依赖 batch 统计：

$$
\hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}},\qquad y=\gamma\odot\hat{x}+\beta
$$

## 公式与 Shape

设隐藏状态为：

$$
X \in \mathbb{R}^{B \times T \times D}
$$

- $B$：batch size
- $T$：序列长度
- $D$：隐藏维度

典型 Pre-LN Block 可写为：

$$
H = X + \operatorname{Attention}(\operatorname{LN}(X))
$$

$$
Y = H + \operatorname{FFN}(\operatorname{LN}(H))
$$

输入和输出通常保持 $[B,T,D]$，便于堆叠多个 Block。Attention 内部的 Shape 变化见 [[Attention]]。

Pre-Norm 使用 $x+\operatorname{Sublayer}(\operatorname{LN}(x))$；Post-Norm 使用 $\operatorname{LN}(x+\operatorname{Sublayer}(x))$。两者均为有效结构，不能把某一课程实现当成 Transformer 的唯一定义。

## 对比与边界

- Transformer 是完整架构；[[Attention]] 只是其中负责跨位置通信的机制。
- [[位置编码]] 提供顺序信息，但不等同于 Transformer，也不负责屏蔽未来 token。
- Decoder-only 模型不代表只有一个 Decoder Block，而是由多层带因果约束的 Block 组成。
- GPT 是以 next-token prediction 训练的 Decoder-only Transformer 模型家族，不是 Attention 或 Transformer 的同义词。
- “LLM”描述模型规模与用途范围；Transformer 描述常用的内部架构，两者不能互换。

## 实现或实验

下面的骨架只表达模块关系，未展开具体 Attention 和 FFN：

```python
class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
```

## 易错点

> [!warning] 常见混淆
> 不能只凭模块名称判断结构细节。Pre-LN 与 Post-LN 的归一化位置不同；Encoder Attention 与带因果掩码的 Decoder Self-Attention 可见范围也不同。Attention 不会天然减少 token 数量，隐藏维 $D$ 也不能与词表大小 $V$ 混用。阅读实现时应沿真实数据流和 Shape 核对。

## 掌握要求

- [ ] 能画出 Decoder-only Transformer 从 token 到 logits 的主数据流。
- [ ] 能说明 Attention、FFN、残差和归一化各自承担的职责。
- [ ] 能追踪一个 Block 输入输出的 $[B,T,D]$ Shape。
- [ ] 能解释 Encoder、Decoder 与因果约束的基本差异。
- [ ] 能实现并测试一个最小 Transformer Block。

## 关联

- 属于：[[基础与模型]]
- 组成：[[Attention]]、[[位置编码]]

## 来源

- 重构前的 `Transformer.md`：保留了 Pre-LN、因果自注意力、FFN 与残差的结构提纲。
- [[大模型概念-重构前]]：用于核对 Attention、Embedding、LLM 等既有概念关系。
- [[精炼-02-Transformer与模型架构]]：迁移 Decoder-only 数据流、Embedding、FFN、LayerNorm、残差和 Block 内容。
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)：Transformer、Multi-Head Attention、位置表示与 FFN 的原始架构。
- [PyTorch Transformer](https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html)：`norm_first` 等实现接口的语义参考。
