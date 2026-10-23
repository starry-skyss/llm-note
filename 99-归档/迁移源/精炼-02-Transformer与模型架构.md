# Transformer 与模型架构（精炼）

> 定位：掌握 Decoder-only Transformer 的完整数据流。复习时始终围绕 `[B,T,D]` 追踪，不把 Attention、Transformer、GPT 和语言模型混为同一概念。

## 全局结构

```text
文本
→ Tokenizer：文本变成 token ids [B,T]
→ Token Embedding：查表得到 [B,T,D]
→ 位置信息：RoPE 或其他位置编码
→ Transformer Block × N
   ├─ Attention：token 之间通信
   └─ FFN：每个 token 独立做非线性计算
→ Final Norm
→ LM Head：[B,T,D] → [B,T,V]
→ 下一个 token 的 logits
```

- Transformer 是整体架构，Attention 是其中让 token 交换信息的机制。
- GPT 是采用因果掩码、按下一个 token 目标训练的 Decoder-only Transformer 家族。
- 一个 Block 内通常包含 Attention、FFN、归一化和残差；多个 Block 堆叠后才构成模型主体。

## Embedding 与上下文化表示

Token ID 只是词表行号，本身没有大小或距离语义。`nn.Embedding(V,D)` 是形状为 `[V,D]` 的可训练查找表：

```text
token ids       [B,T]
Token Embedding [B,T,D]
Transformer     [B,T,D]
LM Head         [B,T,V]
```

同一个 token 的初始 embedding 相同，但经过 Attention 后会因上下文不同而得到不同的隐藏表示。隐藏维度也不应被机械解释成一个个可命名的人类特征；语义通常是分布式表示。

## Self-Attention

输入 $X\in\mathbb{R}^{B\times T\times D}$ 经三个独立线性投影：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

单头缩放点积注意力：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V
$$

其中 $M$ 是 mask。因果 mask 将未来位置的分数设为 $-\infty$，必须在 Softmax 前应用，使对应概率变为 0。

```text
Q [B,H,T,d_h]
K [B,H,T,d_h]
V [B,H,T,d_h]
QKᵀ             → [B,H,T,T]
Softmax          → 每行是一组注意力权重
weights @ V      → [B,H,T,d_h]
拼接所有头        → [B,T,D]
输出投影 W_O      → [B,T,D]
```

核心直觉：

- Q 表示当前位置要查询什么；K 用于判断候选位置是否匹配；V 是匹配后真正汇总的信息。
- $QK^\top$ 生成 token 两两之间的匹配分数；Softmax 将每行变成权重；乘 V 本质是加权求和。
- 除以 $\sqrt{d_h}$ 是为控制点积方差，避免 Softmax 过早饱和。
- Q/K/V 分开，使“用于匹配的特征”和“需要传递的内容”可以学习不同表示。
- Self-Attention 的 Q/K/V 来自同一序列；Cross-Attention 的 Q 与 K/V 来自不同序列或模态。

## 多头、MQA 与 GQA

多头注意力把隐藏维拆成多个子空间，每个头独立学习关系，再拼接并用 $W_O$ 混合：

$$
D=H_qd_h
$$

| 结构 | Query 头 | K/V 头 | 特点 |
| --- | ---: | ---: | --- |
| MHA | $H_q$ | $H_q$ | 表达灵活，KV Cache 最大 |
| MQA | $H_q$ | 1 | 所有 Query 头共享一组 K/V，缓存最小 |
| GQA | $H_q$ | $1<H_{kv}<H_q$ | 若干 Query 头共享一组 K/V，折中质量与推理成本 |

三者主要区别是 K/V 头的共享方式，不是“有没有多个 Query 头”。推理侧影响详见 `精炼-06-推理优化与模型服务.md`。

## 位置编码与 RoPE

Attention 若没有位置信息，本身无法区分 token 的先后顺序。RoPE（Rotary Position Embedding）不把位置向量直接加到隐藏状态，而是在每个 Attention head 中旋转 Q 和 K 的二维特征对，V 通常不旋转。

若位置 $m,n$ 使用旋转矩阵 $R_m,R_n$：

$$
(R_mq)^\top(R_nk)=q^\top R_{n-m}k
$$

因此 Q/K 点积会自然带入相对位置差 $n-m$。完整位置是：

```text
X → Q/K/V → RoPE(Q,K) → QKᵀ → mask → softmax → V
```

关键认识：

- RoPE 改变 Q/K 的数值，不改变 shape。
- RoPE 负责位置与距离；causal mask 负责“能否看见未来”，二者职责不同。
- 最后一维按二维特征对旋转，并使用多组不同频率。单个频率具有周期性，多频率组合可降低位置混叠，但超出训练长度时仍可能外推退化。
- 长上下文扩展不是“RoPE 天然无限外推”，需要单独研究频率缩放和训练分布。

## FFN、门控与残差

Attention 沿 `T` 维聚合其他 token 的信息；FFN 对每个 `[D]` token 向量独立、共享参数地做非线性变换。

经典 FFN：

$$
\operatorname{FFN}(x)=W_2\,\phi(W_1x+b_1)+b_2
$$

形状通常是：

```text
[B,T,D] → [B,T,D_ff] → [B,T,D]
```

ReLU/GELU 是单个激活函数；GeGLU/SwiGLU 是双支路门控 FFN，不应混为同一层级。以 SwiGLU 为例：

$$
\operatorname{SwiGLU}(x)
=W_{down}\left(\operatorname{SiLU}(W_{gate}x)\odot W_{up}x\right)
$$

残差连接写作 $x\leftarrow x+F(x)$。它保留恒等路径，让子层只需学习修正量，并改善深层网络的信息与梯度传播。

## LayerNorm 与 Block

对输入 `[B,T,D]`，LayerNorm 独立归一化每个 `x[b,t,:]` 的最后一维：

$$
\hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}},\qquad
y=\gamma\odot\hat{x}+\beta
$$

它不在 token 之间传递信息，标准 LayerNorm 的计算也不依赖 batch 统计。

常见 Pre-Norm Block：

```python
def forward(self, x):
    x = x + self.attn(self.ln1(x))
    x = x + self.ffn(self.ln2(x))
    return x
```

Pre-Norm：`x + Sublayer(LN(x))`；Post-Norm：`LN(x + Sublayer(x))`。二者都是有效架构，不能把课程中的 Pre-Norm 写成 Transformer 唯一形式。

## 易错点

- Attention 输出仍是一组 token 表示，不会天然减少 token 数量。
- Attention 权重是 Softmax 后的 `[T,T]` 矩阵；`weights @ V` 才得到新的表示。
- `nn.ModuleList` 用于正确注册一组子模块；普通 Python list 中的层不会自动进入参数管理。
- causal mask 应注册为 buffer 或动态构造，不能当作可学习参数。
- `W_O` 负责混合多头输出并返回残差维度；它不同于最终映射到词表的 LM Head。
- `D` 是隐藏维，`V` 是词表大小。输入/隐藏/输出常依次为 `[B,T] → [B,T,D] → [B,T,V]`。

## 待补充

- RoPE 的复数实现、频率设计与长上下文缩放。
- FlashAttention 的精确计算与 IO 优化原理。
- RMSNorm、QK-Norm、残差缩放等现代结构变体。
- Encoder、Encoder-Decoder、Mixture-of-Experts 与多模态 Cross-Attention。

## 来源映射

- `问答.md`：激活函数、GeGLU/SwiGLU、RoPE、MHA/MQA/GQA。
- `问答3.md`：Embedding、Attention、FFN、LayerNorm、残差和 GPT 架构总览。
- `gpt语法解释.md`：字符级 GPT 的 Attention、MultiHead、FFN、Block 与 shape 流。
- `大模型基础概念解析.pdf`：第 5-20、40-72、73-115 页。

