# 数据、Tokenizer 与预训练（精炼）

> 定位：理解语言模型从原始文本到训练样本的完整路径，并能解释 `[B,T] → [B,T,D] → [B,T,V]` 中每一步的含义。

## 文本如何进入模型

```text
原始文本
→ Tokenizer 切分
→ token
→ 词表映射
→ token ids
→ 分批并截取上下文
→ 输入 x / 目标 y
→ Embedding
→ Transformer
→ 每个位置的词表 logits
```

### Token、ID 与词表

- Token 是模型处理文本的离散单位，可以是字符、字节、完整词或子词。
- Tokenizer 同时定义“如何切分”和“token 与整数 ID 如何互相转换”。
- Vocabulary 是可使用 token 的集合，大小记为 `V`。
- ID 只是索引，不代表大小或语义。语义表示来自可训练的 embedding 和后续网络。
- 编码是文本到 IDs；解码是 IDs 到文本。生成过程实际预测的是下一个 ID，再由 Tokenizer 解码。

课程中的字符级 Tokenizer 便于看懂完整流程：词表小、实现直接、没有未知子词问题；但序列较长，不能代表生产模型常用的 BPE、WordPiece、Unigram 或 byte-level 方案。

## 三个必须区分的 shape

```text
token ids     [B,T]    每个位置是一个整数索引
hidden states [B,T,D]  每个位置是一个 D 维表示
logits         [B,T,V] 每个位置对 V 个候选 token 的原始分数
```

`nn.Embedding(V,D)` 的参数矩阵是 `[V,D]`；对 `[B,T]` 中每个 ID 查一行，得到 `[B,T,D]`。Embedding 是可学习参数表，不是预先写死的词义字典。

LM Head 通常把隐藏维映射回词表维：

$$
[B,T,D]\xrightarrow{\text{Linear}}[B,T,V]
$$

注意 `V` 与 `D` 的含义不同，即使某些简化模型让二者数值相同，也不能混用。

## 构造 next-token 训练样本

对连续 token 序列截取长度 `T+1` 的片段，输入和目标错开一位：

```text
原始片段：[18, 7, 4, 9, 3, 6, 2, 5, 11]
x:        [18, 7, 4, 9, 3, 6, 2, 5]
y:        [ 7, 4, 9, 3, 6, 2, 5, 11]
```

每个位置都贡献一次“根据当前位置及其可见前文预测下一 token”的训练目标。批量采样后：

```text
x [B,T]
y [B,T]
```

最小实现：

```python
def get_batch(data, batch_size, block_size, device):
    starts = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
```

这里随机的是片段起点；`block_size` 是取出的上下文长度，不等于模型一定有效利用了这么长的上下文。

## 预训练目标

自回归语言模型最大化语料序列的条件似然：

$$
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t})
$$

训练时等价于最小化各有效 token 的负对数似然/交叉熵：

$$
L=-\frac{1}{N}\sum_t\log p_\theta(x_t\mid x_{<t})
$$

模型输出 `[B,T,V]`，而 `CrossEntropyLoss` 常需要：

```text
logits [B,T,V] → [B*T,V]
targets [B,T]  → [B*T]
```

若存在 padding 或不希望计入训练的区域，应使用 loss mask 或 `ignore_index`，分母也应按有效 token 数计算。

预训练属于自监督学习：标签不是人工逐条标注，而是由原文本向右错一位自动构造。它主要学习语言结构、模式和数据中可预测的知识；“见过大量文本”不等于对每个事实都有稳定、可验证的记忆。

## Bigram 到 Transformer

字符级 Bigram 模型可以把当前 token 直接查成下一 token 的 `V` 个 logits：

```text
Embedding(V,V): [B,T] → [B,T,V]
```

它能演示 Tokenizer、batch、loss、反向传播和生成闭环，但每个位置只看自己：相同当前 token 在任何上下文中得到相同分布。即使输入 shape 是 `[B,T]`，也不代表模型利用了长度 `T` 的上下文。

Transformer 的关键增量是先得到 `[B,T,D]`，再通过因果 Self-Attention 让每个位置读取可见前文，使相同 token 在不同上下文中形成不同表示。

## 数据切分与训练批次

- 训练集用于梯度更新；验证集用于模型选择和观察泛化；测试集应留作最终评估。
- 对长文本按连续区间切分简单直观，但真实工程还需避免文档泄漏、近重复样本跨集合和时间泄漏。
- `batch` 是一次并行处理的样本集合；`iteration/step` 通常对应一次参数更新；`epoch` 是完整遍历训练集一次。随机片段训练可能只按 step 计数，不显式定义 epoch。
- 输入、标签和模型必须位于兼容设备；作为 Embedding 索引的 token IDs 通常使用整数 `torch.long`。

## 易错点

- Tokenizer 负责离散化和映射，Embedding 负责把 ID 查成可训练向量。
- `[B,T,D]` 中 `T` 是 token 数，`D` 是每个 token 的表示维度。
- 位置编码不属于 Tokenizer；它在模型内部告诉网络 token 的顺序或相对距离。
- 训练目标必须错开一位；若 `x` 与 `y` 相同，模型学到的不是 next-token prediction。
- 计算语言模型 loss 时不要先对 logits 做 Softmax。
- 字符级代码是教学实现，不能据此推断现代 LLM 的词表、序列效率和分词质量。

## 待补充

- BPE、SentencePiece/Unigram、byte-level Tokenizer 的训练与实现。
- 特殊 token、聊天模板、截断、padding、packing 和 loss mask。
- 数据清洗、去重、质量过滤、数据混合比例与采样策略。
- 预训练数据泄漏、版权、隐私和评测污染。
- 分布式数据加载、checkpoint 恢复和可复现性。

## 来源映射

- `gpt语法解释.md`：字符级 Tokenizer、`get_batch()`、Bigram 模型、交叉熵和自回归生成。
- `问答2.md`：logits、Softmax、交叉熵及 `[B*T,V]` 分类视角。
- `问答4.md`：mini-batch、随机采样、DataLoader、epoch/iteration。
- `问答6.md`：自监督预训练及 LLM 训练生命周期。
- `大模型基础概念解析.pdf`：第 2-7、21-24、41-72 页。

