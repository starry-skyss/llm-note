---
type: concept
domain: 数据与训练
status: active
aliases:
  - 分词器
  - Tokenization
---

# Tokenizer

> [!abstract] 核心结论
> Tokenizer 定义文本如何切成离散 token，以及 token 与整数 ID 如何互相映射。它连接原始文本和模型输入，但不负责产生上下文化语义；语义表示由 Embedding 和后续网络学习。

## 文本如何变成 ID

```text
原始文本
→ 规范化（可选）
→ 预切分（可选）
→ tokenization model
→ token 序列
→ vocabulary lookup
→ token IDs [T]
→ 添加特殊 token / 截断 / padding
```

- **Token**：模型处理的离散单位，可以是字符、字节、完整词或子词。
- **Vocabulary**：Tokenizer 能输出的 token 集合，大小记为 $V$。
- **Token ID**：token 在词表中的行号，本身没有大小或距离语义。
- **Encode/Decode**：文本到 IDs 与 IDs 到文本。生成模型预测下一个 ID，最后再解码成文本。

`nn.Embedding(V,D)` 接收 ID 并查表得到向量，但 Embedding 属于模型而不是 Tokenizer：

```text
text → Tokenizer → IDs [B,T] → Embedding → hidden states [B,T,D]
```

## 常见粒度与算法

| 方法 | 基本单位/思想 | 优点 | 代价或边界 |
| --- | --- | --- | --- |
| 字符级 | 每个字符一个 token | 实现简单，不依赖分词训练 | 序列长，多语言词表处理困难 |
| BPE | 反复合并高频相邻单元 | 常见、直观、压缩率较好 | 贪心合并不直接优化概率目标 |
| Unigram | 从候选子词集合中删减，按概率模型选切分 | 可比较多种切分，SentencePiece 常用 | 训练和解码逻辑更复杂 |
| Byte-level | 以字节保证基础覆盖，再学习组合 | 几乎没有未知字符 | 人类字符可能拆成多个 token |

WordPiece 与 BPE 外观相似，但合并/选择准则不同，不能只凭 `##` 等显示形式判断算法。SentencePiece 还强调直接从原始句子训练，空格也作为普通符号处理；“SentencePiece”是工具/框架，不等同于单一 Unigram 算法。

## 特殊 token 与聊天模板

生产模型通常还定义：

- BOS/EOS：序列开始与结束；
- PAD：为 batch 对齐长度；
- UNK：无法表示的单元，byte-level 方案可显著减少此问题；
- 对话角色和分隔 token：区分 system、user、assistant 或工具消息。

聊天模板把结构化消息转换为模型训练时见过的 token 序列。相同文本若套用不同模板，会得到不同 IDs 和模型行为。特殊 token 是否自动添加、是否参与 loss、生成何时以 EOS 终止，都必须与模型配置匹配。

## 截断、Padding 与长度

- **截断**改变内容：超出窗口的 token 被删除，必须明确保留左侧还是右侧。
- **Padding**只为组成 batch 补齐长度；Attention mask 防止模型读取 PAD，loss mask 防止 PAD 贡献训练目标。
- 文本字符数、单词数和 token 数不存在固定比例；上下文窗口和计费通常按 token 计算。
- 词表越大不必然越好：可能缩短序列，却增加 Embedding/LM Head 参数，并改变多语言与稀有片段的分配。

## 最小验证

对任意 Tokenizer 至少检查：

```python
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)
print(len(text), len(ids))
print(ids)
print(decoded)
```

验证重点不是要求字节级完全相等，而是弄清规范化、空格、未知字符和特殊 token 是否导致可预期变化。还应分别测试中文、英文、数字、代码、emoji 和极长重复文本。

## 对比与边界

- Tokenizer 的切分不会自动产生词义，ID 的数值大小也没有语义顺序。
- [[位置编码]] 在模型内部提供顺序信息，不属于 Tokenizer。
- 字符级教学实现能展示完整链路，但不能代表现代 LLM 的序列效率、词表和多语言行为。
- 更换 Tokenizer 会改变 Embedding 和 LM Head 的词表维，通常不能直接复用原模型权重。

## 易错点

> [!warning] Token 数量是模型相关的
> 不能用一个 Tokenizer 的 token 数估计另一模型，也不能把“一个 token 等于一个词”作为通用规则。训练、推理和评测必须使用与 checkpoint 配套的词表、特殊 token 和聊天模板。

## 掌握要求

- [ ] 能区分 token、ID、词表、Tokenizer 和 Embedding。
- [ ] 能解释字符级、BPE、Unigram 与 byte-level 的主要取舍。
- [ ] 能检查特殊 token、聊天模板、截断和 padding 的实际输出。
- [ ] 能说明更换 Tokenizer 为什么通常需要改变模型参数。
- [ ] 能用样例统计并比较不同文本的 token 长度。

## 关联

- 属于：[[数据与训练]]

## 来源

- [[精炼-03-数据、Tokenizer与预训练]]：迁移 token、ID、词表、字符级与生产级 Tokenizer 的边界。
- [SentencePiece: A simple and language independent subword tokenizer and detokenizer](https://arxiv.org/abs/1808.06226)：SentencePiece 与 Unigram/BPE 实现背景。
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)：BPE 子词方法在神经机器翻译中的经典来源。

