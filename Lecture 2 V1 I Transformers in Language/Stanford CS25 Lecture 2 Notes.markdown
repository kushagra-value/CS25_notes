# Comprehensive Notes on Stanford CS25: Transformers in Language - The Development of GPT Models, Including GPT-3

## 1. Introduction to Language Models

Language models predict the likelihood of word sequences, enabling tasks like text generation, translation, and speech recognition. They assign probabilities to sequences, helping machines understand and generate human-like text.

### Evolution of Language Models

- **N-gram Models (1950s–2000s)**: These models, like 3-gram, predict the next word based on the previous two words. They’re simple but limited by short context windows and inability to capture complex patterns. For example, in “The cat is,” a 3-gram model might predict “on” based on frequency, ignoring broader context.
- **Neural Network Models (2000s–2010s)**: Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTMs) networks process sequences sequentially, capturing longer dependencies. However, they’re slow to train due to sequential processing and struggle with very long sequences. For instance, an LSTM might forget early words in a long sentence.
- **Transformers (2017–present)**: Introduced in the paper "Attention is All You Need", transformers use self-attention to process entire sequences simultaneously, capturing long-range dependencies efficiently. This makes them ideal for large-scale language models like GPT.

### Why Transformers Matter

Transformers enable parallel processing, speeding up training, and their attention mechanisms allow models to focus on relevant words, regardless of distance. This has led to breakthroughs in NLP, powering models like GPT-3 and ChatGPT.

## 2. Transformer Architecture for Language Modeling

GPT models use the decoder-only part of the transformer architecture, suited for autoregressive tasks where the model predicts the next word given previous words.

### Self-Attention Mechanism

Self-attention lets the model weigh the importance of each word in a sequence when processing a given word. It uses three vectors: queries (Q), keys (K), and values (V), derived from input embeddings via learned weight matrices.

The attention score is computed as:

\[ \\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right) V \]

- **Q**: Query vector for the current token.
- **K**: Key vectors for all tokens.
- **V**: Value vectors for all tokens.
- **(d_k)**: Dimension of key vectors, used for scaling to prevent large values.

#### Example: Self-Attention Calculation

Consider the sequence “hello world” with simplified embeddings (d_model=4):

- “hello”: \[1, 0, 0, 0\]
- “world”: \[0, 1, 0, 0\]

Assume weight matrices are identity for simplicity, so Q = K = V = embeddings. For “hello”:

- Q = \[1, 0, 0, 0\]
- K = \[\[1, 0, 0, 0\], \[0, 1, 0, 0\]\]
- Scores = Q @ K^T = \[1, 0\]
- Scaled by (\\sqrt{d_k} = \\sqrt{4} = 2): \[0.5, 0\]
- Softmax: \[0.622, 0.378\]
- Output = 0.622 \* \[1, 0, 0, 0\] + 0.378 \* \[0, 1, 0, 0\] = \[0.622, 0.378, 0, 0\]

This shows “hello” attends more to itself, capturing its own context.

### Multi-Head Attention

Multiple attention heads (e.g., 96 in GPT-3) compute attention in parallel, capturing different relationships. Outputs are concatenated and transformed, enhancing the model’s ability to understand complex patterns.

### Positional Encoding

Since transformers lack recurrence, positional encodings add information about word order. They use sine and cosine functions:

\[ PE(pos, 2i) = \\sin\\left(\\frac{pos}{10000^{2i/d\_{\\text{model}}}}\\right) \]

\[ PE(pos, 2i+1) = \\cos\\left(\\frac{pos}{10000^{2i/d\_{\\text{model}}}}\\right) \]

#### Example: Positional Encoding

For position pos=0, dimension i=0, d_model=512:

- PE(0, 0) = (\\sin(0 / 10000^{0/512}) = \\sin(0) = 0)
- PE(0, 1) = (\\cos(0 / 10000^{0/512}) = \\cos(0) = 1)

These values are added to input embeddings, ensuring the model knows the sequence order.

### Feed-Forward Networks

Each token is processed through a position-wise feed-forward network:

\[ \\text{FFN}(x) = \\max(0, xW_1 + b_1)W_2 + b_2 \]

This adds non-linearity, allowing the model to learn complex patterns.

### Layer Normalization and Residual Connections

- **Layer Normalization**: Stabilizes training by normalizing inputs across features.
- **Residual Connections**: Add the input to the output of each sub-layer, aiding gradient flow in deep networks.

## 3. Development of GPT Models

The Generative Pre-trained Transformer (GPT) series, developed by OpenAI, has evolved significantly, scaling in size and capability.

### GPT-1 (2018)

- **Parameters**: 117 million.
- **Training Data**: BookCorpus, \~7,000 unpublished books.
- **Architecture**: 12 layers, 12 attention heads, d_model=768.
- **Key Contribution**: Demonstrated that pre-training a transformer on a large corpus, followed by fine-tuning on specific tasks, yields strong NLP performance.
- **Example Use**: Fine-tuned for sentiment analysis, achieving high accuracy on movie review datasets.

### GPT-2 (2019)

- **Parameters**: 1.5 billion.
- **Training Data**: WebText, \~8 million web pages.
- **Architecture**: 48 layers, 48 attention heads, d_model=1600.
- **Improvements**: Scaled up model size and data, enabling coherent text generation over longer passages. Showed potential for zero-shot tasks.
- **Concerns**: OpenAI initially limited release due to misuse risks, like generating fake news.
- **Example Use**: Generating news articles or creative writing, e.g., continuing a story prompt like “Once upon a time…”

### GPT-3 (2020)

- **Parameters**: 175 billion.
- **Training Data**: 570GB from filtered Common Crawl, WebText2, Wikipedia, books; \~500 billion tokens.
- **Architecture**: 96 layers, 96 attention heads, d_model=12288, context window=2048 tokens. Uses alternating dense and sparse attention patterns for efficiency.
- **Key Contribution**: Introduced few-shot learning, where the model performs tasks with minimal examples in the prompt, often without fine-tuning.
- **Training Cost**: Estimated at $14 million, consuming 190,000 kWh (Carbon Footprint).
- **Example Use**: Writing essays, answering questions, or generating code via Codex.

#### GPT-3 Model Sizes

| Model Name | Parameters | Layers | d_model | Heads | Batch Size | Learning Rate |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-3 Small | 125M | 12 | 768 | 12 | 0.5M | 6.0 × 10⁻⁴ |
| GPT-3 Medium | 350M | 24 | 1024 | 16 | 0.5M | 3.0 × 10⁻⁴ |
| GPT-3 Large | 760M | 24 | 1536 | 16 | 0.5M | 2.5 × 10⁻⁴ |
| GPT-3 XL | 1.3B | 24 | 2048 | 24 | 1M | 2.0 × 10⁻⁴ |
| GPT-3 2.7B | 2.7B | 32 | 2560 | 32 | 1M | 1.6 × 10⁻⁴ |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 | 2M | 1.2 × 10⁻⁴ |
| GPT-3 13B | 13.0B | 40 | 5140 | 40 | 2M | 1.0 × 10⁻⁴ |
| GPT-3 175B | 175.0B | 96 | 12288 | 96 | 3.2M | 0.6 × 10⁻⁴ |

## 4. Training and Usage of GPT Models

### Pre-training

GPT models are pre-trained to predict the next token in a sequence, maximizing the likelihood:

\[ \\prod\_{i=1}^n P(x_i | x_1, ..., x\_{i-1}) \]

This is achieved using cross-entropy loss on large, diverse text corpora.

### Fine-tuning vs. In-Context Learning

- **GPT-1 and GPT-2**: Typically fine-tuned on labeled data for specific tasks, like classification or translation.
- **GPT-3**: Leverages in-context learning, where tasks are specified in the prompt with examples, eliminating the need for fine-tuning.

#### Few-Shot Learning Example: Translation

**Prompt**:

```
English: Hello
French: Bonjour
English: World
French: Monde
English: How are you?
French:
```

**Output**: Comment allez-vous?

The model learns the translation pattern from the examples and applies it to the new input.

#### Learning Settings

- **Zero-Shot**: Task description only, no examples. E.g., “Translate ‘Hello’ to French” → “Bonjour.”
- **One-Shot**: One example provided. E.g., “English: Hello, French: Bonjour. Translate ‘World’” → “Monde.”
- **Few-Shot**: Multiple examples. E.g., the translation prompt above.

GPT-3’s performance improves with more examples, often rivaling fine-tuned models (Few-Shot Learners).

## 5. Applications and Impact

GPT models have transformed NLP with versatile applications:

### Key Applications

- **Text Generation**: Writing articles, stories, or poetry. E.g., GPT-3 can continue a story prompt with coherent narrative.
- **Question Answering**: Answering factual or contextual questions. E.g., “What is the capital of France?” → “Paris.”
- **Translation**: Performing translations with few-shot prompts, though less specialized than dedicated models.
- **Summarization**: Condensing long texts into concise summaries. E.g., summarizing a news article into key points.
- **Code Generation**: Generating code via Codex, a GPT-3 derivative. E.g., “Write a Python function to sort a list” yields working code.

### Performance Highlights

- **LAMBADA**: 76% zero-shot accuracy, 8% above state-of-the-art.
- **SuperGLUE**: Marginally outperforms fine-tuned BERT Large, excelling in tasks like COPA.
- **TriviaQA**: Surpasses fine-tuned models in question answering.

### Real-World Impact

- **Chatbots**: Powering conversational agents like ChatGPT.
- **Content Creation**: Assisting in writing emails, reports, or marketing copy.
- **Education**: Tutoring systems that explain concepts or answer student queries.

## 6. Limitations and Ethical Considerations

### Limitations

- **Lack of Bidirectionality**: GPT models are autoregressive, processing text left-to-right, which limits performance on tasks requiring full context, like fill-in-the-blank.
- **Coherence Issues**: Long text generation may lose coherence or repeat ideas.
- **Bias**: Reflects biases in training data, potentially producing biased or harmful outputs.
- **Computational Cost**: Training GPT-3 cost \~$14 million and 190,000 kWh, raising environmental concerns.
- **Interpretability**: Hard to understand why the model produces specific outputs, complicating trust and accountability.

### Ethical Considerations

- **Misuse Potential**: Generating misinformation or deepfakes (Malicious Uses).
- **Fairness**: Biases in gender, race, or religion from internet data can lead to unfair outcomes.
- **Accessibility**: High costs limit access to large-scale models, creating disparities in AI development.

### Future Directions

- **Efficiency**: Developing sparse or distilled models to reduce computational needs.
- **Bias Mitigation**: Improving data curation and training methods to minimize biases.
- **Multimodality**: Combining text with images or audio, as seen in GPT-4.
- **Interpretability**: Enhancing transparency in model decision-making.