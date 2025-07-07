Comprehensive Notes on Stanford CS25: Transformers United (Lecture 1)

1. Introduction to Transformers
Transformers, introduced in 2017 by Vaswani et al. in the seminal paper "Attention is All You Need", are a cornerstone of modern deep learning. Unlike earlier models like recurrent neural networks (RNNs) and long short-term memory (LSTM) networks, transformers rely entirely on attention mechanisms to process sequential data. This shift has led to significant advancements in natural language processing (NLP), computer vision (CV), reinforcement learning (RL), and beyond. Transformers are particularly effective for sequence-to-sequence tasks, such as translating text from one language to another, due to their ability to handle long-range dependencies and process data in parallel.
Key Features

Attention-Based: Transformers use attention to focus on relevant parts of the input, improving context understanding.
Parallel Processing: Unlike RNNs, which process sequences sequentially, transformers handle all elements simultaneously, speeding up training.
Versatility: Applied in NLP (e.g., ChatGPT), CV (e.g., Vision Transformers), and biology (e.g., AlphaFold).

2. Overview of Transformer Architecture
The transformer architecture consists of two main components: an encoder and a decoder, each made up of multiple layers. The encoder processes the input sequence to create a set of contextualized representations, which the decoder uses to generate the output sequence. This structure is particularly suited for tasks like machine translation, where the input and output are different sequences.
High-Level Workflow

Input Processing: Text is tokenized into numerical representations called tokens, which are converted into vectors via an embedding table.
Encoder: Processes the input sequence using self-attention and feed-forward networks to produce encoded representations.
Decoder: Generates the output sequence by attending to the encoder’s output and its own previous outputs, ensuring contextually appropriate results.

3. Attention Mechanisms
Attention mechanisms allow transformers to prioritize relevant parts of the input sequence when processing each element. They were initially developed to enhance RNN-based models for tasks like machine translation but became the core of transformers.
How Attention Works
Attention computes a weighted sum of values based on the similarity between queries and keys. The formula for scaled dot-product attention, the primary attention mechanism in transformers, is:
[\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V]

Q (Query): Represents the current token’s request for information.
K (Key): Represents the information available from other tokens.
V (Value): The actual information to be weighted and summed.
(d_k): Dimension of the key vectors, used for scaling to prevent large values.

Example
Consider the sentence "The cat sat on the mat." When processing "cat," the attention mechanism calculates scores to determine how much focus to give to "The," "sat," "on," etc., based on their relevance to "cat."
4. Self-Attention
Self-attention is a type of attention where queries, keys, and values all come from the same sequence, allowing each token to attend to every other token in the input. This enables the model to capture contextual relationships effectively.
Calculation Steps
Using the example sentence "The animal didn’t cross the street because it was too tired" from The Illustrated Transformer:

Create Vectors: Input embeddings (dimension 512) are transformed into query, key, and value vectors (dimension 64) using learned weight matrices.
Compute Scores: Calculate dot products between the query vector for "it" and key vectors for all tokens (e.g., "animal," "street").
Scale Scores: Divide by (\sqrt{d_k}) (e.g., (\sqrt{64} = 8)) to stabilize gradients.
Apply Softmax: Normalize scores to sum to 1, indicating attention weights.
Weight Values: Multiply value vectors by attention weights and sum to get the output for "it," emphasizing "animal."

Multi-Head Attention
Transformers use multiple attention heads (e.g., 8) to capture different relationships. Each head computes attention independently, and outputs are concatenated and transformed, enhancing the model’s ability to focus on various aspects of the input.
5. Other Necessary Ingredients
Beyond attention, transformers include several components to ensure effective processing and training stability.
Positional Encoding
Since transformers lack recurrence, they use positional encodings to incorporate the order of tokens. These are added to input embeddings and defined as:
[PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)][PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)]

pos: Position of the token in the sequence.
i: Dimension index.
(d_{\text{model}}): Embedding dimension (e.g., 512).

Example: For the first token (pos=0) in a 512-dimensional embedding, the encoding alternates sine and cosine values to represent position uniquely.
Feed-Forward Networks
Each encoder and decoder layer includes a position-wise feed-forward network (FFN) applied identically to each token:
[\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2]
This introduces non-linearity and enhances the model’s capacity to learn complex patterns.
Layer Normalization and Residual Connections

Layer Normalization: Normalizes inputs across features to stabilize training.
Residual Connections: Add the input of a sub-layer to its output, aiding gradient flow in deep networks.

Decoder-Specific Components

Masked Self-Attention: Prevents the decoder from attending to future tokens during generation.
Encoder-Decoder Attention: Allows the decoder to focus on the encoder’s output, crucial for tasks like translation.

6. Encoder-Decoder Architecture
The transformer’s encoder-decoder architecture is designed for sequence-to-sequence tasks. The encoder and decoder each consist of a stack of identical layers (e.g., 6 layers in the original transformer).
Encoder

Input: Tokenized sequence with positional encodings.
Layers: Each layer has a multi-head self-attention sub-layer and a feed-forward sub-layer, with residual connections and layer normalization.
Output: A set of contextualized representations.

Decoder

Input: Output sequence (shifted right during training) with positional encodings.
Layers: Includes masked self-attention, encoder-decoder attention, and feed-forward sub-layers.
Output: Generates the next token iteratively during inference.

Example: Machine Translation
For translating "Hello world" to French ("Bonjour le monde"):

Encoder: Processes "Hello world" to produce encoded representations capturing the meaning.
Decoder: Starts with a start token, attends to the encoder’s output, and generates "Bonjour," then "le," and so on.

7. Advantages and Disadvantages
Transformers have reshaped deep learning, but they come with trade-offs.
Advantages



Advantage
Description



Parallel Processing
Processes entire sequences simultaneously, unlike sequential RNNs, leading to faster training.


Long-Range Dependencies
Attention mechanisms effectively capture relationships between distant tokens.


Scalability
Can be scaled to large models (e.g., GPT-3 with 175 billion parameters) for improved performance.


Versatility
Applicable to NLP, CV, RL, and more, as seen in models like BERT and Vision Transformers.


Disadvantages



Disadvantage
Description



Computational Cost
Requires significant resources, making training expensive.


Data Requirements
Needs large datasets to avoid overfitting, less effective with small data.


Interpretability
Attention weights are complex and hard to interpret.


Energy Consumption
High carbon footprint due to extensive computational needs.


8. Applications of Transformers
Transformers have been applied across diverse domains, showcasing their flexibility.
Natural Language Processing (NLP)

Machine Translation: Translating languages, as in Google Translate.
Text Generation: Creating coherent text, as in GPT-3.
Sentiment Analysis: Determining emotions in text, used in social media monitoring.

Computer Vision (CV)

Image Classification: Vision Transformers (ViT) classify images with high accuracy.
Object Detection: Identifying objects in images, used in autonomous vehicles.

Reinforcement Learning (RL)

Decision-Making: Transformers guide agents in complex environments, as in game-playing AI.
Policy Learning: Optimizing actions in robotics.

Biology

Protein Folding: AlphaFold by DeepMind solves protein structures using transformers.
Genomics: Analyzing DNA sequences for medical research.

Speech

Speech Recognition: Converting audio to text, as in virtual assistants.
Speech Synthesis: Generating human-like speech.

Example: AlphaFold
AlphaFold uses transformers to predict protein structures by modeling relationships between amino acids, solving a decades-old biological challenge.
9. Additional Resources
For deeper understanding, explore:

The original transformer paper for technical details.
Visual explanations in The Illustrated Transformer.
Tutorials on platforms like DataCamp for practical insights.

