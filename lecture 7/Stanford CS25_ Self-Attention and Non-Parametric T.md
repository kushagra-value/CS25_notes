<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Stanford CS25: Self-Attention and Non-Parametric Transformers (NPTs) - Comprehensive Lecture Notes

This detailed set of notes covers the Stanford CS25 lecture on Self-Attention and Non-Parametric Transformers (NPTs), breaking down complex concepts into easy-to-understand explanations with practical examples.

## Introduction to the Lecture

The lecture is divided into two main parts:

1. A 15-minute overview of Transformers by Aidan, covering their origin and fundamental intuitions
2. A presentation on Non-Parametric Transformers (NPTs) by Neil and Jannik, based on their NeurIPS-accepted paper

## Part 1: Transformers Overview

### The Fundamentals of Transformer Architecture

Transformers introduced two revolutionary concepts that changed deep learning forever:

1. Self-attention
2. Multi-head attention

These components were combined with fast auto-regressive decoding to create a powerful new architecture.

#### Historical Context

Before Transformers emerged in 2017, most researchers were using LSTMs (Long Short-Term Memory networks) with attention mechanisms. The shift to Transformer architecture represented a significant leap forward in natural language processing capabilities.

#### Self-Attention Explained

Traditional attention mechanisms worked between two separate sequences (source and target), where each token in the source sequence would be associated with a "soft approximation" of an element in the target sequence.

**Self-attention is different because:**

- It uses the same sequence for both source and target
- It relates each element within the sequence to other elements in the same sequence
- It learns relationships between words within a sentence

For example, in the phrase "the blue ball":

- Self-attention would help relate "blue" (adjective) to "ball" (noun)
- This creates a meaningful connection showing that "blue" is describing "ball"

This helps the model understand internal sequence patterns - how words relate to each other within the same text.

#### Multi-Head Attention Simplified

Multi-head attention takes self-attention to the next level:

1. Each word is represented by an embedding (a vector of numbers)
2. The sentence embeddings are split depth-wise into several groups (e.g., 4 groups)
3. Attention is applied independently to each group
4. Results are concatenated back together to return to the original model dimension

**Why is this helpful?**
Each attention "head" can focus on learning one specific type of relationship:

- One head might learn relationships between adjectives and nouns
- Another might learn relationships between verbs and objects
- Yet another might focus on entity relationships

This creates a "hierarchy" or list of different relationship patterns the model can recognize.

#### Fast Auto-Regressive Decoding

Auto-regressive generation works like this:

1. Generate the first token
2. Based on the first token, generate the second token
3. Based on the first and second tokens, generate the third
4. And so on...

This sequential process is slow because it creates a loop where each step depends on previous steps.

**The transformer innovation:**

- Instead of generating one token at a time in a loop, transformers use a technique called "teacher forcing"
- The model assumes it always generates correct tokens
- During training, it feeds in the entire target sequence and predicts one token ahead
- This allows for parallel processing, making transformers much faster and more scalable

**To prevent "cheating":**

- The model must be prevented from looking at future tokens it shouldn't have seen yet
- This is achieved by creating an "attention mask"
- The mask blocks the model from attending to tokens in the future
- As processing moves through the sequence, the mask gradually reveals more past tokens


### The Transformer Origin Story

Aidan shared his personal experience as an intern at Google in 2017, working alongside key figures in the development of Transformers:

- Transformers were developed in approximately three months as a sprint toward the NeurIPS deadline
- Lukasz Kaiser and Aidan were working on "tensor-to-tensor," a framework for multimodal learning
- The framework already included emerging techniques like layer normalization and learning rate warm-up
- When Noam Shazeer, Ashish Vaswani, Nikki Parmar and Jakob Uszkoreit adopted this framework, these features were enabled by default
- The team spent significant time running ablation studies to determine which components were necessary
- Removing components like layer normalization or learning rate warm-up significantly hurts performance

**Why Transformers became so successful:**

1. Ease of optimization - robust to hyperparameter choices
2. Tailored to modern hardware accelerators (GPUs, TPUs)
3. Highly parallelizable architecture
4. Efficiency allowed for scaling laws that have driven recent AI advances

## Part 2: Non-Parametric Transformers (NPTs)

### A New Paradigm in Deep Learning

#### Challenging Traditional Parametric Prediction

Most supervised deep learning approaches rely on parametric prediction:

1. We have training data of inputs (X) and outcomes (Y)
2. We create a model with tunable parameters (θ)
3. We optimize these parameters using the training data
4. At test time, we use only these optimized parameters to make predictions
5. The prediction is independent of the training data once parameters are set

**Benefits of parametric prediction:**

- Convenient: all learning is summarized in parameters
- Efficient: don't need to store training data at test time
- Scalable: works with large datasets


#### The NPT Innovation

NPTs challenge this dominant paradigm by:

1. Taking the entire dataset as input when possible
2. Learning to predict from interactions between data points
3. Using multi-head self-attention for reasoning about relationships
4. Employing stochastic masking to guide prediction and regularize learning

**Core Concept:** Instead of just learning from features, NPTs learn to use relationships between data points to make predictions.

### How NPTs Differ from Traditional Models

**Traditional model approach:**

- Predict a target value using only features from that specific input row
- Parameters depend on training data, but at test time, only looks at the single instance

**NPT approach:**

- Predict with explicit dependence on all samples in the input
- Look beyond the single data point and consider values from other samples
- Learn patterns across the entire dataset, not just within individual samples

Someone on Twitter called this "KNN 2.0" - a modern take on the k-nearest neighbors algorithm, where the model learns sophisticated ways to use nearby data points for prediction.

### Comparison with Existing Non-Parametric Methods

NPTs build on a tradition of non-parametric models that make predictions with explicit dependence on training data:

- Gaussian processes
- K-nearest neighbors
- Kernel methods

NPTs combine the benefits of non-parametric approaches with representation learning in a novel way.

```python
# Example code showing the conceptual difference between parametric and non-parametric models

# Traditional parametric model (e.g., neural network)
def parametric_model(x_test, trained_parameters):
    # Only uses the test instance and learned parameters
    prediction = apply_model(x_test, trained_parameters)
    return prediction

# Non-parametric transformer approach
def non_parametric_transformer(x_test, training_data, trained_parameters):
    # Combines test instance with training data
    combined_input = concatenate([x_test, training_data])
    
    # Apply self-attention to allow interactions between data points
    attention_outputs = self_attention_layer(combined_input, trained_parameters)
    
    # Extract prediction for test instance
    prediction = extract_test_prediction(attention_outputs)
    return prediction
```


### NPT Architecture

The NPT architecture uses transformer components to implement this new paradigm:

1. **Data Preparation**:
    - The entire dataset is taken as input (both training and test data)
    - Values are embedded into a latent space
2. **Self-Attention Mechanism**:
    - Self-attention is applied between data points rather than within sequences
    - This allows each data point to "look at" and learn from other data points
3. **Masking Strategy**:
    - Uses stochastic masking (similar to techniques in BERT)
    - Tells the model which values to predict
    - Regularizes the learning process
4. **Prediction**:
    - The model learns to generate predictions by attending to relevant information in other data points
```python
# Simplified NPT implementation concept
import torch
import torch.nn as nn

class NonParametricTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # Stack of transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.output_layer = nn.Linear(hidden_dim, 1)  # For regression tasks
        
    def forward(self, x, mask=None):
        # x shape: [batch_size, num_features]
        # Convert to embeddings
        embeddings = self.embedding(x)
        
        # Apply transformer layers - this is where data points attend to each other
        transformer_out = self.transformer(embeddings, src_key_padding_mask=mask)
        
        # Generate predictions
        predictions = self.output_layer(transformer_out)
        
        return predictions
```


### Visual Explanation of NPTs

Let's visualize how NPTs work compared to traditional models:

**Traditional Neural Network:**

```
Input Features (X) → Neural Network → Parameters (θ) → Prediction (Y)
```

**Non-Parametric Transformer:**

```
Test Data Point + Training Dataset → Self-Attention Between Points → Learned Relationships → Prediction
```

Imagine you're trying to predict house prices:

- A traditional model would look at features of house X (size, location, etc.) and make a prediction
- An NPT would look at house X AND similar houses in the dataset, learning which houses are relevant comparisons and how to use that information for better prediction


## Benefits and Applications of NPTs

NPTs excel at tasks where:

1. Relationships between data points matter
2. The dataset has complex patterns that aren't captured by looking at features in isolation
3. Similar examples in the training data can inform predictions

Potential applications include:

- Tabular data prediction with complex relationships
- Few-shot learning tasks
- Data with important contextual relationships
- Tasks where seeing the "big picture" across the dataset helps prediction


## Conclusion

NPTs represent a fascinating shift in deep learning paradigms by:

1. Challenging the dominant parametric prediction approach
2. Leveraging transformer architecture for interactions between data points
3. Learning sophisticated ways to use training data directly in predictions
4. Combining the strengths of representation learning and non-parametric methods

This approach opens new possibilities for machine learning models to reason about relationships within datasets and make more informed predictions by considering the entire data context.

---

**Note:** These notes cover the key concepts from the lecture on Self-Attention and Non-Parametric Transformers. The NPT approach builds on transformer architecture by applying self-attention between data points rather than just within sequences, offering a new paradigm for prediction tasks that benefits from considering relationships across the entire dataset.

<div style="text-align: center">⁂</div>

[^1]: https://www.youtube.com/watch?v=zejXBg-2Vpk\&list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM\&index=7

[^2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC1797516/

[^3]: https://paperswithcode.com/newsletter/11

[^4]: https://glasp.co/youtube/p/stanford-cs25-v1-i-transformers-united-dl-models-that-have-revolutionized-nlp-cv-rl

[^5]: https://www.youtube.com/watch?v=YD-wagrJjhU

[^6]: https://downsub.com

[^7]: https://web.stanford.edu/class/cs25/

[^8]: https://slds-lmu.github.io/seminar_nlp_ss20/attention-and-self-attention-for-nlp.html

[^9]: https://arxiv.org/abs/2106.02584

[^10]: https://pmc.ncbi.nlm.nih.gov/articles/PMC2731626/

[^11]: https://arxiv.org/pdf/2304.10295.pdf

[^12]: https://downsub.com/sites/youtube

[^13]: https://huggingface.co/datasets/jilp00/youtoks-transcripts-Stanford-CS25-Transformers-United

[^14]: https://www.ibm.com/think/topics/self-attention

[^15]: https://github.com/OATML/non-parametric-transformers

[^16]: https://journals.asm.org/doi/abs/10.1128/jvi.01473-06

[^17]: https://www.youtube.com/watch?v=zejXBg-2Vpk

[^18]: https://www.citeab.com/kits/12724510-vpk-141-aav-purification-mega-kit

[^19]: https://www.youtube.com/watch?v=fKMB5UlVY1E

[^20]: https://www.biorxiv.org/content/10.1101/396192v1.full.pdf

[^21]: https://paperswithcode.com/search?q=author%3AHong+Qu\&order_by=date\&order=asc

[^22]: https://www.youtube.com/watch?v=qGkzHFllWDY

[^23]: https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html

[^24]: https://arxiv.org/html/2502.06684v1

[^25]: https://openreview.net/forum?id=nL2lDlsrZU

[^26]: https://proceedings.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf

[^27]: https://arxiv.org/pdf/1801.04520.pdf

[^28]: https://economictimes.indiatimes.com/company/npt-papers-private-limited/U22130DL2007PTC160522

[^29]: https://openreview.net/forum?id=wRXzOa2z5T

[^30]: https://www.american-cse.org/csce2023-ieee/pdfs/CSCE2023-5LlpKs7cpb4k2UysbLCuOx/275900b498/275900b498.pdf

[^31]: https://arxiv.org/abs/2312.00662

