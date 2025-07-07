# Detailed Notes on Self-Attention and Non-Parametric Transformers (NPTs)

These notes are based on the Stanford CS25 video ([YouTube](https://www.youtube.com/watch?v=zejXBg-2Vpk)) and the research paper on Non-Parametric Transformers ([arXiv](https://arxiv.org/pdf/2106.02584)). They are designed for beginners in machine learning, with simple explanations, examples, visualizations, and Python code to make complex concepts clear and interactive.

## 1. Introduction to Transformers and Self-Attention

### What Are Transformers?
Transformers are a type of deep learning model introduced in 2017 in the paper "Attention is All You Need" by Vaswani et al. They are widely used in natural language processing (NLP) tasks like translation and text generation, and have been adapted for images, robotics, and more. Unlike older models that process data sequentially, transformers handle entire sequences at once, making them faster and better at capturing relationships between elements.

**Analogy**: Think of a transformer as a librarian who can instantly find connections between all books in a library, rather than reading them one by one.

### What Is Self-Attention?
Self-attention is the core mechanism of transformers. It allows the model to focus on different parts of the input data when making predictions. For example, in a sentence, self-attention helps the model understand which words are related, even if they are far apart.

**Analogy**: Imagine you’re reading a story and need to understand the word “king.” Self-attention is like looking at all other words in the story to see which ones (like “crown” or “throne”) give clues about what “king” means.

### How Does Self-Attention Work?
Self-attention processes a sequence of items (like words or data points) by creating three vectors for each item:
- **Query (Q)**: What the item is looking for.
- **Key (K)**: What other items offer.
- **Value (V)**: The actual information from other items.

Here’s the step-by-step process:
1. Each item (e.g., a word) is converted into a vector (a list of numbers) called an embedding.
2. For each item, compute Q, K, and V vectors using learned weight matrices.
3. Calculate attention scores by taking the dot product of the item’s Q vector with all K vectors.
4. Scale the scores and apply a softmax function to get attention weights (numbers between 0 and 1 that sum to 1).
5. Compute the output as a weighted sum of V vectors, using the attention weights.

The formula is:
\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
\]
where \( d_k \) is the dimension of the Key vectors.

**Example**: Consider the sentence “The cat sat on the mat.”
- Each word is embedded into a vector.
- For “cat,” its Q vector is compared to K vectors of all words.
- If “sat” and “mat” have high scores, the model focuses on them, creating a new representation of “cat” that includes context from “sat” and “mat.”

### Visualization
To understand self-attention, we can visualize attention weights as a heatmap, where rows and columns are words, and cell values show how much each word attends to others.

**Python Code Example**:
Below is a simplified implementation of self-attention using NumPy, followed by a heatmap visualization using Matplotlib.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulated embeddings for "The cat sat on the mat" (5 words, 4D embeddings)
embeddings = np.random.rand(5, 4)

# Simulated weight matrices for Q, K, V
W_q = np.random.rand(4, 4)
W_k = np.random.rand(4, 4)
W_v = np.random.rand(4, 4)

# Compute Q, K, V
Q = embeddings @ W_q
K = embeddings @ W_k
V = embeddings @ W_v

# Compute attention scores
scores = Q @ K.T
scores = scores / np.sqrt(4)  # Scale by sqrt(d_k)
attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

# Visualize attention weights
plt.figure(figsize=(8, 6))
sns.heatmap(attention_weights, annot=True, cmap='Blues', xticklabels=['The', 'cat', 'sat', 'on', 'mat'], yticklabels=['The', 'cat', 'sat', 'on', 'mat'])
plt.title('Self-Attention Weights')
plt.xlabel('Keys')
plt.ylabel('Queries')
plt.savefig('attention_heatmap.png')
```

This code generates a heatmap showing which words attend to each other, helping you see the model’s focus.

## 2. Non-Parametric Transformers (NPTs)

### What Are Non-Parametric Models?
Machine learning models are either parametric or non-parametric:
- **Parametric**: Have a fixed number of parameters (e.g., neural networks, linear regression). They learn a function based on training data and use it for predictions.
- **Non-Parametric**: Parameters grow with data size (e.g., k-nearest neighbors). They often use the training data directly during prediction.

NPTs are non-parametric because they use the entire dataset to make predictions, rather than just learned parameters.

**Analogy**: A parametric model is like a chef who memorizes a recipe. A non-parametric model is like a chef who checks the pantry (dataset) every time they cook.

### What Are Non-Parametric Transformers (NPTs)?
NPTs are a new type of transformer that take the entire dataset as input, using self-attention to model relationships between all datapoints. This allows NPTs to make predictions by comparing a new input to all training data points, unlike traditional models that process one input at a time.

**Example**: Imagine predicting house prices. A traditional model uses a house’s features (size, location) to predict its price. An NPT looks at all houses in the dataset and predicts the price by focusing on similar houses.

### NPT Architecture
NPTs have a unique architecture with 8 layers, alternating between:
- **Attention Between Datapoints (ABD)**: Models relationships between different datapoints, allowing the model to “look up” relevant data.
- **Attention Between Attributes (ABA)**: Similar to standard self-attention, it models relationships within a datapoint’s features.

Other components include:
- Row-wise feed-forward networks with GeLU activation.
- Dropout (p=0.1) to prevent overfitting.
- Stochastic feature masking (p=0.15), where some features are hidden during training, forcing the model to predict them using other datapoints.

**Analogy**: ABD is like comparing students’ entire test papers to find patterns, while ABA is like comparing answers within one student’s paper.

### Stochastic Masking
NPTs use a masking mechanism inspired by NLP, where some features or targets are randomly hidden during training. The model learns to predict these by attending to other datapoints.

**Example**: In a dataset of student grades, if a student’s math score is masked, the model might predict it by looking at similar students’ scores.

### Visualization
We can visualize which datapoints the model attends to using a heatmap of attention weights, where rows are test datapoints and columns are training datapoints.

**Python Code Example**:
Below is a simplified NPT attention mechanism, visualizing attention weights.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulated dataset: 10 houses with 3 features (size, location, price)
dataset = np.random.rand(10, 3)

# Simulated Q, K, V for ABD
Q = dataset @ np.random.rand(3, 3)
K = dataset @ np.random.rand(3, 3)
V = dataset @ np.random.rand(3, 3)

# Compute attention scores
scores = Q @ K.T
scores = scores / np.sqrt(3)
attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

# Visualize
plt.figure(figsize=(10, 8))
sns.heatmap(attention_weights, annot=True, cmap='Greens')
plt.title('NPT Attention Between Datapoints')
plt.xlabel('Training Datapoints')
plt.ylabel('Test Datapoints')
plt.savefig('npt_attention_heatmap.png')
```

This shows which training houses the model focuses on when predicting prices for test houses.

## 3. Applications and Results

### Tasks NPTs Excel At
NPTs are designed for tasks requiring:
- **Cross-Datapoint Lookup**: Finding relevant datapoints to make predictions.
- **Complex Reasoning**: Capturing causal relationships across data.

**Example**: In a medical dataset, NPTs can predict a patient’s diagnosis by comparing their symptoms to other patients’ records.

### Performance on Tabular Data
NPTs were tested on UCI datasets like Higgs Boson and Poker Hand:
- **Higgs Boson**: 80.7% test accuracy.
- **Poker Hand**: 99.3% test accuracy.

They outperform or match baselines like:
- **MLP**: 99.5% on Poker Hand.
- **XGBoost**: 97.1% on Forest Cover.
- **TabNet**: 79.5% on Higgs Boson.

### Image Classification
NPTs show early promise on image datasets:
- **CIFAR-10**: 93.7% accuracy with a CNN encoder, 68.2% with a linear patching encoder.
- **MNIST**: 98.3% accuracy with a linear patching encoder.

These results suggest NPTs are versatile but may need optimization for images.

### Comparison Table
| Dataset       | NPT Accuracy | MLP Accuracy | XGBoost Accuracy | TabNet Accuracy |
|---------------|--------------|--------------|------------------|-----------------|
| Higgs Boson   | 80.7%        | 78.2%        | 78.9%            | 79.5%           |
| Poker Hand    | 99.3%        | 99.5%        | 98.7%            | 98.9%           |
| Forest Cover  | 96.5%        | 95.8%        | 97.1%            | 96.2%           |

## 4. Additional Topics from the Paper

### Theoretical Aspects
The paper proves that NPTs are **equivariant over datapoints**, meaning predictions remain consistent if datapoints are reordered. This ensures the model’s reliability.

### Ablation Studies
Ablation studies test the impact of components:
- **Without ABD**: Performance drops, as cross-datapoint attention is critical.
- **Without Feature Masking**: Reduces the model’s ability to generalize.

### Computational Cost
NPTs are memory-intensive (e.g., 19.18 GB GPU peak on Higgs Boson vs. 1.18 GB for TabNet). The paper suggests using sparse attention techniques to improve efficiency.

### Societal Impacts
The paper discusses:
- **Explanations**: NPTs can show which datapoints influence predictions, aiding transparency.
- **Fairness**: Potential biases if certain datapoints dominate attention.
- **Privacy**: Using entire datasets raises data security concerns.
- **Environmental Impact**: High computational cost increases energy use.

## 5. Code and Implementations

### GitHub Repository
The NPT code is available at [GitHub](https://github.com/OATML/Non-Parametric-Transformers). It uses PyTorch, NumPy, and Scikit-Learn, and supports training on UCI datasets and images.

### Example Usage
Below is a hypothetical example of training an NPT model.

```python
import torch
from npt.model import NPT
from npt.data import load_dataset

# Load dataset
train_data, test_data = load_dataset('higgs_boson')

# Initialize model
model = NPT(num_layers=8, num_heads=8, embedding_dim=64, dropout=0.1)

# Train
model.train(train_data, epochs=100, batch_size=32)

# Predict
predictions = model.predict(test_data)

# Save attention weights for visualization
attention_weights = model.get_attention_weights()
np.save('attention_weights.npy', attention_weights)
```

### Running the Code
To use the repository:
1. Clone it: `git clone https://github.com/OATML/Non-Parametric-Transformers.git`
2. Install dependencies: `pip install torch numpy pandas scikit-learn`
3. Follow the repository’s instructions for training and evaluation.

## 6. Interactive Elements
To make learning engaging, the notebook includes:
- **Sliders**: Adjust parameters like dropout rate and see their impact on attention weights (using Jupyter widgets).
- **Dropdowns**: Select different datasets to visualize model behavior.
- **Heatmaps**: Visualize attention weights for different tasks.

**Example Interactive Code**:
```python
from ipywidgets import interact
import numpy as np
import matplotlib.pyplot as plt

def plot_attention(dropout=0.1):
    # Simulated attention weights with dropout effect
    weights = np.random.rand(5, 5) * (1 - dropout)
    plt.figure(figsize=(6, 4))
    plt.imshow(weights, cmap='Blues')
    plt.title(f'Attention Weights with Dropout {dropout}')
    plt.colorbar()
    plt.savefig(f'attention_dropout_{dropout}.png')

interact(plot_attention, dropout=(0.0, 0.5, 0.05))
```

This lets you experiment with dropout and see its effect on attention.

## Conclusion
These notes cover self-attention and NPTs in detail, making them accessible for beginners. By combining explanations, examples, visualizations, and code, they aim to help you understand and experiment with these powerful models. Explore the GitHub repository to try NPTs yourself!