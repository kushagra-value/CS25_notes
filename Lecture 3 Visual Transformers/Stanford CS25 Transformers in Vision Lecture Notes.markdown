# Detailed Notes on Stanford CS25: Transformers in Vision

This document provides comprehensive notes on the Stanford CS25 lecture titled "Transformers in Vision: Tackling Problems in Computer Vision," presented by Lucas Beyer on August 11, 2022. The notes cover key topics likely discussed in the lecture, based on related research papers and summaries, including the Vision Transformer (ViT), evaluation challenges, scaling laws, knowledge distillation, multimodal learning, and unified vision models. Each section includes detailed explanations, equations, simplified examples, and descriptions of relevant graphs to aid understanding from beginner to advanced levels.

## 1. Introduction to Transformers

### Overview
Transformers, introduced in the 2017 paper "Attention is All You Need" by Vaswani et al., are deep learning models that excel in processing sequential data, originally for natural language processing (NLP). In computer vision, Transformers offer an alternative to convolutional neural networks (CNNs) by capturing global relationships in images.

### Key Components
- **Self-Attention Mechanism**: Computes the importance of each input element relative to others, allowing the model to focus on relevant parts of the data.
- **Multi-Head Attention**: Runs multiple attention mechanisms in parallel to capture different relationships, improving robustness.
- **Positional Encoding**: Adds information about the position of each input element, as Transformers lack inherent order awareness.
- **Feed-Forward Networks**: Apply transformations to each position independently, followed by layer normalization and residual connections.

### Differences from CNNs
- **Global Context**: Unlike CNNs, which use local receptive fields, Transformers process the entire input simultaneously, capturing long-range dependencies.
- **Flexibility**: Transformers handle variable-length inputs and are not constrained to grid-structured data like images.
- **Data Requirements**: Transformers often require more data to achieve performance comparable to CNNs due to fewer inductive biases (e.g., locality).

### Simplified Example
Imagine a sentence like "The cat sleeps." In NLP, a Transformer processes each word, considering its relationship to others. In vision, an image is treated similarly, but instead of words, the model processes patches of pixels, like small squares of a photo, to understand the whole scene.

### Equations
- **Self-Attention**:
  \[
  \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
  \]
  where \(Q\), \(K\), and \(V\) are query, key, and value matrices, and \(d_k\) is the dimension of the keys.

### Graph Description
A typical diagram of a Transformer encoder shows a stack of layers, each containing a multi-head attention block followed by a feed-forward network, with residual connections and layer normalization. Arrows indicate data flow from input embeddings to output representations.

## 2. Vision Transformer (ViT)

### Overview
The Vision Transformer (ViT), introduced in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" ([ViT Paper](https://arxiv.org/abs/2010.11929)), applies a pure Transformer architecture to image classification by treating images as sequences of patches. It achieves competitive performance against CNNs when pre-trained on large datasets.

### Architecture
1. **Patch Embedding**:
   - Divide an image of size \(H \times W \times C\) into patches of size \(P \times P\).
   - Flatten each patch into a vector of size \(P^2 \cdot C\).
   - Linearly project to a dimension \(D\), resulting in \(N = HW/P^2\) patch embeddings.
2. **Position Embedding**:
   - Add learnable 1D position embeddings to retain spatial information.
3. **Class Token**:
   - Prepend a learnable [CLS] token to the sequence, whose final representation is used for classification.
4. **Transformer Encoder**:
   - Consists of \(L\) layers, each with multi-head self-attention (MSA) and multilayer perceptron (MLP) blocks, plus layer normalization and residual connections.
5. **Classification Head**:
   - A linear layer or MLP applied to the [CLS] token’s output for classification.

### Model Variants
| Model      | Layers | Hidden Size \(D\) | MLP Size | Heads | Parameters |
|------------|--------|-------------------|----------|-------|------------|
| ViT-Base   | 12     | 768               | 3072     | 12    | 86M        |
| ViT-Large  | 24     | 1024              | 4096     | 16    | 307M       |
| ViT-Huge   | 32     | 1280              | 5120     | 16    | 632M       |

### Key Equations
- **Patch Embedding**:
  \[
  \mathbf{z}_0 = [\mathbf{x}_{\text{class}}; \mathbf{x}_p^1 \mathbf{E}; \mathbf{x}_p^2 \mathbf{E}; \dots; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{\text{pos}}
  \]
  where \(\mathbf{x}_p^i\) is a flattened patch, \(\mathbf{E}\) is the projection matrix, and \(\mathbf{E}_{\text{pos}}\) are position embeddings.
- **Transformer Encoder Layer**:
  \[
  \mathbf{z}_\ell' = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}
  \]
  \[
  \mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}_\ell')) + \mathbf{z}_\ell'
  \]
- **Output**:
  \[
  \mathbf{y} = \text{LN}(\mathbf{z}_L^0)
  \]
  where \(\mathbf{z}_L^0\) is the [CLS] token’s state after \(L\) layers.

### Performance
ViT achieves excellent results on benchmarks like ImageNet (up to 88.55% top-1 accuracy for ViT-H/14) and CIFAR-100 when pre-trained on large datasets like JFT-300M (303M images). It requires fewer computational resources than CNNs for training.

### Simplified Example
For a 224x224 RGB image with 16x16 patches:
- Number of patches: \( (224 / 16)^2 = 196 \).
- Each patch is flattened to \( 16 \times 16 \times 3 = 768 \) values.
- A linear layer projects this to \(D = 768\).
- The Transformer processes these 196 embeddings plus a [CLS] token, like a sentence with 197 words, to predict the image’s class (e.g., "cat").

### Graph Description
Figure 1 in the ViT paper illustrates the architecture: an image is split into patches, embedded, and fed into a Transformer encoder. The [CLS] token’s output is shown feeding into a classification head. Another graph (Figure 5) plots accuracy vs. compute, showing ViT’s efficiency compared to CNNs.

## 3. Evaluating Vision Models: Beyond ImageNet

### Overview
The paper "Are we done with ImageNet?" ([ImageNet Paper](https://arxiv.org/abs/2006.07159)) questions the reliability of ImageNet as a benchmark due to potential overfitting to its labeling idiosyncrasies. New human annotations reveal smaller performance gains than reported, suggesting evaluation challenges for vision models like ViT.

### Key Findings
- **Labeling Issues**: Original ImageNet labels may not reflect true generalization, as models exploit dataset-specific patterns.
- **New Annotations**: Reassessed labels ([Reassessed ImageNet](https://github.com/google-research/reassessed-imagenet)) show reduced performance gaps, indicating overfitting.
- **Implications**: Future evaluations should use diverse datasets and out-of-distribution tests (e.g., ObjectNet) to assess robustness.

### Simplified Example
Suppose a model achieves 90% accuracy on ImageNet but only 70% on new annotations. This suggests the model learned dataset quirks rather than general visual features, like recognizing a dog regardless of background.

### Graph Description
The paper likely includes a bar chart comparing model accuracies on original vs. reassessed ImageNet labels, showing smaller performance differences with new labels.

## 4. Scaling Laws for Vision Transformers

### Overview
The paper "Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design" ([Scaling Laws](https://arxiv.org/abs/2305.13035)) explores how to optimize ViT’s architecture (width and depth) for a given compute budget, introducing the SoViT model.

### Key Findings
- **Optimal Shapes**: Balancing width (hidden size) and depth (layers) improves performance for fixed compute.
- **SoViT Performance**: SoViT-400m/14 achieves 90.3% ImageNet accuracy, surpassing larger models like ViT-g/14 with less compute.

### Simplified Example
Think of building a ViT like designing a car engine: you can make it wider (more cylinders) or deeper (more gears). The paper finds the best combination for efficiency, like a compact engine that still performs like a larger one.

### Graph Description
A plot likely shows accuracy vs. compute for different ViT configurations, with SoViT models forming a frontier of optimal performance.

## 5. Knowledge Distillation for Vision Models

### Overview
The paper "Knowledge distillation: A good teacher is patient and consistent" ([Distillation Paper](https://arxiv.org/abs/2106.05237)) describes transferring knowledge from a large teacher model to a smaller student model, making ViTs more practical.

### Key Techniques
- **Teacher-Student Setup**: A large ViT (teacher) guides a smaller ViT or CNN (student).
- **Loss Function**: Combines cross-entropy with KL divergence between teacher and student outputs.
- **Results**: A distilled ResNet-50 achieves 82.8% ImageNet accuracy, competitive with larger models.

### Simplified Example
Imagine a master chef (large ViT) teaching a novice (small model) to cook a complex dish. The novice learns to mimic the master’s techniques, producing similar results with less effort.

### Graph Description
A table or plot compares the accuracy and compute cost of distilled vs. original models, showing minimal performance loss with reduced resources.

## 6. Multimodal Learning with Transformers

### LiT: Locked Image-Text Tuning
- **Overview**: The paper "LiT: Zero-Shot Transfer with Locked-image text Tuning" ([LiT Paper](https://arxiv.org/abs/2111.07991)) introduces contrastive tuning to align pre-trained image and text models.
- **Method**: Locks the image model (e.g., ViT) and tunes the text model to enable zero-shot tasks like classification.
- **Performance**: Achieves 85.2% ImageNet accuracy and 82.5% on ObjectNet.
- **Example**: Given an image of a dog and text prompts like “dog” or “cat,” LiT predicts the correct label without retraining, like a human recognizing objects from descriptions.

### PaLI: Pathways Language and Image Model
- **Overview**: PaLI ([PaLI Model](https://arxiv.org/abs/2209.06794)) combines large-scale vision and language models for tasks like visual question answering and captioning across 109 languages.
- **Architecture**: Uses a 4B parameter ViT and Transformer-based text decoder.
- **Performance**: State-of-the-art on benchmarks like COCO-Captions and VQAv2.
- **Example**: For an image of a beach, PaLI can answer “What’s the weather like?” or generate a caption like “Sunny beach with waves.”

### Graph Description
A bar chart likely compares PaLI’s performance across tasks (e.g., CIDEr scores for captioning) against prior models, showing improvements with scale.

## 7. Unified Vision Models: UViM

### Overview
The paper "UViM: A Unified Modeling Approach for Vision with Learned Guiding Codes" ([UViM Paper](https://arxiv.org/abs/2205.10337)) proposes a single model for multiple vision tasks using a base model guided by a language model.

### Key Features
- **Architecture**: A feed-forward base model predicts outputs, guided by discrete codes from an autoregressive language model.
- **Tasks**: Competitive on panoptic segmentation, depth prediction, and image colorization.
- **Advantage**: Eliminates task-specific modifications, simplifying model design.

### Simplified Example
Think of UViM as a Swiss Army knife: one tool (model) handles different tasks (cutting, screwing) by following instructions (guiding codes) from a manual (language model).

### Graph Description
A table compares UViM’s performance metrics (e.g., IoU for segmentation) against task-specific models, showing near state-of-the-art results.

## Additional Notes
- **Lecture Context**: As part of Stanford CS25, the lecture likely included introductory remarks on Beyer’s background and the evolution of Transformers in vision.
- **Potential Topics**: The lecture may have covered additional applications (e.g., object detection, video analysis) or datasets (e.g., JFT-300M), though not explicitly mentioned in the papers.
- **Visual Aids**: Beyer likely used slides with diagrams of ViT’s architecture, performance plots, and example outputs (e.g., image classifications).

## Key Citations
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Are we done with ImageNet?](https://arxiv.org/abs/2006.07159)
- [Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design](https://arxiv.org/abs/2305.13035)
- [Knowledge distillation: A good teacher is patient and consistent](https://arxiv.org/abs/2106.05237)
- [LiT: Zero-Shot Transfer with Locked-image text Tuning](https://arxiv.org/abs/2111.07991)
- [PaLI: A Jointly-Scaled Multilingual Language-Image Model](https://arxiv.org/abs/2209.06794)
- [UViM: A Unified Modeling Approach for Vision with Learned Guiding Codes](https://arxiv.org/abs/2205.10337)
- [Reassessed ImageNet Labels](https://github.com/google-research/reassessed-imagenet)
- [Vision Transformer Code and Models](https://github.com/google-research/vision_transformer)
- [PaLI Blog Post](https://ai.googleblog.com/2022/09/pali-scaling-language-image-learning-in.html)
- [PaLI GitHub Repository](https://github.com/google-research/google-research/tree/master/pali)