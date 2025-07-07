<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Stanford CS25: Transformers in Vision - Comprehensive Lecture Notes

This detailed set of notes covers Lucas Beyer's lecture on Vision Transformers from Stanford's CS25 course. The lecture explores how transformer architectures have been adapted for computer vision tasks, beginning with foundational concepts in visual representation learning and progressing through the evolution of approaches that led to Vision Transformers.

## 1. Introduction and Motivation for General Visual Representations

### What are General Visual Representations?

Visual representations are the foundation of computer vision systems, serving as the extracted features that enable machines to understand visual information. A general visual representation is one that can be applied across diverse visual tasks with minimal adaptation.

Lucas Beyer, a researcher at Google Brain in Zürich, explains his primary goal: to develop general visual representations that can power a wide range of vision-based applications[^1]. These representations aim to:

- Kickstart tasks requiring visual input
- Enable faster understanding of visual scenes
- Support applications like robotics where visual understanding is crucial
- Allow non-programmers to teach machines visual tasks


### Real-World Applications

The long-term vision involves creating systems (like robots) that can be taught by non-programmers to perform tasks based on visual understanding[^1]. This requires representations that generalize well across various visual domains and scenarios.

### Human Visual System as Inspiration

Humans possess remarkable general visual representations that allow us to:

- Quickly adapt to new visual categories
- Recognize patterns with very few examples
- Transfer knowledge across domains
- Understand abstract visual concepts


## 2. Visual Classification Examples

### Example 1: Flower Classification

Lucas demonstrates how humans can classify flowers into categories (A, B, or C) after seeing just five examples of each class[^1]. When shown a new flower image, we can immediately recognize which class it belongs to based on our powerful visual representation abilities.

### Example 2: Satellite Image Classification

Even with uncommon image types like satellite imagery, humans can quickly learn to distinguish between different classes (e.g., basketball courts) with minimal examples[^1]. This demonstrates our ability to adapt visual understanding to unfamiliar domains.

### Example 3: Abstract Object Counting

In a more abstract example, Lucas shows images containing geometric shapes and demonstrates how humans can quickly determine classification rules based on the number of objects (class A has 3 objects, class B has 5 objects), regardless of their shape, color, or arrangement[^1]. This showcases our ability to extract high-level concepts from visual data.

These examples illustrate the goal for machine learning systems: to develop similar capabilities for quick adaptation and generalization from few examples across diverse visual domains.

## 3. The Visual Task Adaptation Benchmark (VTAB)

### Benchmark Design

To measure progress toward general visual representations, Lucas and his collaborators developed the Visual Task Adaptation Benchmark (VTAB)[^1]. This benchmark:

1. Evaluates a model's ability to adapt to new tasks with limited examples
2. Covers a diverse landscape of visual tasks beyond standard image classification
3. Includes 19 carefully selected tasks spanning different domains and requirements
4. Tests both natural images and specialized imagery (e.g., satellite photos)
5. Incorporates tasks requiring different reasoning capabilities (counting, distance estimation, etc.)

### VTAB Evaluation Process

The benchmark follows this evaluation procedure:

1. A model is pre-trained using any desired approach and data
2. The model is adapted to each of the 19 benchmark tasks using a small number of examples
3. Performance is measured on test sets for each task
4. Final score is the average performance across all tasks[^1]

This approach provides a standardized way to compare different visual representation methods on their generalization capabilities.

## 4. Pre-training and Transfer Learning Terminology

### Key Terms

Lucas clarifies important terminology used throughout the lecture:

- **Pre-training** (also called **upstream** training): The initial training phase where a model learns general visual representations from a large dataset[^1]
- **Transfer learning** (also called **downstream** application): The process of adapting a pre-trained model to new tasks[^1]
- **Adaptation**: The specific method used for transfer learning (e.g., fine-tuning, linear probing, etc.)
- **Fine-tuning**: A common adaptation approach that updates all or some of the pre-trained model's parameters on the new task[^1]

Lucas notes that they generally prefer simple approaches like standard fine-tuning without complex modifications, as these methods typically work well when the pre-training is effective[^1].

## 5. Self-Supervised Pre-training Experiments

### Approach and Results

The research team initially explored self-supervised learning approaches, which have been successful in natural language processing:

- Multiple self-supervised methods were evaluated on the VTAB benchmark
- Results showed that these methods underperformed compared to supervised learning approaches[^1]
- The team found limitations in how well self-supervised representations transferred to diverse tasks


### Limitations

Self-supervised pre-training in vision at the time of the lecture wasn't yet delivering the same breakthroughs as in language models. The representations learned didn't generalize as well across different visual tasks, suggesting the need for alternative approaches.

## 6. Semi-Supervised Learning Approaches

### Method and Improvements

Moving beyond purely self-supervised methods, the team experimented with semi-supervised learning:

- Used a combination of labeled examples (small dataset) and unlabeled examples (large dataset)
- Found significant improvements over self-supervised approaches[^1]
- The addition of even a small amount of labeled data substantially improved representation quality


### Key Insight

This finding suggested that some form of supervision was valuable for learning general visual representations, even if limited. This led to exploring more fully supervised approaches at scale.

## 7. Fully Supervised Pre-training at Scale

### The Data Advantage

In a significant breakthrough, the team discovered that scaling up fully supervised pre-training led to substantially better representations than previous approaches[^1]. This involved:

- Utilizing internet images with weak supervision signals derived from surrounding text/context
- Leveraging internal Google datasets with ~300 million weakly labeled images
- Creating models trained on vastly larger datasets than conventional benchmark datasets


### Implementation Details

The approach required:

1. Collecting and weakly labeling large image datasets
2. Designing efficient training pipelines for massive datasets
3. Developing evaluation methods to measure generalization

This focus on scale marked a departure from the trend toward self-supervised methods and showed the untapped potential of supervised learning when properly scaled.

## 8. The Importance of Patience in Training

### Training Dynamics Observations

Lucas emphasizes that patience is a critical ingredient when training large models on large datasets[^1]. He illustrates this with a training curve showing:

- What appeared to be plateaus that might discourage researchers from continuing training
- Unexpected improvements after extended training periods
- The potential for significant gains after GPU months of training


### Practical Insights

The lecture provides practical advice for training large models:

- Don't give up too early when training appears to stagnate
- Be prepared for long training times (measured in GPU weeks or months)
- Sometimes real breakthroughs happen after what seems like lack of progress[^1]

This insight contradicts common practices where training is often stopped at apparent plateaus, potentially missing significant performance improvements.

## 9. Scaling Laws for Visual Representation Learning

### The Scaling Hypothesis

One of the most important findings presented is that scaling everything simultaneously is crucial for optimal performance[^1]:

- **Data scaling alone** is insufficient - larger datasets with standard models show diminishing returns
- **Model scaling alone** is inefficient - larger models without more data don't improve significantly
- **Joint scaling** of both model size and dataset size yields dramatic improvements


### Evidence from ResNet Experiments

Lucas presents results from experiments with ResNet architectures of various sizes:

- Standard ResNet-50 models show limited improvement when trained on larger datasets
- Larger ResNet models (referred to as "gigantic ResNets") show increasing benefits with larger datasets
- The largest models trained on the largest datasets (300M images) demonstrate remarkable performance on few-shot learning tasks[^1]

The key insight is that model capacity must increase proportionally with dataset size to effectively capture and utilize the additional information.

## 10. Robustness Benefits of Scale

### Object Recognition Robustness

An unexpected benefit of large-scale training was significantly improved robustness to distribution shifts:

- Models were evaluated on ObjectNet, a dataset specifically designed to test robustness
- ObjectNet contains objects in unusual contexts, orientations, and backgrounds (e.g., a chair in a bathtub)[^1]
- Conventional models struggle with these distribution shifts


### Results

The lecture shows that:

- Standard models exhibited poor robustness on ObjectNet
- Models trained at scale (both data and architecture) demonstrated dramatically improved robustness[^1]
- This robustness emerged naturally from scale, without specific robustness-focused training procedures

This finding suggests that scale itself may be a pathway to solving many longstanding challenges in computer vision reliability.

## 11. From ResNets to Vision Transformers

### Limitations of Convolutional Architectures

While the scaling experiments with ResNets showed promising results, convolutional neural networks (CNNs) have inherent limitations:

- Fixed receptive fields limiting global context understanding
- Spatial inductive biases that may be overly restrictive
- Challenges in scaling efficiency
- Limited cross-spatial position interactions


### Transformers as an Alternative

The success of transformers in NLP prompted researchers to adapt them for vision:

- Transformers offer global attention mechanisms for better context modeling
- They provide more flexible receptive fields
- Their self-attention mechanisms can capture complex relationships between image regions
- Transformers have shown excellent scaling properties in language tasks


## 12. Vision Transformer (ViT) Architecture

### ViT Design Principles

The Vision Transformer architecture applies the transformer concept to images by:

1. Splitting images into fixed-size patches (e.g., 16×16 pixels)
2. Linearly embedding these patches into tokens (similar to word embeddings in NLP)
3. Adding position embeddings to retain spatial information
4. Processing the sequence of patch embeddings with standard transformer encoder blocks
5. Using a [CLS] token for classification or other task-specific outputs

### Mathematical Formulation

The patch embedding process can be represented as:

\$ z_0 = [x_{class}; x_p^1 E; x_p^2 E; ...; x_p^N E] + E_{pos} \$

Where:

- $x_p^i$ represents the $i$-th image patch
- $E$ is the patch embedding projection
- $E_{pos}$ is the position embedding
- $x_{class}$ is the class token

The transformer encoder then processes these embeddings with standard transformer layers:

\$ z_l' = MSA(LN(z_{l-1})) + z_{l-1} \$
\$ z_l = MLP(LN(z_l')) + z_l' \$

Where:

- MSA is Multi-head Self-Attention
- LN is Layer Normalization
- MLP is Multi-Layer Perceptron


## 13. Pre-training Strategies for Vision Transformers

### Supervised Pre-training

Building on the scaling insights from ResNet experiments, ViT models are typically pre-trained on large labeled datasets:

- ImageNet-21k (14 million images)
- JFT-300M (300 million weakly labeled images)
- Public and private datasets of increasing scale

The supervised pre-training approach follows similar principles as with ResNets but leverages the transformer architecture's unique capabilities.

### Self-supervised Alternatives

Recent advances have also enabled effective self-supervised pre-training for Vision Transformers:

- MAE (Masked Autoencoders)
- DINO (Self-Distillation with No Labels)
- MoCo-v3 (Momentum Contrast)

These methods have started to close the gap with supervised pre-training, especially at larger scales.

## 14. ViT Transfer Learning Performance

### Few-shot Learning Results

ViT models demonstrate exceptional few-shot learning capabilities:

- Superior performance compared to ResNets of comparable size
- Stronger scaling properties with both model and data size
- Better sample efficiency when adapting to new tasks


### Full Fine-tuning Performance

When fine-tuned on downstream tasks with more data:

- ViT models outperform conventional architectures
- They show better transfer to diverse vision tasks
- They maintain advantages across domains (natural images, medical, satellite, etc.)


## 15. Hybrid Approaches and Architectural Variants

### Hybrid CNN-Transformer Models

Various approaches combine convolutional operations with transformer components:

- Using CNN stem before transformer blocks
- Incorporating convolutional projection layers
- Hierarchical designs with multiple resolution stages


### Efficiency-focused Variants

Several architectural modifications aim to improve efficiency:

- MobileViT: Combining mobile CNN designs with lightweight transformer components
- Swin Transformer: Using shifted windows for more efficient attention computation
- DeiT: Data-efficient training techniques for smaller datasets


## Conclusion

The lecture by Lucas Beyer traces the evolution of visual representation learning from conventional CNNs to Vision Transformers, emphasizing several key insights:

1. General visual representations are crucial for advancing computer vision applications
2. Scale matters tremendously - both dataset size and model capacity need to increase together
3. Patience in training large models can reveal unexpected performance improvements
4. Transformers offer a powerful alternative to CNNs for visual tasks
5. Pre-training strategies significantly impact downstream performance
6. Robustness emerges naturally from scale without explicit optimization

These insights have transformed computer vision research and applications, enabling more powerful, adaptable, and robust visual understanding systems. The Vision Transformer represents a fundamental shift in computer vision architecture design, bringing the field closer to the flexibility and generalizability of human visual perception.

<div style="text-align: center">⁂</div>

[^1]: https://www.youtube.com/watch?v=BP5CM0YxbP8\&list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM\&index=3

[^2]: https://coconote.app/notes/5b8ca492-13d9-4695-9545-938f8290413b

[^3]: https://blog.paperspace.com/vision-transformers/

[^4]: https://www.linkedin.com/posts/steven-feng_stanford-cs25-v4-i-overview-of-transformers-activity-7188983011726503936-ZRog

[^5]: https://glasp.co/youtube/p/stanford-cs25-v1-i-transformers-in-vision-tackling-problems-in-computer-vision

[^6]: https://bulletin.stanford.edu/courses/2233491

[^7]: https://ai-scholar.tech/en/articles/transformer/transformer-vision-1

[^8]: https://www.youtube.com/watch?v=BP5CM0YxbP8

[^9]: https://www.classcentral.com/course/youtube-cs25-i-stanford-seminar-2022-transformers-in-vision-tackling-problems-in-computer-vision-191607

[^10]: https://www.linkedin.com/posts/jpreagan_todays-lecture-for-the-cs25-transformers-activity-7186900630869815296-ahgZ

[^11]: https://web.stanford.edu/class/cs25/

[^12]: https://www.youtube.com/watch?v=JKbtWimlzAE

[^13]: https://glasp.co/youtube/p/stanford-cs25-v2-i-introduction-to-transformers-w-andrej-karpathy

[^14]: https://www.youtube.com/watch?v=XfpMkf4rD6E\&lc=UgxmfjJFHuf1tjyegdh4AaABAg

[^15]: https://cs.nyu.edu/~fergus/teaching/vision/5_transformers.pdf

[^16]: https://www.classcentral.com/course/youtube-stanford-cs25-v4-i-overview-of-transformers-289569

[^17]: https://www.youtube.com/watch?v=j3VNqtJUoz0

[^18]: https://upaspro.com/stanford-cs25-transformers-united/

[^19]: https://machinelearningmastery.com/the-vision-transformer-model/

[^20]: https://www.youtube.com/watch?v=P127jhj-8-Y

[^21]: https://web.stanford.edu/class/cs25/prev_years/2021_fall/

