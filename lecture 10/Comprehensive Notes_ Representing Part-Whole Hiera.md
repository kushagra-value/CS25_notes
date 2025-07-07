<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Comprehensive Notes: Representing Part-Whole Hierarchies in Neural Networks (GLOM)

## Introduction to the Problem

Neural networks are excellent at pattern recognition, but they struggle with a fundamental aspect of human perception: understanding objects as hierarchies of parts and wholes. When we see a face, we automatically recognize it's made of eyes, nose, and mouth, which are themselves made of smaller parts. This hierarchical understanding allows us to recognize objects from different angles or when partially obscured.

Traditional neural networks lack this ability because:

1. They use fixed architectures that can't adapt to different object structures
2. They don't naturally represent part-whole relationships
3. They struggle with coordinate transformations (understanding how parts relate spatially)

Geoffrey Hinton's GLOM architecture attempts to solve this problem by creating a neural network that can dynamically parse images into part-whole hierarchies without changing its underlying structure.

## The GLOM Architecture: Core Concepts

### Columns and Levels

GLOM divides an image into a grid of locations, with each location processed by a "column":

- Each column processes a small patch of the image
- Each column contains multiple levels (typically 5) representing different scales of abstraction
- Lower levels represent small parts (edges, textures)
- Middle levels represent larger parts (nose, eye)
- Higher levels represent whole objects (face, person)

For example, when looking at a face image:

- Level 1: Edge detection, basic shapes
- Level 2: Nostrils, eye corners
- Level 3: Nose, eyes, mouth
- Level 4: Complete face
- Level 5: Person in context


### Embedding Vectors

At each level in each column, GLOM represents information using high-dimensional vectors:

- These vectors encode what the network "thinks" is at that location and level
- Similar vectors = similar parts/objects
- The key insight: locations containing the same part should have similar vectors at that level

For example, all the vectors representing "nose" across different image locations should be similar, forming an "island of agreement."

### Islands of Agreement

The central concept in GLOM is that parts of the same object will form "islands" of similar embedding vectors:

- All locations containing a nose will have similar "nose vectors" at the appropriate level
- All locations containing the face will have similar "face vectors" at a higher level
- These islands effectively create a parse tree for the image without changing the network architecture

This is achieved through two types of interactions:

1. Within-column interactions between levels
2. Between-column interactions at the same level

## How GLOM Works: The Processing Pipeline

### 1. Bottom-Up and Top-Down Processing

Each column processes information bidirectionally:

- **Bottom-up processing**: Lower levels send information to higher levels through encoders
- **Top-down processing**: Higher levels send predictions to lower levels through decoders
- Each pair of adjacent levels has its own autoencoder (shared across all columns)

This creates a loop where:

- Lower levels suggest what parts might be present
- Higher levels provide context about the whole object
- The system iteratively refines its predictions


### 2. Cross-Column Attention

Columns communicate with each other at the same level through a simplified attention mechanism:

- Each embedding vector attends to similar vectors in nearby columns
- This causes similar vectors to become even more similar
- No learned parameters are needed for this step - it's a simple weighted averaging
- The attention weights are determined by vector similarity

This process creates the "islands of agreement" - regions where columns agree on what they're seeing at a particular level.

### 3. Vector Consensus Formation

The embedding vector at level L in a column is influenced by four sources:

1. Bottom-up prediction from level L-1
2. Top-down prediction from level L+1
3. Attention-weighted average of level L vectors in nearby columns
4. The previous state of the level L vector (recurrent connection)

These inputs are combined to update the embedding vector, causing the system to converge toward a consistent interpretation of the image.

## Coordinate Transformations

A crucial aspect of GLOM is handling coordinate transformations between parts and wholes:

- Parts exist in their own coordinate systems (e.g., a nose has its own up/down/left/right)
- Wholes have their own coordinate systems
- The relationship between these coordinate systems must be learned

For example, if a face is tilted, the nose is still "below the eyes" in face coordinates, even though the absolute positions have changed. GLOM learns these transformations through the autoencoders that connect different levels.

## Training GLOM

While Hinton's paper is primarily conceptual, it suggests training GLOM using:

1. **Contrastive Learning**: Making embedding vectors for the same entity similar and different entities dissimilar
2. **Denoising Autoencoders**: Training the system to reconstruct clean images from noisy ones
3. **Self-Supervised Learning**: Using the structure of the data itself to provide supervision

The training process would involve:

- Presenting images to the network
- Allowing the network to converge on its interpretation
- Adjusting weights to improve the consistency and accuracy of the parse trees


## Psychological Evidence Supporting GLOM

Hinton draws on several psychological phenomena that suggest humans use similar mechanisms:

1. **Mental Rotation**: When we mentally rotate objects, we use object-centered coordinate systems
2. **Viewpoint Invariance**: We recognize objects from different angles by understanding their part-whole structure
3. **Perceptual Grouping**: We naturally group parts that belong together (Gestalt principles)
4. **Symmetry Detection**: We easily detect and use symmetry to understand objects

## Detailed Example: Processing a Face Image

Let's walk through how GLOM would process a face image:

1. **Initial Processing**:
    - The image is divided into a grid of patches
    - Each patch is processed by a column
    - Initial bottom-up processing extracts basic features
2. **Early Iterations**:
    - Level 1: Detects edges, textures, and basic shapes
    - Level 2: Begins to identify potential parts (eye corner, nostril)
    - Level 3: Makes tentative guesses about larger parts (eye, nose)
    - Columns begin sharing information through attention
3. **Middle Iterations**:
    - Similar vectors at each level begin to cluster
    - Islands of agreement form for parts like "left eye," "nose," "mouth"
    - Top-down processing helps refine lower-level interpretations
4. **Final Convergence**:
    - Level 4 converges on a consistent "face" interpretation across all face locations
    - Level 3 has distinct islands for each facial feature
    - Level 2 has smaller islands for parts of features
    - The system has effectively parsed the image into a hierarchical structure

## Advantages of GLOM Over Traditional Neural Networks

1. **Dynamic Parsing**: Creates different parse trees for different images without changing architecture
2. **Viewpoint Invariance**: Better handles rotations and transformations through coordinate systems
3. **Occlusion Handling**: Can infer missing parts based on the whole
4. **Interpretability**: The islands of agreement provide a clear interpretation of what the network sees
5. **Biological Plausibility**: More closely mimics how the brain might process visual information

## Implementation Details

While GLOM is still largely conceptual, a practical implementation would involve:

### Network Architecture

- Grid of columns (e.g., 50×50 for a 200×200 image)
- 5 levels per column
- High-dimensional embedding vectors (e.g., 512 dimensions)
- Shared autoencoders between levels
- Attention mechanism for cross-column communication


### Processing Steps

1. Initialize embedding vectors at all levels
2. Perform bottom-up and top-down passes
3. Update embeddings based on attention between columns
4. Repeat until convergence (typically 10-20 iterations)

### Code Example: Simplified GLOM Update Step

```python
import numpy as np

# Simplified GLOM update for one level
def update_embedding(level_vectors, bottom_up, top_down, prev_state):
    # Calculate attention between columns
    similarities = np.dot(level_vectors, level_vectors.T)
    attention_weights = np.exp(similarities) / np.sum(np.exp(similarities), axis=1, keepdims=True)
    
    # Get attention-weighted average
    attention_output = np.dot(attention_weights, level_vectors)
    
    # Combine all inputs (with arbitrary weights for simplicity)
    new_state = (0.3 * bottom_up + 
                 0.3 * top_down + 
                 0.3 * attention_output + 
                 0.1 * prev_state)
    
    # Normalize to unit length
    new_state = new_state / np.linalg.norm(new_state, axis=1, keepdims=True)
    
    return new_state
```


## Challenges and Limitations

1. **Computational Complexity**: Running many iterations across many columns is computationally expensive
2. **Training Difficulty**: The recurrent nature and multiple interacting components make training challenging
3. **Parameter Tuning**: Finding the right balance between bottom-up, top-down, and lateral influences
4. **Evaluation**: Difficult to evaluate the quality of the learned representations

## Extensions and Future Directions

1. **Handling Video**: GLOM could be extended to process temporal sequences by maintaining consistency across frames
2. **Multimodal Integration**: Combining vision with language or other modalities
3. **Unsupervised Learning**: Learning part-whole hierarchies without explicit supervision
4. **Hardware Optimization**: Specialized hardware to efficiently implement GLOM's parallel processing

## Comparison with Related Approaches

### GLOM vs. Capsule Networks

- Both address part-whole hierarchies
- Capsules use routing-by-agreement; GLOM uses attention and consensus
- GLOM is more flexible in creating dynamic parse trees


### GLOM vs. Transformers

- Both use attention mechanisms
- Transformers typically process tokens sequentially; GLOM processes the entire image iteratively
- GLOM has explicit levels representing different scales


### GLOM vs. Thousand Brains Theory (Numenta)

- Both propose columns that model objects at different locations
- Both suggest voting/consensus mechanisms
- Thousand Brains is more biologically motivated; GLOM is more engineered


## Practical Applications

If successfully implemented, GLOM could improve:

1. **Object Recognition**: Better handling of novel viewpoints and occlusions
2. **Scene Understanding**: Parsing complex scenes into meaningful hierarchies
3. **Robotics**: Improved object manipulation through better part understanding
4. **Medical Imaging**: Recognizing anatomical structures as hierarchies of parts
5. **Augmented Reality**: More robust object tracking and annotation

## Conclusion

GLOM represents a significant conceptual advance in neural network design, addressing the fundamental challenge of representing part-whole hierarchies. By combining ideas from transformers, capsule networks, and neuroscience, it offers a promising direction for creating more human-like visual systems.

While still largely theoretical, GLOM's principles could influence the next generation of computer vision systems, potentially bridging the gap between the pattern-matching abilities of current neural networks and the rich hierarchical understanding that characterizes human vision.

<div style="text-align: center">⁂</div>

[^1]: https://syncedreview.com/2021/02/26/better-than-capsules-geoffrey-hintons-glom-idea-represents-part-whole-hierarchies-in-neural-networks/

[^2]: https://www.cs.utoronto.ca/~hinton/absps/glomfinal.pdf

[^3]: https://www.cs.toronto.edu/~hinton/absps/glomculp.pdf

[^4]: https://blog.softtek.com/en/the-future-of-neural-networks-is-hintons-new-glom-model

[^5]: https://www.numenta.com/blog/2021/04/26/comparing-hinton-glom-model-to-numenta-thousand-brains-theory/

[^6]: https://cacm.acm.org/news/glom-teaching-computers-to-see-the-ways-we-do/

[^7]: https://www.reddit.com/r/MachineLearning/comments/ltro4y/d_paper_explained_glom_how_to_represent_partwhole/

[^8]: https://www.classcentral.com/course/youtube-how-to-represent-part-whole-hierarchies-in-a-neural-network-geoff-hinton-s-paper-explained-126682

[^9]: https://www.semanticscholar.org/paper/96e66d407d04ec86ecca2ba40648f82347b69e48

[^10]: https://www.semanticscholar.org/paper/37b1c5b804d412de8b66964c0174a2fbda7454f0

[^11]: https://www.semanticscholar.org/paper/15f1c875c821ee26a791f7e2f81e72fc69313128

[^12]: https://www.semanticscholar.org/paper/92b99fef63d9ed89673440f720405d1f48913e2f

[^13]: https://www.semanticscholar.org/paper/be9ae0f7fd9bdbf0476a60ef36b01511357d3373

[^14]: https://arxiv.org/pdf/2102.12627.pdf

[^15]: http://arxiv.org/pdf/2011.14597v1.pdf

[^16]: https://arxiv.org/pdf/1310.6343.pdf

