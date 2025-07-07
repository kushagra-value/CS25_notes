<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Stanford CS25: Decision Transformers - Comprehensive Lecture Notes

**Lecture by Aditya Grover**
*Transformers for Reinforcement Learning via Sequence Modeling*

---

## 1. **Motivation: Why Transformers for RL?**

### Key Problems in Traditional RL:

1. **Scalability Issues**:
    - RL policies typically use small networks (millions of parameters, ~20 layers) vs. transformers (billions of parameters, ~100 layers).
    - Traditional RL struggles with **training instability** (high variance in reward curves).
    - Example: A robot arm policy might fail catastrophically after minor parameter changes.
2. **Online vs. Offline RL**:
    - **Online RL**: Agent interacts with environment (trial \& error).
    - **Offline RL**: Agent learns from **fixed dataset** of trajectories (no exploration).
    - *Why start with Offline RL?* Avoids exploration challenges, provides controlled setup for testing transformers.

---

## 2. **Decision Transformer Architecture**

### Core Idea:

Reformulate RL as a **sequence modeling problem**:

- Input: Trajectory tokens (states, actions, returns-to-go)
- Output: Optimal actions (autoregressively predicted)


### Mathematical Formulation:

1. **Returns-to-Go (RTG)**: Cumulative reward from current timestep:
\$ RTG_t = \sum_{t'=t}^T r_{t'} \$
    - Example: If total reward = 100, after receiving reward 20 at t=1, RTG at t=2 = 80.
2. **Tokenization**:
Each timestep has 3 tokens:
    - State: \$ s_t \$ (e.g., robot joint angles)
    - Action: \$ a_t \$ (e.g., motor commands)
    - RTG: \$ RTG_t \$
3. **Embedding Layer**:
    - Linear projection + positional encoding:
\$ z_t = Embed(s_t) + Embed(a_t) + Embed(RTG_t) + PositionalEncoding(t) \$
4. **Causal Transformer**:
    - Uses **masked self-attention** to prevent future token visibility.
    - Predicts action \$ \hat{a}_t \$ using context window of **K previous tokens**.

---

## 3. **Key Design Choices**

### A. Returns-to-Go Conditioning

- **Purpose**: Enables **goal-directed behavior** by conditioning on desired cumulative reward.
- Example: To make a robot walk 10 meters, set initial RTG = 10. As it moves, RTG decreases.


### B. Non-Markovian Processing

- Traditional RL assumes Markov property (current state suffices).
- Decision Transformer uses **full history** (K previous tokens) to handle:
    - Partial observability (e.g., noisy sensors)
    - Long-term dependencies (e.g., chess strategies)


### C. Training Objective

- Minimize action prediction error:
\$ \mathcal{L} = \sum_{t=1}^T ||a_t - \hat{a}_t||^2 \$
- No value functions or policy gradients!

---

## 4. **Implementation Details**

### Token Embedding Example:

```python  
# Input: State (s), Action (a), RTG (g)  
s_embed = nn.Linear(state_dim, d_model)(s)  
a_embed = nn.Linear(action_dim, d_model)(a)  
g_embed = nn.Linear(1, d_model)(g)  
z = s_embed + a_embed + g_embed + positional_encoding  
```


### Context Window (K):

- Hyperparameter controlling how much history is used.
- Trade-off: Larger K → better performance but higher compute.

---

## 5. **Experiments \& Results**

### Benchmarks:

1. **Atari Games** (e.g., Breakout)
2. **OpenAI Gym** (e.g., HalfCheetah)
3. **Key-to-Door Task** (sparse rewards)

### Key Findings:

1. Matches/exceeds **model-free offline RL** (CQL, BCQ):
    - 80% higher scores on sparse-reward tasks.
2. Scalable with dataset size:
    - Doubling dataset → 15% performance gain.
3. Stable training curves (vs. erratic traditional RL curves).

---

## 6. **Q\&A Highlights**

### Q: Why add embeddings instead of concatenating?

**A**:

- Addition preserves dimensionality, acts as "feature modulation".
- Concatenation increases parameters → overfitting risk.
- *Empirically*, addition worked better.


### Q: How handle infinite-horizon tasks?

**A**:

- Use discount factor \$ \gamma \$ in RTG:
\$ RTG_t = \sum_{t'=t}^\infty \gamma^{t'-t} r_{t'} \$
- Not tested in initial work but theoretically feasible.


### Q: What if dataset has suboptimal trajectories?

**A**:

- Decision Transformer learns **conditioned behavior**.
- Example: If dataset has both 50/100 RTG trajectories, it can interpolate (e.g., achieve 75 RTG).

---

## 7. **Limitations \& Future Work**

1. **Online RL Extension**:
    - Combine with exploration strategies (e.g., UCB).
2. **Multi-Task Learning**:
    - Condition on task descriptions (text prompts).
3. **Efficiency Improvements**:
    - Sparse attention for long trajectories.

---

## 8. **Simplified Example: Robot Arm Control**

**Task**: Pick up object at position (5,5).

- **State**: (x, y) coordinates of arm tip.
- **Action**: (Δx, Δy) movement.
- **RTG**: Distance remaining to target.

**Trajectory**:


| Timestep | State (x,y) | Action (Δx,Δy) | RTG |
| :-- | :-- | :-- | :-- |
| 1 | (0,0) | (1,1) | 10 |
| 2 | (1,1) | (1,0) | 8 |
| 3 | (2,1) | (2,3) | 5 |

The transformer learns to predict actions that minimize RTG.

---

## 9. **Comparison to Model-Free RL**

| Aspect | Decision Transformer | Model-Free RL (e.g., DQN) |
| :-- | :-- | :-- |
| Training Stability | Stable (no TD error) | Unstable (high variance) |
| Data Efficiency | Better (uses history) | Worse |
| Partial Observability | Handled naturally | Requires RNNs/POMDPs |

---

## 10. **Critical Takeaways**

1. Transformers unify **perception + decision-making**.
2. Sequence modeling bypasses RL’s instability issues.
3. Offline RL is just the beginning – future work will bridge to online settings.

---

Let me know if you need deeper dives into specific equations or experimental setups!

<div style="text-align: center">⁂</div>

[^1]: https://www.youtube.com/watch?v=w4Bw8WYL8Ps\&list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM\&index=4

