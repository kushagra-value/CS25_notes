# Detailed Notes on Stanford CS25: Decision Transformer: Reinforcement Learning via Sequence Modeling

These notes provide a comprehensive overview of the Stanford CS25 lecture titled "Decision Transformer: Reinforcement Learning via Sequence Modeling," presented by Aditya Grover on August 11, 2022, as part of the Transformers United V1 series. Based on the research paper "Decision Transformer: Reinforcement Learning via Sequence Modeling" and related resources, the notes cover all likely topics discussed, including reinforcement learning (RL) fundamentals, sequence modeling, the Decision Transformer’s architecture, training, evaluation, experimental results, and its advantages and limitations. Each section includes detailed explanations, equations, simplified examples, and descriptions of diagrams to ensure understanding from beginner to advanced levels.

## 1. Introduction to Reinforcement Learning

### Overview
Reinforcement Learning (RL) involves an agent learning to make decisions by interacting with an environment to maximize cumulative rewards. The lecture likely began by introducing RL concepts to set the stage for the Decision Transformer.

### Key RL Concepts
- **Agent**: The entity making decisions (e.g., a robot).
- **Environment**: The world the agent interacts with (e.g., a maze).
- **State (\(s_t\))**: A snapshot of the environment at time \(t\).
- **Action (\(a_t\))**: A decision made by the agent affecting the state.
- **Reward (\(r_t\))**: Feedback from the environment, positive or negative.
- **Policy (\(\pi\))**: A strategy mapping states to actions, \(\pi(a|s)\).
- **Value Function**: Estimates expected future rewards, e.g., \(V(s) = \mathbb{E}[\sum_{t} \gamma^t r_t]\), where \(\gamma\) is a discount factor.
- **Return**: Cumulative reward, often discounted, \(\sum_{t} \gamma^t r_t\).

### Challenges in RL
- **Credit Assignment**: Identifying which actions led to delayed rewards.
- **Exploration vs. Exploitation**: Balancing trying new actions versus using known rewarding ones.
- **Offline RL**: Learning from a fixed dataset without further environment interaction, limiting exploration.

### Simplified Example
Imagine a child learning to stack blocks. The child (agent) tries different placements (actions) in a room (environment). Successfully stacking a block earns praise (reward). Over time, the child learns a policy to stack blocks efficiently, but figuring out which moves led to success is challenging, especially if rewards are delayed.

## 2. Sequence Modeling in Reinforcement Learning

### Concept
Traditional RL methods optimize policies using value functions or policy gradients. The Decision Transformer reframes RL as a sequence modeling problem, where trajectories (sequences of states, actions, and rewards) are modeled similarly to sentences in language processing.

### Why Sequence Modeling?
- **Direct Modeling**: Learns the joint distribution of trajectories, capturing complex patterns.
- **Long-term Dependencies**: Transformers excel at modeling relationships across long sequences, aiding credit assignment.
- **Scalability**: Leverages advances in Transformer architectures, like GPT, for large-scale data.

### Decision Transformer Approach
The Decision Transformer uses a Transformer to generate actions conditioned on desired returns, past states, and actions, treating RL as conditional sequence modeling. This avoids explicit value function estimation or policy optimization.

### Simplified Example
Think of RL as writing a story. Each chapter (timestep) includes the setting (state), the character’s choice (action), and the outcome’s value (return-to-go). A Transformer, like a skilled writer, predicts the next choice to achieve a happy ending (desired return).

## 3. Decision Transformer Architecture

### Trajectory Representation
Trajectories are sequences of tuples: \((\hat{R}_t, s_t, a_t)\), where:
- **Return-to-Go (\(\hat{R}_t\))**: Sum of future rewards from time \(t\), \(\hat{R}_t = \sum_{t'=t}^T r_{t'}\).
- **State (\(s_t\))**: Environment state at time \(t\).
- **Action (\(a_t\))**: Action taken at time \(t\).

### Architecture Details
- **Input Sequence**: Processes the last \(K\) timesteps, yielding \(3K\) tokens (return-to-go, state, action per timestep).
- **Embeddings**:
  - Each token is embedded using modality-specific linear layers: \(e_{\hat{R}_t} = W_{\hat{R}} \cdot \hat{R}_t\), similarly for \(s_t\) and \(a_t\).
  - A learned timestep embedding is added to indicate sequence position.
- **Transformer**: A GPT-like model with causal self-attention, ensuring predictions depend only on prior tokens.
- **Output**: Predicts the next action \(a_{t+1}\) given the sequence up to \(a_t\).

### Key Equations
- **Trajectory Sequence**:
  \[
  \tau = (\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \ldots, \hat{R}_T, s_T, a_T)
  \]
- **Embedding**:
  \[
  e_t = W_{\text{modality}} \cdot x_t + e_{\text{timestep}}
  \]
- **Action Prediction**:
  \[
  a_{t+1} = \text{Transformer}(e_1, e_2, \ldots, e_{3t})
  \]

### Diagram Description
Figure 1 in the paper likely depicts the architecture: a sequence of return-to-go, state, and action tokens fed into embedding layers, processed by a Transformer with causal attention, and outputting action predictions. Arrows show the autoregressive flow, with a mask ensuring future tokens are not attended to.

### Simplified Example
Imagine a robot navigating a maze. The trajectory is a sequence like: (remaining points needed, current position, move left), (updated points, new position, move up). The Transformer processes these to predict the next move, like a GPS suggesting turns to reach a destination.

## 4. Training the Decision Transformer

### Dataset
- Uses an offline dataset of trajectories, each containing states, actions, and rewards.
- No interaction with the environment during training, typical of offline RL.

### Training Objective
- **Autoregressive Prediction**: Predict action \(a_t\) given the sequence up to \(a_{t-1}\).
- **Loss Function**:
  - Discrete actions: Cross-entropy loss.
  - Continuous actions: Mean-squared error.
  \[
  \mathcal{L} = \begin{cases} 
  -\sum \log P(a_t | \text{sequence}) & \text{(discrete)} \\
  \frac{1}{N} \sum (a_t - \hat{a}_t)^2 & \text{(continuous)}
  \end{cases}
  \]
- Loss is averaged across timesteps in a minibatch.

### Key Points
- No value function or policy gradient computation.
- Training is supervised, leveraging Transformer’s sequence modeling capabilities.
- Predicting states or returns-to-go was tested but found unnecessary.

### Simplified Example
Training is like teaching a student to play chess by showing past games. The student sees board positions (states), moves (actions), and game scores (returns-to-go), learning to predict the next move. The Decision Transformer learns similarly, using a dataset of “games” to predict actions.

## 5. Evaluation and Inference

### Inference Process
1. Initialize with desired return \(\hat{R}_0\) and initial state \(s_0\).
2. Predict action \(a_0\) using the Transformer.
3. Execute \(a_0\), observe reward \(r_0\) and state \(s_1\).
4. Update return-to-go: \(\hat{R}_1 = \hat{R}_0 - r_0\).
5. Repeat until the episode ends.

### Key Equations
- **Return-to-Go Update**:
  \[
  \hat{R}_{t+1} = \hat{R}_t - r_t
  \]
- **Action Generation**:
  \[
  a_t = \text{Transformer}(\hat{R}_t, s_t, a_{t-1}, \ldots)
  \]

### Simplified Example
In a video game, set a high score goal (desired return). The Decision Transformer suggests moves (actions) based on the current game state and past moves, adjusting the goal as points are earned, like a coach guiding a player to a target score.

### Diagram Description
A figure likely illustrates inference: a loop showing the model taking desired return and state, outputting an action, receiving environment feedback, and updating the return-to-go, continuing autoregressively.

## 6. Experimental Results

### Benchmarks
The Decision Transformer was evaluated on three offline RL tasks:
- **Atari**: Games like Breakout, Qbert, Pong, Seaquest using DQN-replay dataset.
- **OpenAI Gym**: Continuous control tasks (HalfCheetah, Hopper, Walker, Reacher) from D4RL datasets (Medium, Medium-Replay, Medium-Expert).
- **Key-to-Door**: A grid-based environment with sparse rewards, testing long-term credit assignment.

### Experimental Setup
- **Architecture**: GPT-based with causal self-attention, context lengths \(K=30\) (Atari, except \(K=50\) for Pong), \(K=20\) (Gym locomotion), \(K=5\) (Reacher).
- **Training**: Supervised loss on offline datasets, no state/return prediction.
- **Metrics**: Normalized scores (0 for random, 100 for expert) for Atari and Gym; success rates for Key-to-Door.

### Results
#### Atari (Normalized Scores)
| Game      | Decision Transformer | CQL    | QR-DQN | REM    | Behavior Cloning |
|-----------|----------------------|--------|--------|--------|------------------|
| Breakout  | 267.5 ± 97.5         | 211.1  | 17.1   | 8.9    | 138.9 ± 61.7     |
| Qbert     | 15.4 ± 11.4          | 104.2  | 0.0    | 0.0    | 17.3 ± 14.7      |
| Pong      | 106.1 ± 8.1          | 111.9  | 18.0   | 0.5    | 85.2 ± 20.0      |
| Seaquest  | 2.5 ± 0.4            | 1.7    | 0.4    | 0.7    | 2.1 ± 0.3        |

- Competitive with CQL in 3/4 games, outperforms others in most cases.

#### OpenAI Gym (Normalized Scores)
| Dataset         | Environment  | Decision Transformer | CQL    | BEAR   | BRAC-v | AWR    | BC   |
|-----------------|--------------|---------------------|--------|--------|--------|--------|------|
| Medium-Expert   | HalfCheetah  | 86.8 ± 1.3          | 62.4   | 53.4   | 41.9   | 52.7   | 59.9 |
| Medium-Expert   | Hopper       | 107.6 ± 1.8         | 111.0  | 96.3   | 0.8    | 27.1   | 79.6 |
| Medium-Expert   | Walker       | 108.1 ± 0.2         | 98.7   | 40.1   | 81.6   | 53.8   | 36.6 |
| Medium-Expert   | Reacher      | 89.1 ± 1.3          | 30.6   | -      | -      | -      | 73.3 |
| Medium          | HalfCheetah  | 42.6 ± 0.1          | 44.4   | 41.7   | 46.3   | 37.4   | 43.1 |
| Medium          | Hopper       | 67.6 ± 1.0          | 58.0   | 52.1   | 31.1   | 35.9   | 63.9 |
| Medium          | Walker       | 74.0 ± 1.4          | 79.2   | 59.1   | 81.1   | 17.4   | 77.3 |
| Medium-Replay   | Hopper       | 82.7 ± 7.0          | 48.6   | 33.7   | 0.6    | 28.4   | 27.6 |

- Highest scores in most tasks, especially Medium-Expert datasets.

#### Key-to-Door (Success Rates)
| Dataset             | Decision Transformer | CQL    | BC    | %BC   | Random |
|---------------------|---------------------|--------|-------|-------|--------|
| 1K Random Traj.     | 71.8%               | 13.1%  | 1.4%  | 69.9% | 3.1%   |
| 10K Random Traj.    | 94.6%               | 13.3%  | 1.6%  | 95.1% | 3.1%   |

- Excels in sparse reward settings, outperforming CQL and BC.

### Key Findings
- **Extrapolation**: Can achieve higher returns than in training data (e.g., Seaquest).
- **Context Length**: Longer contexts improve performance (e.g., Breakout: 267.5 at \(K=30\) vs. 73.9 at \(K=1\)).
- **Sparse Rewards**: Robust in delayed reward scenarios, unlike some baselines.

### Diagram Description
Figure 4 likely plots desired vs. observed returns, showing high correlation and extrapolation in Seaquest. Table 5 compares performance across context lengths.

## 7. Advantages and Limitations

### Advantages
- **Simplicity**: Eliminates complex RL components, using supervised sequence modeling.
- **Scalability**: Benefits from large datasets and Transformer advancements.
- **Flexibility**: Applicable to diverse tasks without task-specific tuning.
- **Extrapolation**: Can generate policies achieving higher returns than training data.

### Limitations
- **Data Dependency**: Requires large, diverse offline datasets.
- **Limited Generalization**: May struggle if optimal policies are absent from data.
- **Computational Cost**: Transformers are resource-intensive.

### Simplified Example
The Decision Transformer is like a GPS trained on past trips. It suggests routes (actions) to reach a destination (desired return) based on current location (state) and travel history. If trained on suboptimal routes, it may still find better paths but struggles without diverse trip data.

## 8. Applications and Future Directions

### Applications
- **Robotics**: Planning actions for navigation or manipulation.
- **Gaming**: Generating strategies for complex games.
- **Autonomous Systems**: Decision-making in vehicles or drones.

### Future Directions
- **Online Fine-Tuning**: Combining with online RL for real-time learning.
- **Multi-Agent RL**: Extending to cooperative or competitive settings.
- **Generalization**: Improving performance on unseen tasks or environments.

### Simplified Example
In robotics, the Decision Transformer could guide a warehouse robot to move packages efficiently, learning from past human-operated trajectories to optimize paths, with future work enabling real-time adjustments.

## Additional Notes
- **Lecture Context**: As lecture 4 in CS25 V1, it likely included an introduction by Aditya Grover, possibly discussing his research background and the paper’s motivation.
- **Visual Aids**: Slides likely featured diagrams of the architecture, result tables, and example trajectories.
- **Q&A**: The lecture may have ended with audience questions, possibly addressing practical implementation or comparisons with other RL methods.