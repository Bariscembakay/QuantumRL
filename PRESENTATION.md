---
marp: true
theme: gaia
_class: lead
paginate: true
backgroundColor: #ffffff
color: #1a1a1a
style: |
  section {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    padding: 40px;
  }
  h1 {
    color: #8B0000; /* Dark Red matching the project color system */
  }
  h2 {
    color: #8B0000;
    border-bottom: 2px solid #8B0000;
  }
  footer {
    font-size: 0.5em;
    color: #777;
  }
  code {
    background-color: #f4f4f4;
    color: #d14;
  }
---

# Quantum vs. Classical RL for Grid World Navigation

## A Comparative Analysis

**Course Project: ELEC/PHYS 450/550 Quantum Computing**

**Presenter**: Barış Cem Bakay
**Student ID**: 0082990
**Institution**: Koç University
**Instructor**: Assoc. Prof. Mehmet Cengiz Onbaşlı

---

## 1. Background & Motivation

- **Deep Reinforcement Learning (DRL)**: Mnih et al. (2015) demonstrated superhuman Atari control using Deep Q-Networks, but these models require **millions of parameters** and are computationally expensive to train.
- **The NISQ Opportunity**: Near-term Intermediate-Scale Quantum (NISQ) devices offer parameterized quantum circuits that can represent complex functions with **exponentially fewer parameters** by leveraging superposition and entanglement.
- **Variational Quantum Circuits (VQCs)**: Chen et al. (2020) and Lockwood & Si (2020) showed VQCs can serve as Q-value function approximators, achieving convergence on simple control tasks with orders-of-magnitude parameter reduction.

---

## 2. Project Objectives

- **The Core Research Question**: Can near-term Variational Quantum Circuits (VQCs) replace deep neural networks for reinforcement learning with high parameter efficiency?
- **Key Deliverables & Milestones**:
  - **Classical Baseline**: Deep Q-Network (DQN) implementation in PyTorch (4,612 weights).
  - **Quantum Policy**: Hybrid VQC Deep Q-learning using Qiskit and `TorchConnector` (36 weights).
  - **Comparative Benchmarking**: Training on 3x3 and 4x4 Grid Worlds to analyze parameter scaling.
  - **Physical Verification**: Performing exact **Quantum State Tomography** during agent navigation.

---

## 3. Markov Decision Process (MDP) Formulation

- **State Space ($\mathcal{S}$)**: Normalized agent coordinates $s = (x, y) \in [0, N-1]^2$.
- **Action Space ($\mathcal{A}$)**: 4 discrete actions representing movement direction:
  $$\mathcal{A} = \{\text{Up (0), Down (1), Left (2), Right (3)}\}$$
- **Reward Structure ($\mathcal{R}$)**:
  - Goal state reach: $+1.0$ (Terminates episode)
  - Obstacle collision: $-1.0$ (Terminates episode at state $(1,1)$)
  - Step penalty: $-0.04$ per action (Encourages shortest path discovery)
- **Optimization Goal**: Solve Bellman Optimality Equation:
  $$Q^*(s, a) = R(s, a) + \gamma \max_{a'} Q^*(s', a')$$

---

## 4. Environment Visualization

- Navigation tasks are trained and evaluated on two grid environments:
  - **(a) 3x3 Grid**: Start at $(0,0)$, Obstacle at $(1,1)$, Goal at $(2,2)$. Optimal path = 3 steps (Reward: $+0.88$).
  - **(b) 4x4 Grid**: Start at $(0,0)$, Obstacle at $(1,1)$, Goal at $(3,3)$. Optimal path = 6 steps (Reward: $+0.76$).

![center width:750px](results/grid_world_visualization.png)

---

## 5. Classical DQN Baseline

- **Network Structure**: Dense feedforward network in PyTorch.
- **Architecture**: Input (2, coordinates) $\to$ Hidden 1 (64, ReLU) $\to$ Hidden 2 (64, ReLU) $\to$ Output (4, Q-values).
- **Trainable Parameters**: **4,612 weights**.
- **Exploration vs. Exploitation Trade-off**:
  - **Exploration (Random Search)**: With probability $\epsilon$, the agent selects a random action $a \in \mathcal{A}$. This allows discovery of new paths, states, and the terminal goal, avoiding local minima (e.g., staying still to avoid the obstacle).
  - **Exploitation (Greedy Policy)**: With probability $1 - \epsilon$, the agent selects the action maximizing the Q-value prediction: $a = \arg\max_{a'} Q(s, a'; \theta)$. This leverages current model knowledge.
- **Epsilon Decay Schedule**: Starts at $\epsilon = 1.0$ (complete exploration) and decays exponentially after each episode: $\epsilon \leftarrow \max(\epsilon_{\text{min}}, \epsilon \times \epsilon_{\text{decay}})$ where $\epsilon_{\text{decay}} = 0.995$ and $\epsilon_{\text{min}} = 0.01$.
- **Stability Mechanisms**: Replay Buffer (capacity: 5,000, batch size: 64) to break sample correlation, and a separate Target Network $\theta^-$ updated every 10 steps to stabilize learning targets.

![center width:700px](results/classical_dqn_agent.png)

---

## 6. Quantum Policy Network (VQC)

- **Register**: 4-qubit register (one qubit corresponding to each discrete action).
- **State Encoding**: 2D coordinate normalization followed by **Angle Encoding**:
  $$\theta_x = \frac{\pi}{N-1} x, \quad \theta_y = \frac{\pi}{N-1} y \implies |\psi_{\text{enc}}(s)\rangle = R_y(\theta_x)^{\otimes 2} \otimes R_y(\theta_y)^{\otimes 2} |0000\rangle$$
- **Ansatz**: `RealAmplitudes` with linear CNOT entanglement ladders (3 repetitions).
- **Action Output**: Expectation values $\langle Z_i \rangle$ scaled by classical layer:
  $$Q(s, a_i) = \sum_{j=0}^{3} W_{ij} \langle Z_j \rangle + b_i \quad (W \in \mathbb{R}^{4 \times 4}, b \in \mathbb{R}^4)$$
- **Total Parameters**: **36 weights** (16 quantum angles + 20 classical weights) — a **99.2% reduction**.

---

## 7. Circuit Diagram & Pipeline

- Quantum circuit is simulated using exact statevector methods.
- Trained using PyTorch integration via Qiskit's `TorchConnector` and parameter-shift gradients.

![center width:900px](results/quantum_vqc_agent.png)

---

## 8. Quantum Policy Expressibility

- **Input Duplication ($q_0, q_2 \to x$ and $q_1, q_3 \to y$)**:
  Allows entangling CNOT gates to construct high-order cross-terms like $\sin(\theta_x/2)\sin(\theta_y/2)$. This acts as a non-linear feature mapping to locate boundaries and obstacles.
- **Ansatz Depth ($R = 3$ repetitions)**:
  Balances model expressibility against the **Barren Plateau Theorem** (McClean et al., 2018). For random circuits, gradient variance decays exponentially with qubit size:
  $$\text{Var}\left[\frac{\partial \langle Z_i \rangle}{\partial \theta_k}\right] \sim \mathcal{O}(2^{-n})$$
  Choosing $R=3$ for $n=4$ qubits avoids vanishing gradients while retaining capacity.

---

## 9. Training Pipeline, Loss, & Optimization

- **Shared Hyperparameters** (both agents):
  | Hyperparameter | Value |
  |---|---|
  | Discount Factor $\gamma$ | $0.99$ |
  | Learning Rate $\eta$ (Adam) | $0.001$ |
  | $\epsilon$-start / $\epsilon$-min / $\epsilon$-decay | $1.0$ / $0.01$ / $0.995$ |
  | Target Network Sync | Every $10$ episodes |
  | Gradient Clipping | $\|\nabla\|_{\max} = 1.0$ |
- **Differences**:
  | | Classical DQN | Quantum VQC |
  |---|---|---|
  | Batch Size | $64$ | $16$ |
  | Replay Buffer | $10{,}000$ | $5{,}000$ |
  | Parameters | $4{,}612$ | $36$ |
- **Loss**: MSE of TD error: $L(\theta) = \frac{1}{|B|}\sum\left(y - Q(s,a;\theta)\right)^2$, where $y = r + \gamma(1-d)\max_{a'}Q(s',a';\theta^-)$
- **Quantum Gradients**: Parameter-Shift Rule: $\frac{\partial \langle Z_i\rangle}{\partial \theta_j} = \frac{1}{2}\left(\langle Z_i\rangle_{\theta_j+\pi/2} - \langle Z_i\rangle_{\theta_j-\pi/2}\right)$

---

## 10. Accuracy Verification (3x3 Grid)

- **Convergence Results**:
  - Both agents converge to the optimal 3-step path with reward **$+0.88$**.
  - Classical DQN stabilizes by Episode 50 (higher noise due to 4,612 parameter updates).
  - Quantum VQC stabilizes by Episode 110 (extremely stable convergence due to 36-parameter regularization).

![center width:700px](results/reward_plot.png)

---

## 11. Scaling Analysis (3x3 vs. 4x4)

- **The Scaling Test**: Double the state space (9 to 16 states) and increase shortest path to 6 steps.
- **Empirical Convergence**:
  - **Classical 4x4**: Converges by Episode 180 (Reward: $+0.72$, near-optimal 7-step path).
  - **Quantum 4x4**: Converges by Episode 220 (Reward: $+0.72$, near-optimal 7-step path).

![center width:750px](results/scaling_analysis_plot.png)

---

## 12. The Computational Bottleneck

- **Parametric Scaling**: The VQC requires **zero additional parameters** ($36$ weights) when scaling from 3x3 to 4x4, showing extreme parameter scaling efficiency.
- **Classical Simulation Cost**: The training duration scales poorly under CPU simulation.
  - 3x3 Grid: ~2 minutes (Classical) vs. ~15 minutes (Quantum).
  - 4x4 Grid: ~5 minutes (Classical) vs. ~1.5 hours (Quantum).
- **The Cause**: The **Parameter-Shift Rule** requires $2N$ circuit simulations per parameter update ($2 \times 16 \times 64 = 2,048$ simulations per training step), leading to over 15 million simulations per 250-episode run.

---

## 13. Quantum State Tomography

- **Dimensionality**: Reconstructed density matrix $\rho$ is a $2^4 \times 2^4 = 16 \times 16$ complex matrix (4-qubit system).
- **Physical Coordinates Evaluated**:
  - **Start $(0, 0)$**: High-amplitude state corresponding to Down/Right actions.
  - **Obstacle Adjacent $(1, 0)$**: Reconstructed state exhibits superposition corresponding to actions that steer away from the obstacle at $(1, 1)$.
  - **Near Goal $(2, 1)$ & Goal $(2, 2)$**: Finalized convergence states pointing to termination.
- **Physical Quality Metrics**:
  - **Purity ($\gamma = \text{Tr}(\rho^2)$)**: Exactly **$1.0$** at all test coordinates, indicating a pure state ($|\psi\rangle\langle\psi|$) instead of a mixed state ($\gamma \in [1/16, 1.0]$).
  - **Von Neumann Entropy ($S = -\text{Tr}(\rho \ln \rho)$)**: Exactly **$0.0$**, confirming zero statistical disorder.
- **Coherence Verification**: Confirms the quantum policy network executes as a fully coherent superposition, unaffected by decoherence under simulator conditions.

![center width:800px](results/tomography_analysis.png)

---

## 14. Conclusion & Outlook

- **Core Takeaways**:
  1. Hybrid Quantum RL is viable and matches classical accuracy with a **99.2% parameter reduction**.
  2. The VQC's parameter constraint acts as a natural regularizer, reducing policy variance.
  3. Classical simulation of gradients is the primary training bottleneck.
- **Future Work**:
  - Deploy on physical QPUs (IBM Quantum) to evaluate resilience under thermal relaxation noise and gate errors.
  - Utilize **Adjoint Quantum Differentiation** to bypass the parameter-shift simulation bottleneck.
- **Q&A**: Open the floor to questions.
