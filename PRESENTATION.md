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
**Course Project: ELEC/PHYS 450/550**

**Presenter**: Barış Cem Bakay
**Student ID**: 0082990
**Institution**: Koç University
**Instructor**: Assoc. Prof. Mehmet Cengiz Onbaşlı

---

## 1. Project Objectives
* **The Core Goal**: Explore whether near-term Quantum Neural Networks (Variational Quantum Circuits) can solve sequential decision-making tasks compared to classical deep learning.
* **Core Deliverables**:
  1. Classical Deep Q-Network (DQN) implementation as baseline.
  2. Hybrid Quantum DQN using Qiskit and PyTorch integration.
  3. Accuracy verification and scaling analysis (3x3 vs. 4x4 grids).
  4. Physical verification via exact Quantum State Tomography.

---

## 2. Environment & Problem Formulation
* **State Space**: Discrete 2D Grid World ($3\times3$ and $4\times4$) containing coordinates $(x,y)$.
* **Action Space**: 4 discrete movement directions: $\mathcal{A} = \{\text{Up, Down, Left, Right}\}$.
* **Reward Dynamics**: Goal state $(N-1, N-1)$ yields $+1.0$; Obstacle $(1,1)$ yields $-1.0$. Step penalty is $-0.04$ per move.

![center width:750px](results/grid_world_visualization.png)

---

## 3. Classical DQN Architecture (Baseline)
* **Structure**: Feedforward Deep Q-Network implemented in PyTorch.
* **Architecture**: Input layer (2) $\to$ Hidden 1 (64, ReLU) $\to$ Hidden 2 (64, ReLU) $\to$ Output (4, Linear).
* **Parameter Cost**: **4,612 trainable weights**.
* **Exploration Policy**: $\epsilon$-Greedy exploration with a replay buffer (capacity = 5,000).

![center width:700px](results/classical_dqn_agent.png)

---

## 4. Quantum Policy Network (VQC)
* **Qubits**: 4-qubit circuit replacing the classical neural network.
* **Data Encoding**: Normalize coordinates to $\theta \in [0, \pi]$ and encode via $R_y$ rotation gates.
* **Ansatz**: `RealAmplitudes` with linear entangling CNOT ladders (3 repetitions).
* **Total Parameters**: **36 trainable weights** ($16$ quantum angles + $20$ classical weights) — a **99.2% parameter reduction**!

![center width:900px](results/quantum_vqc_agent.png)

---

## 5. Accuracy Verification (3x3 Grid)
* **Key Finding**: The Quantum agent successfully matches classical accuracy!
* **Optimal Path**: Shortest path to goal takes 3 steps. Theoretical maximum reward is $+0.88$.
* **Convergence**: Classical DQN converges by Episode 50; Quantum VQC stabilizes by Episode 110.

![center width:700px](results/reward_plot.png)

---

## 6. Scaling Analysis (3x3 vs. 4x4 Grids)
* **The Scaling Challenge**: State space nearly doubles (9 to 16 states) and optimal path length doubles (3 to 6 steps).
* **Empirical Results**:
  * **Classical 4x4**: Converges by Episode 200 (Total reward: $-0.08$ due to sub-optimal 27 steps).
  * **Quantum 4x4**: Converges by Episode 220 (Total reward: $+0.72$ matching the optimal 7 steps).

![center width:750px](results/scaling_analysis_plot.png)

---

## 7. The Core Trade-off
* **Quantum Advantage (Memory)**: $99.2\%$ Parameter Reduction. Moving from 3x3 to 4x4 required **zero extra parameters** in the VQC ($36$ weights), while classical networks scaled from $4,612$ parameters.
* **Classical Advantage (Speed)**: VQC simulation on CPUs is computationally heavy.
* **The Bottleneck**: Parameter-shift gradient rules require simulating $2N$ quantum circuits per parameter per step (over 15 million simulations per training run).

---

## 8. Quantum State Tomography
* **Verification**: Exact state tomography performed on the final density matrix $\rho$ across key grid coordinates.
* **State Purity ($\gamma$)**: $\text{Tr}(\rho^2) = 1.0$. Confirms unitary simulation preserved 100% state purity.
* **Von Neumann Entropy ($S$)**: $-\text{Tr}(\rho \ln \rho) = 0.0$. Verifies coherent, pure-state policy execution.

![center width:850px](results/tomography_analysis.png)

---

## 9. Conclusion & Outlook
* **Summary**: Quantum RL is highly viable and extremely parameter-efficient for discrete navigation tasks.
* **Future Work**:
  1. Deploying policy networks on physical QPU hardware (e.g., IBM Quantum).
  2. Utilizing adjoint quantum differentiation methods to eliminate the parameter-shift simulation bottleneck.
* **Q&A**: Open the floor to questions.
