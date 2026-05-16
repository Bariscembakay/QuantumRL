# Presentation Outline: Quantum vs. Classical RL for Grid World Navigation
**Course Project: Phys450**

---

## Slide 1: Title Slide
*   **Title**: Quantum Reinforcement Learning for Grid World Navigation: A Comparative Analysis
*   **Presenter**: Baris Cem Bakay
*   **Course**: Phys450 - Quantum Computing Project
*   **Visual**: A simple graphic or icon of a quantum circuit alongside a grid world.

---

## Slide 2: Project Objectives
*   **The Goal**: Explore whether near-term Quantum Neural Networks (VQC) can successfully solve sequential decision-making tasks compared to classical deep learning.
*   **Core Deliverables**:
    1. Classical Deep Q-Network (DQN) implementation and baseline.
    2. Quantum DQN implementation in Qiskit using Variational Quantum Circuits.
    3. Accuracy verification and scaling analysis (3x3 vs 4x4 grids).
    4. Exact Quantum State Tomography of the policy network during navigation.

---

## Slide 3: Environment & Problem Formulation
*   **The Environment**: Discrete 2D Grid World (Sizes: 3x3 and 4x4).
*   **State Space**: Agent Cartesian coordinates $(x, y)$.
*   **Action Space**: 4 discrete moves (Up, Down, Left, Right).
*   **Reward Dynamics**:
    *   Goal State $(N-1, N-1)$: $+1.0$ (Termination)
    *   Obstacle State $(1, 1)$: $-1.0$ (Termination)
    *   Step Penalty: $-0.04$ per move (encourages shortest pathfinding).
*   **Visual Idea**: Screenshot of the terminal environment grid (S . . / . X . / . . G).

---

## Slide 4: Classical DQN Architecture (Baseline)
*   **Structure**: Standard Feedforward Deep Q-Network implemented in PyTorch.
*   **Layers**: 2 hidden layers with 64 neurons each + ReLU activations.
*   **Total Parameters**: **4,612 trainable weights**.
*   **Exploration**: Epsilon-Greedy exploration policy with experience replay buffer (capacity = 5,000).

---

## Slide 5: Quantum Policy Network (VQC)
*   **The Paradigm Shift**: Replacing the 4,612-parameter classical neural network with a compact 4-qubit quantum circuit.

```
     ┌──────────────┐ ░ ┌──────────────────────────────────────────┐
q_0: ┤ Ry(input[0]) ├─░─┤0                                         ├ ── Z ── Q(s, Up)
     ├──────────────┤ ░ │                                          │
q_1: ┤ Ry(input[1]) ├─░─┤1                                         ├ ── Z ── Q(s, Down)
     └──────────────┘ ░ │  RealAmplitudes (3 layers, 12 weights)   │
q_2: ─────────────────░─┤2                                         ├ ── Z ── Q(s, Left)
                      ░ │                                          │
q_3: ─────────────────░─┤3                                         ├ ── Z ── Q(s, Right)
                      ░ └──────────────────────────────────────────┘
```

*   **Data Encoding (Angle Encoding)**: $(x,y)$ coordinates normalized to angles $\theta \in [0, \pi]$ and applied via $R_y$ rotation gates.
*   **Ansatz**: `RealAmplitudes` with linear entangling CNOT ladders (3 repetition layers).
*   **Measurement**: Q-values calculated from the Pauli-Z expectation values of each qubit.
*   **Total Parameters**: **28 trainable weights** ($12$ quantum rotation angles + $16$ classical post-processing weights).

---

## Slide 6: Accuracy Verification (3x3 Grid)
*   **Key Finding**: The Quantum VQC successfully matches classical accuracy on the 3x3 grid!
*   **Optimal Path**: Shortest path to goal is 3 steps. Theoretical maximum reward = $1.0 - (3 \times 0.04) = +0.88$.
*   **Results**:
    *   Classical DQN reached $+0.88$ by Episode 50.
    *   Quantum VQC reached $+0.88$ by Episode 110.
*   **Visual to Include**: `results/reward_plot.png` (Comparative learning curves).

---

## Slide 7: Scaling Analysis (3x3 vs. 4x4 Grids)
*   **The Scaling Question**: How do both models react when the state space nearly doubles (9 states $\to$ 16 states) and optimal path length doubles (3 $\to$ 6 steps)?
*   **Empirical Results**:
    *   Classical 4x4 reached goal by Episode 200 (Total reward: $-0.08 = 27 \text{ steps}$).
    *   Quantum 4x4 reached goal by Episode 230 (Total reward: $+0.72 = 7 \text{ steps}$).
*   **Visual to Include**: `results/scaling_analysis_plot.png`.

---

## Slide 8: The Core Trade-off (Parametric vs. Computational)
*   **Quantum Advantage (Memory)**: 99.4% Parameter Reduction! Moving from 3x3 to 4x4 required **zero extra parameters** in the quantum circuit ($28$ weights), while classical networks required $>4,000$ weights.
*   **Classical Advantage (Speed)**: VQC simulation is computationally heavy.
*   **The Bottleneck**: Parameter-Shift rule required simulating over 1,500 quantum circuits per step in Python during backpropagation (over 15 million simulations per run).

---

## Slide 9: Quantum State Tomography Verification
*   **Physical Verification**: Exact state tomography performed on the final density matrix $\rho$ across key grid coordinates.
*   **State Purity ($\gamma$)**: $\text{Tr}(\rho^2) = 1.0$. Confirms unitary simulation preserved 100% state purity without decoherence.
*   **Von Neumann Entropy ($S$)**: $-\text{Tr}(\rho \ln \rho) = 0.0$. Verifies coherent pure-state policy execution.
*   **Visual to Include**: `results/tomography_analysis.png` (Density matrix heatmaps).

---

## Slide 10: Conclusion & Future Outlook
*   **Summary**: Quantum RL is highly viable and extremely parameter-efficient for discrete navigation tasks.
*   **Future Work**: Executing this policy network directly on physical QPU hardware (like IBM Quantum chips) or using quantum adjoint differentiation would eliminate the parameter-shift simulation bottleneck, unlocking scalable quantum advantage.
*   **Q&A**: Open the floor to questions.
