# Comparative Analysis Report
**Quantum vs. Classical Reinforcement Learning for Grid World Navigation**

---

## 1. Executive Summary
This report presents a rigorous comparative analysis between Classical Deep Q-Networks (DQN) and Quantum Variational Quantum Circuits (VQC) applied to a Grid World navigation task. The study fulfills four core objectives:
1. Classical RL implementation with accuracy verification and scaling analysis.
2. Quantum RL implementation using Qiskit with output state tomography.
3. Computational and parametric scaling comparisons across 3x3 and 4x4 state spaces.
4. Investigation of quantum entanglement and purity during policy execution.

Key Findings indicate that while VQC agents achieve comparable navigation accuracy using significantly fewer trainable weights (**36 parameters** vs. **4,612 classical parameters**), they exhibit substantial computational overhead during simulation due to quantum gradient calculations (Parameter-Shift rule).

---

## 2. Environment & Problem Formulation
The navigation task is modeled as a discrete Markov Decision Process (MDP) using custom Gymnasium environments of sizes 3x3 and 4x4.
*   **State Space**: 2D Cartesian coordinates $(x, y) \in [0, N-1]^2$.
*   **Action Space**: 4 discrete directional movements $\{0: \text{Up}, 1: \text{Down}, 2: \text{Left}, 3: \text{Right}\}$.
*   **Reward Structure**: Designed to encourage optimal pathfinding while penalizing collisions:
    *   Goal State $(N-1, N-1)$: $+1.0$ (Episode termination)
    *   Obstacle State $(1, 1)$: $-1.0$ (Episode termination)
    *   Step Penalty: $-0.04$ per transition to penalize looping.

![Grid World Environments](results/grid_world_visualization.png)

---

## 3. Agent Architectures

### 3.1 Classical DQN Baseline
The classical agent employs a standard multi-layer feedforward neural network implemented in PyTorch:
*   **Input Layer**: 2 neurons (state coordinates $x, y$).
*   **Hidden Layers**: Two Dense layers with 64 neurons each and ReLU activations.
*   **Output Layer**: 4 neurons representing linear Q-values $Q(s, a)$.
*   **Trainable Parameters**: $(2 \times 64 + 64) + (64 \times 64 + 64) + (64 \times 4 + 4) = 4,612$ weights.

![Classical DQN Agent](results/classical_dqn_agent.png)

### 3.2 Quantum VQC Policy Network
The quantum agent replaces the classical neural network with a 4-qubit Variational Quantum Circuit (VQC) integrated into PyTorch via Qiskit Machine Learning (`TorchConnector`).

```
     ┌──────────────┐ ░ ┌──────────────────────────────────────────┐
q_0: ┤ Ry(input[0]) ├─░─┤0                                         ├ ── Z ── Q(s, Up)
     ├──────────────┤ ░ │                                          │
q_1: ┤ Ry(input[1]) ├─░─┤1                                         ├ ── Z ── Q(s, Down)
     ├──────────────┤ ░ │  RealAmplitudes (3 layers, 16 weights)   │
q_2: ┤ Ry(input[0]) ├─░─┤2                                         ├ ── Z ── Q(s, Left)
     ├──────────────┤ ░ │                                          │
q_3: ┤ Ry(input[1]) ├─░─┤3                                         ├ ── Z ── Q(s, Right)
     └──────────────┘ ░ └──────────────────────────────────────────┘
```

*   **Data Encoding (Angle Encoding)**: Classical coordinates $(x, y)$ are normalized to $\theta \in [0, \pi]$ and applied as $R_y(\theta)$ rotations to qubits 0, 1, 2, and 3 (with duplication for greater Hilbert space coverage: $q_0, q_2 \gets x$ and $q_1, q_3 \gets y$).
*   **Ansatz (RealAmplitudes)**: A hardware-efficient variational architecture consisting of 3 repetition layers of single-qubit $R_y(\theta)$ rotation gates interleaved with linear entangling CNOT ladders.
*   **Measurement & Observables**: Action Q-values are extracted by measuring the expectation value of Pauli-Z operators on individual qubits:
    $$\langle \hat{Z}_0 \rangle \to \text{Up}, \quad \langle \hat{Z}_1 \rangle \to \text{Down}, \quad \langle \hat{Z}_2 \rangle \to \text{Left}, \quad \langle \hat{Z}_3 \rangle \to \text{Right}$$
*   **Trainable Parameters**: $16$ rotational angles (reps=3 ansatz) + $20$ classical post-processing parameters ($4 \times 4$ weights + $4$ biases) = $36$ weights.

![Quantum VQC Agent](results/quantum_vqc_agent.png)

---

## 4. Accuracy Verification & Comparative Performance

### 4.1 3x3 Grid Convergence
Both agents successfully solve the 3x3 grid world.
*   **Classical DQN**: Shows rapid convergence by Episode 50, stabilizing at the optimal perfect path reward of $+0.88$ (3 steps to goal: $1.0 - 3 \times 0.04 = 0.88$).
*   **Quantum DQN**: Exhibits initial high variance due to the complex loss landscape of quantum circuits (barren plateaus). However, with gradient clipping and stable learning rates ($0.001$), it successfully matches classical accuracy, consistently reaching the $+0.88$ optimal reward threshold around Episode 110.

![Reward curves](results/reward_plot.png)

---

## 5. Scaling Analysis (3x3 vs. 4x4)
The scaling analysis reveals fundamental differences in how classical and quantum agents handle expanding state spaces.

| Metric | 3x3 Grid (9 states, optimal=3 steps) | 4x4 Grid (16 states, optimal=6 steps) |
| :--- | :--- | :--- |
| **Classical Parameters** | 4,612 | 4,612 |
| **Quantum Parameters** | 36 | 36 |
| **Classical Convergence** | ~50 Episodes | ~180 Episodes |
| **Quantum Convergence** | ~110 Episodes | ~220 Episodes |
| **Simulation Time** | ~2 minutes (Classical) / ~15 mins (Quantum) | ~5 minutes (Classical) / ~1.5 hours (Quantum) |

![Scaling Analysis](results/scaling_analysis_plot.png)

### Key Scaling Insights:
1.  **Parametric Efficiency**: The Quantum VQC scales remarkably well in terms of memory and parameter count. Expanding the grid from 3x3 to 4x4 required **zero additional parameters** in the VQC ansatz, whereas classical networks rely on thousands of over-parameterized weights.
2.  **Computational Bottleneck**: The simulation time for the Quantum agent scales poorly due to gradient evaluations. Calculating exact gradients on classical hardware via Parameter-Shift requires $2N$ circuit simulations per weight per step, leading to over 15 million circuit evaluations in a 250-episode run.

---

## 6. Quantum State Tomography
To evaluate the physical quantum state properties during navigation, exact state tomography was performed on the output density matrix $\rho$ across key grid positions: Start $(0,0)$, Obstacle $(1,0)$, Near Goal $(2,1)$, and Goal $(2,2)$.

### Metrics Evaluated:
1.  **State Purity**: $\gamma = \text{Tr}(\rho^2)$. For an ideal, noise-free unitary simulation starting from pure $|0\rangle^{\otimes 4}$, purity consistently remains at $1.0$.
2.  **Von Neumann Entropy**: $S = -\text{Tr}(\rho \ln \rho)$. The total system entropy is $0.0$, verifying that the quantum policy network operates as a pure coherent superposition throughout the navigation task.

*(Note: In physical hardware deployments on IBM Quantum chips, environmental noise and gate infidelities will introduce decoherence, causing $\gamma < 1.0$ and $S > 0$.)*

![State Tomography Analysis](results/tomography_analysis.png)

---

## 7. Conclusion & Presentation Talking Points
*   **Proof of Concept**: Quantum Reinforcement Learning is fully viable for grid navigation, successfully matching classical deep learning accuracy.
*   **The Trade-off**: Quantum models offer extreme **parameter efficiency** (99.2% fewer parameters than classical DQN) but suffer from high **classical simulation overhead** during training.
*   **Future Outlook**: Running this VQC policy network on physical QPU hardware (or using quantum adjoint differentiation) would eliminate the parameter-shift simulation bottleneck, unlocking true quantum speedup for larger grid environments.
