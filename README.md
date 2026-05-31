# Quantum vs. Classical Reinforcement Learning for Grid World Navigation

This repository contains the codebase and comparative analysis report for the **Phys450: Quantum Computing** course project. The project explores, implements, and benchmarks a Quantum Variational Quantum Circuit (VQC) agent against a Classical Deep Q-Network (DQN) agent on a Grid World navigation task.

---

## 🚀 Project Overview

The core objective of this project is to implement a hybrid quantum-classical reinforcement learning algorithm to navigate a grid world with static obstacles.

### Key Objectives
* **Classical Baseline**: Standard Deep Q-Network (DQN) implemented in PyTorch.
* **Quantum Agent**: Variational Quantum Circuit (VQC) policy network using Qiskit and PyTorch integration via `qiskit-machine-learning`.
* **State Tomography**: Exact extraction of the output quantum density matrices, evaluating state purity and Von Neumann entropy to verify coherence.
* **Scaling Analysis**: Performance benchmarking across 3x3 and 4x4 grids to assess sample and parameter efficiency vs. simulation overhead.

---

## 📂 Repository Structure

```
├── pyproject.toml         # Python project configuration (uv managed)
├── requirements.txt       # Direct dependency list
├── README.md              # Project documentation
├── REPORT.md              # Detailed academic report of comparative findings
├── PRESENTATION.md        # Slide-by-slide project presentation outline
├── main.py                # Unified classical and quantum training script
├── models/                # Saved weights (.pth) for classical and quantum models
├── results/               # Saved plots, training CSVs, and tomography images
└── src/
    ├── environment.py     # Custom Gymnasium GridWorld environment (3x3 and 4x4)
    ├── quantum_policy.py  # VQC construction (Angle Encoding, RealAmplitudes ansatz)
    ├── classical_dqn.py   # Classical DQN agent class and training functions
    ├── quantum_dqn.py     # Quantum DQN agent class and Qiskit-PyTorch training functions
    ├── tomography.py      # Output density matrix and entropy analysis
    ├── analysis.py        # Post-training curve generation
    └── scaling_analysis.py# Main script to execute multi-grid comparison runs
```

---

## 🛠️ Installation & Setup

This project uses `uv` for lightning-fast virtual environment and dependency management.

### Prerequisites
* Python 3.13
* (Optional) `uv` package manager

### Standard Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/QuantumRL.git
   cd QuantumRL
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏃 How to Run

### 1. Unified Baseline Training
To train both the classical and quantum agents on a 3x3 grid, save their final weights, and write the training rewards to a CSV:
```bash
python main.py
```

### 2. Quantum State Tomography
To load the trained quantum model, construct its physical density matrices across key grid states, and output purity, entropy, and heatmaps:
```bash
python src/tomography.py
```
*Outputs: Density matrix heatmaps saved to `results/tomography_analysis.png`.*

### 3. Scaling & Comparative Analysis
To run the complete comparative benchmark across 3x3 and 4x4 grids:
```bash
python src/scaling_analysis.py
```
*Outputs: Learning curve plot saved to `results/scaling_analysis_plot.png` and CSV data to `results/scaling_history.csv`.*

---

## 📊 Key Findings Summary

Detailed findings can be found in [REPORT.md](REPORT.md). 

1. **Parameter Efficiency**: The Quantum agent solves the navigation tasks using only **36 trainable weights** (99.2% fewer than the classical DQN network which uses 4,612 weights).
2. **Path Optimality**: Both agents successfully converge to optimal navigation trajectories on 3x3 grids (reward $+0.88$, matching the 3-step perfect path) and 4x4 grids (reward $+0.72$, matching the 7-step near-optimal path).
3. **Simulation Trade-off**: Quantum training on classical CPUs requires calculating gradients using the Parameter-Shift rule, scaling simulation time substantially (1.5+ hours for 4x4). This highlighted the critical need for native QPU acceleration or adjoint method integration in real-world deployments.