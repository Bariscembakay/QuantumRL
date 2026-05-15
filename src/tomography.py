import torch
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, state_fidelity, purity, entropy
from qiskit_aer import AerSimulator
from environment import GridWorldEnv
from quantum_dqn import QuantumQNetwork
import matplotlib.pyplot as plt

def perform_tomography(agent_path="models/quantum_dqn.pth"):
    # 1. Load the trained agent
    print(f"Loading trained quantum agent from {agent_path}...")
    model = QuantumQNetwork()
    model.load_state_dict(torch.load(agent_path))
    model.eval()
    
    # 2. Select states for analysis
    # (row, col) coordinates
    test_states = [
        (0, 0), # Start
        (1, 0), # Next to obstacle
        (2, 1), # Near goal
        (2, 2)  # Goal
    ]
    
    results = []

    for pos in test_states:
        print(f"\nAnalyzing state: {pos}")
        state_tensor = torch.FloatTensor(np.array(pos)).unsqueeze(0)
        
        # Scale input as we did in training
        x_scaled = state_tensor * (np.pi / 2.0)
        
        # To get the state vector/density matrix, we need to run the circuit 
        # with the specific weights and inputs.
        # TorchConnector weights are stored in self.qnn_connector.weight
        weights = model.qnn_connector.weight.detach().numpy()
        
        # Build the specific circuit for this state
        # (Using the parameters from the model)
        qc = model.circuit.assign_parameters({
            model.input_params[0]: x_scaled[0, 0].item(),
            model.input_params[1]: x_scaled[0, 1].item()
        })
        
        # Assign the trained weights
        # Note: RealAmplitudes parameters are in a specific order
        # We assign the weight parameters from our list
        param_dict = {p: weights[i] for i, p in enumerate(model.weight_params)}
        qc = qc.assign_parameters(param_dict)
        
        # 3. Simulate to get the Density Matrix
        # We add save_density_matrix to get the full state info
        qc.save_density_matrix()
        
        sim = AerSimulator()
        job = sim.run(qc)
        rho = job.result().data()['density_matrix']
        
        # 4. Calculate metrics
        p = purity(rho)
        s = entropy(rho)
        
        print(f"  Purity: {p:.4f}")
        print(f"  Entropy: {s:.4f}")
        
        results.append({
            'pos': pos,
            'density_matrix': rho,
            'purity': p,
            'entropy': s
        })

    # 5. Visualization
    fig, axes = plt.subplots(1, len(test_states), figsize=(18, 5))
    for i, res in enumerate(results):
        axes[i].set_title(f"Pos: {res['pos']}\nS={res['entropy']:.2f}")
        # Plot the real part of the density matrix as a heatmap
        im = axes[i].imshow(np.real(res['density_matrix'].data), cmap='viridis')
        plt.colorbar(im, ax=axes[i])
        
    plt.suptitle("Quantum State Tomography (Density Matrices) across Grid")
    plt.tight_layout()
    plt.savefig("results/tomography_analysis.png")
    print("\nTomography analysis saved to results/tomography_analysis.png")

if __name__ == "__main__":
    perform_tomography()
