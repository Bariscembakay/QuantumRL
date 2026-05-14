from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes

def create_qrl_circuit(num_qubits=4, layers=3):
    """
    Creates a Variational Quantum Circuit (VQC) for Reinforcement Learning.
    
    This circuit combines Angle Encoding (for state data) and the 
    RealAmplitudes ansatz (for the trainable policy weights).
    
    Args:
        num_qubits (int): Total number of qubits. We'll use 4 qubits since 
                          we have 4 possible actions (Up, Down, Left, Right).
        layers (int): Number of repetition layers in the ansatz.
        
    Returns:
        tuple: (QuantumCircuit, ParameterVector for inputs)
    """
    # 1. Initialize Circuit
    qc = QuantumCircuit(num_qubits)
    
    # 2. Data Encoding (Angle Encoding)
    # We create a ParameterVector for the classical input data.
    # We need 2 parameters: one for the agent's X coordinate, one for Y.
    input_params = ParameterVector('input', 2)
    
    # Apply Ry rotations to encode the scaled X and Y coordinates.
    # We duplicate the inputs to give the circuit more expressivity across all 4 qubits.
    qc.ry(input_params[0], 0)
    qc.ry(input_params[1], 1)
    qc.ry(input_params[0], 2)
    qc.ry(input_params[1], 3)
    
    # Barrier to visually separate encoding from the ansatz
    qc.barrier()
    
    # 3. Ansatz (RealAmplitudes)
    # This acts as the 'neural network' layers. It contains trainable parameters.
    # 'linear' entanglement means qubit 0 is entangled with 1, 1 with 2, etc.
    from qiskit.circuit.library import real_amplitudes
    ansatz = real_amplitudes(num_qubits, entanglement='linear', reps=layers)
    
    # Combine the encoding circuit and the ansatz
    qc.compose(ansatz, inplace=True)
    
    return qc, input_params

if __name__ == "__main__":
    # Test the circuit generation
    print("--- Generating Quantum Policy Network ---")
    circuit, inputs = create_qrl_circuit(num_qubits=4, layers=2)
    
    # Draw the circuit in the terminal
    print("\nCircuit Diagram:")
    print(circuit.draw(output='text'))
    
    # Calculate the number of weights the RL agent will need to learn
    total_params = circuit.num_parameters
    input_size = len(inputs)
    trainable_weights = total_params - input_size
    print(f"\nTotal parameters: {total_params}")
    print(f"Input parameters (State): {input_size}")
    print(f"Trainable Weights (Policy): {trainable_weights}")
