import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

qc = QuantumCircuit(4)
input_params = ParameterVector('input', 2)
weight_params = ParameterVector('weight', 4)

qc.ry(input_params[0], 0)
qc.ry(input_params[1], 1)
for i in range(4):
    qc.ry(weight_params[i], i)

observables = [
    SparsePauliOp.from_list([("IIIZ", 1)]),
    SparsePauliOp.from_list([("IIZI", 1)]),
    SparsePauliOp.from_list([("IZII", 1)]),
    SparsePauliOp.from_list([("ZIII", 1)])
]

qnn = EstimatorQNN(
    circuit=qc,
    observables=observables,
    input_params=input_params,
    weight_params=weight_params
)

model = TorchConnector(qnn)
x = torch.tensor([[0.5, 0.5]])
y = model(x)
print("Output shape:", y.shape)
print("Output:", y)
