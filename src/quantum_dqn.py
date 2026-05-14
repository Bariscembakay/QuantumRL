import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from environment import GridWorldEnv
from quantum_policy import create_qrl_circuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

# 1. Quantum Neural Network for Q-Value Approximation
class QuantumQNetwork(nn.Module):
    def __init__(self):
        super(QuantumQNetwork, self).__init__()
        # We use 3 layers to ensure the VQC has enough trainable parameters (expressivity)
        self.circuit, self.input_params = create_qrl_circuit(num_qubits=4, layers=3)
        
        # The weight parameters are all parameters after the input parameters
        all_params = list(self.circuit.parameters)
        self.weight_params = [p for p in all_params if p not in self.input_params]
        
        # Define observables for the 4 actions
        # Each action corresponds to measuring Pauli-Z on one of the 4 qubits
        self.observables = [
            SparsePauliOp.from_list([("IIIZ", 1)]), # Action 0: Up (Qubit 0)
            SparsePauliOp.from_list([("IIZI", 1)]), # Action 1: Down (Qubit 1)
            SparsePauliOp.from_list([("IZII", 1)]), # Action 2: Left (Qubit 2)
            SparsePauliOp.from_list([("ZIII", 1)])  # Action 3: Right (Qubit 3)
        ]
        
        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=self.input_params,
            weight_params=self.weight_params
        )
        
        # Wrap QNN into a PyTorch layer
        self.qnn_connector = TorchConnector(self.qnn)
        
        # Replace scalar scale with a Linear layer to fix gradient issues
        self.output_layer = nn.Linear(4, 4)
        
        # Initialize small weights so Q-values start near 0
        nn.init.uniform_(self.output_layer.weight, -0.1, 0.1)
        nn.init.constant_(self.output_layer.bias, 0.0)
        
    def forward(self, x):
        # GridWorld states are coordinates in [0, 2].
        # We scale them by pi/2 to fit into the [0, pi] range for angle encoding.
        x_scaled = x * (np.pi / 2.0)
        qnn_out = self.qnn_connector(x_scaled)
        return self.output_layer(qnn_out)

# 2. Replay Buffer to store experiences
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# 3. Quantum DQN Agent
class QuantumDQNAgent:
    def __init__(self):
        self.action_dim = 4
        self.policy_net = QuantumQNetwork()
        self.target_net = QuantumQNetwork()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # VQC usually benefits from a slightly higher learning rate than classical networks
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.01)
        self.memory = ReplayBuffer(5000)
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995  # Restore to classical DQN's decay rate for proper exploration
        self.epsilon_min = 0.01
        self.batch_size = 16 
        self.target_update = 10

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            return self.policy_net(state).argmax().item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        state_batch = torch.FloatTensor(np.array(batch[0]))
        action_batch = torch.LongTensor(batch[1]).unsqueeze(1)
        reward_batch = torch.FloatTensor(batch[2]).unsqueeze(1)
        next_state_batch = torch.FloatTensor(np.array(batch[3]))
        done_batch = torch.FloatTensor(batch[4]).unsqueeze(1)

        # Current Q values
        current_q = self.policy_net(state_batch).gather(1, action_batch)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            expected_q = reward_batch + (1 - done_batch) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, expected_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_quantum_dqn(episodes=150):
    env = GridWorldEnv()
    agent = QuantumDQNAgent()
    rewards_history = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        step_count = 0
        while not done and step_count < 50:
            action = agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            
            # Train every step, simulating QNNs is slow, but necessary for convergence
            agent.train()
                
            state = next_state
            total_reward += reward
            step_count += 1
        
        # Update target network
        if episode % agent.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
        rewards_history.append(total_reward)
        if episode % 10 == 0:
            print(f"Episode {episode:3d}, Total Reward: {total_reward:6.2f}, Epsilon: {agent.epsilon:.2f}")
            
    return agent, rewards_history

if __name__ == "__main__":
    print("--- Starting Quantum DQN Training ---")
    # VQCs often need more episodes than classical networks because the loss landscape 
    # (Barren Plateaus) is harder to navigate. We use 250 episodes.
    agent, history = train_quantum_dqn(episodes=250)
    
    # Final Test
    print("\n--- Final Test Run ---")
    env = GridWorldEnv()
    state, _ = env.reset()
    env.render()
    done = False
    steps = 0
    while not done and steps < 20:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = agent.policy_net(state_tensor)
            action = q_values.argmax().item()
        print(f"State: {state}, Q-values: {q_values.numpy()[0]}")
        print(f"Action taken: {action}")
        state, reward, done, _, _ = env.step(action)
        env.render()
        steps += 1
    
    if done and reward == 1.0:
        print("Agent reached the goal!")
    else:
        print("Agent did not reach the goal within 20 steps.")
