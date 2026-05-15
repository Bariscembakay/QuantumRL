import os
import sys

# Ensure we can import from src
sys.path.append(os.path.join(os.getcwd(), 'src'))

import torch
import pandas as pd
from classical_dqn import train_dqn
from quantum_dqn import train_quantum_dqn

def main():
    print("========================================")
    print("   Quantum vs Classical RL Training     ")
    print("========================================\n")
    
    # 1. Train Classical DQN
    print("--- Training Classical DQN ---")
    c_agent, c_history = train_dqn(episodes=200) # Reduced episodes for quicker data collection
    torch.save(c_agent.policy_net.state_dict(), "models/classical_dqn.pth")
    print("Classical training complete.\n")
    
    # 2. Train Quantum DQN
    print("--- Training Quantum DQN ---")
    # Quantum simulation is slower, we keep it to a comparable level
    q_agent, q_history = train_quantum_dqn(episodes=200) 
    torch.save(q_agent.policy_net.state_dict(), "models/quantum_dqn.pth")
    print("Quantum training complete.\n")
    
    # 3. Save Results to CSV
    print("--- Saving Results ---")
    # Pad histories with NaN if they have different lengths (though here they are both 200)
    df = pd.DataFrame({
        'episode': range(len(c_history)),
        'classical_reward': c_history,
        'quantum_reward': q_history
    })
    df.to_csv("results/reward_history.csv", index=False)
    print("Results saved to results/reward_history.csv")
    print("Models saved to models/ directory.")

if __name__ == "__main__":
    main()
