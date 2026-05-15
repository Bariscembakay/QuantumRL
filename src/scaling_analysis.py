import torch
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from classical_dqn import train_dqn
from quantum_dqn import train_quantum_dqn

# Ensure we can import from src
sys.path.append(os.path.join(os.getcwd(), 'src'))

def run_scaling_analysis():
    print("========================================")
    print("    Scaling Analysis: 3x3 vs 4x4        ")
    print("========================================\n")
    
    episodes = 250
    grid_sizes = [3, 4]
    
    results = {}

    for size in grid_sizes:
        print(f"--- Training on {size}x{size} Grid ---")
        
        # 1. Classical
        print(f"  Training Classical DQN ({size}x{size})...")
        _, c_history = train_dqn(episodes=episodes, grid_size=size)
        results[f'classical_{size}x{size}'] = c_history
        
        # 2. Quantum
        print(f"  Training Quantum DQN ({size}x{size})...")
        _, q_history = train_quantum_dqn(episodes=episodes, grid_size=size)
        results[f'quantum_{size}x{size}'] = q_history
        print(f"Finished {size}x{size} Grid.\n")

    # Save data
    df = pd.DataFrame(results)
    df.to_csv("results/scaling_history.csv", index=False)
    print("Scaling data saved to results/scaling_history.csv")

    # Plot Comparison
    plt.figure(figsize=(14, 8))
    for col in df.columns:
        # Smooth the data
        smooth_data = df[col].rolling(window=15).mean()
        style = '-' if 'quantum' in col else '--'
        color = 'green' if '3x3' in col else 'red'
        label = col.replace('_', ' ').title()
        plt.plot(smooth_data, label=label, linestyle=style, color=color if 'classical' not in col else None)

    plt.xlabel('Episode')
    plt.ylabel('Total Reward (Smoothed)')
    plt.title('Scaling Analysis: Grid World Navigation (3x3 vs 4x4)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("results/scaling_analysis_plot.png")
    print("Scaling plot saved to results/scaling_analysis_plot.png")

if __name__ == "__main__":
    run_scaling_analysis()
