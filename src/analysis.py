import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_results(csv_path="results/reward_history.csv", output_path="results/reward_plot.png"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Calculate rolling average for smoother visualization
    window_size = 10
    df['classical_smooth'] = df['classical_reward'].rolling(window=window_size).mean()
    df['quantum_smooth'] = df['quantum_reward'].rolling(window=window_size).mean()

    plt.figure(figsize=(12, 6))
    
    # Plot raw data (faded)
    plt.plot(df['episode'], df['classical_reward'], color='blue', alpha=0.1)
    plt.plot(df['episode'], df['quantum_reward'], color='green', alpha=0.1)
    
    # Plot smoothed data
    plt.plot(df['episode'], df['classical_smooth'], color='blue', label='Classical DQN (Moving Avg)')
    plt.plot(df['episode'], df['quantum_smooth'], color='green', label='Quantum DQN (Moving Avg)')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reinforcement Learning Performance: Classical vs Quantum')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set y-axis limits to focus on the interesting part (rewards usually between -2 and 1)
    # But keep it flexible if there are huge outliers
    plt.ylim(max(-5, df['classical_reward'].min()), 1.2)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_results()
