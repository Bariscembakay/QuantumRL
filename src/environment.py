import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    """
    A 3x3 Grid World Environment for Reinforcement Learning.
    
    States:
    The grid is 3x3. Coordinates are (row, col) from (0,0) to (2,2).
    
    Actions:
    0: Up
    1: Down
    2: Left
    3: Right
    
    Rewards:
    +1.0 for reaching the Goal (2, 2)
    -1.0 for hitting an Obstacle (1, 1)
    -0.04 for every other step (to encourage efficiency)
    """
    def __init__(self, size=3):
        super(GridWorldEnv, self).__init__()
        self.size = size
        self.observation_space = spaces.Box(low=0, high=size-1, shape=(2,), dtype=np.int32)
        self.action_space = spaces.Discrete(4)
        
        # Define Fixed Positions
        self.start_pos = np.array([0, 0])
        self.goal_pos = np.array([2, 2])
        self.obstacle_pos = np.array([1, 1])
        
        self.state = self.start_pos.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start_pos.copy()
        return self.state, {}

    def step(self, action):
        # 0: Up, 1: Down, 2: Left, 3: Right
        move = np.array([0, 0])
        if action == 0:   # Up
            move = np.array([-1, 0])
        elif action == 1: # Down
            move = np.array([1, 0])
        elif action == 2: # Left
            move = np.array([0, -1])
        elif action == 3: # Right
            move = np.array([0, 1])
            
        # Update State
        new_state = self.state + move
        
        # Boundary Check
        new_state[0] = np.clip(new_state[0], 0, self.size - 1)
        new_state[1] = np.clip(new_state[1], 0, self.size - 1)
        
        self.state = new_state
        
        # Calculate Reward and Done
        done = False
        reward = -0.04 # Step penalty
        
        if np.array_equal(self.state, self.goal_pos):
            reward = 1.0
            done = True
        elif np.array_equal(self.state, self.obstacle_pos):
            reward = -1.0
            done = True # Episode ends if we hit an obstacle
            
        return self.state, reward, done, False, {}

    def render(self):
        grid = np.zeros((self.size, self.size), dtype=str)
        grid[:] = "."
        grid[self.start_pos[0], self.start_pos[1]] = "S"
        grid[self.goal_pos[0], self.goal_pos[1]] = "G"
        grid[self.obstacle_pos[0], self.obstacle_pos[1]] = "X"
        grid[self.state[0], self.state[1]] = "A"
        
        print("\n".join([" ".join(row) for row in grid]))
        print("-" * 10)

if __name__ == "__main__":
    # Test the environment
    print("--- Testing GridWorld Environment ---")
    env = GridWorldEnv()
    state, _ = env.reset()
    env.render()
    
    # Take a few steps (Path to goal avoiding obstacle at 1,1)
    actions = [1, 3, 1, 3] # Down, Right, Down, Right
    for a in actions:
        state, reward, done, _, _ = env.step(a)
        print(f"Action: {a}, State: {state}, Reward: {reward}, Done: {done}")
        env.render()
        if done:
            break
