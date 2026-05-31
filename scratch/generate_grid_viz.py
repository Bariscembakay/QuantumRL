import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def draw_grid(ax, size, title, start, goal, obstacle, path):
    # Set limits and grid lines
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5) # Row 0 is at the top
    
    # Draw cells
    for r in range(size):
        for c in range(size):
            # Check cell type for coloring
            if [r, c] == start:
                color = '#E8F0F8' # Soft Blue
                label = 'Start\n(0,0)'
                text_color = '#1f77b4'
            elif [r, c] == goal:
                color = '#E8F8F0' # Soft Green
                label = f'Goal\n({size-1},{size-1})'
                text_color = '#2ca02c'
            elif [r, c] == obstacle:
                color = '#FDE8E8' # Soft Red
                label = 'Obstacle\n(1,1)'
                text_color = '#d62728'
            else:
                color = '#FFFFFF'
                label = f'({r},{c})'
                text_color = '#7f7f7f'
                
            # Draw cell box
            rect = plt.Rectangle((c - 0.5, r - 0.5), 1, 1, 
                                 facecolor=color, edgecolor='#CCCCCC', linewidth=1.5)
            ax.add_patch(rect)
            
            # Add label
            ax.text(c, r, label, ha='center', va='center', 
                     fontsize=9, fontweight='bold' if [r, c] in [start, goal, obstacle] else 'normal',
                     color=text_color)
            
    # Draw path arrows
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        # Calculate arrow start and direction
        dr = r2 - r1
        dc = c2 - c1
        # Shorter arrow to fit inside cell
        ax.annotate('', xy=(c2, r2), xytext=(c1, r1),
                    arrowprops=dict(arrowstyle="->", color='#8B0000', lw=2.5, ls='-',
                                    connectionstyle="arc3,rad=0.1"))

    # Style the axes
    ax.set_xticks(np.arange(size))
    ax.set_yticks(np.arange(size))
    ax.set_xticklabels([f"Col {i}" for i in range(size)], fontsize=9)
    ax.set_yticklabels([f"Row {i}" for i in range(size)], fontsize=9)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15, color='#333333')
    ax.grid(False)
    ax.set_aspect('equal')

# Define paths
# 3x3 optimal path: Down, Down, Right, Right
path_3x3 = [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]]

# 4x4 optimal path: Down, Down, Down, Right, Right, Right (avoiding Obstacle at 1,1)
path_4x4 = [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1], [3, 2], [3, 3]]

# 1. Create side-by-side figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
draw_grid(ax1, 3, "3x3 Grid World Navigation", [0,0], [2,2], [1,1], path_3x3)
draw_grid(ax2, 4, "4x4 Grid World Navigation", [0,0], [3,3], [1,1], path_4x4)
plt.tight_layout()
plt.savefig('results/grid_world_visualization.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("Saved side-by-side visualization to results/grid_world_visualization.png")

# 2. Create standalone 3x3 figure
fig3, ax3 = plt.subplots(figsize=(5, 5))
draw_grid(ax3, 3, "3x3 Grid World Navigation", [0,0], [2,2], [1,1], path_3x3)
plt.tight_layout()
plt.savefig('results/grid_world_3x3.png', dpi=300, bbox_inches='tight')
plt.close(fig3)
print("Saved standalone 3x3 visualization to results/grid_world_3x3.png")

# 3. Create standalone 4x4 figure
fig4, ax4 = plt.subplots(figsize=(5, 5))
draw_grid(ax4, 4, "4x4 Grid World Navigation", [0,0], [3,3], [1,1], path_4x4)
plt.tight_layout()
plt.savefig('results/grid_world_4x4.png', dpi=300, bbox_inches='tight')
plt.close(fig4)
print("Saved standalone 4x4 visualization to results/grid_world_4x4.png")
