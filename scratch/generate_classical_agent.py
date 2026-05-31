import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Setup figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0.0, 6.5)
ax.axis('off')

# Layer configurations
layer_x = [1.5, 4.0, 6.5, 9.0]
layer_names = [
    "Input Layer\n(State Space)",
    "Hidden Layer 1\n(64 Neurons, ReLU)",
    "Hidden Layer 2\n(64 Neurons, ReLU)",
    "Output Layer\n(Action Space)"
]

# Coordinates for node positions in each layer
# Input layer: 2 nodes
y_input = [2.5, 3.5]
# Hidden 1: Draw top 3 and bottom 3 nodes, with a gap for vertical ellipsis
y_hidden1 = [0.5, 1.25, 2.0, 4.0, 4.75, 5.5]
# Hidden 2: Same as Hidden 1
y_hidden2 = [0.5, 1.25, 2.0, 4.0, 4.75, 5.5]
# Output layer: 4 nodes
y_output = [1.5, 2.5, 3.5, 4.5]

all_y = [y_input, y_hidden1, y_hidden2, y_output]
node_colors = ['#1f77b4', '#7f7f7f', '#7f7f7f', '#2ca02c']

# 1. Draw connections (fully connected line patterns)
# Input to Hidden 1
for yi in y_input:
    for yh in y_hidden1:
        ax.plot([layer_x[0], layer_x[1]], [yi, yh], color='#EAEAEA', lw=0.8, zorder=1)

# Hidden 1 to Hidden 2
for yh1 in y_hidden1:
    for yh2 in y_hidden2:
        ax.plot([layer_x[1], layer_x[2]], [yh1, yh2], color='#EAEAEA', lw=0.8, zorder=1)

# Hidden 2 to Output
for yh2 in y_hidden2:
    for yo in y_output:
        ax.plot([layer_x[2], layer_x[3]], [yh2, yo], color='#EAEAEA', lw=0.8, zorder=1)

# 2. Draw nodes and Ellipses
for l in range(4):
    y_coords = all_y[l]
    for y in y_coords:
        circle = plt.Circle((layer_x[l], y), 0.18, facecolor=node_colors[l], edgecolor='#333333', linewidth=1.2, zorder=3)
        ax.add_patch(circle)
        
    # Draw vertical ellipses for hidden layers
    if l in [1, 2]:
        ax.text(layer_x[l], 3.0, r"$\vdots$", ha='center', va='center', fontsize=18, fontweight='bold', color='#555555', zorder=4)

    # Layer Header Text
    ax.text(layer_x[l], 6.0, layer_names[l], ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1a1a1a')

# 3. Label Specific Nodes
# Input State Labels
ax.text(layer_x[0] - 0.4, y_input[1], "x (row)", ha='right', va='center', fontsize=9, fontweight='semibold', color='#1f77b4')
ax.text(layer_x[0] - 0.4, y_input[0], "y (col)", ha='right', va='center', fontsize=9, fontweight='semibold', color='#1f77b4')

# Output Action Labels
actions = ["Up", "Down", "Left", "Right"]
for i, y in enumerate(y_output):
    ax.text(layer_x[3] + 0.4, y, f"Q(s, {actions[i]})", ha='left', va='center', fontsize=9, fontweight='semibold', color='#2ca02c')

# 4. Parameter Equations & Labels
# Input -> Hidden 1
ax.text((layer_x[0] + layer_x[1])/2, 4.8, 
        "$W_1 \\in \\mathbb{R}^{64 \\times 2}$\n$b_1 \\in \\mathbb{R}^{64}$\n\n$\\mathbf{192}$ params", 
        ha='center', va='center', fontsize=8.5, bbox=dict(facecolor='#FFFFFF', edgecolor='#CCCCCC', boxstyle='round,pad=0.4', alpha=0.9))

# Hidden 1 -> Hidden 2
ax.text((layer_x[1] + layer_x[2])/2, 4.8, 
        "$W_2 \\in \\mathbb{R}^{64 \\times 64}$\n$b_2 \\in \\mathbb{R}^{64}$\n\n$\\mathbf{4,160}$ params", 
        ha='center', va='center', fontsize=8.5, bbox=dict(facecolor='#FFFFFF', edgecolor='#CCCCCC', boxstyle='round,pad=0.4', alpha=0.9))

# Hidden 2 -> Output
ax.text((layer_x[2] + layer_x[3])/2, 4.8, 
        "$W_3 \\in \\mathbb{R}^{4 \\times 64}$\n$b_3 \\in \\mathbb{R}^{4}$\n\n$\\mathbf{260}$ params", 
        ha='center', va='center', fontsize=8.5, bbox=dict(facecolor='#FFFFFF', edgecolor='#CCCCCC', boxstyle='round,pad=0.4', alpha=0.9))
plt.tight_layout()
plt.savefig('results/classical_dqn_agent.png', dpi=300, bbox_inches='tight')
print("Saved classical DQN agent structure to results/classical_dqn_agent.png")
