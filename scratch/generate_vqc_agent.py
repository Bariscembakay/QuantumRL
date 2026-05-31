import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Setup figure and axis with extra width and height for breathing room
fig, ax = plt.subplots(figsize=(13.5, 7))
ax.set_xlim(0, 13.5)
ax.set_ylim(0.5, 6.5)
ax.axis('off')

# Helper to draw a box
def draw_box(ax, x, y, w, h, title, subtitle="", bg_color='#F5F5F5', border_color='#CCCCCC', lw=2):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                  facecolor=bg_color, edgecolor=border_color, linewidth=lw)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + (0.25 if subtitle else 0), title, 
            ha='center', va='center', fontsize=10, fontweight='bold', color='#1a1a1a')
    if subtitle:
        ax.text(x + w/2, y + h/2 - 0.25, subtitle, 
                ha='center', va='center', fontsize=8.5, color='#555555', linespacing=1.3)

# 1. Inputs (x, y)
ax.text(0.4, 4.0, "$x$ (row)", ha='left', va='center', fontsize=11, fontweight='bold', color='#1f77b4')
ax.text(0.4, 2.0, "$y$ (col)", ha='left', va='center', fontsize=11, fontweight='bold', color='#1f77b4')

# 2. Scaling block (positioned at x=1.3, width=1.6)
draw_box(ax, 1.3, 1.2, 1.6, 3.6, "State Scaling", r"$\theta = \frac{\pi}{N-1} \cdot s$", bg_color='#E8F0F8', border_color='#1f77b4')
ax.annotate('', xy=(1.2, 4.0), xytext=(0.9, 4.0), arrowprops=dict(arrowstyle="->", color='#1f77b4', lw=1.5))
ax.annotate('', xy=(1.2, 2.0), xytext=(0.9, 2.0), arrowprops=dict(arrowstyle="->", color='#1f77b4', lw=1.5))

# 3. Quantum Circuit Border / Background (x=3.4 to x=9.1)
qc_bg = patches.FancyBboxPatch((3.4, 0.8), 5.7, 4.4, boxstyle="round,pad=0.15",
                               facecolor='#FAFAFA', edgecolor='#8B0000', linewidth=1.5, ls='--')
ax.add_patch(qc_bg)
ax.text(3.6, 5.0, "Variational Quantum Circuit (VQC)", fontsize=11, fontweight='bold', color='#8B0000')

# Qubit lines inside VQC
y_qubits = [1.5, 2.3, 3.1, 3.9]
for i, y in enumerate(y_qubits):
    ax.plot([3.6, 8.8], [y, y], color='#888888', lw=1.5, zorder=1)
    ax.text(3.5, y, f"$q_{i}$", ha='right', va='center', fontsize=10, fontweight='bold')

# Angle Encoding gates (Ry) (x=3.9 to x=4.7)
draw_box(ax, 3.9, 3.6, 0.8, 0.6, r"$R_y(x_s)$", bg_color='#D1E8E2', border_color='#2c3e50', lw=1)
draw_box(ax, 3.9, 2.8, 0.8, 0.6, r"$R_y(y_s)$", bg_color='#D1E8E2', border_color='#2c3e50', lw=1)
draw_box(ax, 3.9, 2.0, 0.8, 0.6, r"$R_y(x_s)$", bg_color='#D1E8E2', border_color='#2c3e50', lw=1)
draw_box(ax, 3.9, 1.2, 0.8, 0.6, r"$R_y(y_s)$", bg_color='#D1E8E2', border_color='#2c3e50', lw=1)

# Connect scaling output to Ry inputs (x=2.9 to x=3.8)
ax.annotate('', xy=(3.8, 3.9), xytext=(3.0, 3.9), arrowprops=dict(arrowstyle="->", color='#1f77b4', lw=1.5))
ax.annotate('', xy=(3.8, 3.1), xytext=(3.0, 1.9), arrowprops=dict(arrowstyle="->", color='#1f77b4', lw=1.5))
ax.annotate('', xy=(3.8, 2.3), xytext=(3.0, 3.9), arrowprops=dict(arrowstyle="->", color='#1f77b4', lw=1.5))
ax.annotate('', xy=(3.8, 1.5), xytext=(3.0, 1.9), arrowprops=dict(arrowstyle="->", color='#1f77b4', lw=1.5))

# Barrier line (x=5.0)
ax.plot([5.0, 5.0], [1.0, 4.4], color='#555555', ls='--', lw=1.5, zorder=2)
ax.text(5.0, 4.5, "Barrier", ha='center', va='bottom', fontsize=8, color='#555555', style='italic')

# Ansatz Block (RealAmplitudes reps=3) (x=5.3 to x=7.5)
draw_box(ax, 5.3, 1.1, 2.2, 3.6, "RealAmplitudes Ansatz", "reps=3, linear CNOT\n(16 Trainable Angles)", bg_color='#E8EAF6', border_color='#3f51b5', lw=1.5)

# Measurement block (x=7.9 to x=8.5)
for y in y_qubits:
    rect = patches.Rectangle((7.9, y - 0.25), 0.6, 0.5, facecolor='#E0E0E0', edgecolor='#555555', zorder=3)
    ax.add_patch(rect)
    ax.plot([7.95, 8.25], [y - 0.15, y + 0.15], color='#333333', lw=1.5, zorder=4)
    ax.plot([7.95, 8.45], [y - 0.2, y - 0.2], color='#333333', lw=1.5, zorder=4)

# 4. Measurement Expectation Labels (x=8.9, centered to avoid arrow overlap)
for y in y_qubits:
    ax.text(8.9, y, r"$\langle Z \rangle$", ha='center', va='center', fontsize=9, color='#2ca02c', fontweight='bold')

# Connect Qubits to Expectation Labels, and Labels to Linear Layer
for y in y_qubits:
    # Qubit to label
    ax.plot([8.5, 8.7], [y, y], color='#888888', lw=1.5, zorder=1)
    # Label to Linear Layer
    ax.annotate('', xy=(9.7, y), xytext=(9.1, y), arrowprops=dict(arrowstyle="->", color='#2ca02c', lw=1.5))

# 5. Classical Post-processing Box (x=9.8 to x=11.6, height extended to 3.9)
draw_box(ax, 9.8, 1.0, 1.8, 3.9, "Linear Layer", "$W \\in \\mathbb{R}^{4 \\times 4}$\n$b \\in \\mathbb{R}^4$\n\n(20 Trainable\nWeights & Biases)", bg_color='#FCF3CF', border_color='#F1C40F')

# 6. Output Q-Values (flowing directly from the Linear Layer output)
y_out = [1.5, 2.5, 3.5, 4.5]
actions = ["Up", "Down", "Left", "Right"]
for i, y in enumerate(y_out):
    ax.annotate('', xy=(12.2, y), xytext=(11.7, y), arrowprops=dict(arrowstyle="->", color='#1a1a1a', lw=1.5))
    ax.text(12.3, y, f"$Q(s, \\text{{{actions[i]}}})$", ha='left', va='center', fontsize=10, fontweight='bold', color='#1a1a1a')

plt.tight_layout()
plt.savefig('results/quantum_vqc_agent.png', dpi=300, bbox_inches='tight')
print("Saved quantum VQC agent structure to results/quantum_vqc_agent.png")
