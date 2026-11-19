#!/usr/bin/env python3
"""
Generate Diagram 1: System Overview (Headless)
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(18, 14))
ax.set_xlim(0, 18)
ax.set_ylim(0, 14)
ax.axis('off')

# Colors
c1, c2, c3, c_comp, c_met, c_arr = '#C8E6C9', '#81D4FA', '#FFE082', '#E1BEE7', '#FFCCBC', '#424242'

def box(x, y, w, h, t, c, s=10):
    b = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.1",
                       edgecolor='#333', facecolor=c, linewidth=2.5)
    ax.add_patch(b)
    ax.text(x, y, t, ha='center', va='center', fontsize=s, fontweight='bold')

def arrow(x1, y1, x2, y2, l='', curved=False):
    if curved:
        a = FancyArrowPatch((x1, y1), (x2, y2), connectionstyle="arc3,rad=0.4",
                           arrowstyle='->', mutation_scale=35, linewidth=3, color=c_arr)
    else:
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->',
                           mutation_scale=35, linewidth=3, color=c_arr)
    ax.add_patch(a)
    if l:
        ax.text((x1+x2)/2+0.3, (y1+y2)/2+0.3, l, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Title
ax.text(9, 13.5, 'BALL-IN-CUP LEARNING PIPELINE', fontsize=20, fontweight='bold', ha='center')
ax.text(9, 13, 'From Hand-Coded FSM through Training to Evaluation', fontsize=12, ha='center', style='italic')

# PHASE 1
ax.text(0.5, 12.3, 'PHASE 1: DATA GENERATION', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=c1, alpha=0.7, edgecolor='#2E7D32', linewidth=2))

box(3, 11.2, 4, 1.5, 'HAND-CODED FSM\nREACH->ASSESS->PUSH->VERIFY\n100% Success', c1, 11)
box(9, 11.2, 3.5, 1.5, 'MuJoCo Simulation\n500 Hz Control\nRealistic Physics', c1, 11)
box(15, 11.2, 3, 1.5, 'Run 300 Trials\n3 Language Variants\n95% Success Rate', c1, 11)

arrow(5, 11.2, 7.5, 11.2)
arrow(10.75, 11.2, 13.5, 11.2)

box(9, 9.3, 6, 1.3, 'DATASET: 300 Rollouts (6GB)\nImages + Joint State + Force + Targets', '#D1C4E9', 11)
arrow(9, 10.65, 9, 9.95)

# PHASE 2
ax.text(0.5, 8.8, 'PHASE 2: TRAINING', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=c2, alpha=0.7, edgecolor='#0277BD', linewidth=2))

box(3, 7.8, 2.5, 1, 'Vision Encoder\n(ViT)', c_comp, 10)
box(5.5, 7.8, 2.5, 1, 'Proprioception\nEncoder (MLP)', c_comp, 10)
box(8, 7.8, 2.5, 1, 'Language Encoder\n(CLIP)', c_comp, 10)

box(6.5, 6.5, 4, 1, 'Multimodal Fusion Transformer\n(Cross-attention)', c2, 10)
arrow(3, 7.3, 5, 7)
arrow(5.5, 7.3, 6, 7)
arrow(8, 7.3, 7, 7)

box(6.5, 5.2, 3, 0.8, 'Output: 7D Joint Positions', c2, 10)
arrow(6.5, 6, 6.5, 5.6)

box(11, 7.5, 3, 1.5, 'MSE Loss Training\n20 Epochs\nAdam Optimizer', c2, 10)
box(11, 5.5, 3, 1.5, 'Convergence\nLoss: 0.15->0.02\nError: ~0.05 rad', c_met, 10)

box(15, 6.3, 2.5, 1.2, 'Best Model\nCheckpoint\npi05_best.pt', '#F8BBD0', 10)

arrow(9, 9.3, 6.5, 7.3, 'Load Data')
arrow(8, 5.2, 13.75, 6.3)

# PHASE 3
ax.text(0.5, 4.8, 'PHASE 3: EVALUATION', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=c3, alpha=0.7, edgecolor='#F57F17', linewidth=2))

box(2.5, 3.8, 2.3, 1, 'Hand-Coded\nBaseline\n100% Success', c3, 10)
box(5.5, 3.8, 2.3, 1, 'Learned Pi05\nPolicy\n85-95% Success', c3, 10)
box(9, 3.8, 3.5, 1, 'Test Scenarios\nSeen | Unseen | Language', c_met, 10)

arrow(15, 5.7, 8, 4.3, 'Load Model')

# Metrics
metrics = [('Success\n88% vs 100%', 1.5), ('Time\n4.8 vs 4.5s', 4),
           ('Force\nLearning', 6.5), ('Generalization', 9), ('Failures', 11.5)]
for txt, x in metrics:
    box(x, 2.2, 1.8, 0.8, txt, c_met, 8)

# Results
box(9, 0.8, 7, 0.9, 'RESULTS: Pi05 achieves 88% success | Learns language conditioning | Generalizes', '#C5CAE9', 11)

# Main flow
arrow(9, 8.95, 9, 8.2)
arrow(9, 7, 9, 5.8)
arrow(9, 4.3, 9, 2.7)

# Save
plt.savefig('diagram_1_system_overview.jpg', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: diagram_1_system_overview.jpg (18x14 inches, 300 DPI)")
plt.close()
