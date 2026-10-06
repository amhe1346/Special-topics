#!/usr/bin/env python3
"""
Generate Diagram 1: End-to-End System Architecture
From raw FSM through training to evaluation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(18, 14))
ax.set_xlim(0, 18)
ax.set_ylim(0, 14)
ax.axis('off')

# Color scheme
color_phase1 = '#C8E6C9'
color_phase2 = '#81D4FA'
color_phase3 = '#FFE082'
color_component = '#E1BEE7'
color_metric = '#FFCCBC'
color_arrow = '#424242'

def draw_box(ax, x, y, width, height, text, color, fontsize=10, fontweight='normal', bold_title=False):
    """Draw a rounded box with text"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.1",
                         edgecolor='#333333', facecolor=color,
                         linewidth=2.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight=fontweight, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label='', curved=False, style='->'):
    """Draw an arrow between two points"""
    if curved:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                              connectionstyle="arc3,rad=0.4",
                              arrowstyle=style, mutation_scale=35,
                              linewidth=3, color=color_arrow)
    else:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle=style, mutation_scale=35,
                              linewidth=3, color=color_arrow)
    ax.add_patch(arrow)

    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        offset_x, offset_y = 0.3, 0.3
        if abs(x2 - x1) < 0.5:  # Vertical arrow
            offset_x = 0.5
        ax.text(mid_x + offset_x, mid_y + offset_y, label, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#666'),
                fontweight='bold')

# ============================================================================
# TITLE
# ============================================================================
ax.text(9, 13.5, 'BALL-IN-CUP LEARNING PIPELINE',
        fontsize=20, fontweight='bold', ha='center')
ax.text(9, 13, 'From Hand-Coded FSM through Training to Evaluation',
        fontsize=12, ha='center', style='italic', color='#555')

# ============================================================================
# PHASE 1: DATA GENERATION
# ============================================================================
ax.text(0.5, 12.3, 'PHASE 1: DATA GENERATION', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=color_phase1, alpha=0.7, edgecolor='#2E7D32', linewidth=2))

# FSM Controller Box
draw_box(ax, 3, 11.2, 4, 1.5,
         'HAND-CODED FSM\nREACH -> ASSESS -> PUSH -> VERIFY\n100% Success Rate\nPerfect Ground Truth Labels',
         color_phase1, fontsize=11, fontweight='bold')

# MuJoCo Simulation
draw_box(ax, 9, 11.2, 3.5, 1.5,
         'MuJoCo Simulation\n500 Hz Control Loop\nRealistic Physics\nClean Sensors',
         color_phase1, fontsize=11, fontweight='bold')

# 300 Trials
draw_box(ax, 15, 11.2, 3, 1.5,
         'Run 300 Trials\n3 Language Variants\nRandom Configurations\n95% Success Rate',
         color_phase1, fontsize=11, fontweight='bold')

draw_arrow(ax, 5, 11.2, 7.5, 11.2)
draw_arrow(ax, 10.75, 11.2, 13.5, 11.2)

# Dataset output
draw_box(ax, 9, 9.3, 6, 1.3,
         'DATASET (300 Rollouts)\nImages (224×224 RGB) | Joint State | End-Effector Force\nTarget Positions (Ground Truth) | Language Tags\nSize: ~6 GB | Success Rate: 95%',
         '#D1C4E9', fontsize=11, fontweight='bold')

draw_arrow(ax, 9, 10.65, 9, 9.95)

# ============================================================================
# PHASE 2: TRAINING
# ============================================================================
ax.text(0.5, 8.8, 'PHASE 2: TRAINING', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=color_phase2, alpha=0.7, edgecolor='#0277BD', linewidth=2))

# Pi 0.5 Model
draw_box(ax, 3, 7.8, 2.5, 1,
         'Vision Encoder\n(ViT)',
         color_component, fontsize=10, fontweight='bold')

draw_box(ax, 5.5, 7.8, 2.5, 1,
         'Proprioception Encoder\n(MLP)',
         color_component, fontsize=10, fontweight='bold')

draw_box(ax, 8, 7.8, 2.5, 1,
         'Language Encoder\n(CLIP)',
         color_component, fontsize=10, fontweight='bold')

# Fusion
draw_box(ax, 6.5, 6.5, 4, 1,
         'Multimodal Fusion Transformer\n(Cross-attention & Self-attention)',
         color_phase2, fontsize=10, fontweight='bold')

draw_arrow(ax, 3, 7.3, 5, 7)
draw_arrow(ax, 5.5, 7.3, 6, 7)
draw_arrow(ax, 8, 7.3, 7, 7)

# Output
draw_box(ax, 6.5, 5.2, 3, 0.8,
         'Output: 7D Joint Positions',
         color_phase2, fontsize=10, fontweight='bold')

draw_arrow(ax, 6.5, 6, 6.5, 5.6)

# Training metrics
draw_box(ax, 11, 7.5, 3, 1.5,
         'MSE Loss Training\n20 Epochs\nAdam Optimizer\nBatch Size 4',
         color_phase2, fontsize=10, fontweight='bold')

draw_box(ax, 11, 5.5, 3, 1.5,
         'Convergence\nLoss: 0.15 -> 0.02\nError: ~0.05 rad\nValidation Track',
         color_metric, fontsize=10, fontweight='bold')

draw_arrow(ax, 9, 9.3, 6.5, 7.3, 'Load Dataset')

# Checkpoint
draw_box(ax, 15, 6.3, 2.5, 1.2,
         'Best Model\nCheckpoint\npi05_best.pt\n250K params',
         '#F8BBD0', fontsize=10, fontweight='bold')

draw_arrow(ax, 8, 5.2, 13.75, 6.3)

# ============================================================================
# PHASE 3: EVALUATION
# ============================================================================
ax.text(0.5, 4.8, 'PHASE 3: EVALUATION', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=color_phase3, alpha=0.7, edgecolor='#F57F17', linewidth=2))

# Baseline
draw_box(ax, 2.5, 3.8, 2.3, 1,
         'Hand-Coded\nBaseline\n100% Success\n(Oracle)',
         color_phase3, fontsize=10, fontweight='bold')

# Learned Policy
draw_box(ax, 5.5, 3.8, 2.3, 1,
         'Learned Pi05\nPolicy\n85-95%\nSuccess',
         color_phase3, fontsize=10, fontweight='bold')

# Test Scenarios
draw_box(ax, 9, 3.8, 3.5, 1,
         'Test Scenarios\nSeen Configs | Unseen Positions\nLanguage Variants | Ball Sizes',
         color_metric, fontsize=10, fontweight='bold')

draw_arrow(ax, 15, 5.7, 8, 4.3, 'Load Model')

# Metrics
metrics_list = [
    ('Success Rate\n88% vs 100%', 1.5, 2.2),
    ('Completion Time\n4.8s vs 4.5s', 4, 2.2),
    ('Force Learning\nGently/Normal/Force', 6.5, 2.2),
    ('Generalization\nNew Positions', 9, 2.2),
    ('Failure Analysis\nUnderstand Modes', 11.5, 2.2),
]

for metric_text, x, y in metrics_list:
    draw_box(ax, x, y, 1.8, 0.8, metric_text, color_metric, fontsize=8)

# Results Summary
draw_box(ax, 9, 0.8, 7, 0.9,
         'RESULTS: Pi05 achieves 88% success (vs 100% baseline) | Learns language conditioning | Generalizes reasonably',
         '#C5CAE9', fontsize=11, fontweight='bold')

# ============================================================================
# INFORMATION FLOW
# ============================================================================
# Main vertical flow
draw_arrow(ax, 9, 8.95, 9, 8.2)
draw_arrow(ax, 9, 7, 9, 5.8)
draw_arrow(ax, 9, 4.3, 9, 2.7)

# Phase labels
phase_y = [12.8, 8.3, 4.3]
phase_colors = [color_phase1, color_phase2, color_phase3]

plt.tight_layout()
plt.savefig('diagram_1_system_overview.jpg', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("[OK] Saved: diagram_1_system_overview.jpg")
print("   Location: mujoco_force_control/")
print("   Size: 18x14 inches @ 300 DPI")
print("   Format: JPEG (high quality)")

# Also display
plt.show()
