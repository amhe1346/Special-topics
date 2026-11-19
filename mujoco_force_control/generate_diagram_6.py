#!/usr/bin/env python3
"""
Generate Diagram 6: Information Flow - From Observation to Action
Creates a visual representation of the control loop with boxes and arrows
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Color scheme
color_sense = '#E8F4F8'
color_process = '#B3E5FC'
color_compute = '#81D4FA'
color_act = '#4FC3F7'
color_loop = '#FFE082'
color_arrow = '#424242'

def draw_box(ax, x, y, width, height, text, color, fontsize=10, fontweight='normal'):
    """Draw a rounded box with text"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.1",
                         edgecolor='black', facecolor=color,
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight=fontweight, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label='', curved=False):
    """Draw an arrow between two points"""
    if curved:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                              connectionstyle="arc3,rad=0.3",
                              arrowstyle='->', mutation_scale=30,
                              linewidth=2.5, color=color_arrow)
    else:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle='->', mutation_scale=30,
                              linewidth=2.5, color=color_arrow)
    ax.add_patch(arrow)

    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Title
ax.text(5, 13.5, 'Diagram 6: Real-Time Control Loop (500 Hz)',
        fontsize=18, fontweight='bold', ha='center')
ax.text(5, 13, 'From Observation to Action at 2ms Timestep',
        fontsize=12, ha='center', style='italic', color='#666')

# ============================================================================
# CYCLE OVERVIEW
# ============================================================================
ax.text(0.5, 12.2, 'CYCLE k (at time t = k × 2ms):',
        fontsize=12, fontweight='bold')

# ============================================================================
# STEP 1: SENSE
# ============================================================================
y_pos = 11.2
ax.text(0.5, y_pos + 0.5, 'STEP 1: SENSE', fontsize=11, fontweight='bold',
        color='#0277BD')

draw_box(ax, 1.2, y_pos, 1.8, 0.7, 'RGB Camera\n224×224\nCapture', color_sense)
draw_box(ax, 3.2, y_pos, 1.8, 0.7, 'Joint Encoders\nRead q(7)', color_sense)
draw_box(ax, 5.2, y_pos, 1.8, 0.7, 'Velocity\nDifferentiate', color_sense)
draw_box(ax, 7.2, y_pos, 1.8, 0.7, 'F/T Sensor\nRead F(6)', color_sense)

ax.text(9.2, y_pos, '~10ms\nlatency', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#FFCDD2'))

draw_arrow(ax, 1.2, y_pos-0.4, 1.2, y_pos-1)
draw_arrow(ax, 3.2, y_pos-0.4, 3.2, y_pos-1)
draw_arrow(ax, 5.2, y_pos-0.4, 5.2, y_pos-1)
draw_arrow(ax, 7.2, y_pos-0.4, 7.2, y_pos-1)

# ============================================================================
# STEP 2: REPRESENT
# ============================================================================
y_pos = 9.2
ax.text(0.5, y_pos + 0.7, 'STEP 2: REPRESENT (in π₀.₅ input space)',
        fontsize=11, fontweight='bold', color='#0277BD')

draw_box(ax, 2, y_pos, 3.2, 1.2,
         'images[k] = Normalize(RGB) / 255\nSize: (3, 224, 224)\nLatency: included in SENSE',
         color_process, fontsize=9)

draw_box(ax, 5.5, y_pos, 3.2, 1.2,
         'proprio[k] = [q_pos, q_vel, F_ee]\nSize: (20,)\nq ∈ [-π,π], F ∈ [-100,100]N',
         color_process, fontsize=9)

draw_box(ax, 8.5, y_pos, 1.2, 1.2,
         'language\nCLIP emb.\nSize: (256,)',
         color_process, fontsize=9)

draw_arrow(ax, 5, y_pos-0.7, 5, y_pos-1.3)

# ============================================================================
# STEP 3: FORWARD PASS
# ============================================================================
y_pos = 7.2
ax.text(0.5, y_pos + 0.5, 'STEP 3: FORWARD PASS (π₀.₅ Inference)',
        fontsize=11, fontweight='bold', color='#0277BD')

draw_box(ax, 5, y_pos, 6, 1.2,
         'target_pos[k] = π₀.₅(images[k], proprio[k], language)\n\n' +
         'Processing: Vision ViT + Proprioception MLP + Language CLIP + Fusion Transformer\n' +
         'Output: 7D joint position targets ∈ ℝ⁷',
         color_compute, fontsize=9)

ax.text(9.2, y_pos, 'Latency:\n~50ms (CPU)\n~10ms (GPU)',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='#FFCDD2'))

draw_arrow(ax, 5, y_pos-0.7, 5, y_pos-1.3)

# ============================================================================
# STEP 4: INNER CONTROL LOOP
# ============================================================================
y_pos = 5.2
ax.text(0.5, y_pos + 0.7, 'STEP 4: INNER CONTROL LOOP (PD Position Servo)',
        fontsize=11, fontweight='bold', color='#0277BD')

# Error calculation
draw_box(ax, 2, y_pos, 2.8, 1,
         'error_pos = target_pos[k] - q_actual[k]\nerror_vel = 0 - q_vel_actual[k]',
         color_act, fontsize=9)

# PD calculation
draw_box(ax, 5.5, y_pos, 2.8, 1,
         'τ[k] = Kp ⊙ error_pos + Kd ⊙ error_vel\nKp = 100, Kd = 20',
         color_act, fontsize=9)

# Saturation
draw_box(ax, 8.5, y_pos, 1.2, 1,
         'Saturate\nτ ∈ [-87,87]\nN⋅m',
         color_act, fontsize=9)

draw_arrow(ax, 2, y_pos-0.6, 5.5, y_pos-0.6)
draw_arrow(ax, 6.4, y_pos-0.6, 8.5, y_pos-0.6)
draw_arrow(ax, 8.5, y_pos-0.6, 8.5, y_pos-1.2)

ax.text(9.2, y_pos, '<1ms\nlatency',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='#C8E6C9'))

# ============================================================================
# STEP 5: ACTUATE
# ============================================================================
y_pos = 3.2
ax.text(0.5, y_pos + 0.5, 'STEP 5: ACTUATE',
        fontsize=11, fontweight='bold', color='#0277BD')

draw_box(ax, 2.5, y_pos, 2.5, 0.8,
         'Command Motors\nτ[k] → drivers',
         color_loop, fontsize=9)

draw_box(ax, 5.5, y_pos, 2.5, 0.8,
         'Physics Integration\nq[k+1] = f(q[k], τ[k], dt)',
         color_loop, fontsize=9)

draw_box(ax, 8, y_pos, 1.8, 0.8,
         'Motor Electronics\n~5ms delay',
         color_loop, fontsize=9)

draw_arrow(ax, 2.5, y_pos-0.5, 5.5, y_pos-0.5)
draw_arrow(ax, 6.75, y_pos-0.5, 8, y_pos-0.5)

# ============================================================================
# TOTAL LATENCY SUMMARY
# ============================================================================
y_pos = 1.8
ax.text(0.5, y_pos, 'TOTAL LATENCY BUDGET (2ms per control cycle):',
        fontsize=11, fontweight='bold')

latency_data = [
    ('Sensing', '~10ms', '#E8F4F8'),
    ('Inference', '~50ms (CPU)\n~10ms (GPU)', '#81D4FA'),
    ('PD Loop', '<1ms', '#C8E6C9'),
    ('Actuate', '~5ms', '#FFE082'),
]

x_start = 0.8
for i, (name, latency, color) in enumerate(latency_data):
    x = x_start + i * 2.3
    draw_box(ax, x, y_pos-0.8, 1.8, 0.6, f'{name}\n{latency}', color, fontsize=8)

ax.text(5, y_pos-1.8, '⚠ Bottleneck: Inference latency >> 2ms control cycle',
        fontsize=10, fontweight='bold', color='#D32F2F')
ax.text(5, y_pos-2.2, '✓ Solution: Run inference asynchronously, use previous prediction while new one computes',
        fontsize=10, color='#388E3C')

# ============================================================================
# FINAL NOTE
# ============================================================================
y_pos = 0.1
ax.text(5, y_pos, 'Total Effective Latency: ~60-70ms (Acceptable: 60ms / 2000ms task = 2.5% of task time)',
        fontsize=10, ha='center', style='italic',
        bbox=dict(boxstyle='round', facecolor='#E1BEE7', alpha=0.8))

plt.tight_layout()
plt.savefig('diagram_6_control_loop.jpg', dpi=300, bbox_inches='tight', facecolor='white')
print("[OK] Saved: diagram_6_control_loop.jpg")
print("  Location: C:\\Users\\aehee\\OneDrive - UCB-O365\\HIRO\\Special topics\\mujoco_force_control\\")
print("  Size: 16x12 inches @ 300 DPI")
print("  Format: JPEG (high quality)")

plt.show()
