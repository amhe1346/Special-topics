# π₀.₅ Learning Project: MuJoCo Position Control

Integrated Vision-Language-Action (VLA) learning pipeline combining:
- **Hand-coded controller** (Phase 1 PD) for baseline data collection
- **Data generation** from MuJoCo simulator
- **π₀.₅ imitation learning** to replace hand-coded control
- **Benchmarking** against baseline

## Quick Start

### 1. Setup Environment

```bash
cd mujoco_force_control

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Collect Training Data (Phase 4.5)

Run the PD controller in MuJoCo to generate training data:

```bash
# Collect 300 rollouts (takes ~6-8 hours on CPU, ~2-3 hours on 8-core parallel)
python scripts/collect_data.py --num_rollouts 300 --output_dir dataset/rollouts

# Or quick test with 50 rollouts
python scripts/collect_data.py --num_rollouts 50 --output_dir dataset/rollouts_test
```

**What gets saved:**
- `dataset/rollouts/rollout_000000.h5` through `rollout_000299.h5`
- Each HDF5 file contains:
  - RGB images (T, 224, 224, 3)
  - Joint positions & velocities (T, 7)
  - End-effector force (T, 3)
  - Motor current (T, 7)
  - **Target joint positions** (ground truth from PD controller) ← used for training
  - Language labels ("move to home", "oscillate smoothly", etc.)

### 3. Train π₀.₅ (Phase 5.5)

Fine-tune the π₀.₅ model on collected data using imitation learning:

```bash
# Train on full dataset (20 epochs, ~30 min on GPU, ~2-3 hours on CPU)
python scripts/train_vla.py \
    --data_dir dataset/rollouts \
    --output_dir checkpoints \
    --num_epochs 20 \
    --batch_size 4 \
    --device cpu  # Use 'cuda' if available

# Quick test with smaller batch
python scripts/train_vla.py \
    --data_dir dataset/rollouts_test \
    --num_epochs 5 \
    --batch_size 2 \
    --device cpu
```

**Output:**
- `checkpoints/pi05_best.pt` — Best model checkpoint (based on validation loss)
- Training curves printed to console (train/test loss, L2 error in radians)

### 4. Evaluate π₀.₅ vs. Baseline (Phase 5.6)

Compare learned policy against PD baseline:

```bash
# Evaluate on 5 trials each
python scripts/evaluate_vla.py \
    --model_path checkpoints/pi05_best.pt \
    --num_trials 5
```

**Metrics computed:**
- **RMSE** (Root Mean Squared Error): Position tracking accuracy
- **Max Error**: Worst-case tracking error
- **Settling Time**: How fast the arm reaches target (in steps)

**Success criteria:**
- π₀.₅ RMSE within 20% of PD baseline → ✓ Good
- π₀.₅ RMSE within 50% of baseline → ~ Acceptable for first training
- π₀.₅ RMSE > 50% of baseline → ✗ More training needed

## Architecture

### Environment & Control

```
ForceControlEnv (envs/base_env.py)
├─ Loads Panda MJCF (envs/assets/panda/panda.xml)
├─ 7-DOF arm with sensors:
│  ├─ Joint position/velocity (encoder)
│  ├─ End-effector force (load cell)
│  └─ Motor current (actuator force)
└─ Supports RGB rendering (224×224)

PDController (controllers/pd_position_controller.py)
├─ Proportional-Derivative control
├─ Maps target positions → joint torques
└─ Used as baseline for comparison
```

### Data Collection

```
collect_data.py
├─ Runs PD controller on 3 scenarios:
│  ├─ Sinusoid (oscillating motion)
│  ├─ Step (move to fixed target)
│  └─ Random walk (explore workspace)
├─ Records: images, joint state, target positions
├─ Generates language labels
└─ Saves to HDF5 for efficient training
```

### π₀.₅ Model

```
SimplePi05 (vla/pi05_wrapper.py)
├─ Vision encoder: CNN on RGB images
├─ Proprioceptive encoder: Dense layers on joint/force state
├─ Language encoder: Dense embedding layer
├─ Fusion: Concatenate + fuse all modalities
└─ Output head: Predict 7-DOF joint positions

Training Loop (scripts/train_vla.py)
├─ Loss: MSE(predicted_positions, actual_positions)
├─ Optimizer: Adam (lr=1e-4)
├─ Data: Train/test split 80/20
└─ Best model saved based on validation loss
```

## File Structure

```
mujoco_force_control/
├── envs/
│   ├── base_env.py          ← ForceControlEnv wrapper
│   └── assets/panda/
│       └── panda.xml        ← 7-DOF arm MJCF model
├── controllers/
│   └── pd_position_controller.py  ← PD control law
├── sensors/
├── data/
│   ├── language_sampler.py  ← Language label generation
│   └── dataset.py           ← HDF5 I/O + train/test split
├── vla/
│   └── pi05_wrapper.py      ← π₀.₅ model & inference
├── scripts/
│   ├── collect_data.py      ← Data collection entry point
│   ├── train_vla.py         ← Training entry point
│   └── evaluate_vla.py      ← Evaluation entry point
├── checkpoints/             ← Saved models
├── dataset/
│   └── rollouts/            ← HDF5 training data
├── requirements.txt
├── setup.py
└── README.md
```

## Troubleshooting

### "Model not found" error
- Make sure `envs/assets/panda/panda.xml` exists
- Check the file path in `base_env.py`

### Slow data collection
- CPU rendering is slow; use multiple processes (see `collect_data.py` with Ray)
- Or reduce `rollout_steps` (default 500) to collect shorter trajectories

### Out of memory during training
- Reduce `batch_size` (default 4 → try 2)
- Reduce image resolution (224×224 → 128×128)
- Use fewer epochs

### π₀.₅ training loss not decreasing
- Check that data is being loaded: print first batch in `train_vla.py`
- Increase learning rate slightly (1e-4 → 5e-4)
- Verify HDF5 files contain expected data: use `h5py` to inspect

### Evaluation showing poor π₀.₅ performance
- More training epochs (try 50-100)
- Larger dataset (collect 500+ rollouts)
- Check that language embeddings are being passed correctly

## Extending the Pipeline

### Add more scenarios
Edit `generate_target_trajectory()` in `collect_data.py`:
```python
elif scenario == 'lissajous':
    targets[:, 0] = 0.5 * np.sin(2 * np.pi * 0.3 * np.arange(num_steps) * dt)
    targets[:, 1] = 0.5 * np.cos(2 * np.pi * 0.4 * np.arange(num_steps) * dt)
```

### Use real π₀.₅ model
- Clone [OpenPi repository](https://github.com/Physical-Intelligence/openpi)
- Load official model in `pi05_wrapper.py`
- Fine-tune on collected data (as shown in `train_vla.py`)

### Add force control objectives
- Modify reward/loss in `train_vla.py` to include force tracking
- Add end-effector force targets to language labels
- Collect data with force-controlled scenarios

## References

- **π₀.₅ Paper**: [Physical Intelligence](https://www.pi.website/download/pi05.pdf)
- **OpenPi Repository**: https://github.com/Physical-Intelligence/openpi
- **MuJoCo Docs**: https://mujoco.readthedocs.io/
- **MuJoCo Menagerie** (robot models): https://github.com/google-deepmind/mujoco_menagerie

## Timeline & Compute

| Phase | Task | Time (CPU) | Time (GPU) |
|-------|------|-----------|-----------|
| **4.5** | Collect 300 rollouts | 8-12 hrs | 3-6 hrs |
| **5.5** | Train π₀.₅ (20 epochs) | 2-3 hrs | 30 min |
| **5.6** | Evaluate (5 trials) | 30 min | 10 min |
| **Total** | Full pipeline | ~11-16 hrs | ~4-7 hrs |

**Recommendation:** Start with 50 rollouts + 10 epochs for quick validation (1-2 hrs), then scale to full dataset.

---

**Next Steps:**
1. Collect data: `python scripts/collect_data.py --num_rollouts 50`
2. Train: `python scripts/train_vla.py --data_dir dataset/rollouts`
3. Evaluate: `python scripts/evaluate_vla.py`
4. Iterate on architecture/hyperparameters based on results
