# Quick Start Command Reference

## One-Command Test Run (10 min, CPU)

```bash
cd mujoco_force_control

# 1. Collect 10 test rollouts
python scripts/collect_data.py --num_rollouts 10 --output_dir dataset/test

# 2. Train for 3 epochs
python scripts/train_vla.py --data_dir dataset/test --num_epochs 3 --batch_size 2

# 3. Evaluate
python scripts/evaluate_vla.py --model_path checkpoints/pi05_best.pt --num_trials 2
```

## Full Run (6-16 hrs depending on CPU/GPU)

```bash
cd mujoco_force_control

# 1. Collect data (6-12 hrs CPU, 3-6 hrs GPU)
python scripts/collect_data.py --num_rollouts 300 --output_dir dataset/rollouts

# 2. Train (2-3 hrs CPU, 30 min GPU)
python scripts/train_vla.py \
    --data_dir dataset/rollouts \
    --num_epochs 20 \
    --batch_size 4 \
    --device cuda  # Use 'cpu' if no GPU

# 3. Evaluate (30 min CPU, 10 min GPU)
python scripts/evaluate_vla.py \
    --model_path checkpoints/pi05_best.pt \
    --num_trials 10
```

## Debugging

```bash
# Test environment loads correctly
python -c "from mujoco_force_control.envs.base_env import ForceControlEnv; env = ForceControlEnv(); env.reset(); print('✓ OK')"

# Test data collection generates valid HDF5
python -c "from mujoco_force_control.data.dataset import Dataset; d = Dataset('dataset/test'); print(f'✓ Loaded {len(d)} rollouts')"

# Test model loads and runs
python -c "from mujoco_force_control.vla.pi05_wrapper import Pi05Wrapper; m = Pi05Wrapper(); import numpy as np; print(m.predict(np.random.randn(1,1,3,224,224), np.random.randn(1,1,16), np.random.randn(1,256)).shape)"
```

## Key Files

| File | Purpose |
|------|---------|
| `envs/base_env.py` | MuJoCo environment wrapper |
| `controllers/pd_position_controller.py` | Baseline PD controller |
| `scripts/collect_data.py` | Generate training data |
| `vla/pi05_wrapper.py` | π₀.₅ model & inference |
| `scripts/train_vla.py` | Training loop |
| `scripts/evaluate_vla.py` | Benchmark vs. baseline |
| `data/dataset.py` | HDF5 I/O utilities |

## Success Metrics

After training:
- ✓ π₀.₅ RMSE **< 1.2× baseline** = Good fit
- ~ π₀.₅ RMSE **1.2–1.5× baseline** = Acceptable
- ✗ π₀.₅ RMSE **> 1.5× baseline** = More training needed

Run `python scripts/evaluate_vla.py` to check.
