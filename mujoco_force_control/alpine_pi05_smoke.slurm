#!/bin/bash
#SBATCH --job-name=pi05-oracle-smoke
#SBATCH --partition=aa100
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail
umask 077

: "${PRIVATE_WORK_DIR:?Set PRIVATE_WORK_DIR to an existing private absolute directory}"
: "${EXPERIMENT_NAME:?Set EXPERIMENT_NAME to a new experiment name}"
: "${TERMS_REVIEWED:?Set TERMS_REVIEWED=yes only after reviewing the checkpoint terms}"
if [[ "$TERMS_REVIEWED" != "yes" ]]; then
    echo "TERMS_REVIEWED must be yes after operator review" >&2
    exit 1
fi
if [[ "$PRIVATE_WORK_DIR" != /* || ! -d "$PRIVATE_WORK_DIR" || -L "$PRIVATE_WORK_DIR" || ! -O "$PRIVATE_WORK_DIR" ]]; then
    echo "PRIVATE_WORK_DIR must be an existing, owned, non-symlink absolute directory" >&2
    exit 1
fi
permissions=$(stat -c '%a' "$PRIVATE_WORK_DIR")
if (( (8#$permissions & 077) != 0 )); then
    echo "PRIVATE_WORK_DIR must not grant group or other access" >&2
    exit 1
fi
: "${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to the Apptainer SIF image}"
cd "${SLURM_SUBMIT_DIR:?Submit from the mujoco_force_control directory}"
case "$(pwd -P)/" in
    "$(realpath "$PRIVATE_WORK_DIR")/"*) ;;
    *) echo "Submit the job from a project inside PRIVATE_WORK_DIR" >&2; exit 1 ;;
esac
: "${CUDA_VISIBLE_DEVICES:?Slurm must provide CUDA_VISIBLE_DEVICES}"
if [[ ! -f "$CONTAINER_IMAGE" ]]; then
    echo "Container image not found: $CONTAINER_IMAGE" >&2
    exit 1
fi
if [[ ! -f "$PRIVATE_WORK_DIR/model-cache/openpi-assets/checkpoints/pi05_droid/params/_METADATA" ||
      ! -f "$PRIVATE_WORK_DIR/model-cache/openpi-assets/checkpoints/pi05_droid/params/manifest.ocdbt" ]]; then
    echo "Pretrained checkpoint cache is missing required metadata; prepare and verify it before submitting" >&2
    exit 1
fi
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
export APPTAINERENV_OPENPI_DATA_HOME="$PRIVATE_WORK_DIR/model-cache"
export APPTAINERENV_XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export APPTAINERENV_HF_HOME="$PRIVATE_WORK_DIR/huggingface"
python_path="$PWD/.openpi/.venv/bin/python"
if [[ ! -f "$PWD/.openpi/.venv/pyvenv.cfg" ]]; then
    echo "OpenPI container environment not found: $python_path" >&2
    exit 1
fi

apptainer exec --cleanenv --nv --bind "$PRIVATE_WORK_DIR:$PRIVATE_WORK_DIR" \
    "$CONTAINER_IMAGE" "$python_path" -c \
    'import jax; devices = jax.devices("gpu"); print("JAX GPU devices:", devices, flush=True); assert devices'

apptainer exec --cleanenv --nv --bind "$PRIVATE_WORK_DIR:$PRIVATE_WORK_DIR" \
    "$CONTAINER_IMAGE" "$python_path" -u run_oracle_finetune.py \
    --dataset results/stage_c_oracle_lerobot \
    --train --terms-reviewed \
    --checkpoint-base-dir "$PRIVATE_WORK_DIR/checkpoints" \
    --exp-name "$EXPERIMENT_NAME" --steps 100 --batch-size 2
