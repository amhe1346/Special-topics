"""Preflight or launch the pinned OpenPI DROID fine-tuning recipe.

Run with the pinned OpenPI environment. Training is opt-in and requires a
separate acknowledgement that the checkpoint terms have been reviewed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "results/stage_c_oracle_lerobot")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--checkpoint-base-dir", type=Path)
    parser.add_argument("--exp-name")
    parser.add_argument("--train", action="store_true", help="run training rather than just a data preflight")
    parser.add_argument(
        "--terms-reviewed", action="store_true",
        help="confirm you have reviewed the applicable checkpoint terms and authorize fine-tuning",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 1:
        parser.error("--batch-size and --steps must be positive")
    if args.train and not args.terms_reviewed:
        parser.error("training requires --terms-reviewed")
    if args.train and (args.checkpoint_base_dir is None or not args.exp_name):
        parser.error("training requires --checkpoint-base-dir and --exp-name")
    if args.exp_name and (
        args.exp_name in (".", "..") or Path(args.exp_name).name != args.exp_name
    ):
        parser.error("--exp-name must be a single directory name")

    dataset_root = args.dataset.expanduser().resolve()
    manifest_path = dataset_root / "conversion_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "stage-c-oracle-lerobot-v1" or manifest.get("status") != "complete":
        raise ValueError(f"{manifest_path}: expected a completed oracle LeRobot conversion")
    if not (dataset_root / "train/meta/info.json").is_file():
        raise FileNotFoundError(f"training metadata missing from {dataset_root}")

    # OpenPI's loader looks up repo_id under HF_LEROBOT_HOME, not an arbitrary root.
    os.environ["HF_LEROBOT_HOME"] = str(dataset_root)
    os.environ["HF_HUB_OFFLINE"] = "1"

    from openpi.training import config as openpi_config
    from openpi.training import data_loader

    base = openpi_config.get_config("pi05_droid_finetune")
    data = dataclasses.replace(base.data, repo_id="train")
    config = dataclasses.replace(
        base,
        name="peg_oracle_pi05_droid",
        data=data,
        batch_size=args.batch_size,
        num_workers=0,
        num_train_steps=args.steps,
        wandb_enabled=False,
        save_interval=50,
        exp_name=args.exp_name or "preflight",
        checkpoint_base_dir=str(args.checkpoint_base_dir or ROOT / "checkpoints"),
    )
    loader = data_loader.create_data_loader(config, num_batches=1, shuffle=False)
    observation, actions = next(iter(loader))
    expected_shape = (args.batch_size, config.model.action_horizon, config.model.action_dim)
    if tuple(actions.shape) != expected_shape:
        raise ValueError(f"unexpected action batch shape {tuple(actions.shape)}; expected {expected_shape}")
    if loader.data_config().norm_stats is None:
        raise ValueError("the pretrained DROID normalization assets were not loaded")
    expected_image_shape = (args.batch_size, 224, 224, 3)
    if any(tuple(image.shape) != expected_image_shape for image in observation.images.values()):
        raise ValueError(f"expected image batches with shape {expected_image_shape}")
    print(json.dumps({
        "status": "preflight_passed",
        "dataset": str(dataset_root / "train"),
        "action_shape": list(actions.shape),
        "prompt_token_shape": list(observation.tokenized_prompt.shape),
        "normalization": "pretrained DROID assets",
        "training_requested": args.train,
        "checkpoint_dir": str(config.checkpoint_dir) if args.train else None,
    }), flush=True)

    if args.train:
        openpi_root = ROOT / ".openpi"
        if not (openpi_root / "scripts/train.py").is_file():
            raise FileNotFoundError(f"pinned OpenPI training script not found in {openpi_root}")
        if config.checkpoint_dir.exists():
            raise FileExistsError(f"checkpoint directory already exists: {config.checkpoint_dir}")
        sys.path.insert(0, str(openpi_root))
        from scripts.train import main as train_main

        train_main(config)


if __name__ == "__main__":
    main()
