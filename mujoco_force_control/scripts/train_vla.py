#!/usr/bin/env python3
"""Training harness for π₀.₅ imitation learning."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from pathlib import Path
import argparse
from tqdm import tqdm

from mujoco_force_control.vla.pi05_wrapper import Pi05Wrapper
from mujoco_force_control.data.dataset import Dataset


class RolloutDataset(TorchDataset):
    """PyTorch dataset wrapper for HDF5 rollouts."""

    def __init__(self, dataset, language_to_embedding):
        """
        Args:
            dataset: mujoco_force_control.data.Dataset instance
            language_to_embedding: Dict mapping language strings to embeddings
        """
        self.dataset = dataset
        self.language_to_embedding = language_to_embedding

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Load and prepare a rollout for training."""
        rollout = self.dataset[idx]

        # Normalize observations
        images = rollout['images'].astype(np.float32) / 255.0  # [0, 1]
        images = np.transpose(images, (0, 3, 1, 2))  # (T, H, W, 3) -> (T, 3, H, W)

        # Concatenate proprioceptive state
        proprio = np.concatenate([
            rollout['joint_pos'],
            rollout['joint_vel'],
            rollout['ee_force'],
        ], axis=1)  # (T, 16)

        # Get language embedding
        language = rollout['language']
        if language not in self.language_to_embedding:
            # Dummy embedding if language not in vocab
            lang_emb = np.random.randn(256).astype(np.float32)
        else:
            lang_emb = self.language_to_embedding[language]

        # Target actions (from hand-coded controller)
        actions = rollout['target_joint_pos']  # (T, 7)

        return {
            'images': images.astype(np.float32),
            'proprio': proprio.astype(np.float32),
            'language': lang_emb.astype(np.float32),
            'actions': actions.astype(np.float32),
        }


def build_language_embeddings():
    """Build simple language embeddings for training."""
    language_vocab = [
        "oscillate smoothly",
        "move in a wave pattern",
        "sinusoidal motion",
        "move to home position",
        "reach the target pose",
        "go to home",
        "explore the workspace",
        "move around freely",
        "random trajectory",
    ]

    # Simple random embeddings (in real π₀.₅, these would be from CLIP or similar)
    embeddings = {}
    np.random.seed(42)
    for text in language_vocab:
        embeddings[text] = np.random.randn(256).astype(np.float32)

    return embeddings


def train_epoch(model, dataloader, optimizer, loss_fn, device):
    """Train for one epoch."""
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch['images'].to(device)  # (B, T, 3, 224, 224)
        proprio = batch['proprio'].to(device)  # (B, T, 16)
        language = batch['language'].to(device)  # (B, 256)
        actions = batch['actions'].to(device)  # (B, T, 7)

        # Forward pass
        predicted_actions = model.model(images, proprio, language)

        # Loss
        loss = loss_fn(predicted_actions, actions)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def evaluate(model, dataloader, loss_fn, device):
    """Evaluate on validation set."""
    total_loss = 0.0
    total_error = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch['images'].to(device)
            proprio = batch['proprio'].to(device)
            language = batch['language'].to(device)
            actions = batch['actions'].to(device)

            # Forward pass
            predicted_actions = model.model(images, proprio, language)

            # Loss
            loss = loss_fn(predicted_actions, actions)
            total_loss += loss.item()

            # L2 tracking error
            error = torch.norm(predicted_actions - actions, dim=-1).mean()
            total_error += error.item()

            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_error = total_error / max(num_batches, 1)
    return avg_loss, avg_error


def train_vla(
    data_dir="dataset/rollouts",
    output_dir="checkpoints",
    num_epochs=20,
    batch_size=4,
    learning_rate=1e-4,
    device='cpu',
):
    """Train π₀.₅ on collected rollouts.

    Args:
        data_dir: Directory with HDF5 rollouts
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: 'cpu' or 'cuda'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {data_dir}...")
    dataset = Dataset(data_dir)

    # Build language embeddings
    language_embs = build_language_embeddings()

    # Train/test split
    train_indices, test_indices = dataset.train_test_split(train_ratio=0.8)
    train_dataset = RolloutDataset(dataset, language_embs)
    test_dataset = RolloutDataset(dataset, language_embs)

    # Custom subset (for train/test split)
    class Subset(TorchDataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]

    train_dataset = Subset(train_dataset, train_indices)
    test_dataset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Initialize model
    print("Initializing π₀.₅ model...")
    model = Pi05Wrapper(device=device)

    # Training setup
    optimizer = torch.optim.Adam(model.model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    print(f"\nTraining for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Train set: {len(train_dataset)} rollouts, Test set: {len(test_dataset)} rollouts\n")

    best_test_loss = float('inf')

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)

        # Evaluate
        test_loss, test_error = evaluate(model, test_loader, loss_fn, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train loss: {train_loss:.6f}")
        print(f"  Test loss:  {test_loss:.6f}")
        print(f"  Test error: {test_error:.6f} rad")

        # Save best checkpoint
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            checkpoint_path = output_dir / f"pi05_best.pt"
            model.save(checkpoint_path)
            print(f"  ✓ Saved best model to {checkpoint_path}")

    print(f"\n✓ Training complete!")
    print(f"  Best model saved to {output_dir}/pi05_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train π₀.₅ on collected data")
    parser.add_argument("--data_dir", type=str, default="dataset/rollouts", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", choices=['cpu', 'cuda'], help="Device")

    args = parser.parse_args()

    train_vla(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
