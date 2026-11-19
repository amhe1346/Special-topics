"""Trajectory recorder and HDF5 dataset utilities."""
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Any


class TrajectoryRecorder:
    """Record trajectory data from environment/controller."""

    def __init__(self):
        """Initialize empty trajectory."""
        self.trajectory = {}
        self.reset()

    def reset(self):
        """Clear trajectory data."""
        self.trajectory = {
            'images': [],
            'joint_pos': [],
            'joint_vel': [],
            'ee_force': [],
            'motor_current': [],
            'target_joint_pos': [],
            'timestamps': [],
        }

    def append(self, image, obs, target_pos, timestamp):
        """Record one step of trajectory.

        Args:
            image: RGB image (H, W, 3)
            obs: Observation dict from env
            target_pos: Target joint positions (7D)
            timestamp: Time in seconds
        """
        self.trajectory['images'].append(image)
        self.trajectory['joint_pos'].append(obs['joint_pos'])
        self.trajectory['joint_vel'].append(obs['joint_vel'])
        self.trajectory['ee_force'].append(obs['ee_force'])
        self.trajectory['motor_current'].append(obs['motor_current'])
        self.trajectory['target_joint_pos'].append(target_pos)
        self.trajectory['timestamps'].append(timestamp)

    def to_arrays(self):
        """Convert to numpy arrays (for HDF5 saving)."""
        return {
            'images': np.stack(self.trajectory['images']),
            'joint_pos': np.stack(self.trajectory['joint_pos']),
            'joint_vel': np.stack(self.trajectory['joint_vel']),
            'ee_force': np.stack(self.trajectory['ee_force']),
            'motor_current': np.stack(self.trajectory['motor_current']),
            'target_joint_pos': np.stack(self.trajectory['target_joint_pos']),
            'timestamps': np.array(self.trajectory['timestamps']),
        }

    def save_hdf5(self, filepath, metadata: Dict[str, Any]):
        """Save trajectory to HDF5 file.

        Args:
            filepath: Output HDF5 path
            metadata: Dict of metadata (scenario, language, etc.)
        """
        arrays = self.to_arrays()

        with h5py.File(filepath, 'w') as f:
            # Save trajectory data
            for key, value in arrays.items():
                f.create_dataset(key, data=value)

            # Save metadata
            for key, value in metadata.items():
                if isinstance(value, str):
                    f.attrs[key] = value
                else:
                    f.attrs[key] = value

    def length(self):
        """Get trajectory length."""
        return len(self.trajectory['timestamps'])


class Dataset:
    """HDF5 dataset loader for training."""

    def __init__(self, data_dir):
        """Initialize dataset from directory of HDF5 files.

        Args:
            data_dir: Path to directory containing .h5 rollout files
        """
        self.data_dir = Path(data_dir)
        self.file_paths = sorted(self.data_dir.glob("rollout_*.h5"))
        print(f"Loaded {len(self.file_paths)} rollouts from {self.data_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """Load a single rollout.

        Returns:
            Dict with keys: images, joint_pos, joint_vel, ee_force, motor_current,
                           target_joint_pos, language, scenario
        """
        filepath = self.file_paths[idx]

        with h5py.File(filepath, 'r') as f:
            data = {
                'images': f['images'][:],  # (T, 224, 224, 3)
                'joint_pos': f['joint_pos'][:],
                'joint_vel': f['joint_vel'][:],
                'ee_force': f['ee_force'][:],
                'motor_current': f['motor_current'][:],
                'target_joint_pos': f['target_joint_pos'][:],
                'language': f.attrs.get('language', ''),
                'scenario': f.attrs.get('scenario', ''),
            }

        return data

    def train_test_split(self, train_ratio=0.8):
        """Split dataset into train/test.

        Args:
            train_ratio: Fraction for training

        Returns:
            (train_indices, test_indices)
        """
        n = len(self)
        indices = np.arange(n)
        np.random.shuffle(indices)
        split = int(n * train_ratio)
        return indices[:split], indices[split:]
