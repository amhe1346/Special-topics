"""π₀.₅ Vision-Language-Action model wrapper for robotic manipulation."""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class SimplePi05(nn.Module):
    """
    Simplified π₀.₅ model for joint position control.

    In reality, π₀.₅ is a large vision transformer trained on multimodal data.
    For this learning project, we use a simpler architecture that:
    - Takes RGB images (224x224x3)
    - Takes proprioceptive state (joint_pos, joint_vel, force, motor_current)
    - Takes language embeddings
    - Outputs joint position targets (7D)

    This is a proxy for the real π₀.₅ model from OpenPi.
    """

    def __init__(self, obs_dim=16, language_dim=256, action_dim=7, hidden_dim=256):
        """
        Args:
            obs_dim: Proprioceptive observation dimension (joint_pos + joint_vel + force + current)
            language_dim: Language embedding dimension
            action_dim: Output action dimension (7 for Franka Panda)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        # Vision encoder (simplified CNN)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Compute vision features dimension
        with torch.no_grad():
            dummy_img = torch.randn(1, 3, 224, 224)
            vision_dim = self.vision_encoder(dummy_img).shape[1]

        # Proprioceptive encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Language encoder (simple embedding)
        self.language_encoder = nn.Linear(language_dim, hidden_dim)

        # Fusion layer (combine vision + proprioception + language)
        fused_dim = vision_dim + hidden_dim + hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        # Action head (output joint positions)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, images, proprio, language_emb):
        """
        Args:
            images: (B, T, 3, 224, 224) or (B, 3, 224, 224)
            proprio: (B, T, obs_dim) or (B, obs_dim)
            language_emb: (B, language_dim)

        Returns:
            actions: (B, T, action_dim) or (B, action_dim)
        """
        # Handle time dimension
        squeeze_time = False
        if images.dim() == 4:
            squeeze_time = True
            images = images.unsqueeze(1)
            proprio = proprio.unsqueeze(1)

        B, T, C, H, W = images.shape
        images = images.view(B * T, C, H, W)

        # Encode vision
        vision_features = self.vision_encoder(images)

        # Encode proprioception
        proprio_features = self.proprio_encoder(proprio.view(B * T, -1))

        # Encode language (broadcast across time)
        language_features = self.language_encoder(language_emb)
        language_features = language_features.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)

        # Fuse modalities
        fused = torch.cat([vision_features, proprio_features, language_features], dim=-1)
        fused = self.fusion(fused)

        # Predict actions
        actions = self.action_head(fused)
        actions = actions.view(B, T, -1)

        if squeeze_time:
            actions = actions.squeeze(1)

        return actions


class Pi05Wrapper:
    """Wrapper for π₀.₅ model inference."""

    def __init__(self, model_path=None, device='cpu'):
        """
        Args:
            model_path: Path to saved model checkpoint (or None to create fresh model)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model = SimplePi05()

        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")

        self.model.to(device)
        self.model.eval()

    def predict(self, images, proprio, language_emb):
        """
        Predict joint position targets.

        Args:
            images: (B, T, 3, 224, 224) RGB images
            proprio: (B, T, 16) proprioceptive state [joint_pos (7), joint_vel (7), force (3), current (7)]
            language_emb: (B, 256) language embeddings (or raw text labels)

        Returns:
            actions: (B, T, 7) predicted joint position targets
        """
        with torch.no_grad():
            # Convert to tensors if needed
            if not isinstance(images, torch.Tensor):
                images = torch.from_numpy(images).float().to(self.device)
            if not isinstance(proprio, torch.Tensor):
                proprio = torch.from_numpy(proprio).float().to(self.device)
            if not isinstance(language_emb, torch.Tensor):
                language_emb = torch.from_numpy(language_emb).float().to(self.device)

            actions = self.model(images, proprio, language_emb)

        return actions.cpu().numpy()

    def save(self, filepath):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), filepath)
        print(f"Saved model to {filepath}")
