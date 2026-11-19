"""Fuse video embeddings with optional person-tracking features."""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger("ViolenceClassifier.FeatureFusion")


class TrackingMotionFusionModel(nn.Module):
    """Combine video features with motion features from person tracks."""

    def __init__(
        self,
        video_feature_dim: int = 512,
        tracking_feature_dim: int = 64,
        hidden_dim: int = 128,
        num_classes: int = 1,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.video_proj = nn.Sequential(
            nn.Linear(video_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.tracking_proj = nn.Sequential(
            nn.Linear(tracking_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Gated fusion mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        video_features: torch.Tensor,
        tracking_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        v_h = self.video_proj(video_features)

        if tracking_features is None:
            # The video embedding remains usable when no tracks are available.
            fused = v_h
        else:
            t_h = self.tracking_proj(tracking_features)
            gate_weight = self.gate(torch.cat([v_h, t_h], dim=-1))
            fused = gate_weight * v_h + (1.0 - gate_weight) * t_h

        return self.classifier(fused)
