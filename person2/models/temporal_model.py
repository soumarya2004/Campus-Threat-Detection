"""
Spatiotemporal video models for behavior recognition and violence classification.
Supports torchvision video ResNets, a lightweight 3D CNN baseline, and optional
fine-tuning of the full backbone or its top residual stages.
"""

import logging
from typing import Any, Dict

import torch
import torch.nn as nn
import torchvision.models.video as vmodels

logger = logging.getLogger("ViolenceClassifier.Model")


class MotionExcitationModule(nn.Module):
    """
    Learn channel attention from adjacent-frame differences.
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        reduced_ch = max(4, in_channels // reduction)
        self.channel_squeeze = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(
                in_channels,
                reduced_ch,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                bias=False,
            ),
            nn.BatchNorm3d(reduced_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced_ch, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, T, H, W)
        # Compute adjacent frame difference
        diff = torch.zeros_like(x)
        diff[:, :, 1:, :, :] = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
        diff[:, :, 0, :, :] = diff[:, :, 1, :, :]  # boundary copy

        # Channel attention based on motion
        attention = self.channel_squeeze(diff)
        return x * attention + x


class Lightweight3DCNN(nn.Module):
    """
    Compact 3D CNN baseline with a smaller parameter count than Video ResNet.
    """

    def __init__(
        self, num_classes: int = 1, in_channels: int = 3, dropout_rate: float = 0.5
    ):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: Spatial downsampling in stem (224x224 -> 56x56)
            nn.Conv3d(
                in_channels,
                32,
                kernel_size=(3, 5, 5),
                stride=(1, 2, 2),
                padding=(1, 2, 2),
                bias=False,
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            # Block 2: (B, 32, T, 56, 56) -> (B, 64, T/2, 28, 28)
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            # Block 3: (B, 64, T/2, 28, 28) -> (B, 128, T/4, 14, 14)
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            # Block 4: (B, 128, T/4, 14, 14) -> (B, 256, T/8, 7, 7)
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
        )

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.global_pool(feat).flatten(1)
        return self.classifier(pooled)


class TemporalViolenceModel(nn.Module):
    """
    Video classifier with optional motion attention and backbone fine-tuning.
    """

    def __init__(
        self,
        architecture: str = "r2plus1d_18",
        pretrained: bool = False,
        num_classes: int = 1,
        dropout_rate: float = 0.5,
        fine_tune_mode: str = "partial",
        use_motion_excitation: bool = True,
    ):
        super().__init__()
        self.architecture = architecture.lower()
        self.num_classes = num_classes
        self.fine_tune_mode = fine_tune_mode
        self.use_motion_excitation = use_motion_excitation

        if self.architecture == "custom_3d_cnn":
            self.backbone = Lightweight3DCNN(
                num_classes=num_classes, dropout_rate=dropout_rate
            )
            self.in_features = 256
            self.is_custom = True
        else:
            self.is_custom = False
            self.backbone, self.in_features = self._build_torchvision_backbone(
                pretrained
            )
            self._apply_finetune_strategy(self.fine_tune_mode)

            # Custom classification head
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(self.in_features, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate / 2),
                nn.Linear(128, num_classes),
            )

        # Motion Excitation
        self.motion_excitation = (
            MotionExcitationModule(in_channels=3) if use_motion_excitation else None
        )

    def _build_torchvision_backbone(self, pretrained: bool):
        arch = self.architecture
        weights = "DEFAULT" if pretrained else None

        constructors = {
            "r2plus1d_18": vmodels.r2plus1d_18,
            "r3d_18": vmodels.r3d_18,
            "mc3_18": vmodels.mc3_18,
        }
        if arch not in constructors:
            raise ValueError(f"Unsupported video architecture: {arch}")
        try:
            model = constructors[arch](weights=weights)
        except Exception as exc:
            if pretrained:
                raise RuntimeError(
                    f"Could not load pretrained weights for {arch}. Check the torchvision install "
                    "and network access, or set model.pretrained=false to train from scratch."
                ) from exc
            raise

        in_feat = model.fc.in_features
        # Remove default linear head
        model.fc = nn.Identity()
        return model, in_feat

    def _apply_finetune_strategy(self, mode: str):
        if mode == "head_only":
            logger.info("Finetune mode 'head_only': Freezing all backbone layers.")
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif mode == "partial":
            logger.info(
                "Finetune mode 'partial': Freezing stem & layers 1-2, unfreezing layer 3-4."
            )
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Unfreeze top layers if present
            if hasattr(self.backbone, "layer3"):
                for param in self.backbone.layer3.parameters():
                    param.requires_grad = True
            if hasattr(self.backbone, "layer4"):
                for param in self.backbone.layer4.parameters():
                    param.requires_grad = True
        elif mode == "full":
            logger.info(
                "Finetune mode 'full': Unfreezing entire model for end-to-end training."
            )
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown fine_tune_mode: {mode}")

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled video embedding before classification."""
        if self.motion_excitation is not None:
            x = self.motion_excitation(x)

        if self.is_custom:
            feat = self.backbone.features(x)
            return self.backbone.global_pool(feat).flatten(1)

        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, C, T, H, W)
        Returns:
            Logits of shape (B, num_classes)
        """
        if self.is_custom:
            if self.motion_excitation is not None:
                x = self.motion_excitation(x)
            return self.backbone(x)

        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits


def build_temporal_model(config_dict: Dict[str, Any]) -> TemporalViolenceModel:
    """
    Factory function to instantiate TemporalViolenceModel from configuration dictionary.
    """
    m_cfg = config_dict.get("model", {})
    return TemporalViolenceModel(
        architecture=m_cfg.get("architecture", "r2plus1d_18"),
        pretrained=m_cfg.get("pretrained", False),
        num_classes=m_cfg.get("num_classes", 1),
        dropout_rate=m_cfg.get("dropout_rate", 0.5),
        fine_tune_mode=m_cfg.get("fine_tune_mode", "partial"),
        use_motion_excitation=m_cfg.get("use_motion_excitation", True),
    )
