"""
Unit tests for 3D CNN architectures, fine-tuning modes, and feature fusion.
"""

import torch

from models.feature_fusion import TrackingMotionFusionModel
from models.temporal_model import (
    Lightweight3DCNN,
    MotionExcitationModule,
    TemporalViolenceModel,
)


def test_lightweight_3d_cnn_shape():
    model = Lightweight3DCNN(num_classes=1)
    model.eval()

    # Input: (batch=2, channels=3, temporal=16, height=112, width=112)
    # Using 112x112 for rapid testing
    x = torch.randn(2, 3, 16, 112, 112)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 1)


def test_motion_excitation_module():
    module = MotionExcitationModule(in_channels=3)
    x = torch.randn(2, 3, 8, 56, 56)
    out = module(x)

    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_temporal_violence_model_modes():
    # Test head_only mode
    model_frozen = TemporalViolenceModel(
        architecture="r2plus1d_18",
        pretrained=False,
        fine_tune_mode="head_only",
    )
    # Backbone weights should not require grad
    for name, param in model_frozen.backbone.named_parameters():
        assert not param.requires_grad, (
            f"Parameter {name} should be frozen in head_only mode"
        )

    # Classifier head should require grad
    for param in model_frozen.classifier.parameters():
        assert param.requires_grad

    # Test partial fine-tune mode
    model_partial = TemporalViolenceModel(
        architecture="r2plus1d_18",
        pretrained=False,
        fine_tune_mode="partial",
    )
    if hasattr(model_partial.backbone, "layer4"):
        for param in model_partial.backbone.layer4.parameters():
            assert param.requires_grad


def test_tracking_feature_fusion_bridge():
    fusion_model = TrackingMotionFusionModel(
        video_feature_dim=512, tracking_feature_dim=64, num_classes=1
    )
    fusion_model.eval()

    v_feat = torch.randn(3, 512)
    t_feat = torch.randn(3, 64)

    # Combined forward pass
    with torch.no_grad():
        out_fused = fusion_model(v_feat, t_feat)
    assert out_fused.shape == (3, 1)

    # Fallback without tracking features
    with torch.no_grad():
        out_fallback = fusion_model(v_feat, None)
    assert out_fallback.shape == (3, 1)
