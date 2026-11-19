"""Train clip-level or weakly supervised video-level violence classifiers."""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.behavior.data.dataset import VideoClipDataset, create_dataloader
from src.behavior.data.synthetic_generator import create_synthetic_benchmark_dataset
from src.behavior.data.video_bag_dataset import VideoBagDataset, create_video_bag_dataloader
from src.behavior.models.temporal_model import build_temporal_model
from src.behavior.utils.device import set_seed, setup_device
from src.behavior.utils.metrics import evaluate_predictions, find_optimal_threshold

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("TrainViolenceModel")


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for highly imbalanced surveillance video anomaly detection.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        focal_weight = alpha_t * torch.pow((1.0 - p_t).clamp(min=1e-6), self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    grad_clip: float = 1.0,
    use_amp: bool = False,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    all_targets = []
    all_probs = []

    pbar = tqdm(loader, desc="  Train Batch", leave=False)
    for clips, labels, meta in pbar:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp and device.type == "cuda":
            with autocast():
                logits = model(clips)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()

        all_probs.extend(probs.flatten())
        all_targets.extend(targets.flatten())
        total_loss += loss.item() * clips.size(0)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader.dataset)
    metrics = evaluate_predictions(
        np.array(all_targets), np.array(all_probs), threshold=0.50
    )
    return avg_loss, metrics.f1, metrics.accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.50,
) -> Tuple[float, Any]:
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []

    for clips, labels, meta in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(clips)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits).cpu().numpy()
        targets = labels.cpu().numpy()

        all_probs.extend(probs.flatten())
        all_targets.extend(targets.flatten())
        total_loss += loss.item() * clips.size(0)

    avg_loss = total_loss / len(loader.dataset)
    metrics = evaluate_predictions(
        np.array(all_targets), np.array(all_probs), threshold=threshold
    )
    return avg_loss, metrics


def pool_video_bag_logits(
    model: nn.Module,
    bags: torch.Tensor,
    top_k: int = 1,
) -> torch.Tensor:
    """Score video bags with top-k multiple-instance pooling.

    A video label supervises the bag, not every sampled clip. Top-k pooling
    allows a positive untrimmed video to contain many normal-looking windows.
    """
    batch_size, bag_size, channels, frames, height, width = bags.shape
    clip_logits = model(
        bags.reshape(batch_size * bag_size, channels, frames, height, width)
    )
    clip_logits = clip_logits.reshape(batch_size, bag_size, -1).squeeze(-1)
    k = max(1, min(int(top_k), bag_size))
    return torch.topk(clip_logits, k=k, dim=1).values.mean(dim=1, keepdim=True)


def train_video_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    grad_clip: float = 1.0,
    use_amp: bool = False,
    top_k: int = 1,
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    all_targets = []
    all_probs = []
    for bags, labels, _meta in tqdm(loader, desc="  Train Videos", leave=False):
        bags = bags.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).reshape(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        if use_amp and device.type == "cuda":
            with autocast():
                logits = pool_video_bag_logits(model, bags, top_k=top_k)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = pool_video_bag_logits(model, bags, top_k=top_k)
            loss = criterion(logits, labels)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().flatten())
        all_targets.extend(labels.detach().cpu().numpy().flatten())
        total_loss += loss.item() * bags.size(0)

    avg_loss = total_loss / max(1, len(loader.dataset))
    metrics = evaluate_predictions(
        np.asarray(all_targets), np.asarray(all_probs), threshold=0.50
    )
    return avg_loss, metrics.f1, metrics.accuracy


@torch.no_grad()
def validate_videos(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.50,
    top_k: int = 1,
) -> Tuple[float, Any, float]:
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []
    for bags, labels, _meta in loader:
        bags = bags.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).reshape(-1, 1)
        logits = pool_video_bag_logits(model, bags, top_k=top_k)
        loss = criterion(logits, labels)
        all_probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
        all_targets.extend(labels.cpu().numpy().flatten())
        total_loss += loss.item() * bags.size(0)
    avg_loss = total_loss / max(1, len(loader.dataset))
    targets = np.asarray(all_targets)
    probabilities = np.asarray(all_probs)
    metrics = evaluate_predictions(targets, probabilities, threshold=threshold)
    selected_threshold, _ = find_optimal_threshold(
        targets, probabilities, criterion="max_f1"
    )
    return avg_loss, metrics, selected_threshold


def main():
    parser = argparse.ArgumentParser(
        description="Train Temporal Violence Classification Model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/behavior/training_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument("--arch", type=str, default=None, help="Override architecture")
    parser.add_argument(
        "--device", type=str, default=None, help="Compute device ('cuda', 'cpu')"
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Generate synthetic dataset if needed"
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    epochs = args.epochs or cfg["training"]["epochs"]
    batch_size = args.batch_size or cfg["training"]["batch_size"]
    arch = args.arch or cfg["model"]["architecture"]
    cfg["training"]["epochs"] = epochs
    cfg["training"]["batch_size"] = batch_size
    cfg["model"]["architecture"] = arch

    # Seed training and select the compute device.
    seed = cfg["experiment"].get("seed", 42)
    set_seed(seed)
    req_dev = args.device or "auto"
    device, dev_mgr = setup_device(requested=req_dev)

    # Generate synthetic media only when explicitly requested. Real XD-Violence
    # manifests are prepared with `python -m src.behavior.scripts.prepare_xd_violence`.
    train_manifest = cfg["dataset"]["train_manifest"]
    val_manifest = cfg["dataset"]["val_manifest"]

    if args.synthetic:
        logger.info(
            "Creating synthetic pipeline-demo data (not benchmark training data)..."
        )
        paths = create_synthetic_benchmark_dataset()
        train_manifest = paths["train"]
        val_manifest = paths["val"]
        dataset_provenance = {
            "dataset": "synthetic fixture",
            "training_source": "generated",
            "training_scope": "synthetic_fixture",
        }
    elif not os.path.exists(train_manifest) or not os.path.exists(val_manifest):
        raise FileNotFoundError(
            "Training manifests are missing. Prepare XD-Violence with "
            "`python -m src.behavior.scripts.prepare_xd_violence` or pass --synthetic for the demo only."
        )
    else:
        summary_path = Path(train_manifest).parent / "xd_manifest_summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            dataset_provenance = {
                key: summary.get(key)
                for key in (
                    "dataset",
                    "training_source",
                    "training_scope",
                    "test_scope",
                    "source_video_files",
                )
            }
        else:
            dataset_provenance = {
                "dataset": cfg["dataset"].get("name", "unknown"),
                "training_source": "unspecified",
                "training_scope": "unknown",
            }
            logger.warning(
                "Manifest summary not found at %s; dataset scope will be marked unknown.",
                summary_path,
            )

    # Datasets and Loaders
    clip_len = cfg["dataset"]["clip_length"]
    stride = cfg["dataset"]["sampling_stride"]
    res = tuple(cfg["dataset"]["spatial_resolution"])

    logger.info(
        "Initializing datasets (Clip len=%d, Stride=%d, Res=%s)...",
        clip_len,
        stride,
        res,
    )
    use_video_mil = cfg["dataset"].get("supervision", "clip") == "weak_video_mil"
    if use_video_mil:
        common_dataset_args = {
            "project_root": Path.cwd(),
            "clip_length": clip_len,
            "sampling_stride": stride,
            "spatial_resolution": res,
            "clips_per_video": cfg["dataset"].get("clips_per_video", 4),
            "horizontal_flip_prob": cfg.get("augmentation", {})
            .get("train", {})
            .get("horizontal_flip_prob", 0.5),
        }
        train_dataset = VideoBagDataset(
            train_manifest, is_training=True, **common_dataset_args
        )
        val_dataset = VideoBagDataset(
            val_manifest, is_training=False, **common_dataset_args
        )
        train_loader = create_video_bag_dataloader(
            train_dataset,
            batch_size=batch_size,
            is_training=True,
            num_workers=cfg["dataset"].get("num_workers", 0),
            pin_memory=dev_mgr.is_cuda,
            seed=seed,
        )
        val_loader = create_video_bag_dataloader(
            val_dataset,
            batch_size=batch_size,
            is_training=False,
            num_workers=cfg["dataset"].get("num_workers", 0),
            pin_memory=dev_mgr.is_cuda,
        )
    else:
        train_dataset = VideoClipDataset(
            manifest_path_or_records=train_manifest,
            clip_length=clip_len,
            sampling_stride=stride,
            spatial_resolution=res,
            is_training=True,
            enable_augmentation=True,
        )
        val_dataset = VideoClipDataset(
            manifest_path_or_records=val_manifest,
            clip_length=clip_len,
            sampling_stride=stride,
            spatial_resolution=res,
            is_training=False,
            enable_augmentation=False,
        )
        train_loader = create_dataloader(
            train_dataset,
            batch_size=batch_size,
            is_training=True,
            balance_classes=True,
            num_workers=cfg["dataset"].get("num_workers", 0),
            pin_memory=dev_mgr.is_cuda,
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=batch_size,
            is_training=False,
            balance_classes=False,
            num_workers=cfg["dataset"].get("num_workers", 0),
            pin_memory=dev_mgr.is_cuda,
        )

    # Model instantiation
    logger.info("Building model architecture: %s", arch)
    model = build_temporal_model(cfg).to(device)

    # Criterion setup
    loss_name = cfg["training"]["loss"].get("name", "focal")
    if loss_name == "focal":
        alpha = cfg["training"]["loss"].get("focal_alpha", 0.25)
        gamma = cfg["training"]["loss"].get("focal_gamma", 2.0)
        criterion = BinaryFocalLoss(alpha=alpha, gamma=gamma)
    elif loss_name == "weighted_bce":
        pos_weight = torch.tensor([cfg["training"]["loss"].get("pos_weight", 2.0)]).to(
            device
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Differential learning rate parameter groups
    opt_cfg = cfg["training"]["optimizer"]
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": float(opt_cfg.get("backbone_lr", 1e-5))},
        {"params": head_params, "lr": float(opt_cfg.get("head_lr", 1e-4))},
    ]
    optimizer = AdamW(
        param_groups, weight_decay=float(opt_cfg.get("weight_decay", 1e-4))
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(cfg["training"]["scheduler"].get("eta_min", 1e-6)),
    )
    scaler = GradScaler(
        enabled=(dev_mgr.is_cuda and cfg["training"].get("mixed_precision", True))
    )

    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = output_dir / "best_model.pt"

    # Training Loop
    best_val_f1 = -1.0
    patience = cfg["training"]["early_stopping"].get("patience", 6)
    patience_counter = 0

    logger.info("Training for %d epochs on %s", epochs, device)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_fn = train_video_epoch if use_video_mil else train_epoch
        mil_args = (
            {"top_k": cfg["dataset"].get("mil_top_k", 1)} if use_video_mil else {}
        )
        train_loss, train_f1, train_acc = train_fn(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip=cfg["training"].get("gradient_clip_norm", 1.0),
            use_amp=cfg["training"].get("mixed_precision", True),
            **mil_args,
        )

        if use_video_mil:
            val_loss, val_metrics, val_threshold = validate_videos(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                threshold=cfg["evaluation"].get("decision_threshold", 0.50),
                **mil_args,
            )
        else:
            val_loss, val_metrics = validate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                threshold=cfg["evaluation"].get("decision_threshold", 0.50),
            )
            val_threshold = cfg["evaluation"].get("decision_threshold", 0.50)

        scheduler.step()
        epoch_sec = time.time() - t0

        logger.info(
            "Epoch [%02d/%02d] (%.1fs) | Train loss: %.4f, accuracy: %.4f, F1: %.4f | "
            "Val loss: %.4f, F1: %.4f, threshold: %.3f, AUROC: %.4f, FAR: %.2f%%",
            epoch,
            epochs,
            epoch_sec,
            train_loss,
            train_acc,
            train_f1,
            val_loss,
            val_metrics.f1,
            val_threshold,
            val_metrics.auroc,
            val_metrics.false_alarm_rate * 100,
        )

        # Checkpoint if best metric
        if val_metrics.f1 > best_val_f1:
            best_val_f1 = val_metrics.f1
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics.to_dict(),
                "validation_threshold": float(val_threshold),
                "dataset_provenance": dataset_provenance,
                "config": cfg,
            }
            torch.save(checkpoint, best_checkpoint_path)
            logger.info(
                "--> Checkpoint saved! New best Val F1: %.4f at %s",
                best_val_f1,
                best_checkpoint_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience and cfg["training"]["early_stopping"].get(
                "enabled", True
            ):
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    patience,
                )
                break

    total_duration = time.time() - start_time
    logger.info(
        "Training complete in %.2f minutes. Best Validation F1: %.4f",
        total_duration / 60.0,
        best_val_f1,
    )


if __name__ == "__main__":
    main()
