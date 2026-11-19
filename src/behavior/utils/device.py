"""Select a compute device and set reproducible random seeds."""

import logging
import random
from typing import Dict, Tuple

import numpy as np
import torch

logger = logging.getLogger("ViolenceClassifier.Device")


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Ensure reproducible experiments across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    logger.info("Random seeds set to %d (deterministic=%s)", seed, deterministic)


class DeviceManager:
    """
    Handles hardware resource allocation, mixed precision, and GPU memory monitoring.
    """

    def __init__(self, requested_device: str = "auto", memory_fraction: float = 0.85):
        self.device = self._resolve_device(requested_device)
        self.memory_fraction = memory_fraction
        self.is_cuda = self.device.type == "cuda"
        self._configure_gpu()

    def _resolve_device(self, requested: str) -> torch.device:
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda:0")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(requested)

    def _configure_gpu(self) -> None:
        if self.is_cuda:
            device_idx = self.device.index or 0
            device_name = torch.cuda.get_device_name(device_idx)
            total_vram_gb = torch.cuda.get_device_properties(
                device_idx
            ).total_memory / (1024**3)
            logger.info("Using GPU: %s [%.2f GB VRAM]", device_name, total_vram_gb)

            # Cap PyTorch memory allocation to avoid system OOM
            try:
                torch.cuda.set_per_process_memory_fraction(
                    self.memory_fraction, device_idx
                )
            except Exception as e:
                logger.warning("Could not set VRAM memory fraction: %s", e)
        else:
            logger.info("Using compute device: %s", self.device)

    def empty_cache(self) -> None:
        if self.is_cuda:
            torch.cuda.empty_cache()

    def get_memory_stats(self) -> Dict[str, float]:
        if not self.is_cuda:
            return {"allocated_mb": 0.0, "reserved_mb": 0.0}
        dev = self.device.index or 0
        return {
            "allocated_mb": torch.cuda.memory_allocated(dev) / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved(dev) / (1024**2),
            "max_allocated_mb": torch.cuda.max_memory_allocated(dev) / (1024**2),
        }


def setup_device(
    requested: str = "auto", memory_fraction: float = 0.85
) -> Tuple[torch.device, DeviceManager]:
    manager = DeviceManager(requested_device=requested, memory_fraction=memory_fraction)
    return manager.device, manager
