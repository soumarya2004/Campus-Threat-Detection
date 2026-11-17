"""Check installed packages and whether CPU or CUDA inference is available."""

import argparse
import importlib
from importlib.metadata import version
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail if CUDA is unavailable; otherwise a CPU environment is valid",
    )
    args = parser.parse_args()
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("ultralytics", "ultralytics"),
        ("cv2", "opencv-python"),
        ("yaml", "PyYAML"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
    ]
    all_ok = True
    for module_name, distribution in packages:
        try:
            importlib.import_module(module_name)
            print(f"[OK] {distribution}: {version(distribution)}")
        except Exception as exc:
            print(f"[FAIL] {distribution}: {exc}")
            all_ok = False

    try:
        import torch
        from torchvision.ops import nms

        print(f"PyTorch CUDA runtime: {torch.version.cuda or 'CPU-only build'}")
        # Importing a package alone does not prove its native NMS operator works.
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        scores = torch.tensor([0.9])
        nms(boxes, scores, 0.5)
        print("[OK] CPU tensor and NMS operations")
        if torch.cuda.is_available():
            nms(boxes.cuda(), scores.cuda(), 0.5)
            print(f"[OK] CUDA NMS operation: {torch.cuda.get_device_name(0)}")
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            print(
                f"GPU memory: {free_bytes / 2**30:.2f} GiB free / {total_bytes / 2**30:.2f} GiB total"
            )
        elif args.require_cuda:
            print("[FAIL] CUDA was requested but is unavailable.")
            all_ok = False
        else:
            print("[OK] CPU mode available; CUDA is unavailable.")
    except Exception as exc:
        print(f"[FAIL] Tensor/NMS operation: {exc}")
        all_ok = False

    print(
        "[PASS] Environment checks passed."
        if all_ok
        else "[FAIL] Environment checks failed."
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
