import sys
import importlib
from importlib.metadata import PackageNotFoundError, version

def check_package(package_name:str, distribution_name:str | None=None)->bool:
    distribution_name=distribution_name or package_name
    try:
        installed_version=version(distribution_name)
        print(f"[OK] {distribution_name}: {installed_version}")
        return True
    except PackageNotFoundError:
        print(f"[FAIL] {distribution_name}: not installed")
        return False

def main()->int:
    print("=== Campus Threat Detection: Environment Check ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print()
    packages=[
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("ultralytics", "ultralytics"),
        ("cv2", "opencv-python"),
        ("yaml", "PyYAML"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
    ]
    all_ok=True
    for module_name, distribution_name in packages:
        try:
            importlib.import_module(module_name)
            all_ok &= check_package(module_name, distribution_name)
        except Exception as exc:
            print(f"[FAIL] {distribution_name}: import error: {exc}")
            all_ok = False
    print("\n=== CUDA / GPU ===")
    try:
        import torch
        print(f"PyTorch CUDA build: {torch.version.cuda}")
        if torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.cuda.is_available()}")
            print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[OK] GPU count: {torch.cuda.device_count()}")
            #verify that a tensor operation executes on the GPU
            test_tensor=torch.ones((2, 2), device="cuda")
            result=test_tensor @ test_tensor
            print(f"[OK] GPU tensor operation: {result.device}")
        else:
            print("[FAIL] CUDA is unavailable to PyTorch.")
            all_ok=False
    except Exception as exc:
        print(f"[FAIL] CUDA/GPU check failed: {exc}")
        all_ok=False
    print("\n=== Result ===")
    if all_ok:
        print("[PASS] Environment checks passed.")
        return 0
    print("[FAIL] One or more environment checks failed.")
    return 1

if __name__=="__main__":
    raise SystemExit(main())
