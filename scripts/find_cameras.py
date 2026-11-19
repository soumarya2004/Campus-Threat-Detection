"""List camera indices that OpenCV can open on this computer."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.camera_source import (
    CameraSource,
    CaptureError,
    load_camera_config,
)  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-index", type=int, default=9)
    parser.add_argument("--backend", choices=("auto", "any", "msmf", "dshow"))
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show one captured frame for each available index",
    )
    args = parser.parse_args(argv)
    if args.max_index < 0:
        parser.error("--max-index must be nonnegative")

    settings = load_camera_config()
    backend = args.backend or settings.backend
    found = 0
    try:
        for index in range(args.max_index + 1):
            source = CameraSource(
                replace(
                    settings,
                    source=index,
                    backend=backend,
                    max_read_failures=1,
                    reconnect_attempts=0,
                )
            )
            try:
                source.open()
                packet = source.read()
                if packet is None:
                    print(f"Index {index}: opened but returned no frame")
                    continue
                found += 1
                print(
                    f"Index {index}: {packet.metadata['width']}x"
                    f"{packet.metadata['height']} via {source.backend}"
                )
                if args.preview:
                    window = f"Camera index {index}"
                    cv2.imshow(window, packet.frame)
                    print("Press Q to continue scanning.")
                    cv2.waitKey(0)
                    cv2.destroyWindow(window)
            except CaptureError as exc:
                print(f"Index {index}: unavailable ({exc})")
            finally:
                source.close()
    except KeyboardInterrupt:
        print("\nCamera scan stopped.", file=sys.stderr)
        return 130
    finally:
        if args.preview:
            cv2.destroyAllWindows()

    print(f"Found {found} camera source(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
