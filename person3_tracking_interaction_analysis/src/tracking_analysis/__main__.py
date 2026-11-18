"""Command-line interface for the recorded-video experiment."""

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("run", help="analyze a recorded video")
    command.add_argument("--video", required=True, help="path to the input video")
    command.add_argument("--output", required=True, help="path to a new run directory")
    command.add_argument("--mode", required=True, choices=("continuous", "gated"))
    command.add_argument("--person-weights", help="local YOLOv8n person-model weights")
    command.add_argument("--item-weights", help="local candidate-object detector weights")
    command.add_argument("--replay", help="JSON fixture of synthetic person boxes and triggers")
    command.add_argument("--device", default="cpu")
    command.add_argument("--grace", type=float, default=0.8, help="gate grace period in seconds")
    command.add_argument("--proximity-px", type=float, default=150)
    command.add_argument("--distance-change-px", type=float, default=5)
    command.add_argument("--max-speed-px-s", type=float, default=600)

    args = vars(parser.parse_args(argv))
    args.pop("command")
    try:
        import cv2

        from .pipeline import run
    except ImportError as error:
        print(f"Install the project requirements before running the demo. {error}", file=sys.stderr)
        return 1

    try:
        result = run(**args)
    except (OSError, ValueError, RuntimeError, ImportError, cv2.error) as error:
        print(f"Tracking run failed: {error}", file=sys.stderr)
        return 1

    print(
        f'Processed {result["frame_count"]} frames '
        f'({result["active_frames"]} active) in {result["elapsed_s"]:.2f} s.'
    )
    print(f'Run saved to {args["output"]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
