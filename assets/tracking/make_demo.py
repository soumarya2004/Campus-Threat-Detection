"""Generate the submission's synthetic clip and matching detection inputs."""

import json
from pathlib import Path

import cv2
import numpy as np


WIDTH = 480
HEIGHT = 270
FPS = 20
FRAME_COUNT = 54


def generate(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    video_path = output / "approach-and-separate.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not create the demo video at {video_path}")

    replay = {}
    try:
        for frame_id in range(FRAME_COUNT):
            step = min(frame_id, FRAME_COUNT - 1 - frame_id)
            people = [
                [78 + 4 * step, 97, 112 + 4 * step, 163],
                [368 - 4 * step, 115, 402 - 4 * step, 181],
            ]
            trigger = frame_id in (4, 30)
            error = frame_id == 9
            frame = np.full((HEIGHT, WIDTH, 3), (38, 33, 28), dtype=np.uint8)
            cv2.line(frame, (0, 185), (WIDTH - 1, 185), (86, 78, 71), 2)
            cv2.putText(
                frame, "SYNTHETIC  /  20 FPS", (16, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 237, 225), 1, cv2.LINE_AA,
            )
            for (x1, y1, x2, y2), color in zip(people, [(235, 170, 81), (130, 212, 162)]):
                cv2.circle(frame, ((x1 + x2) // 2, y1 + 10), 9, color, -1)
                cv2.rectangle(frame, (x1 + 5, y1 + 22), (x2 - 5, y2), color, -1)
            note = (
                "Scripted trigger" if trigger else
                "Scripted inference error" if error else "No scripted trigger"
            )
            cv2.putText(
                frame, f"Frame {frame_id:02}  /  {note}", (16, 245),
                cv2.FONT_HERSHEY_SIMPLEX, 0.53, (240, 237, 225), 1, cv2.LINE_AA,
            )
            writer.write(frame)
            # Reversing the rows exercises independence from incoming box order.
            replay[str(frame_id)] = {
                "people": people if frame_id % 2 == 0 else people[::-1],
                "trigger": trigger,
                "inference_error": error,
            }
    finally:
        writer.release()

    for name, frames in (
        ("triggers.json", replay),
        ("no-trigger.json", {key: {"people": value["people"]} for key, value in replay.items()}),
    ):
        (output / name).write_text(
            json.dumps({"frames": frames}, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent / "demo")
