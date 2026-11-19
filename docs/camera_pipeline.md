# Shared camera and video input

Person 4's capture component lives in `src/camera_source.py`. It gives the
detector, temporal classifier, and tracking pipeline a common source format.
Frames stay in OpenCV BGR order, frame IDs begin at zero and increase across a
camera reconnect, and timestamps travel with each frame.

## Configure a source

Edit `configs/camera.yaml` for the default input. A numeric `source` selects a
camera index; a path selects a recorded clip. Relative video paths are resolved
from the repository root. The defaults request a one-frame capture buffer and
retry a live camera after five consecutive failed reads.

Use command-line overrides for one-off runs:

```powershell
python scripts/run_camera_demo.py --camera 1 --backend msmf --with-people --device cpu
python scripts/run_camera_demo.py --video path/to/clip.mp4 --device cpu --max-frames 120 --no-display --report outputs/camera/clip.json
python -m src.tracking_analysis run --video path/to/clip.mp4 --mode continuous --device cpu --output outputs/tracking/clip-continuous
```

`--camera` and `--video` are mutually exclusive. Available backends are
`auto`, `any`, `msmf`, and `dshow`. Requested dimensions, FPS, and capture
buffer size are best-effort because camera drivers can ignore them. The camera
demo report records whether OpenCV accepted each requested setting.

For DroidCam, connect the phone using the method supported by the installed
DroidCam version, confirm the virtual camera appears in Windows, then locate its
device index:

```powershell
python scripts/find_cameras.py --backend msmf
```

Add `--preview` to display one captured frame for each available index. Select
the working index in `configs/camera.yaml` or pass it to `--camera`.

## Frame metadata

`CameraSource.read()` returns a `CapturedFrame` or `None` at recorded-video EOF.
Each packet contains the BGR frame and these metadata fields:

- `frame_id`: zero-based ID that does not reset on reconnect
- `timestamp_s`: source-relative seconds; advancing media time for recorded clips when available, otherwise host monotonic elapsed time
- `timestamp_utc`: capture time in UTC for logs and event ordering across modules
- `monotonic_time`: host monotonic clock value for latency calculations
- `fps`, `width`, `height`: measured capture rate and decoded frame dimensions
- `source`, `backend`, `reconnect_count`, `read_failures`: capture status

Live camera timestamps are host times, not hardware sensor timestamps. A video
file falls back to host monotonic time when OpenCV reports missing or repeated
media timestamps. The timestamp source is included in the packet.

## Runtime behavior

`scripts/run_camera_demo.py` uses the shared source directly, runs the item model
and optional person model on the same frame, and closes models and capture in a
`finally` path. `src.behavior.interfaces.frame_provider.StreamFrameProvider`
uses a one-frame latest-only queue for a live camera, counting discarded stale
frames; for recorded clips it uses a bounded FIFO queue with backpressure so
every decoded frame is preserved. The Person 3 analysis command also reads
recorded video through `CameraSource`.

The input module reports capture errors separately from model inference errors.
Its retry policy is bounded by `reconnect_attempts`; a source that does not
recover stops with a `CaptureReadError`. The capture class owns no windows and
does not mutate or copy frames.

## Project limits

The capture interface is integrated with the existing detector, temporal
inference adapter, and tracking runner. The supplied archive does not contain a
loadable Person 2 checkpoint: `models/behavior/best_model.pt` is a Git LFS
pointer. Fetch the LFS object from the configured repository remote or train a
checkpoint before running the behavior demo. Capture success and interface
tests do not measure model accuracy or establish that a real incident occurred.
