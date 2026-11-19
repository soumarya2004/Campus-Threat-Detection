"""Local operator dashboard for pipeline sessions and evaluation reports."""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.monitoring.session import load_dashboard_session  # noqa: E402

st.set_page_config(
    page_title="Campus Threat Detection",
    page_icon="🛡️",
    layout="wide",
)

OUTPUT_ROOT = (ROOT / "outputs").resolve()


def resolve_session(value: str) -> Path:
    path = Path(value).expanduser()
    resolved = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(OUTPUT_ROOT)
    except ValueError as exc:
        raise ValueError(
            "Choose a session folder inside the repository outputs directory"
        ) from exc
    return resolved


def label(value, missing="Unavailable"):
    if value is None or value == "":
        return missing
    return str(value).replace("_", " ").title()


def render_status(snapshot: dict) -> None:
    status = snapshot.get("status") or {}
    if not status:
        st.info(
            "No live-session status is available. An evaluation report can still be reviewed below."
        )
        return

    capture = status.get("capture") or {}
    models = status.get("models") or {}
    performance = status.get("performance") or {}
    current = status.get("current_frame") or {}
    assessment = status.get("assessment") or {"level": "UNASSESSED"}

    updated_text = status.get("updated_at_utc")
    stale = False
    if updated_text:
        try:
            updated = datetime.fromisoformat(updated_text.replace("Z", "+00:00"))
            stale = (datetime.now(timezone.utc) - updated).total_seconds() > 8
        except ValueError:
            st.warning("The session update timestamp is malformed.")

    if status.get("status") == "error":
        st.error(
            f"Pipeline stopped with an error: {status.get('error', 'No details supplied.')}"
        )
    elif status.get("status") == "running" and stale:
        st.warning(
            "The session has stopped updating. Its last prediction may be stale."
        )
    elif status.get("status") == "running":
        st.success("Pipeline is updating")
    else:
        st.info(f"Session status: {label(status.get('status'))}")

    level = assessment.get("level", "UNASSESSED")
    if level == "HIGH_RISK":
        st.error(
            "HIGH RISK — operator review required. This is a system assessment, not a verified incident."
        )
    elif level == "SUSPICIOUS":
        st.warning("SUSPICIOUS — review the available evidence.")
    elif level == "NORMAL":
        st.success("NORMAL — assessment supplied by the upstream policy.")
    else:
        st.info(
            "UNASSESSED — no final threat policy output was supplied. Missing predictions are not negative predictions."
        )
    if assessment.get("reason"):
        st.caption(assessment["reason"])

    columns = st.columns(5)
    columns[0].metric("Camera / video", label(capture.get("status")))
    columns[1].metric("Object model", label(models.get("object_detection")))
    columns[2].metric("Behavior model", label(models.get("behavior_classification")))
    columns[3].metric("Tracking", label(models.get("tracking")))
    fps = performance.get("processing_fps")
    fps_target = performance.get("minimum_fps_target", 30)
    columns[4].metric(
        "Processing FPS",
        f"{fps:.1f}" if isinstance(fps, (int, float)) else "—",
        delta=(f"{fps - fps_target:+.1f} vs {fps_target} FPS target")
        if isinstance(fps, (int, float))
        else None,
        delta_color="normal",
    )

    cols = st.columns([1, 1, 2])
    cols[0].metric("Frame", label(current.get("frame_id"), "No frame"))
    latency = performance.get("capture_to_dashboard_ms")
    cols[1].metric(
        "Capture to dashboard",
        f"{latency:.1f} ms" if isinstance(latency, (int, float)) else "—",
    )
    cols[2].caption(
        f"Input: {label(capture.get('source_label'))} · "
        f"Updated: {updated_text or 'unknown'} · "
        f"Reconnects: {capture.get('reconnect_count', '—')} · "
        f"Dropped frames: {performance.get('dropped_frames', '—')}"
    )

    behavior = current.get("behavior")
    if isinstance(behavior, dict) or models.get("behavior_classification") in {
        "ready",
        "error",
    }:
        behavior = behavior or {}
        probability = behavior.get("violence_probability")
        behavior_state = behavior.get("model_status", "buffering")
        behavior_cols = st.columns(3)
        behavior_cols[0].metric(
            "Behavior score",
            f"{probability:.1%}"
            if isinstance(probability, (int, float))
            else label(behavior_state),
        )
        behavior_cols[1].metric(
            "Behavior inference",
            label(behavior_state),
        )
        behavior_cols[2].metric(
            "Active person tracks",
            str((current.get("tracking") or {}).get("active_tracks", 0)),
        )

    image_path = snapshot.get("latest_image")
    if image_path:
        st.image(str(image_path), caption="Latest processed frame", width="stretch")
    else:
        st.info("No processed frame is available in this session.")

    detections = current.get("detections")
    st.markdown("**Current frame evidence**")
    if not isinstance(detections, list):
        st.caption("The producer has not supplied current-frame detections.")
    elif not detections:
        st.caption("No detections were returned for this frame.")
    else:
        rows = []
        for detection in detections:
            if not isinstance(detection, dict):
                continue
            box = detection.get("bbox")
            rows.append(
                {
                    "Source": detection.get("source", "unknown"),
                    "Class": detection.get("class_name", "unknown"),
                    "Confidence": detection.get("confidence"),
                    "Bounding box": ", ".join(map(str, box))
                    if isinstance(box, (list, tuple))
                    else None,
                    "Track ID": detection.get("track_id"),
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.caption("No valid detection records were supplied for this frame.")


def render_event_log(snapshot: dict) -> None:
    st.subheader("Event history")
    if not snapshot.get("events_available"):
        st.info("The upstream pipeline has not supplied an event log.")
        return
    events = snapshot.get("events", [])
    if not events:
        st.caption("The event log is present and contains no assessed events.")
        return

    levels = sorted({str(event.get("level", "UNASSESSED")) for event in events})
    selected = st.multiselect("Show event levels", levels, default=levels)
    visible = [
        event for event in events if str(event.get("level", "UNASSESSED")) in selected
    ]
    if not visible:
        st.caption("No events match the selected levels.")
        return
    rows = [
        {
            "Time": event.get("recorded_at_utc", ""),
            "Event": event.get("event_id", ""),
            "Level": event.get("level", "UNASSESSED"),
            "Evidence": ", ".join(map(str, event.get("evidence_categories", []))),
            "Start (s)": event.get("start_s"),
            "End (s)": event.get("end_s"),
            "Summary": event.get("summary", ""),
        }
        for event in visible
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    selected_id = st.selectbox(
        "Review event details",
        options=[
            event.get("event_id", str(index)) for index, event in enumerate(visible)
        ],
    )
    chosen = next(
        (
            event
            for index, event in enumerate(visible)
            if event.get("event_id", str(index)) == selected_id
        ),
        None,
    )
    if chosen:
        st.json(chosen)


def render_evaluation(report: dict | None) -> None:
    st.subheader("External evaluation")
    if not report:
        st.info("No evaluation report is present in this session folder.")
        return

    st.caption(
        f"{report.get('dataset', 'Evaluation')} · {report.get('split', 'split not recorded')}"
    )
    if report.get("evaluation_scope"):
        st.write(report["evaluation_scope"])

    video = report.get("video_metrics") or {}
    event = report.get("event_metrics") or {}
    runtime = report.get("runtime_metrics") or {}
    metric_rows = [
        ("Video F1", video.get("f1"), "Thresholded video-level score"),
        ("Video AUROC", video.get("auroc"), "Ranking metric"),
        ("Event F1", event.get("f1"), "Temporal-IoU matched events"),
        (
            "False alarms / normal hour",
            event.get("false_alarms_per_normal_hour"),
            "Normal UCF-Crime videos only",
        ),
        (
            "Mean detection delay",
            event.get("mean_detection_delay_s"),
            "Seconds from annotated start",
        ),
        (
            "Capture to score p95",
            runtime.get("p95_capture_to_score_ms"),
            "Milliseconds",
        ),
    ]
    columns = st.columns(3)
    for index, (title, value, help_text) in enumerate(metric_rows):
        formatted = (
            "—"
            if value is None
            else (f"{value:.3f}" if isinstance(value, (int, float)) else str(value))
        )
        columns[index % 3].metric(title, formatted, help=help_text)

    st.caption(
        f"Scored videos: {video.get('scored_videos', '—')} · "
        f"Unscored in-scope videos: {video.get('unscored_in_scope_videos', '—')} · "
        f"Temporal false positives / misses: {event.get('false_positive', '—')} / {event.get('false_negative', '—')} · "
        f"Processed FPS: {runtime.get('video_processing_fps', '—')}"
    )

    per_video = report.get("per_video")
    if isinstance(per_video, list) and per_video:
        predicted_events = []
        for video in per_video:
            for event_item in video.get("predicted_events", []):
                predicted_events.append(
                    {
                        "Video": video.get("video_id"),
                        "Category": video.get("category"),
                        "Start (s)": event_item.get("start_s"),
                        "End (s)": event_item.get("end_s"),
                        "Peak behavior score": event_item.get("peak_score"),
                        "Assessment": "UNASSESSED",
                    }
                )
        if predicted_events:
            st.markdown("**Predicted event intervals**")
            st.dataframe(
                pd.DataFrame(predicted_events), hide_index=True, width="stretch"
            )
            st.caption(
                "These are thresholded behavior-model outputs, not verified incidents or final threat assessments."
            )
        frame = pd.DataFrame(per_video)
        keep = [
            name
            for name in (
                "video_id",
                "category",
                "target_label",
                "annotation_status",
                "scored",
                "max_violence_probability",
                "prediction",
            )
            if name in frame.columns
        ]
        st.markdown("**Per-video review**")
        st.dataframe(frame[keep], hide_index=True, width="stretch")
        st.download_button(
            "Download evaluation report",
            data=__import__("json").dumps(report, indent=2, allow_nan=False),
            file_name="ucf_crime_evaluation_report.json",
            mime="application/json",
        )
    for note in report.get("limitations", []):
        st.caption(f"Scope: {note}")


def render_video_upload() -> None:
    st.subheader("Recorded video demo")
    st.caption(
        "Drop a short video to generate an annotated output with object detections, "
        "person tracks, and a behavior-model score. Predictions are evidence for "
        "human review, not confirmed incidents."
    )
    st.caption(
        "The clip is processed on the server running this dashboard. Use the "
        "laptop-local dashboard for private footage."
    )

    with st.form("recorded_video_form"):
        uploaded_video = st.file_uploader(
            "Choose or drop a video",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            help="Short clips finish sooner and are easier to present.",
        )
        process_video = st.form_submit_button(
            "Process video", type="primary", disabled=uploaded_video is None
        )

    video_bytes = uploaded_video.getvalue() if uploaded_video is not None else None
    upload_digest = (
        hashlib.sha256(video_bytes).hexdigest() if video_bytes is not None else None
    )
    previous_result = st.session_state.get("recorded_video_result")
    if previous_result and previous_result.get("upload_digest") != upload_digest:
        st.session_state.pop("recorded_video_result", None)
        previous_result = None

    if process_video and uploaded_video is not None and video_bytes is not None:
        progress_bar = st.progress(0, text="Preparing the video…")

        def report_progress(processed: int, total: int) -> None:
            if total > 0:
                fraction = min(0.95, 0.9 * processed / total)
                progress_bar.progress(
                    fraction, text=f"Processed {processed:,} of {total:,} frames"
                )
            else:
                progress_bar.progress(
                    0.0, text=f"Processed {processed:,} frames"
                )

        try:
            from src.video_upload import process_uploaded_video

            with st.spinner("Loading models and rendering the annotated video…"):
                result = process_uploaded_video(
                    video_bytes,
                    uploaded_video.name,
                    progress_callback=report_progress,
                )
        except Exception as exc:
            progress_bar.empty()
            st.error(f"Could not process this video: {exc}")
        else:
            progress_bar.empty()
            result["upload_digest"] = upload_digest
            result["filename"] = uploaded_video.name
            st.session_state["recorded_video_result"] = result
            previous_result = result

    if previous_result and previous_result.get("upload_digest") == upload_digest:
        st.markdown("#### Annotated output")
        st.video(previous_result["video_bytes"], format="video/mp4")
        first_row = st.columns(3)
        second_row = st.columns(3)
        first_row[0].metric("Frames", f"{previous_result['frames']:,}")
        first_row[1].metric(
            "Clip duration", f"{previous_result['duration_seconds']:.1f} s"
        )
        first_row[2].metric(
            "Object detections", f"{previous_result['object_detections']:,}"
        )
        second_row[0].metric(
            "Person detections", f"{previous_result['person_detections']:,}"
        )
        second_row[1].metric("Active tracks", previous_result["active_tracks"])
        second_row[2].metric(
            "Peak behavior score",
            f"{previous_result['peak_behavior_score']:.1%}"
            if previous_result["peak_behavior_score"] is not None
            else "Warming up",
        )
        st.download_button(
            "Download annotated video",
            data=previous_result["video_bytes"],
            file_name=f"{Path(previous_result['filename']).stem}_annotated.mp4",
            mime="video/mp4",
            key=f"download_{upload_digest[:12]}",
        )
        st.caption(
            f"Processed on {previous_result['device']} at "
            f"{previous_result['processing_fps']:.1f} FPS. "
            "The output is silent; the original audio is not retained. "
            "Behavior scores are unverified model outputs."
        )
        if previous_result["inference_errors"]:
            st.warning(
                "The object detector reported inference errors in "
                f"{previous_result['inference_errors']} frames."
            )
        if previous_result["behavior_status"] == "buffering":
            st.info(
                "This clip was too short to fill the behavior model's temporal window. "
                "Upload a longer clip for a behavior score."
            )


def render() -> None:
    st.title("Campus Threat Detection")
    st.caption(
        "Local monitoring and evaluation viewer · predictions are evidence for human review"
    )
    render_video_upload()

    with st.sidebar:
        st.header("Session")
        session_value = st.text_input("Output folder", "outputs/dashboard/latest")
        refresh = st.slider("Dashboard refresh interval (seconds)", 1, 10, 1)
        st.caption(
            "The viewer reads output files only. It does not start model inference."
        )

    try:
        session_path = resolve_session(session_value)
        snapshot = load_dashboard_session(session_path)
    except (OSError, ValueError) as exc:
        st.error(str(exc))
        return

    for warning in snapshot.get("warnings", []):
        st.warning(warning)

    @st.fragment(run_every=refresh)
    def live_panel(path: str) -> None:
        current = load_dashboard_session(path)
        render_status(current)
        render_event_log(current)
        render_evaluation(current.get("evaluation"))

    live_panel(str(session_path))
    st.caption(
        "Latest frames can contain identifiable people. Keep session folders on a trusted local disk and remove them after review."
    )


render()
