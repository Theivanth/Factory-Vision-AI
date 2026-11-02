import csv
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st

from src.alerts import (
    Alert,
    alert_record,
    assess_ppe,
    build_alerts,
    quality_disposition,
)
from src.domain import Detection, PPEAssessment, QualityPrediction
from src.preprocess import decode_image, iter_video_frames
from src.quality import CHECKPOINTS, QualityClassifier
from src.safety import SafetyPipeline


PROJECT_ROOT = Path(__file__).resolve().parent


@st.cache_resource
def get_safety_pipeline(
    image_size: int = 640,
    confidence_threshold: float = 0.25,
) -> SafetyPipeline:
    return SafetyPipeline(
        image_size=image_size,
        confidence_threshold=confidence_threshold,
    )


@st.cache_resource
def get_quality_classifier(checkpoint_path: str) -> QualityClassifier:
    return QualityClassifier(checkpoint_path)


def build_training_command(
    model_name: str,
    epochs: int,
    batch_size: int,
    device: str,
    image_size: int = 640,
    learning_rate: float = 0.001,
    seed: int = 42,
    weights: str = "yolo11n.pt",
) -> list[str]:
    command = [sys.executable, "-u"]
    if model_name == "PPE YOLO":
        command.extend(
            [
                str(PROJECT_ROOT / "scripts" / "train_safety.py"),
                "--confirm-training",
                "--weights",
                weights,
                "--epochs",
                str(epochs),
                "--image-size",
                str(image_size),
                "--batch-size",
                str(batch_size),
                "--seed",
                str(seed),
            ]
        )
        if device != "Automatic":
            command.extend(["--device", "0" if device == "CUDA:0" else "cpu"])
        return command

    architectures = {
        "Custom CNN": "cnn",
        "MobileNetV3-Small": "mobilenet_v3_small",
    }
    if model_name not in architectures:
        raise ValueError(f"Unknown training model: {model_name}")
    architecture = architectures[model_name]
    return command + [
        "-m",
        "scripts.train_quality",
        "--architecture",
        architecture,
        "--confirm-training",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--seed",
        str(seed),
        "--device",
        {"Automatic": "auto", "CPU": "cpu", "CUDA:0": "cuda"}[device],
    ]


def _stream_training_output(command: list[str], output_area: Any) -> None:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    output_lines: list[str] = []
    with subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    ) as process:
        if process.stdout is None:
            raise RuntimeError("Training process output could not be captured.")
        for line in process.stdout:
            output_lines.append(line.rstrip())
            output_area.code("\n".join(output_lines[-60:]), language="text")
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"Training failed with exit code {return_code}.\n"
            + "\n".join(output_lines[-60:])
        )


def _render_training_controls() -> None:
    with st.sidebar.expander("Local model training", expanded=False):
        st.caption(
            "Training runs locally and can take substantial time and compute. "
            "It is opt-in and may replace the selected checkpoint."
        )
        model_name = st.selectbox(
            "Model to train",
            ["PPE YOLO", "Custom CNN", "MobileNetV3-Small"],
            key="training_model",
        )
        with st.form("local_training_form"):
            epochs = st.slider("Training epochs", 1, 100, 10)
            batch_size = st.slider(
                "Batch size",
                1,
                64,
                16 if model_name == "PPE YOLO" else 32,
            )
            device = st.selectbox(
                "Training device",
                ["Automatic", "CPU", "CUDA:0"],
                help="Automatic uses the training script's available device.",
            )
            image_size = 640
            learning_rate = 0.001
            seed = 42
            weights = "yolo11n.pt"
            if model_name == "PPE YOLO":
                weights = st.selectbox(
                    "Initial YOLO weights",
                    ["yolo11n.pt", "yolov8n.pt"],
                )
                image_size = st.slider(
                    "Training image size",
                    320,
                    1280,
                    640,
                    step=32,
                )
                seed = st.slider("Training seed", 0, 9999, 0)
            else:
                learning_rate = st.slider(
                    "Learning rate",
                    min_value=0.00001,
                    max_value=0.01,
                    value=0.001,
                    step=0.00001,
                    format="%.5f",
                )
                seed = st.slider("Training seed", 0, 9999, 42)

            confirmed = st.checkbox(
                "I understand this starts local training and may overwrite a checkpoint."
            )
            submitted = st.form_submit_button(
                "Start local training",
                type="primary",
                disabled=not confirmed,
            )

        if not submitted:
            return

        command = build_training_command(
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            image_size=image_size,
            learning_rate=learning_rate,
            seed=seed,
            weights=weights,
        )
        log_area = st.empty()
        status = st.status(f"Training {model_name}...", expanded=True)
        try:
            _stream_training_output(command, log_area)
        except RuntimeError as error:
            status.update(label=f"{model_name} training failed", state="error")
            st.error(str(error))
            return

        get_safety_pipeline.clear()
        get_quality_classifier.clear()
        status.update(label=f"{model_name} training completed", state="complete")
        st.success("Training completed. The dashboard will load the updated checkpoint.")


def _inference_controls() -> tuple[int, float]:
    with st.sidebar.expander("Inference settings", expanded=False):
        image_size = st.slider(
            "Inference image size",
            320,
            1280,
            640,
            step=32,
            help="Larger inputs may improve small-object visibility but use more compute.",
        )
        confidence = st.slider(
            "Detection confidence",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            help="Lower values return more detections and may increase false alerts.",
        )
    return image_size, confidence


def _annotate_image(
    image_bgr: np.ndarray,
    detections: list[Detection],
    ppe_assessments: list[PPEAssessment],
) -> np.ndarray:
    annotated = image_bgr.copy()
    assessment_by_box = {assessment.box: assessment for assessment in ppe_assessments}

    for detection in detections:
        x1, y1, x2, y2 = (int(value) for value in detection.box)
        assessment = assessment_by_box.get(detection.box)
        normalized_label = detection.label.lower().replace("-", " ").replace("_", " ")
        if assessment is not None:
            color = (0, 0, 255) if assessment.status == "Potential missing helmet" else (0, 200, 0)
            label = f"{detection.label}: {assessment.status}"
        elif normalized_label in {
            "hardhat",
            "helmet",
            "safety helmet",
            "no hardhat",
            "no helmet",
            "no safety helmet",
            "no safety vest",
        }:
            is_violation = normalized_label.startswith("no ")
            color = (0, 0, 255) if is_violation else (0, 200, 0)
            label = f"{detection.label} {detection.confidence:.2f}"
        else:
            color = (0, 190, 255)
            label = f"{detection.label} {detection.confidence:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    for detection in detections:
        if detection.label.lower().replace("-", " ") not in {"hardhat", "helmet"}:
            continue
        x1, y1, _, _ = (int(value) for value in detection.box)
        cv2.putText(
            annotated,
            "Helmet",
            (x1, max(y1 - 24, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 0),
            2,
        )
    return annotated


def _display_frame_results(
    detections: list[Detection],
    ppe_assessments: list[PPEAssessment],
    ppe_model_available: bool,
    quality_prediction: QualityPrediction | None,
) -> None:
    if not ppe_model_available:
        st.info(
            "The fine-tuned PPE model is not available. PPE status cannot be assessed; "
            "general-object detections are shown without PPE alerts."
        )
    elif ppe_assessments:
        for assessment in ppe_assessments:
            if assessment.status == "Helmet detected":
                st.success(
                    f"Helmet detected for this worker "
                    f"({assessment.confidence:.1%}); no PPE alert."
                )
            else:
                st.warning(
                    f"{assessment.status} "
                    f"({assessment.confidence:.1%}); sent for human review."
                )
    elif any(
        detection.label.lower() in {"person", "worker", "head"}
        for detection in detections
    ):
        st.info("A person was detected, but the PPE model did not return a helmet assessment.")
    else:
        st.info("No person or head was detected in this frame.")

    if quality_prediction is None:
        return
    disposition = quality_disposition(quality_prediction)
    if disposition == "Acceptable":
        st.success(f"Cast acceptable ({quality_prediction.confidence:.1%} confidence).")
    elif disposition == "Manual inspection required":
        st.warning(
            f"Defect detected — manual quality inspection required "
            f"({quality_prediction.confidence:.1%} confidence)."
        )
    else:
        st.info(
            f"Unrecognized quality class '{quality_prediction.label}' "
            f"({quality_prediction.confidence:.1%} confidence)."
        )


def _run_frame(
    image_bgr: np.ndarray,
    source: str,
    frame_index: int | None,
    quality_model: QualityClassifier | None,
    image_size: int = 640,
    detection_confidence: float = 0.25,
) -> list[Alert]:
    pipeline = get_safety_pipeline(image_size, detection_confidence)
    safety_detections = pipeline.predict(image_bgr)
    ppe_assessments = assess_ppe(
        safety_detections,
        ppe_model_available=pipeline.ppe_model_available,
    )
    quality_prediction = (
        quality_model.predict(image_bgr) if quality_model is not None else None
    )
    frame_alerts = build_alerts(
        safety_detections,
        quality_prediction,
        source=source,
        frame_index=frame_index,
        ppe_model_available=pipeline.ppe_model_available,
    )
    _display_frame_results(
        safety_detections,
        ppe_assessments,
        pipeline.ppe_model_available,
        quality_prediction,
    )
    annotated = _annotate_image(image_bgr, safety_detections, ppe_assessments)
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
    return frame_alerts


def _quality_model_selector() -> QualityClassifier | None:
    selected = st.sidebar.selectbox(
        "Casting classifier",
        ["Custom CNN", "MobileNetV3-Small", "PPE only (no casting)"],
    )
    if selected == "PPE only (no casting)":
        return None
    checkpoint = CHECKPOINTS[selected]
    if not checkpoint.is_file():
        st.sidebar.info(
            f"{selected} is not trained yet. Quality inspection will be skipped."
        )
        return None
    return get_quality_classifier(str(checkpoint))


def _render_review_queue(records: list[dict[str, object]]) -> None:
    st.subheader("Human review queue")
    if not records:
        st.info("No alerts yet. Process an image, camera snapshot, or video.")
        return

    display_fields = (
        "created_at",
        "source",
        "frame_index",
        "category",
        "message",
        "severity",
        "confidence",
        "review_status",
    )
    st.dataframe(
        [{key: record[key] for key in display_fields} for record in records],
        use_container_width=True,
        hide_index=True,
    )
    selected_id = st.selectbox(
        "Select alert to review",
        options=[record["alert_id"] for record in records],
        format_func=lambda alert_id: next(
            f"{record['category']}: {record['message']} ({record['review_status']})"
            for record in records
            if record["alert_id"] == alert_id
        ),
    )
    decision = st.radio(
        "Review decision",
        ["Confirmed", "Dismissed"],
        horizontal=True,
        key="review_decision",
    )
    if st.button("Save review"):
        for record in records:
            if record["alert_id"] == selected_id:
                record["review_status"] = decision
                break
        st.rerun()

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    st.download_button(
        "Download alert log",
        output.getvalue(),
        file_name="factoryvision_alerts.csv",
        mime="text/csv",
    )


def _process_image_input(
    quality_model: QualityClassifier | None,
    image_size: int,
    detection_confidence: float,
) -> list[Alert]:
    input_mode = st.sidebar.radio("Image source", ["Image", "Camera snapshot"])
    if input_mode == "Image":
        uploaded_image = st.file_uploader(
            "Choose an image",
            type=("jpg", "jpeg", "png", "bmp"),
        )
        image_bytes = uploaded_image.getvalue() if uploaded_image else None
        source_name = uploaded_image.name if uploaded_image else "image"
    else:
        captured_image = st.camera_input("Take a camera snapshot")
        image_bytes = captured_image.getvalue() if captured_image else None
        source_name = "camera snapshot"

    if image_bytes is None or not st.button("Inspect image"):
        return []
    image = decode_image(image_bytes)
    return _run_frame(
        image,
        source_name,
        None,
        quality_model,
        image_size,
        detection_confidence,
    )


def _process_video_input(
    quality_model: QualityClassifier | None,
    image_size: int,
    detection_confidence: float,
) -> list[Alert]:
    uploaded_video = st.file_uploader(
        "Choose a video",
        type=("mp4", "avi", "mov", "mkv"),
    )
    sample_every = st.sidebar.slider("Sample every N frames", 1, 60, 15)
    if uploaded_video is None or not st.button("Inspect video"):
        return []

    suffix = Path(uploaded_video.name).suffix or ".mp4"
    temporary_path: Path | None = None
    frame_alerts: list[Alert] = []
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as video_file:
            video_file.write(uploaded_video.getvalue())
            temporary_path = Path(video_file.name)
        for frame_index, timestamp, frame in iter_video_frames(
            temporary_path,
            sample_every=sample_every,
            max_frames=30,
        ):
            st.markdown(f"**Frame {frame_index}** · {timestamp:.2f} seconds")
            frame_alerts.extend(
                _run_frame(
                    frame,
                    uploaded_video.name,
                    frame_index,
                    quality_model,
                    image_size,
                    detection_confidence,
                )
            )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return frame_alerts


def main() -> None:
    st.set_page_config(page_title="FactoryVisionAI", layout="wide")
    st.title("FactoryVisionAI")
    st.caption("Human-reviewed PPE and casting inspection prototype")
    st.error(
        "NOT A CERTIFIED INDUSTRIAL SAFETY SYSTEM. This educational prototype is "
        "not certified, validated, or approved for workplace safety or product "
        "release. Never use it as a safety control or as a substitute for qualified "
        "human inspection."
    )
    st.session_state.setdefault("alerts", [])

    _render_training_controls()
    image_size, detection_confidence = _inference_controls()
    input_mode = st.sidebar.radio("Input", ["Image / camera", "Video"])
    quality_model = _quality_model_selector()
    if input_mode == "Image / camera":
        new_alerts = _process_image_input(
            quality_model,
            image_size,
            detection_confidence,
        )
    else:
        new_alerts = _process_video_input(
            quality_model,
            image_size,
            detection_confidence,
        )

    if new_alerts:
        st.session_state["alerts"].extend(
            alert_record(alert) for alert in new_alerts
        )
        st.success(f"Added {len(new_alerts)} alert(s) for human review.")
    _render_review_queue(st.session_state["alerts"])


if __name__ == "__main__":
    main()
