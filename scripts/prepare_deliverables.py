import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from src.alerts import alert_record, assess_ppe, build_alerts, quality_disposition
from src.domain import Detection
from src.quality import CHECKPOINTS, QUALITY_CLASSES, QualityClassifier
from src.safety import SafetyPipeline
from scripts.deploy_kaggle import (
    NOTEBOOK_NAME,
    TRAINING_SCRIPT,
    make_notebook,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "runs"
    / "kaggle"
    / "20261102_140021"
    / "kaggle_output"
    / "factoryvision_artifacts"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports"
PPE_TEST_IMAGES = (
    PROJECT_ROOT / "datasets" / "raw" / "ppe" / "css-data" / "test" / "images"
)
CASTING_TEST = (
    PROJECT_ROOT
    / "datasets"
    / "raw"
    / "casting"
    / "casting_data"
    / "casting_data"
    / "test"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required run evidence is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return data


def image_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset image directory is missing: {directory}")
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not paths:
        raise ValueError(f"No supported images found in {directory}.")
    return paths


def add_banner(image: Any, lines: list[str]) -> Any:
    scale = 0.48
    available_width = image.shape[1] - 24
    wrapped_lines = []
    for line in lines:
        words = line.split()
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            text_width = cv2.getTextSize(
                candidate,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                1,
            )[0][0]
            if current and text_width > available_width:
                wrapped_lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            wrapped_lines.append(current)
    line_height = 23
    banner_height = max(50, line_height * len(wrapped_lines) + 10)
    canvas = cv2.copyMakeBorder(
        image,
        banner_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(32, 32, 32),
    )
    for index, line in enumerate(wrapped_lines):
        cv2.putText(
            canvas,
            line,
            (12, 22 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def save_actual_ppe_predictions(
    output_dir: Path,
    test_images: list[Path],
) -> dict[str, str]:
    from app import _annotate_image

    pipeline = SafetyPipeline(
        pretrained_weights=None,
        ppe_weights=PROJECT_ROOT / "checkpoints" / "ppe_yolo.pt",
    )
    evidence: dict[str, str] = {}
    targets = {
        "ppe_potential_alert.png": "Potential missing helmet",
        "ppe_helmet_detected.png": "Helmet detected",
    }
    candidates: dict[str, tuple[int, Any, list[Detection], list[Any], Path, float]] = {}

    for image_path in test_images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Unable to decode PPE test image: {image_path}")
        detections = pipeline.predict(frame)
        assessments = assess_ppe(
            detections,
            ppe_model_available=pipeline.ppe_model_available,
        )
        label_path = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"PPE test annotation is missing: {label_path}")
        annotated_classes = {
            int(line.split(maxsplit=1)[0])
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        statuses = {assessment.status for assessment in assessments}
        if not {0, 2}.issubset(annotated_classes) or not set(targets.values()).issubset(
            statuses
        ):
            continue
        for filename, target_status in targets.items():
            matches = [
                assessment
                for assessment in assessments
                if assessment.status == target_status
            ]
            if not matches:
                continue
            confidence = max(assessment.confidence for assessment in matches)
            score = len(detections)
            if filename not in candidates or score < candidates[filename][0]:
                candidates[filename] = (
                    score,
                    frame.copy(),
                    detections,
                    assessments,
                    image_path,
                    confidence,
                )

    for filename, target_status in targets.items():
        if filename not in candidates:
            continue
        _, frame, detections, assessments, image_path, confidence = candidates[filename]
        annotated = _annotate_image(frame, detections, assessments)
        banner = add_banner(
            annotated,
            [
                "ACTUAL FINE-TUNED PPE YOLO | HELD-OUT TEST",
                f"{target_status} | confidence {confidence:.3f}",
                image_path.name,
            ],
        )
        destination = output_dir / filename
        if not cv2.imwrite(str(destination), banner):
            raise OSError(f"Could not write prediction evidence: {destination}")
        evidence[filename] = destination.relative_to(PROJECT_ROOT).as_posix()

    missing = set(targets) - set(evidence)
    if missing:
        raise RuntimeError(
            "The fine-tuned PPE model produced no annotation-supported held-out "
            "test examples for: "
            + ", ".join(sorted(missing))
        )
    return evidence


def save_actual_quality_predictions(
    output_dir: Path,
) -> dict[str, str]:
    checkpoint = CHECKPOINTS["MobileNetV3-Small"]
    classifier = QualityClassifier(checkpoint, device="cpu")
    evidence: dict[str, str] = {}

    for class_name in QUALITY_CLASSES:
        class_dir = CASTING_TEST / class_name
        for image_path in image_paths(class_dir):
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise ValueError(f"Unable to decode casting test image: {image_path}")
            prediction = classifier.predict(frame)
            if prediction.label != class_name:
                continue
            banner = add_banner(
                frame,
                [
                    "ACTUAL MOBILENETV3-SMALL PREDICTION | HELD-OUT TEST IMAGE",
                    (
                        f"Actual: {class_name} | Predicted: {prediction.label} | "
                        f"{prediction.confidence:.3f} | {quality_disposition(prediction)}"
                    ),
                ],
            )
            filename = f"casting_{class_name}.png"
            destination = output_dir / filename
            if not cv2.imwrite(str(destination), banner):
                raise OSError(f"Could not write prediction evidence: {destination}")
            evidence[filename] = destination.relative_to(PROJECT_ROOT).as_posix()
            break
        if f"casting_{class_name}.png" not in evidence:
            raise RuntimeError(
                f"MobileNetV3 did not correctly predict a {class_name} test sample."
            )
    return evidence


def save_live_webcam_frame(video_path: Path, destination: Path) -> None:
    from app import _annotate_image

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open recorded webcam demo: {video_path}")
    try:
        success, frame = capture.read()
    finally:
        capture.release()
    if not success:
        raise ValueError(f"Recorded webcam demo has no readable frames: {video_path}")

    pipeline = SafetyPipeline()
    detections = pipeline.predict(frame)
    assessments = assess_ppe(
        detections,
        ppe_model_available=pipeline.ppe_model_available,
    )
    alert_assessments = [
        assessment
        for assessment in assessments
        if assessment.status == "Potential missing helmet"
    ]
    alert_line = (
        f"Potential missing helmet | review required | "
        f"{max(item.confidence for item in alert_assessments):.3f}"
        if alert_assessments
        else "No potential missing-helmet alert on this frame"
    )
    annotated = _annotate_image(frame, detections, assessments)
    evidence_image = add_banner(
        annotated,
        [
            "ACTUAL LOCAL WEBCAM VIDEO | FRAME 0",
            alert_line,
            "Unverified model suggestion; human review required",
        ],
    )
    if not cv2.imwrite(str(destination), evidence_image):
        raise OSError(f"Could not write live webcam evidence: {destination}")


def benchmark_callable(
    name: str,
    predict: Any,
    warmup_runs: int,
    measured_runs: int,
) -> dict[str, Any]:
    for _ in range(warmup_runs):
        predict()
    durations_ms = []
    for _ in range(measured_runs):
        started = time.perf_counter()
        predict()
        durations_ms.append((time.perf_counter() - started) * 1000)
    return {
        "model": name,
        "device": "CPU",
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "mean_ms": statistics.mean(durations_ms),
        "median_ms": statistics.median(durations_ms),
        "p95_ms": sorted(durations_ms)[math.ceil(0.95 * len(durations_ms)) - 1],
        "images_per_second": 1000 / statistics.mean(durations_ms),
    }


def run_cpu_benchmarks(
    test_images: list[Path],
    warmup_runs: int,
    measured_runs: int,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    frame = cv2.imread(str(test_images[0]))
    if frame is None:
        raise ValueError(f"Unable to decode PPE benchmark image: {test_images[0]}")

    detector = SafetyPipeline(
        pretrained_weights=None,
        ppe_weights=PROJECT_ROOT / "checkpoints" / "ppe_yolo.pt",
    )
    ppe_result = benchmark_callable(
        "Fine-tuned PPE YOLO (preprocess, inference, and postprocess)",
        lambda: detector.predict(frame),
        warmup_runs,
        measured_runs,
    )

    results = [ppe_result]
    for model_name, checkpoint in (
        ("Custom PyTorch CNN", CHECKPOINTS["Custom CNN"]),
        ("MobileNetV3-Small", CHECKPOINTS["MobileNetV3-Small"]),
    ):
        classifier = QualityClassifier(checkpoint, device="cpu")
        results.append(
            benchmark_callable(
                f"{model_name} (preprocess and inference)",
                lambda active_classifier=classifier: active_classifier.predict(frame),
                warmup_runs,
                measured_runs,
            )
        )

    return {
        "benchmark_date_utc": datetime.now(timezone.utc).isoformat(),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "logical_cpu_count": __import__("os").cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "ultralytics": __import__("ultralytics").__version__,
            "opencv": cv2.__version__,
        },
        "timings": results,
        "notes": (
            "Single-image CPU latency includes image preprocessing and model "
            "postprocessing. Local workstation timings are hardware-specific and "
            "are not Kaggle GPU measurements."
        ),
    }


def write_demo_alert_csv(
    test_images: list[Path],
    destination: Path,
) -> int:
    pipeline = SafetyPipeline(ppe_weights=PROJECT_ROOT / "checkpoints" / "ppe_yolo.pt")
    classifier = QualityClassifier(CHECKPOINTS["Custom CNN"], device="cpu")
    image_path = next(
        (path for path in test_images if path.name.startswith("004063")),
        test_images[0],
    )
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Unable to decode CSV demonstration image: {image_path}")
    detections = pipeline.predict(frame)
    quality_prediction = classifier.predict(frame)
    alerts = build_alerts(
        detections,
        quality_prediction,
        source=image_path.name,
        ppe_model_available=pipeline.ppe_model_available,
    )
    if not alerts:
        raise RuntimeError(
            "The selected demonstration image produced no alert records to export."
        )
    records = [alert_record(alert) for alert in alerts]
    records[0]["review_status"] = "Confirmed"
    with destination.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def build_model_metadata(
    run_dir: Path,
    metrics: dict[str, Any],
    dataset_inspection: dict[str, Any],
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_paths = {
        "ppe_yolo": PROJECT_ROOT / "checkpoints" / "ppe_yolo.pt",
        "casting_cnn": PROJECT_ROOT / "checkpoints" / "casting_cnn.pth",
        "casting_mobilenet_v3_small": (
            PROJECT_ROOT / "checkpoints" / "casting_mobilenet_v3_small.pth"
        ),
    }
    missing = [str(path) for path in checkpoint_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Imported checkpoints are missing: " + ", ".join(missing))

    yolo_run_dirs = sorted((run_dir / "yolo_runs").glob("ppe_yolo_*"))
    yolo_args_path = next(
        (path / "args.yaml" for path in yolo_run_dirs if (path / "args.yaml").is_file()),
        None,
    )
    if yolo_args_path is None:
        raise FileNotFoundError("PPE YOLO args.yaml is missing from Kaggle artifacts.")
    yolo_args = yaml.safe_load(yolo_args_path.read_text(encoding="utf-8"))

    quality_histories: dict[str, int] = {}
    for architecture in ("cnn", "mobilenet_v3_small"):
        history_path = run_dir / f"{architecture}_history.csv"
        if not history_path.is_file():
            raise FileNotFoundError(f"Training history is missing: {history_path}")
        with history_path.open(encoding="utf-8", newline="") as history_file:
            quality_histories[architecture] = sum(1 for _ in csv.DictReader(history_file))

    runtime_metadata_path = run_dir / "runtime_metadata.json"
    kaggle_runtime = (
        read_json(runtime_metadata_path)
        if runtime_metadata_path.is_file()
        else {
            "note": (
                "The completed Kaggle run did not capture package/runtime versions. "
                "The training script now records them for future runs."
            )
        }
    )
    return {
        "kaggle_kernel": "theivanth/factoryvisionai",
        "kaggle_run_dir": run_dir.relative_to(PROJECT_ROOT).as_posix(),
        "training_metrics": metrics,
        "dataset_integrity": {
            "ppe_split_counts": dataset_inspection["ppe"]["splits"],
            "casting_split_counts": dataset_inspection["casting"]["split_counts"],
            "casting_effective_train_counts": (
                dataset_inspection["casting"]["effective_train_counts"]
            ),
            "casting_exact_duplicate_pairs_excluded": (
                dataset_inspection["casting"]["exact_duplicates_across_train_test"]
            ),
            "casting_excluded_training_paths": (
                dataset_inspection["casting"][
                    "train_paths_excluded_to_prevent_test_leakage"
                ]
            ),
        },
        "training_configuration": {
            "ppe": {
                "initial_weights": yolo_args["model"],
                "epochs": yolo_args["epochs"],
                "image_size": yolo_args["imgsz"],
                "batch_size": yolo_args["batch"],
                "seed": yolo_args["seed"],
                "deterministic": yolo_args["deterministic"],
                "optimizer": yolo_args["optimizer"],
                "device": yolo_args["device"],
            },
            "casting": {
                "cnn_and_mobilenet_epochs": quality_histories,
                "validation_fraction": 0.2,
                "seed": 42,
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "input_size": 224,
                "mobilenet_initial_weights": "MobileNet_V3_Small_Weights.DEFAULT",
                "quality_model_selection": "highest validation macro-F1",
                "test_split_used_for_selection": False,
            },
            "kaggle_notebook_metadata": (
                "GPU and Internet enabled; NvidiaTeslaT4; both datasets attached. "
                "Exact Kaggle runtime package versions were not captured in this run."
            ),
        },
        "kaggle_runtime": kaggle_runtime,
        "checkpoints": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for name, path in checkpoint_paths.items()
        },
        "cpu_benchmark": benchmark,
    }


def write_confusion_matrix(
    matrix: list[list[int]],
    classes: list[str],
    destination: Path,
    title: str,
) -> None:
    cell_size = 180
    margin_left = 230
    margin_top = 95
    canvas = np.full((560, 690, 3), 255, dtype=np.uint8)
    cv2.putText(
        canvas,
        title,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Predicted class",
        (margin_left + 120, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    for column, class_name in enumerate(classes):
        cv2.putText(
            canvas,
            class_name,
            (margin_left + column * cell_size + 15, margin_top - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
    for row, class_name in enumerate(classes):
        y = margin_top + row * cell_size
        cv2.putText(
            canvas,
            class_name,
            (25, y + 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        for column, count in enumerate(matrix[row]):
            x = margin_left + column * cell_size
            maximum = max(max(row_values) for row_values in matrix)
            shade = 235 - min(int(count / maximum * 150), 150)
            cv2.rectangle(canvas, (x, y), (x + cell_size, y + cell_size), (shade, 220, 175), -1)
            cv2.rectangle(canvas, (x, y), (x + cell_size, y + cell_size), (80, 80, 80), 1)
            cv2.putText(
                canvas,
                str(count),
                (x + 75, y + 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
    cv2.putText(
        canvas,
        "Actual class",
        (25, margin_top + 2 * cell_size + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(destination), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write confusion matrix image: {destination}")


def render_report(
    metrics: dict[str, Any],
    metadata: dict[str, Any],
    evidence: dict[str, str],
    report_path: Path,
) -> None:
    quality = metrics["quality_models"]
    cnn = quality["cnn"]["test"]
    mobile = quality["mobilenet_v3_small"]["test"]
    baseline = metrics["pretrained_person_baseline"]
    ppe = metrics["ppe_test"]
    duplicate_count = metadata["dataset_integrity"][
        "casting_exact_duplicate_pairs_excluded"
    ]
    benchmarks = metadata["cpu_benchmark"]["timings"]
    kaggle_runtime_note = (
        "Exact Kaggle runtime package versions were not captured by the completed "
        "run; they are now recorded for future runs."
        if "note" in metadata["kaggle_runtime"]
        else "Kaggle runtime package versions are recorded in `model_metadata.json`."
    )
    benchmark_rows = "\n".join(
        f"| {item['model']} | {item['mean_ms']:.2f} | {item['median_ms']:.2f} | "
        f"{item['p95_ms']:.2f} | {item['images_per_second']:.2f} |"
        for item in benchmarks
    )
    evidence_rows = "\n".join(
        f"| {name} | [Open](./{path.removeprefix('reports/')}) |"
        for name, path in evidence.items()
    )
    run_link = f"../{metadata['kaggle_run_dir']}"
    text = f"""# FactoryVisionAI engineering report

**Status:** prototype trained and locally smoke-tested<br>
**Kaggle run:** [theivanth/factoryvisionai](https://www.kaggle.com/code/theivanth/factoryvisionai)<br>
**Local run archive:** [`{metadata['kaggle_run_dir']}`]({run_link})

## Deliverable checklist

- [x] Reproducible Kaggle training notebook, GPU/runtime settings and attached
  datasets archived in [`../notebooks/factoryvisionai.ipynb`](../notebooks/factoryvisionai.ipynb)
  and [`../notebooks/kernel-metadata.json`](../notebooks/kernel-metadata.json).
- [x] Evidence of training, histories, logs, evaluation output and saved model bundle archived in [`{metadata['kaggle_run_dir']}`]({run_link})
- [x] Pretrained YOLO person-only baseline and fine-tuned PPE YOLO.
- [x] Custom PyTorch CNN and fine-tuned MobileNetV3-Small, with checkpoint metadata, test metrics and confusion matrices.
- [x] Streamlit application supports image upload, camera snapshot and video upload, annotation, review queue, review decisions and CSV export.
- [x] Actual held-out test predictions saved below; these are model outputs, not mock images.
- [x] CPU inference benchmark recorded on this workstation.
- [{'x' if 'webcam_demo.mp4' in evidence else ' '}] Short real webcam clip processed through the live dashboard; see the linked demo video.

## Architecture and implementation

OpenCV decodes image/video input and samples video frames. The safety pipeline
runs the fine-tuned YOLO model, associates helmet detections with individual
workers, and raises review alerts for potential missing helmets. A pretrained
COCO person detector is evaluated separately as a baseline; it is not treated as
a PPE classifier. The quality pipeline can select either the custom CNN or
MobileNetV3-Small, or be disabled for PPE-only camera/video inspection.
Predictions feed a shared decision engine, Streamlit review queue, and CSV
export. Alerts are human-review suggestions, not release or workplace-safety
decisions.

## Dataset provenance, integrity and license

- PPE: [Construction Site Safety Image Dataset](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow),
  2,605 train / 114 validation / 82 test images, 10 YOLO classes. Kaggle lists
  CC BY 4.0; retain attribution and inspect the dataset's included attribution.
- Casting: [Real-Life Industrial Dataset of Casting Product](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product),
  original train/test folders, binary `def_front` and `ok_front`. Kaggle lists
  CC BY-NC-ND 4.0; non-commercial/no-derivatives restrictions apply. Check that
  training, local model use, and any distribution comply before sharing.
- SHA-256 integrity inspection found **{duplicate_count}** exact image duplicates
  crossing casting train/test. The test copies were retained and matching train
  copies excluded from both casting models. Duplicate files within a split and
  conflicting labels cause training to stop. See the dataset inspection JSON.

## Training and reproducibility

PPE YOLO starts from `{metadata['training_configuration']['ppe']['initial_weights']}`,
uses {metadata['training_configuration']['ppe']['epochs']} epochs, image size
{metadata['training_configuration']['ppe']['image_size']}, batch size
{metadata['training_configuration']['ppe']['batch_size']}, seed
{metadata['training_configuration']['ppe']['seed']}, and deterministic training.
Quality models use a stratified 20% validation holdout from the training split,
seed 42, Adam at 0.001 and 15 epochs each. MobileNetV3-Small uses the default
ImageNet pretrained weights. The quality winner is chosen by validation macro-F1;
the test split is not used for model selection.

The archived Kaggle metadata enables Internet and an Nvidia Tesla T4 GPU and
attaches both datasets. {kaggle_runtime_note} Local package versions and the CPU
benchmark environment are recorded in `model_metadata.json`. The local
[`factoryvisionai.ipynb`](../notebooks/factoryvisionai.ipynb) is generated from
the current training script; the completed run's metrics and weights remain
those in the archived Kaggle output.

## Held-out evaluation

| Model / task | Precision | Recall | Macro-F1 / mAP50 | Test accuracy |
|---|---:|---:|---:|---:|
| Pretrained YOLO11n COCO person-only baseline | {baseline['precision']:.3f} | {baseline['recall']:.3f} | F1 {baseline['f1']:.3f} | n/a |
| Fine-tuned PPE YOLO, all detection classes | {ppe['metrics/precision(B)']:.3f} | {ppe['metrics/recall(B)']:.3f} | mAP50 {ppe['metrics/mAP50(B)']:.3f} | n/a |
| Custom PyTorch CNN, casting | {cnn['precision_macro']:.3f} | {cnn['recall_macro']:.3f} | {cnn['macro_f1']:.3f} | {cnn['accuracy']:.3f} |
| MobileNetV3-Small, casting | {mobile['precision_macro']:.3f} | {mobile['recall_macro']:.3f} | {mobile['macro_f1']:.3f} | {mobile['accuracy']:.3f} |

Casting class order is `def_front`, `ok_front`. On test data, the CNN confusion
matrix is `[[{cnn['confusion_matrix'][0][0]}, {cnn['confusion_matrix'][0][1]}], [{cnn['confusion_matrix'][1][0]}, {cnn['confusion_matrix'][1][1]}]]`;
MobileNetV3-Small is
`[[{mobile['confusion_matrix'][0][0]}, {mobile['confusion_matrix'][0][1]}], [{mobile['confusion_matrix'][1][0]}, {mobile['confusion_matrix'][1][1]}]]`.
The generated confusion matrix images are included in this folder.

For the selected MobileNetV3 model, **{mobile['confusion_matrix'][0][1]}** defective
test items were classified as acceptable (false negatives), and
**{mobile['confusion_matrix'][1][0]}** acceptable items were flagged defective
(false positives). These counts are from the supplied held-out split and do not
estimate factory performance. The PPE baseline had {baseline['false_positive']}
false positive and {baseline['false_negative']} false negative person boxes at
IoU 0.5. PPE detection metrics are aggregate detector metrics, not compliance
classification error counts.

## CPU benchmark

Measured locally with batch size 1, {benchmarks[0]['warmup_runs']} warmups and
{benchmarks[0]['measured_runs']} timed calls per model. The exact workstation,
runtime versions and measurement notes are in `model_metadata.json`.

| Model | Mean ms/image | Median ms/image | P95 ms/image | Images/s |
|---|---:|---:|---:|---:|
{benchmark_rows}

These are local CPU measurements, not real-time guarantees. Camera resolution,
threading, thermal conditions, frame sampling and model loading affect deployed
latency.

## Actual model prediction captures

| Capture | File |
|---|---|
{evidence_rows}

The dashboard review-queue screenshot and a CSV exported from the same
prediction/alert-record pipeline are included as application evidence.

## Known limitations and error risks

- Casting results are unusually high on this benchmark dataset. Exact duplicates
  were removed, but near-duplicates, capture backgrounds, class-specific image
  artifacts, and the small benchmark domain may still inflate scores.
- Three defective samples were missed by MobileNetV3 on this test split. Defect
  misses can cause unsafe acceptance; keep mandatory manual inspection and
  validate on independently collected factory data.
- PPE recall is lower than precision; missed people/PPE objects can suppress
  alerts. Crowding, occlusion, lighting, camera angle, PPE class imbalance and
  worker-to-helmet association can cause false alerts and missed violations.
- The webcam demonstration is a short local smoke test, not representative
  factory footage; do not interpret its safety or quality predictions as
  evaluation results.
- The pretrained baseline detects people only, not helmets or PPE compliance.
- No factory deployment or real factory-footage validation is claimed. The
  local webcam smoke test is not a representative safety evaluation. Review all
  alerts; do not use this prototype as an autonomous safety control or
  product-release authority.

## Evidence and artifacts

- Kaggle run output, training histories, logs, test metrics, Ultralytics curves,
  detector confusion matrices, and checkpoint bundle: [`{metadata['kaggle_run_dir']}`]({run_link})
- Current Kaggle notebook source and kernel settings: [`../notebooks/factoryvisionai.ipynb`](../notebooks/factoryvisionai.ipynb), [`../notebooks/kernel-metadata.json`](../notebooks/kernel-metadata.json)
- Machine-readable metadata and local CPU timings: [`model_metadata.json`](./model_metadata.json)
- Streamlit launch command and application workflow: [project README](../README.md)
"""
    report_path.write_text(text, encoding="utf-8")


def prepare_deliverables(
    run_dir: Path,
    output_dir: Path,
    warmup_runs: int,
    measured_runs: int,
) -> None:
    metrics = read_json(run_dir / "training_metrics.json")
    dataset_inspection = read_json(run_dir / "dataset_inspection.json")
    notebook_backup = run_dir.parent.parent / "notebook_backup"
    kernel_metadata = notebook_backup / "kernel-metadata.json"
    if not kernel_metadata.is_file():
        raise FileNotFoundError("The archived Kaggle kernel metadata is missing.")
    if not TRAINING_SCRIPT.is_file():
        raise FileNotFoundError(f"Training source is missing: {TRAINING_SCRIPT}")

    notebooks_dir = PROJECT_ROOT / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    notebook_data = make_notebook(
        TRAINING_SCRIPT.read_text(encoding="utf-8"),
        epochs=15,
        ppe_epochs=30,
    )
    (notebooks_dir / NOTEBOOK_NAME).write_text(
        json.dumps(notebook_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    shutil.copy2(kernel_metadata, notebooks_dir / "kernel-metadata.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir = output_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    ppe_test_images = image_paths(PPE_TEST_IMAGES)
    evidence = save_actual_ppe_predictions(screenshots_dir, ppe_test_images)
    evidence.update(save_actual_quality_predictions(screenshots_dir))
    webcam_demo = output_dir / "live_demo" / "webcam_demo.mp4"
    if webcam_demo.is_file():
        evidence[webcam_demo.name] = webcam_demo.relative_to(PROJECT_ROOT).as_posix()
        webcam_frame = screenshots_dir / "live_webcam_frame.png"
        save_live_webcam_frame(webcam_demo, webcam_frame)
        evidence[webcam_frame.name] = webcam_frame.relative_to(PROJECT_ROOT).as_posix()
    webcam_alerts = output_dir / "live_demo" / "webcam_alerts.csv"
    if webcam_alerts.is_file():
        evidence[webcam_alerts.name] = webcam_alerts.relative_to(PROJECT_ROOT).as_posix()

    for architecture in ("cnn", "mobilenet_v3_small"):
        quality_metrics = read_json(run_dir / f"{architecture}_metrics.json")
        matrix_path = screenshots_dir / f"{architecture}_confusion_matrix.png"
        write_confusion_matrix(
            quality_metrics["test"]["confusion_matrix"],
            quality_metrics["test"]["classes"],
            matrix_path,
            f"{architecture} | held-out casting test confusion matrix",
        )
        evidence[matrix_path.name] = matrix_path.relative_to(PROJECT_ROOT).as_posix()

    ppe_confusion_path = next(
        (run_dir / "yolo_runs").glob("ppe_yolo_*_test/confusion_matrix.png"),
        None,
    )
    if ppe_confusion_path is not None:
        destination = screenshots_dir / "ppe_yolo_test_confusion_matrix.png"
        shutil.copy2(ppe_confusion_path, destination)
        evidence[destination.name] = destination.relative_to(PROJECT_ROOT).as_posix()
    dashboard_screenshot = screenshots_dir / "dashboard_review_queue.png"
    if dashboard_screenshot.is_file():
        evidence[dashboard_screenshot.name] = (
            dashboard_screenshot.relative_to(PROJECT_ROOT).as_posix()
        )

    benchmark = run_cpu_benchmarks(
        ppe_test_images,
        warmup_runs,
        measured_runs,
    )
    csv_path = output_dir / "factoryvision_alerts.csv"
    alert_count = write_demo_alert_csv(ppe_test_images, csv_path)
    evidence[csv_path.name] = csv_path.relative_to(PROJECT_ROOT).as_posix()
    metadata = build_model_metadata(
        run_dir,
        metrics,
        dataset_inspection,
        benchmark,
    )
    metadata["application_demo"] = {
        "source_image": next(
            (path.name for path in ppe_test_images if path.name.startswith("004063")),
            ppe_test_images[0].name,
        ),
        "alert_records_in_csv": alert_count,
        "review_status_set_to_confirmed": True,
        "dashboard_screenshot": (
            dashboard_screenshot.relative_to(PROJECT_ROOT).as_posix()
            if dashboard_screenshot.is_file()
            else None
        ),
    }
    (output_dir / "cpu_benchmark.json").write_text(
        json.dumps(benchmark, indent=2),
        encoding="utf-8",
    )
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    render_report(
        metrics,
        metadata,
        evidence,
        output_dir / "ENGINEERING_REPORT.md",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package Kaggle run evidence, actual test-set prediction images, "
            "model metadata, and a local CPU benchmark."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--measured-runs", type=int, default=30)
    args = parser.parse_args()
    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative.")
    if args.measured_runs < 1:
        parser.error("--measured-runs must be positive.")
    return args


def main() -> None:
    args = parse_arguments()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not run_dir.is_relative_to(PROJECT_ROOT):
        raise ValueError("Keep the Kaggle run archive inside the project directory.")
    if not output_dir.is_relative_to(PROJECT_ROOT):
        raise ValueError("Write deliverables inside the project directory.")
    prepare_deliverables(
        run_dir,
        output_dir,
        args.warmup_runs,
        args.measured_runs,
    )
    print(f"Engineering report: {output_dir / 'ENGINEERING_REPORT.md'}")
    print(f"Evidence screenshots: {output_dir / 'screenshots'}")


if __name__ == "__main__":
    main()
