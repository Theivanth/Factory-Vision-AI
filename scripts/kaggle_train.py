import csv
import hashlib
import json
import logging
import platform
import random
import shutil
import sys
import time
import zipfile
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, TextIO

import torch
import torchvision
import ultralytics
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


QUALITY_CLASSES = ("def_front", "ok_front")
PPE_CLASSES = (
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "machinery",
    "vehicle",
)
OUTPUT_ROOT = Path("/kaggle/working/factoryvision_artifacts")
IMAGE_SIZE = 224
LOGGER = logging.getLogger("factoryvision")


class TeeStream:
    def __init__(self, console: TextIO, log_file: TextIO) -> None:
        self.console = console
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.console.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self.console.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.console.isatty()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.console, name)


@contextmanager
def capture_console_output(log_path: Path) -> Generator[None, None, None]:
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        with redirect_stdout(TeeStream(sys.stdout, log_file)):
            with redirect_stderr(TeeStream(sys.stderr, log_file)):
                yield


class CastingCNN(nn.Module):
    def __init__(self, num_classes: int = len(QUALITY_CLASSES)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


class TransformedSubset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        dataset: datasets.ImageFolder,
        indices: list[int],
        transform: transforms.Compose,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.dataset.samples[self.indices[index]]
        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
        return self.transform(image_rgb), label


def build_quality_model(architecture: str, pretrained: bool = False) -> nn.Module:
    if architecture == "cnn":
        return CastingCNN()
    if architecture == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(input_features, len(QUALITY_CLASSES))
        return model
    raise ValueError(f"Unknown quality architecture: {architecture}")


def discover_ppe_root(input_root: Path) -> Path:
    candidates = []
    for train_path in input_root.rglob("train"):
        if not train_path.is_dir():
            continue
        candidate = train_path.parent
        if all(
            (candidate / split / kind).is_dir()
            for split in ("train", "valid", "test")
            for kind in ("images", "labels")
        ):
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(
            f"No PPE train/valid/test images and labels found under {input_root}."
        )
    return sorted(candidates, key=lambda path: len(path.parts))[0]


def discover_casting_root(input_root: Path) -> Path:
    candidates = []
    for train_path in input_root.rglob("train"):
        if not train_path.is_dir():
            continue
        candidate = train_path.parent
        if all(
            (candidate / split / class_name).is_dir()
            for split in ("train", "test")
            for class_name in QUALITY_CLASSES
        ):
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(
            f"No casting train/test class folders found under {input_root}."
        )
    return sorted(candidates, key=lambda path: len(path.parts))[0]


def validate_ppe_dataset(dataset_root: Path) -> None:
    for split in ("train", "valid", "test"):
        for kind in ("images", "labels"):
            directory = dataset_root / split / kind
            if not directory.is_dir() or not any(directory.iterdir()):
                raise ValueError(f"PPE dataset directory is missing or empty: {directory}")


def validate_casting_dataset(dataset_root: Path) -> None:
    for split in ("train", "test"):
        for class_name in QUALITY_CLASSES:
            directory = dataset_root / split / class_name
            if not directory.is_dir() or not any(directory.iterdir()):
                raise ValueError(
                    f"Casting dataset directory is missing or empty: {directory}"
                )


def setup_logging(output_root: Path) -> Path:
    log_path = output_root / "training.log"
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)
    return log_path


def inspect_ppe_annotations(dataset_root: Path) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    for split in ("train", "valid", "test"):
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"
        class_counts = {class_name: 0 for class_name in PPE_CLASSES}
        image_stems = {
            path.stem
            for path in image_dir.iterdir()
            if path.is_file()
        }
        label_stems = {path.stem for path in label_dir.glob("*.txt")}
        if image_stems != label_stems:
            raise ValueError(
                f"PPE {split} images and label files do not match "
                f"(images without labels: {len(image_stems - label_stems)}, "
                f"labels without images: {len(label_stems - image_stems)})."
            )
        for label_path in label_dir.glob("*.txt"):
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 5:
                    raise ValueError(f"Malformed YOLO annotation: {label_path}:{line_number}")
                class_id = int(fields[0])
                coordinates = [float(value) for value in fields[1:]]
                if not 0 <= class_id < len(PPE_CLASSES):
                    raise ValueError(f"Invalid PPE class id at {label_path}:{line_number}")
                if not all(0 <= value <= 1 for value in coordinates):
                    raise ValueError(
                        f"Out-of-range YOLO coordinates at {label_path}:{line_number}"
                    )
                class_counts[PPE_CLASSES[class_id]] += 1
        splits[split] = {
            "image_count": len(image_stems),
            "annotation_count": sum(class_counts.values()),
            "class_counts": class_counts,
        }
    return {
        "root": str(dataset_root),
        "classes": list(PPE_CLASSES),
        "splits": splits,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_casting_dataset(dataset_root: Path) -> dict[str, Any]:
    split_hashes: dict[str, dict[str, str]] = {}
    split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "test"):
        split_hashes[split] = {}
        split_counts[split] = {}
        for class_name in QUALITY_CLASSES:
            paths = [
                path
                for path in (dataset_root / split / class_name).rglob("*")
                if path.is_file()
            ]
            if not paths:
                raise ValueError(f"No images found for {split}/{class_name}.")
            split_counts[split][class_name] = len(paths)
            for path in sorted(paths):
                digest = sha256_file(path)
                previous = split_hashes[split].get(digest)
                if previous is not None:
                    raise ValueError(
                        f"Duplicate casting image within the {split} split: "
                        f"{previous} and {path}"
                    )
                split_hashes[split][digest] = str(path)

    leaked_hashes = set(split_hashes["train"]) & set(split_hashes["test"])
    excluded_train_paths = []
    for digest in sorted(leaked_hashes):
        train_path = Path(split_hashes["train"][digest])
        test_path = Path(split_hashes["test"][digest])
        if train_path.parent.name != test_path.parent.name:
            raise ValueError(
                "An exact duplicate casting image has conflicting train/test labels: "
                f"{train_path} ({train_path.parent.name}) and "
                f"{test_path} ({test_path.parent.name})."
            )
        excluded_train_paths.append(
            train_path.relative_to(dataset_root).as_posix()
        )
    excluded_by_class = {
        class_name: sum(
            Path(path).parts[1] == class_name
            for path in excluded_train_paths
        )
        for class_name in QUALITY_CLASSES
    }
    return {
        "root": str(dataset_root),
        "classes": list(QUALITY_CLASSES),
        "split_counts": split_counts,
        "effective_train_counts": {
            class_name: split_counts["train"][class_name] - excluded_by_class[class_name]
            for class_name in QUALITY_CLASSES
        },
        "exact_duplicates_across_train_test": len(leaked_hashes),
        "train_paths_excluded_to_prevent_test_leakage": excluded_train_paths,
    }


def inspect_datasets(
    ppe_root: Path,
    casting_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    report = {
        "ppe": inspect_ppe_annotations(ppe_root),
        "casting": inspect_casting_dataset(casting_root),
        "casting_validation_split": (
            "Stratified 20% holdout from train; original test split remains untouched."
        ),
    }
    report_path = output_root / "dataset_inspection.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Dataset inspection report: %s", report_path)
    LOGGER.info("Dataset counts: %s", json.dumps(report, sort_keys=True))
    return report


def calculate_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection_width = max(min(first[2], second[2]) - max(first[0], second[0]), 0.0)
    intersection_height = max(
        min(first[3], second[3]) - max(first[1], second[1]),
        0.0,
    )
    intersection = intersection_width * intersection_height
    first_area = max(first[2] - first[0], 0.0) * max(first[3] - first[1], 0.0)
    second_area = max(second[2] - second[0], 0.0) * max(
        second[3] - second[1],
        0.0,
    )
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def match_boxes(
    predictions: list[tuple[float, float, float, float, float]],
    ground_truth: list[tuple[float, float, float, float]],
    iou_threshold: float,
) -> tuple[int, int, int]:
    matched: set[int] = set()
    true_positive = 0
    false_positive = 0
    for prediction in sorted(predictions, key=lambda box: box[4], reverse=True):
        overlaps = [
            (calculate_iou(prediction[:4], truth), index)
            for index, truth in enumerate(ground_truth)
            if index not in matched
        ]
        best_iou, best_index = max(overlaps, default=(0.0, -1))
        if best_iou >= iou_threshold:
            true_positive += 1
            matched.add(best_index)
        else:
            false_positive += 1
    false_negative = len(ground_truth) - len(matched)
    return true_positive, false_positive, false_negative


def read_person_annotations(label_path: Path, width: int, height: int) -> list[tuple[float, float, float, float]]:
    boxes = []
    person_class_id = PPE_CLASSES.index("Person")
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Malformed YOLO annotation: {label_path}:{line_number}")
        if int(fields[0]) != person_class_id:
            continue
        center_x, center_y, box_width, box_height = (
            float(value)
            for value in fields[1:]
        )
        boxes.append(
            (
                (center_x - box_width / 2) * width,
                (center_y - box_height / 2) * height,
                (center_x + box_width / 2) * width,
                (center_y + box_height / 2) * height,
            )
        )
    return boxes


def evaluate_pretrained_person_baseline(
    ppe_root: Path,
    output_root: Path,
    image_size: int,
    max_visualizations: int = 12,
) -> dict[str, Any]:
    import cv2
    from ultralytics import YOLO

    test_image_dir = ppe_root / "test" / "images"
    test_label_dir = ppe_root / "test" / "labels"
    image_paths = sorted(
        path
        for path in test_image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_paths:
        raise ValueError(f"No PPE test images found in {test_image_dir}.")

    baseline_dir = output_root / "baseline_pretrained_yolo"
    visualization_dir = baseline_dir / "sample_predictions"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolo11n.pt")
    class_names = model.names
    class_items = class_names.items() if isinstance(class_names, dict) else enumerate(class_names)
    person_id = next(
        (int(index) for index, label in class_items if str(label).lower() == "person"),
        None,
    )
    if person_id is None:
        raise ValueError("Pretrained YOLO model does not expose a person class.")

    true_positive = false_positive = false_negative = 0
    ground_truth_count = prediction_count = image_count = visualizations = 0
    latency_total_ms = 0.0
    for result in model.predict(
        source=[str(path) for path in image_paths],
        classes=[person_id],
        imgsz=image_size,
        conf=0.25,
        stream=True,
        verbose=False,
    ):
        image_path = Path(str(result.path))
        image = result.orig_img
        height, width = image.shape[:2]
        annotation_path = test_label_dir / f"{image_path.stem}.txt"
        ground_truth = (
            read_person_annotations(annotation_path, width, height)
            if annotation_path.is_file()
            else []
        )
        predictions = []
        if result.boxes is not None:
            predictions = [
                (
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                    float(confidence),
                )
                for box, confidence in zip(
                    result.boxes.xyxy.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(),
                )
            ]
        tp, fp, fn = match_boxes(predictions, ground_truth, iou_threshold=0.5)
        true_positive += tp
        false_positive += fp
        false_negative += fn
        ground_truth_count += len(ground_truth)
        prediction_count += len(predictions)
        image_count += 1
        latency_total_ms += float(result.speed.get("inference", 0.0))
        if visualizations < max_visualizations:
            output_path = visualization_dir / image_path.name
            if not cv2.imwrite(str(output_path), result.plot()):
                raise OSError(f"Could not save baseline prediction to {output_path}.")
            visualizations += 1

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    metrics = {
        "model": "YOLO11n pretrained COCO",
        "task": "person-only baseline (not PPE compliance)",
        "split": "test",
        "images": image_count,
        "ground_truth_persons": ground_truth_count,
        "predicted_persons": prediction_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "mean_inference_latency_ms": latency_total_ms / image_count if image_count else 0.0,
        "sample_visualizations": visualizations,
    }
    (baseline_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Pretrained YOLO baseline: %s", json.dumps(metrics, sort_keys=True))
    return metrics


def build_ppe_config(ppe_root: Path, output_root: Path) -> Path:
    config = {
        "path": str(ppe_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(PPE_CLASSES),
        "names": list(PPE_CLASSES),
    }
    config_path = output_root / "ppe_data.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def train_ppe_yolo(
    ppe_root: Path,
    output_root: Path,
    epochs: int,
    image_size: int,
    batch_size: int,
) -> tuple[Path, dict[str, float]]:
    from ultralytics import YOLO

    config_path = build_ppe_config(ppe_root, output_root)
    run_name = f"ppe_yolo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = output_root / "yolo_runs"
    device = 0 if torch.cuda.is_available() else "cpu"
    LOGGER.info(
        "Fine-tuning PPE YOLO: epochs=%d image_size=%d batch_size=%d device=%s",
        epochs,
        image_size,
        batch_size,
        device,
    )
    model = YOLO("yolo11n.pt")
    model.train(
        data=str(config_path),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        workers=2,
        project=str(run_root),
        name=run_name,
        patience=10,
        save=True,
    )
    best_weights = run_root / run_name / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Best PPE weights were not produced: {best_weights}")

    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_dir / "ppe_yolo.pt"
    shutil.copy2(best_weights, destination)
    test_results = YOLO(str(destination)).val(
        data=str(config_path),
        split="test",
        imgsz=image_size,
        batch=batch_size,
        device=device,
        project=str(run_root),
        name=f"{run_name}_test",
    )
    metrics = {
        str(key): float(value)
        for key, value in test_results.results_dict.items()
    }
    (output_root / "ppe_test_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Fine-tuned PPE test metrics: %s", json.dumps(metrics, sort_keys=True))
    return destination, metrics


def make_transforms(training: bool) -> transforms.Compose:
    steps: list[Any] = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if training:
        steps.append(transforms.RandomHorizontalFlip())
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(steps)


def make_quality_loaders(
    casting_root: Path,
    batch_size: int,
    seed: int,
    excluded_train_paths: set[str],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = datasets.ImageFolder(casting_root / "train")
    test_dataset = datasets.ImageFolder(
        casting_root / "test",
        transform=make_transforms(training=False),
    )
    if tuple(train_dataset.classes) != QUALITY_CLASSES:
        raise ValueError(
            f"Expected casting classes {QUALITY_CLASSES}, got {train_dataset.classes}."
        )
    if tuple(test_dataset.classes) != QUALITY_CLASSES:
        raise ValueError(
            f"Train/test casting class order differs: {train_dataset.classes}, "
            f"{test_dataset.classes}."
        )
    excluded_samples = [
        sample
        for sample in train_dataset.samples
        if Path(sample[0]).relative_to(casting_root).as_posix()
        in excluded_train_paths
    ]
    if len(excluded_samples) != len(excluded_train_paths):
        raise ValueError(
            "Casting inspection reported duplicate training paths that do not match "
            "the loaded ImageFolder samples."
        )
    retained_train_samples = [
        sample
        for sample in train_dataset.samples
        if Path(sample[0]).relative_to(casting_root).as_posix()
        not in excluded_train_paths
    ]
    train_dataset.samples = retained_train_samples
    train_dataset.imgs = retained_train_samples
    train_dataset.targets = [label for _, label in retained_train_samples]

    class_indices = {
        class_id: [
            index
            for index, (_, label) in enumerate(train_dataset.samples)
            if label == class_id
        ]
        for class_id in range(len(QUALITY_CLASSES))
    }
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for class_id, indices in class_indices.items():
        random.Random(seed + class_id).shuffle(indices)
        validation_count = max(1, round(len(indices) * 0.2))
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])

    train_data = TransformedSubset(
        train_dataset,
        train_indices,
        make_transforms(training=True),
    )
    validation_data = TransformedSubset(
        train_dataset,
        validation_indices,
        make_transforms(training=False),
    )
    worker_count = min(4, torch.get_num_threads())
    return (
        DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=worker_count,
            pin_memory=torch.cuda.is_available(),
        ),
        DataLoader(
            validation_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=worker_count,
            pin_memory=torch.cuda.is_available(),
        ),
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=worker_count,
            pin_memory=torch.cuda.is_available(),
        ),
    )


def calculate_quality_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    confusion = torch.zeros(
        len(QUALITY_CLASSES),
        len(QUALITY_CLASSES),
        dtype=torch.int64,
    )
    inference_seconds = 0.0
    image_count = 0
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started_at = time.perf_counter()
            predictions = model(images).argmax(dim=1).cpu()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - started_at
            labels = labels.cpu()
            image_count += int(labels.size(0))
            for actual, predicted in zip(labels, predictions):
                confusion[int(actual), int(predicted)] += 1

    precision_by_class: dict[str, float] = {}
    recall_by_class: dict[str, float] = {}
    f1_by_class: dict[str, float] = {}
    for index, class_name in enumerate(QUALITY_CLASSES):
        true_positive = int(confusion[index, index])
        false_positive = int(confusion[:, index].sum()) - true_positive
        false_negative = int(confusion[index, :].sum()) - true_positive
        precision_by_class[class_name] = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall_by_class[class_name] = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1_denominator = 2 * true_positive + false_positive + false_negative
        f1_by_class[class_name] = (
            2 * true_positive / f1_denominator if f1_denominator else 0.0
        )

    total = int(confusion.sum())
    return {
        "accuracy": int(confusion.diag().sum()) / total if total else 0.0,
        "precision_macro": sum(precision_by_class.values()) / len(QUALITY_CLASSES),
        "recall_macro": sum(recall_by_class.values()) / len(QUALITY_CLASSES),
        "macro_f1": sum(f1_by_class.values()) / len(QUALITY_CLASSES),
        "per_class_precision": precision_by_class,
        "per_class_recall": recall_by_class,
        "per_class_f1": f1_by_class,
        "confusion_matrix": confusion.tolist(),
        "classes": list(QUALITY_CLASSES),
        "mean_inference_latency_ms": (
            inference_seconds * 1000 / image_count if image_count else 0.0
        ),
    }


def train_quality_model(
    architecture: str,
    casting_root: Path,
    output_root: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    excluded_train_paths: set[str],
) -> tuple[Path, dict[str, Any]]:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_loader, validation_loader, test_loader = make_quality_loaders(
        casting_root,
        batch_size,
        seed,
        excluded_train_paths,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_quality_model(
        architecture,
        pretrained=architecture == "mobilenet_v3_small",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    checkpoint_dir = output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"casting_{architecture}.pth"
    history_path = output_root / f"{architecture}_history.csv"
    history_fields = [
        "epoch",
        "train_loss",
        "validation_accuracy",
        "validation_precision_macro",
        "validation_recall_macro",
        "validation_macro_f1",
        "validation_def_front_f1",
        "validation_ok_front_f1",
    ]
    best_validation_f1 = -1.0
    LOGGER.info(
        "Training quality model %s for %d epochs on %s",
        architecture,
        epochs,
        device,
    )
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history_fields)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(model(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())

            validation_metrics = calculate_quality_metrics(
                model,
                validation_loader,
                device,
            )
            average_loss = total_loss / max(len(train_loader), 1)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": average_loss,
                    "validation_accuracy": validation_metrics["accuracy"],
                    "validation_precision_macro": validation_metrics["precision_macro"],
                    "validation_recall_macro": validation_metrics["recall_macro"],
                    "validation_macro_f1": validation_metrics["macro_f1"],
                    "validation_def_front_f1": validation_metrics["per_class_f1"]["def_front"],
                    "validation_ok_front_f1": validation_metrics["per_class_f1"]["ok_front"],
                }
            )
            file.flush()
            LOGGER.info(
                "%s epoch %d/%d loss=%.4f validation_accuracy=%.4f "
                "validation_macro_f1=%.4f",
                architecture,
                epoch,
                epochs,
                average_loss,
                validation_metrics["accuracy"],
                validation_metrics["macro_f1"],
            )
            if validation_metrics["macro_f1"] > best_validation_f1:
                best_validation_f1 = validation_metrics["macro_f1"]
                torch.save(
                    {
                        "architecture": architecture,
                        "classes": list(QUALITY_CLASSES),
                        "state_dict": {
                            name: tensor.detach().cpu()
                            for name, tensor in model.state_dict().items()
                        },
                        "image_size": IMAGE_SIZE,
                        "validation_macro_f1": best_validation_f1,
                        "seed": seed,
                    },
                    checkpoint_path,
                )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    validation_metrics = calculate_quality_metrics(model, validation_loader, device)
    test_metrics = calculate_quality_metrics(model, test_loader, device)
    results = {
        "checkpoint": checkpoint_path.name,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    metrics_path = output_root / f"{architecture}_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("%s results: %s", architecture, json.dumps(results, sort_keys=True))
    return checkpoint_path, results


def package_artifacts(
    output_root: Path,
    ppe_metrics: dict[str, float],
    baseline_metrics: dict[str, Any],
    quality_results: dict[str, dict[str, Any]],
    best_architecture: str,
) -> Path:
    best_quality_checkpoint = output_root / "checkpoints" / "casting_best.pth"
    shutil.copy2(
        output_root / "checkpoints" / f"casting_{best_architecture}.pth",
        best_quality_checkpoint,
    )
    report = {
        "pretrained_person_baseline": baseline_metrics,
        "ppe_test": ppe_metrics,
        "quality_models": quality_results,
        "best_quality_architecture": best_architecture,
        "selection_metric": "validation macro-F1",
        "test_split_used_for_selection": False,
    }
    report_path = output_root / "training_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    included_files: dict[str, Path] = {}
    for folder_name in ("checkpoints", "baseline_pretrained_yolo", "yolo_runs"):
        folder = output_root / folder_name
        if folder.is_dir():
            for file_path in folder.rglob("*"):
                if file_path.is_file():
                    included_files[file_path.relative_to(output_root).as_posix()] = file_path
    for path in output_root.iterdir():
        if path.is_file() and path.name != "artifact_manifest.json":
            included_files[path.relative_to(output_root).as_posix()] = path
    included_files["training_metrics.json"] = report_path

    required = {
        "checkpoints/ppe_yolo.pt",
        "checkpoints/casting_best.pth",
        "training.log",
        "console.log",
        "dataset_inspection.json",
        "baseline_pretrained_yolo/metrics.json",
    }
    if missing := required - included_files.keys():
        raise FileNotFoundError(
            f"Training output bundle is incomplete: {sorted(missing)}"
        )

    manifest = {
        "format_version": 1,
        "quality_architecture": best_architecture,
        "files": {
            name: sha256_file(path)
            for name, path in included_files.items()
        },
    }
    manifest_path = output_root / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    included_files["artifact_manifest.json"] = manifest_path
    archive_path = output_root / "factoryvision_kaggle_models.zip"
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for name, path in included_files.items():
            archive.write(path, arcname=name)
    return archive_path


def run_training() -> Path:
    import argparse
    from ultralytics import YOLO

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--ppe-epochs", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--ppe-batch-size", type=int, default=16)
    parser.add_argument("--quality-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.ppe_epochs, args.image_size, args.ppe_batch_size, args.quality_batch_size) < 1:
        parser.error("Epoch counts, image size, and batch sizes must be positive.")
    if args.learning_rate <= 0:
        parser.error("Learning rate must be greater than zero.")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "training.log"
    runtime_metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__,
        "pillow": Image.__version__,
        "pyyaml": yaml.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "training_arguments": {
            "epochs": args.epochs,
            "ppe_epochs": args.ppe_epochs,
            "image_size": args.image_size,
            "ppe_batch_size": args.ppe_batch_size,
            "quality_batch_size": args.quality_batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "reproducibility": {
            "ppe_seed": 0,
            "ppe_deterministic": True,
            "quality_seed": args.seed,
            "quality_validation_fraction": 0.2,
            "quality_selection_metric": "validation macro-F1",
            "test_split_used_for_model_selection": False,
        },
    }
    (output_root / "runtime_metadata.json").write_text(
        json.dumps(runtime_metadata, indent=2),
        encoding="utf-8",
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(console_handler)

    with capture_console_output(output_root / "console.log"):
        try:
            ppe_root = discover_ppe_root(args.input_root)
            casting_root = discover_casting_root(args.input_root)
            validate_ppe_dataset(ppe_root)
            validate_casting_dataset(casting_root)
            inspection = inspect_datasets(ppe_root, casting_root, output_root)

            baseline_metrics = evaluate_pretrained_person_baseline(
                ppe_root,
                output_root,
                args.image_size,
            )
            config_path = build_ppe_config(ppe_root, output_root)
            run_name = f"ppe_yolo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run_root = output_root / "yolo_runs"
            device = 0 if torch.cuda.is_available() else "cpu"
            LOGGER.info(
                "Starting PPE YOLO fine-tuning for %d epochs on %s",
                args.ppe_epochs,
                device,
            )
            detector = YOLO("yolo11n.pt")
            detector.train(
                data=str(config_path),
                epochs=args.ppe_epochs,
                imgsz=args.image_size,
                batch=args.ppe_batch_size,
                device=device,
                workers=2,
                project=str(run_root),
                name=run_name,
                patience=10,
                save=True,
            )
            best_ppe = run_root / run_name / "weights" / "best.pt"
            if not best_ppe.is_file():
                raise FileNotFoundError(f"Best PPE checkpoint missing: {best_ppe}")
            checkpoint_dir = output_root / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ppe_checkpoint = checkpoint_dir / "ppe_yolo.pt"
            shutil.copy2(best_ppe, ppe_checkpoint)
            ppe_test = YOLO(str(ppe_checkpoint)).val(
                data=str(config_path),
                split="test",
                imgsz=args.image_size,
                batch=args.ppe_batch_size,
                device=device,
                project=str(run_root),
                name=f"{run_name}_test",
            )
            ppe_metrics = {
                str(key): float(value)
                for key, value in ppe_test.results_dict.items()
            }
            (output_root / "ppe_test_metrics.json").write_text(
                json.dumps(ppe_metrics, indent=2),
                encoding="utf-8",
            )
            LOGGER.info("PPE test results: %s", json.dumps(ppe_metrics, sort_keys=True))

            quality_results: dict[str, dict[str, Any]] = {}
            for architecture in ("cnn", "mobilenet_v3_small"):
                _, results = train_quality_model(
                    architecture,
                    casting_root,
                    output_root,
                    epochs=args.epochs,
                    batch_size=args.quality_batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                    excluded_train_paths=set(
                        inspection["casting"][
                            "train_paths_excluded_to_prevent_test_leakage"
                        ]
                    ),
                )
                quality_results[architecture] = results
            best_architecture = max(
                quality_results,
                key=lambda architecture: quality_results[architecture]["validation"]["macro_f1"],
            )
            archive_path = package_artifacts(
                output_root,
                ppe_metrics,
                baseline_metrics,
                quality_results,
                best_architecture,
            )
            LOGGER.info("Input inspection: %s", json.dumps(inspection, sort_keys=True))
            LOGGER.info("Best quality architecture: %s", best_architecture)
            LOGGER.info("Training outputs: %s", output_root)
            LOGGER.info("Model bundle: %s", archive_path)
            return archive_path
        except Exception:
            LOGGER.exception("FactoryVisionAI Kaggle training failed.")
            raise


def main() -> None:
    run_training()


if __name__ == "__main__":
    main()
