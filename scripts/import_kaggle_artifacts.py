import argparse
import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MAX_ARTIFACT_BYTES = 500 * 1024 * 1024
QUALITY_CLASSES = ("def_front", "ok_front")


def read_manifest(archive: zipfile.ZipFile) -> dict[str, Any]:
    try:
        manifest = json.loads(archive.read("artifact_manifest.json"))
    except (KeyError, json.JSONDecodeError) as error:
        raise ValueError("Artifact ZIP is missing a valid artifact_manifest.json.") from error
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported Kaggle artifact manifest version.")
    if not isinstance(manifest.get("files"), dict):
        raise ValueError("Artifact manifest has no valid files map.")
    return manifest


def get_verified_artifact(
    archive: zipfile.ZipFile,
    manifest: dict[str, Any],
    archive_name: str,
) -> bytes:
    expected_hash = manifest["files"].get(archive_name)
    if not isinstance(expected_hash, str):
        raise ValueError(f"Artifact manifest does not declare {archive_name}.")
    try:
        info = archive.getinfo(archive_name)
    except KeyError as error:
        raise ValueError(f"Artifact ZIP is missing {archive_name}.") from error
    if info.file_size > MAX_ARTIFACT_BYTES:
        raise ValueError(f"Artifact is unexpectedly large: {archive_name}.")
    data = archive.read(archive_name)
    actual_hash = hashlib.sha256(data).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"SHA-256 verification failed for {archive_name}.")
    return data


def validate_quality_checkpoint(
    checkpoint_data: bytes,
    expected_architecture: str,
) -> None:
    checkpoint = torch.load(
        io.BytesIO(checkpoint_data),
        map_location="cpu",
        weights_only=True,
    )
    if checkpoint.get("architecture") != expected_architecture:
        raise ValueError("Quality checkpoint architecture does not match the manifest.")
    if tuple(checkpoint.get("classes", ())) != QUALITY_CLASSES:
        raise ValueError(
            f"Quality checkpoint classes must be {QUALITY_CLASSES} in that order."
        )
    if not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("Quality checkpoint has no valid state_dict.")


def import_artifacts(archive_path: Path, checkpoint_dir: Path) -> list[Path]:
    with zipfile.ZipFile(archive_path) as archive:
        manifest = read_manifest(archive)
        ppe_data = get_verified_artifact(
            archive,
            manifest,
            "checkpoints/ppe_yolo.pt",
        )
        cnn_data = get_verified_artifact(
            archive,
            manifest,
            "checkpoints/casting_cnn.pth",
        )
        mobilenet_data = get_verified_artifact(
            archive,
            manifest,
            "checkpoints/casting_mobilenet_v3_small.pth",
        )
        best_quality_data = get_verified_artifact(
            archive,
            manifest,
            "checkpoints/casting_best.pth",
        )
        architecture = manifest.get("quality_architecture")
        if architecture not in {"cnn", "mobilenet_v3_small"}:
            raise ValueError(f"Unsupported quality architecture in manifest: {architecture}")
        validate_quality_checkpoint(cnn_data, "cnn")
        validate_quality_checkpoint(mobilenet_data, "mobilenet_v3_small")
        validate_quality_checkpoint(best_quality_data, architecture)

    artifacts = {
        "ppe_yolo.pt": ppe_data,
        "casting_cnn.pth": cnn_data,
        "casting_mobilenet_v3_small.pth": mobilenet_data,
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    imported_paths: list[Path] = []
    for filename, data in artifacts.items():
        destination = checkpoint_dir / filename
        with tempfile.NamedTemporaryFile(
            dir=checkpoint_dir,
            prefix=f".{filename}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(data)
        try:
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)
        imported_paths.append(destination)
    return imported_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and install model checkpoints downloaded from Kaggle."
    )
    parser.add_argument("archive", type=Path, help="factoryvision_kaggle_models.zip")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=CHECKPOINT_DIR,
    )
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"Artifact ZIP does not exist: {args.archive}")
    for path in import_artifacts(args.archive, args.checkpoint_dir.resolve()):
        print(f"Imported checkpoint: {path}")


if __name__ == "__main__":
    main()
