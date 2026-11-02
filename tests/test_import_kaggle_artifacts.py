import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch

from scripts.import_kaggle_artifacts import import_artifacts


def checkpoint_bytes(architecture: str) -> bytes:
    buffer = io.BytesIO()
    torch.save(
        {
            "architecture": architecture,
            "classes": ["def_front", "ok_front"],
            "state_dict": {},
        },
        buffer,
    )
    return buffer.getvalue()


class KaggleArtifactImportTests(unittest.TestCase):
    def create_bundle(self, archive_path: Path) -> None:
        contents = {
            "checkpoints/ppe_yolo.pt": b"ppe weights",
            "checkpoints/casting_cnn.pth": checkpoint_bytes("cnn"),
            "checkpoints/casting_mobilenet_v3_small.pth": checkpoint_bytes(
                "mobilenet_v3_small"
            ),
            "checkpoints/casting_best.pth": checkpoint_bytes(
                "mobilenet_v3_small"
            ),
        }
        manifest = {
            "format_version": 1,
            "quality_architecture": "mobilenet_v3_small",
            "files": {
                name: hashlib.sha256(data).hexdigest()
                for name, data in contents.items()
            },
        }
        with zipfile.ZipFile(archive_path, "w") as archive:
            for name, data in contents.items():
                archive.writestr(name, data)
            archive.writestr("artifact_manifest.json", json.dumps(manifest))

    def test_imports_ppe_and_both_casting_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "models.zip"
            checkpoint_dir = root / "checkpoints"
            self.create_bundle(archive_path)

            imported_paths = import_artifacts(archive_path, checkpoint_dir)

            self.assertEqual(
                {path.name for path in imported_paths},
                {
                    "ppe_yolo.pt",
                    "casting_cnn.pth",
                    "casting_mobilenet_v3_small.pth",
                },
            )
            self.assertTrue(all(path.is_file() for path in imported_paths))

    def test_import_rejects_hash_mismatch_before_installing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "models.zip"
            self.create_bundle(archive_path)
            with zipfile.ZipFile(archive_path, "a") as archive:
                archive.writestr("checkpoints/casting_cnn.pth", b"tampered")
            checkpoint_dir = root / "checkpoints"

            with self.assertRaisesRegex(ValueError, "SHA-256 verification failed"):
                import_artifacts(archive_path, checkpoint_dir)

            self.assertFalse(checkpoint_dir.exists())


if __name__ == "__main__":
    unittest.main()
