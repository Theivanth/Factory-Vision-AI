import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.kaggle_train import (
    QUALITY_CLASSES,
    inspect_casting_dataset,
    make_quality_loaders,
)


class CastingDatasetInspectionTests(unittest.TestCase):
    def create_dataset(self, root: Path) -> None:
        for split in ("train", "test"):
            for class_name in QUALITY_CLASSES:
                directory = root / split / class_name
                directory.mkdir(parents=True)

        Image.new("RGB", (8, 8), (1, 2, 3)).save(
            root / "train" / "ok_front" / "duplicate.png"
        )
        Image.new("RGB", (8, 8), (1, 2, 3)).save(
            root / "test" / "ok_front" / "duplicate.png"
        )
        for class_index, class_name in enumerate(QUALITY_CLASSES):
            for index in range(2):
                Image.new(
                    "RGB",
                    (8, 8),
                    (index + 10 + class_index * 100, 1, 1),
                ).save(
                    root / "train" / class_name / f"train_{index}.png"
                )
            Image.new("RGB", (8, 8), (60 + class_index * 100, 1, 1)).save(
                root / "test" / class_name / "held_out.png"
            )

    def test_train_copy_is_excluded_and_reported_when_test_contains_same_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.create_dataset(root)

            inspection = inspect_casting_dataset(root)

        self.assertEqual(inspection["exact_duplicates_across_train_test"], 1)
        self.assertEqual(
            inspection["train_paths_excluded_to_prevent_test_leakage"],
            ["train/ok_front/duplicate.png"],
        )
        self.assertEqual(inspection["effective_train_counts"]["ok_front"], 2)

    def test_quality_loaders_remove_reported_training_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.create_dataset(root)

            train_loader, _, _ = make_quality_loaders(
                root,
                batch_size=2,
                seed=42,
                excluded_train_paths={"train/ok_front/duplicate.png"},
            )

            self.assertEqual(len(train_loader.dataset.dataset.samples), 4)
            self.assertNotIn(
                str(root / "train" / "ok_front" / "duplicate.png"),
                {path for path, _ in train_loader.dataset.dataset.samples},
            )

    def test_duplicate_with_conflicting_class_labels_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.create_dataset(root)
            Image.new("RGB", (8, 8), (250, 1, 1)).save(
                root / "test" / "ok_front" / "duplicate.png"
            )
            (root / "test" / "def_front" / "held_out.png").write_bytes(
                (root / "train" / "ok_front" / "duplicate.png").read_bytes()
            )

            with self.assertRaisesRegex(ValueError, "conflicting train/test labels"):
                inspect_casting_dataset(root)


if __name__ == "__main__":
    unittest.main()
