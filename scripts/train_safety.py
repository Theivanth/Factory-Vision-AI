import argparse
import shutil
import tempfile
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "raw" / "ppe" / "css-data"
SOURCE_CONFIG = PROJECT_ROOT / "configs" / "ppe_data.yaml"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a YOLO detector on the PPE dataset."
    )
    parser.add_argument("--confirm-training", action="store_true")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="For example: 0 or cpu.")
    args = parser.parse_args()

    if not args.confirm_training:
        parser.error("Training is opt-in. Add --confirm-training to start training.")
    for split in ("train", "valid", "test"):
        for kind in ("images", "labels"):
            directory = DATASET_ROOT / split / kind
            if not directory.is_dir() or not any(directory.iterdir()):
                parser.error(f"Required PPE dataset directory is missing or empty: {directory}")
    if args.epochs < 1 or args.image_size < 1 or args.batch_size < 1:
        parser.error("epochs, image-size, and batch-size must all be positive.")

    from ultralytics import YOLO

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
    config["path"] = DATASET_ROOT.resolve().as_posix()
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".yaml",
        delete=False,
    ) as temporary_config:
        yaml.safe_dump(config, temporary_config, sort_keys=False)
        config_path = Path(temporary_config.name)

    try:
        model = YOLO(args.weights)
        results = model.train(
            data=str(config_path),
            epochs=args.epochs,
            imgsz=args.image_size,
            batch=args.batch_size,
            device=args.device,
            seed=args.seed,
            deterministic=True,
            workers=0,
            project=str(PROJECT_ROOT / "runs"),
            name="ppe_yolo",
            exist_ok=True,
        )
    finally:
        config_path.unlink(missing_ok=True)

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"YOLO training did not produce expected weights: {best_weights}")
    destination = CHECKPOINT_DIR / "ppe_yolo.pt"
    shutil.copy2(best_weights, destination)
    print(f"Saved fine-tuned PPE detector to {destination}")


if __name__ == "__main__":
    main()
