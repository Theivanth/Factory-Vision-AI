import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.quality import QUALITY_CLASSES, build_quality_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASTING_ROOT = (
    PROJECT_ROOT / "datasets" / "raw" / "casting" / "casting_data" / "casting_data"
)
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
IMAGE_SIZE = 224


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the casting CNN or fine-tune MobileNetV3-Small."
    )
    parser.add_argument(
        "--architecture",
        choices=("cnn", "mobilenet_v3_small"),
        required=True,
    )
    parser.add_argument("--confirm-training", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()

    if not args.confirm_training:
        parser.error("Training is opt-in. Add --confirm-training to start training.")
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs and batch-size must be positive; learning-rate must be greater than 0.")
    for split in ("train", "test"):
        for class_name in QUALITY_CLASSES:
            directory = CASTING_ROOT / split / class_name
            if not directory.is_dir() or not any(directory.iterdir()):
                parser.error(f"Required casting dataset directory is missing or empty: {directory}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but no CUDA-capable device is available.")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    normalization = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    training_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalization,
        ]
    )
    train_root = CASTING_ROOT / "train"
    training_data = datasets.ImageFolder(train_root, transform=training_transform)
    validation_data = datasets.ImageFolder(train_root, transform=evaluation_transform)
    if tuple(training_data.classes) != QUALITY_CLASSES:
        parser.error(
            f"Expected casting classes {QUALITY_CLASSES}, got {tuple(training_data.classes)}."
        )

    indices = list(range(len(training_data)))
    random.Random(args.seed).shuffle(indices)
    validation_size = max(1, round(len(indices) * 0.2))
    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]
    train_loader = DataLoader(
        Subset(training_data, training_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        Subset(validation_data, validation_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        datasets.ImageFolder(CASTING_ROOT / "test", transform=evaluation_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_quality_model(
        args.architecture,
        len(QUALITY_CLASSES),
        pretrained=args.architecture == "mobilenet_v3_small",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    best_validation_accuracy = -1.0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / (
        "casting_cnn.pth"
        if args.architecture == "cnn"
        else "casting_mobilenet_v3_small.pth"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()

        validation_accuracy = _accuracy(model, validation_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"validation accuracy={validation_accuracy:.4f}"
        )
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(
                {
                    "architecture": args.architecture,
                    "classes": list(QUALITY_CLASSES),
                    "state_dict": model.state_dict(),
                },
                checkpoint_path,
            )

    best_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint["state_dict"])
    test_accuracy = _accuracy(model, test_loader, device)
    print(f"Test accuracy={test_accuracy:.4f}")
    print(f"Saved quality checkpoint to {checkpoint_path}")


def _accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.size(0))
    if total == 0:
        raise ValueError("Cannot calculate accuracy for an empty dataset.")
    return correct / total


if __name__ == "__main__":
    main()
