from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from src.domain import QualityPrediction


QUALITY_CLASSES = ("def_front", "ok_front")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "Custom CNN": PROJECT_ROOT / "checkpoints" / "casting_cnn.pth",
    "MobileNetV3-Small": PROJECT_ROOT / "checkpoints" / "casting_mobilenet_v3_small.pth",
}


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


def build_quality_model(
    architecture: str,
    num_classes: int = len(QUALITY_CLASSES),
    pretrained: bool = False,
) -> nn.Module:
    if architecture == "cnn":
        return CastingCNN(num_classes=num_classes)
    if architecture == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(input_features, num_classes)
        return model
    raise ValueError(f"Unsupported quality model architecture: {architecture}")


class QualityClassifier:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Quality checkpoint not found: {self.checkpoint_path}. "
                "Train a quality model before running quality inference."
            )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        architecture = checkpoint["architecture"]
        self.class_names = tuple(checkpoint["classes"])
        if set(self.class_names) != set(QUALITY_CLASSES):
            raise ValueError(
                f"Checkpoint classes must be {QUALITY_CLASSES}, got {self.class_names}."
            )

        self.model = build_quality_model(architecture, len(self.class_names))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def predict(self, image_bgr: np.ndarray) -> QualityPrediction:
        image_rgb = image_bgr[:, :, ::-1].copy()
        image = Image.fromarray(image_rgb)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        class_index = int(torch.argmax(probabilities).item())
        return QualityPrediction(
            label=self.class_names[class_index],
            confidence=float(probabilities[class_index].item()),
        )