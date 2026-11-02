from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.domain import Detection

if TYPE_CHECKING:
    from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PPE_WEIGHTS = PROJECT_ROOT / "checkpoints" / "ppe_yolo.pt"


class SafetyPipeline:
    def __init__(
        self,
        pretrained_weights: str | Path | None = "yolo11n.pt",
        ppe_weights: str | Path = DEFAULT_PPE_WEIGHTS,
        image_size: int = 640,
        confidence_threshold: float = 0.25,
    ) -> None:
        self.pretrained_weights = pretrained_weights
        self.ppe_weights = Path(ppe_weights)
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self._pretrained_model: YOLO | None = None
        self._ppe_model: YOLO | None = None

    @property
    def ppe_model_available(self) -> bool:
        return self.ppe_weights.is_file()

    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        from ultralytics import YOLO

        detections: list[Detection] = []
        if self.pretrained_weights is not None:
            if self._pretrained_model is None:
                self._pretrained_model = YOLO(str(self.pretrained_weights))
            detections.extend(
                self._predict_model(
                    self._pretrained_model,
                    image_bgr,
                    source="pretrained",
                )
            )

        if self.ppe_model_available:
            if self._ppe_model is None:
                self._ppe_model = YOLO(str(self.ppe_weights))
            detections.extend(
                self._predict_model(self._ppe_model, image_bgr, source="fine-tuned PPE")
            )
        return detections

    def _predict_model(
        self,
        model: YOLO,
        image_bgr: np.ndarray,
        source: str,
    ) -> list[Detection]:
        results = model.predict(
            source=image_bgr,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            names = result.names
            for box, confidence, class_id in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
            ):
                label = names[int(class_id)]
                detections.append(
                    Detection(
                        label=str(label),
                        confidence=float(confidence),
                        box=tuple(float(value) for value in box),
                        source=source,
                    )
                )
        return detections