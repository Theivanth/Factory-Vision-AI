from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]
    source: str


@dataclass(frozen=True)
class QualityPrediction:
    label: str
    confidence: float


@dataclass(frozen=True)
class PPEAssessment:
    box: tuple[float, float, float, float]
    status: str
    confidence: float
