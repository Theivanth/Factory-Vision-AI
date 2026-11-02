import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from uuid import uuid4

from src.domain import Detection, PPEAssessment, QualityPrediction


@dataclass
class Alert:
    alert_id: str
    created_at: str
    source: str
    frame_index: int | None
    category: str
    message: str
    severity: str
    confidence: float
    review_status: str = "Pending"


def _normalized_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()


def _is_head(label: str) -> bool:
    return _normalized_label(label) == "head"


def _is_helmet(label: str) -> bool:
    normalized = _normalized_label(label)
    return normalized in {"hardhat", "helmet", "safety helmet"}


def _is_missing_helmet(label: str) -> bool:
    normalized = _normalized_label(label)
    return normalized in {"no hardhat", "no helmet", "no safety helmet"}


def _center_in_head_region(
    person: Detection,
    item: Detection,
) -> bool:
    person_x1, person_y1, person_x2, person_y2 = person.box
    item_x1, item_y1, item_x2, item_y2 = item.box
    person_width = max(person_x2 - person_x1, 1.0)
    person_height = max(person_y2 - person_y1, 1.0)
    item_center_x = (item_x1 + item_x2) / 2
    item_center_y = (item_y1 + item_y2) / 2

    if _is_head(person.label):
        horizontal_margin = max((person_x2 - person_x1) * 0.35, 4.0)
        vertical_margin = max((person_y2 - person_y1) * 0.5, 4.0)
        return (
            person_x1 - horizontal_margin <= item_center_x <= person_x2 + horizontal_margin
            and person_y1 - vertical_margin <= item_center_y <= person_y2 + vertical_margin
        )

    horizontal_margin = person_width * 0.1
    head_region_bottom = person_y1 + person_height * 0.35
    return (
        person_x1 - horizontal_margin <= item_center_x <= person_x2 + horizontal_margin
        and person_y1 - person_height * 0.05 <= item_center_y <= head_region_bottom
    )


def _box_iou(
    first: Detection,
    second: Detection,
) -> float:
    first_x1, first_y1, first_x2, first_y2 = first.box
    second_x1, second_y1, second_x2, second_y2 = second.box
    intersection_width = max(min(first_x2, second_x2) - max(first_x1, second_x1), 0.0)
    intersection_height = max(
        min(first_y2, second_y2) - max(first_y1, second_y1),
        0.0,
    )
    intersection = intersection_width * intersection_height
    first_area = max(first_x2 - first_x1, 0.0) * max(first_y2 - first_y1, 0.0)
    second_area = max(second_x2 - second_x1, 0.0) * max(
        second_y2 - second_y1,
        0.0,
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _head_anchors(detections: list[Detection]) -> list[Detection]:
    people: list[Detection] = []
    for candidate in sorted(
        (
            item
            for item in detections
            if _normalized_label(item.label) in {"person", "worker"}
        ),
        key=lambda item: item.confidence,
        reverse=True,
    ):
        if not any(_box_iou(candidate, person) >= 0.5 for person in people):
            people.append(candidate)
    anchors = list(people)
    for head in (item for item in detections if _is_head(item.label)):
        head_center_x = (head.box[0] + head.box[2]) / 2
        head_center_y = (head.box[1] + head.box[3]) / 2
        covered_by_person = any(
            person.box[0] <= head_center_x <= person.box[2]
            and person.box[1] <= head_center_y <= person.box[3]
            for person in people
        )
        if not covered_by_person:
            anchors.append(head)
    return anchors


def assess_ppe(
    safety_detections: list[Detection],
    ppe_model_available: bool,
    confidence_threshold: float = 0.5,
) -> list[PPEAssessment]:
    if not ppe_model_available:
        return []

    helmets = [
        item
        for item in safety_detections
        if _is_helmet(item.label) and item.confidence >= confidence_threshold
    ]
    violations = [
        item
        for item in safety_detections
        if _is_missing_helmet(item.label) and item.confidence >= confidence_threshold
    ]
    assessments: list[PPEAssessment] = []
    for anchor in _head_anchors(safety_detections):
        if anchor.confidence < confidence_threshold:
            continue
        associated_helmet = any(
            _center_in_head_region(anchor, helmet) for helmet in helmets
        )
        associated_violation = next(
            (
                violation
                for violation in violations
                if _center_in_head_region(anchor, violation)
            ),
            None,
        )
        if associated_helmet and associated_violation is None:
            status = "Helmet detected"
            confidence = anchor.confidence
        elif associated_violation is not None:
            status = "Potential missing helmet"
            confidence = associated_violation.confidence
        else:
            status = "Potential missing helmet"
            confidence = anchor.confidence
        assessments.append(
            PPEAssessment(
                box=anchor.box,
                status=status,
                confidence=confidence,
            )
        )
    return assessments


def quality_disposition(prediction: QualityPrediction) -> str:
    label = _normalized_label(prediction.label)
    if label in {"ok", "ok front"}:
        return "Acceptable"
    if label in {"defective", "def front"}:
        return "Manual inspection required"
    return "Unrecognized quality class"


def build_alerts(
    safety_detections: list[Detection],
    quality_prediction: QualityPrediction | None,
    source: str,
    frame_index: int | None = None,
    confidence_threshold: float = 0.5,
    ppe_model_available: bool = False,
) -> list[Alert]:
    alerts: list[Alert] = []
    ppe_detector_available = ppe_model_available or any(
        detection.source.lower() == "fine-tuned ppe"
        for detection in safety_detections
    )
    ppe_assessments = assess_ppe(
        safety_detections,
        ppe_model_available=ppe_detector_available,
        confidence_threshold=confidence_threshold,
    )
    for assessment in ppe_assessments:
        if assessment.status == "Potential missing helmet":
            alerts.append(
                _new_alert(
                    source,
                    frame_index,
                    category="PPE",
                    message="Potential missing helmet",
                    severity="High",
                    confidence=assessment.confidence,
                )
            )

    for detection in safety_detections:
        label = _normalized_label(detection.label)
        if label == "no safety vest" and detection.confidence >= confidence_threshold:
            alerts.append(
                _new_alert(
                    source,
                    frame_index,
                    category="PPE",
                    message="Potential missing safety vest",
                    severity="High",
                    confidence=detection.confidence,
                )
            )
        elif (
            ppe_detector_available
            and _is_missing_helmet(detection.label)
            and detection.confidence >= confidence_threshold
            and not any(
                anchor.confidence >= confidence_threshold
                and _center_in_head_region(anchor, detection)
                for anchor in _head_anchors(safety_detections)
            )
        ):
            alerts.append(
                _new_alert(
                    source,
                    frame_index,
                    category="PPE",
                    message="Potential missing helmet",
                    severity="High",
                    confidence=detection.confidence,
                )
            )

    if (
        quality_prediction is not None
        and quality_disposition(quality_prediction) == "Manual inspection required"
    ):
        alerts.append(
            _new_alert(
                source,
                frame_index,
                category="Quality",
                message="Defective casting requires manual quality inspection",
                severity="Review",
                confidence=quality_prediction.confidence,
            )
        )
    return alerts


def _new_alert(
    source: str,
    frame_index: int | None,
    category: str,
    message: str,
    severity: str,
    confidence: float,
) -> Alert:
    return Alert(
        alert_id=uuid4().hex,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source=source,
        frame_index=frame_index,
        category=category,
        message=message,
        severity=severity,
        confidence=round(confidence, 4),
    )


def alert_record(alert: Alert) -> dict[str, object]:
    return asdict(alert)