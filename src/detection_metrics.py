"""Dependency-free bounding-box metrics for object detection evaluation.

Boxes use the ``(x1, y1, x2, y2)`` convention; predictions append a
confidence score. A ground-truth box can be matched at most once.
"""

from __future__ import annotations


def calculate_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Compute intersection over union, returning zero for empty boxes."""
    intersection_width = max(min(first[2], second[2]) - max(first[0], second[0]), 0.0)
    intersection_height = max(min(first[3], second[3]) - max(first[1], second[1]), 0.0)
    intersection = intersection_width * intersection_height
    first_area = max(first[2] - first[0], 0.0) * max(first[3] - first[1], 0.0)
    second_area = max(second[2] - second[0], 0.0) * max(
        second[3] - second[1], 0.0
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def match_detection_boxes(
    predictions: list[tuple[float, float, float, float, float]],
    ground_truth: list[tuple[float, float, float, float]],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Greedily match predictions by confidence; return (TP, FP, FN)."""
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be greater than 0 and at most 1")

    matched_ground_truth: set[int] = set()
    true_positive = 0
    false_positive = 0
    for prediction in sorted(predictions, key=lambda box: box[4], reverse=True):
        overlaps = (
            (calculate_iou(prediction[:4], truth), index)
            for index, truth in enumerate(ground_truth)
            if index not in matched_ground_truth
        )
        best_iou, best_index = max(overlaps, default=(0.0, -1))
        if best_index >= 0 and best_iou >= iou_threshold:
            true_positive += 1
            matched_ground_truth.add(best_index)
        else:
            false_positive += 1
    false_negative = len(ground_truth) - len(matched_ground_truth)
    return true_positive, false_positive, false_negative
