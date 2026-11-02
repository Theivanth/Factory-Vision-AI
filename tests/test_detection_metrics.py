import unittest

from src.detection_metrics import calculate_iou, match_detection_boxes


class DetectionMetricTests(unittest.TestCase):
    def test_identical_boxes_have_iou_one(self) -> None:
        box = (0.0, 0.0, 10.0, 10.0)

        self.assertEqual(calculate_iou(box, box), 1.0)

    def test_non_overlapping_boxes_have_iou_zero(self) -> None:
        first = (0.0, 0.0, 10.0, 10.0)
        second = (20.0, 20.0, 30.0, 30.0)

        self.assertEqual(calculate_iou(first, second), 0.0)

    def test_one_ground_truth_matches_only_one_prediction(self) -> None:
        predictions = [
            (0.0, 0.0, 10.0, 10.0, 0.9),
            (0.0, 0.0, 10.0, 10.0, 0.8),
        ]
        ground_truth = [(0.0, 0.0, 10.0, 10.0)]

        counts = match_detection_boxes(predictions, ground_truth, iou_threshold=0.5)

        self.assertEqual(counts, (1, 1, 0))

    def test_unmatched_ground_truth_is_a_false_negative(self) -> None:
        predictions = [(20.0, 20.0, 30.0, 30.0, 0.9)]
        ground_truth = [(0.0, 0.0, 10.0, 10.0)]

        counts = match_detection_boxes(predictions, ground_truth, iou_threshold=0.5)

        self.assertEqual(counts, (0, 1, 1))

    def test_invalid_iou_threshold_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            match_detection_boxes([], [], iou_threshold=1.1)


if __name__ == "__main__":
    unittest.main()
