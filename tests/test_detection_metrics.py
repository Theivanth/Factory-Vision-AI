import unittest
from src.detection_metrics import calculate_iou, match_detection_boxes


class DetectionMetricTests(unittest.TestCase):
    def test_identical_boxes_have_iou_one(self):
        self.assertEqual(calculate_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)

    def test_non_overlapping_boxes_have_iou_zero(self):
        self.assertEqual(calculate_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_one_ground_truth_matches_only_one_prediction(self):
        predictions = [(0, 0, 10, 10, 0.9), (0, 0, 10, 10, 0.8)]
        self.assertEqual(match_detection_boxes(predictions, [(0, 0, 10, 10)]), (1, 1, 0))

    def test_unmatched_ground_truth_is_a_false_negative(self):
        self.assertEqual(match_detection_boxes([(20, 20, 30, 30, 0.9)], [(0, 0, 10, 10)]), (0, 1, 1))

    def test_invalid_iou_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            match_detection_boxes([], [], 1.1)

    def test_empty_predictions(self):
        self.assertEqual(match_detection_boxes([], [(0, 0, 10, 10)]), (0, 0, 1))

    def test_empty_ground_truth(self):
        self.assertEqual(match_detection_boxes([(0, 0, 10, 10, 0.9)], []), (0, 1, 0))

    def test_zero_area_boxes(self):
        self.assertEqual(calculate_iou((0, 0, 0, 5), (0, 0, 10, 10)), 0.0)

    def test_threshold_boundary(self):
        with self.assertRaises(ValueError):
            match_detection_boxes([], [], 0)
        self.assertEqual(match_detection_boxes([(0, 0, 10, 10, 0.9)], [(0, 0, 10, 10)], 1.0), (1, 0, 0))


if __name__ == '__main__':
    unittest.main()
