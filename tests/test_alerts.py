import unittest

from src.alerts import assess_ppe, build_alerts, quality_disposition
from src.domain import Detection, QualityPrediction


def detection(
    label: str,
    confidence: float,
    box: tuple[float, float, float, float],
    source: str = "fine-tuned PPE",
) -> Detection:
    return Detection(label, confidence, box, source)


class DecisionEngineTests(unittest.TestCase):
    def test_worker_with_associated_helmet_is_annotated_without_alert(self) -> None:
        detections = [
            detection("Person", 0.96, (10, 10, 110, 210)),
            detection("Hardhat", 0.93, (40, 12, 72, 38)),
        ]

        assessments = assess_ppe(detections, ppe_model_available=True)
        alerts = build_alerts(
            detections,
            None,
            source="line-1",
            ppe_model_available=True,
        )

        self.assertEqual(len(assessments), 1)
        self.assertEqual(assessments[0].status, "Helmet detected")
        self.assertEqual(alerts, [])

    def test_worker_without_associated_helmet_creates_review_alert(self) -> None:
        detections = [detection("Person", 0.91, (10, 10, 110, 210))]

        assessments = assess_ppe(detections, ppe_model_available=True)
        alerts = build_alerts(
            detections,
            None,
            source="line-1",
            ppe_model_available=True,
        )

        self.assertEqual(assessments[0].status, "Potential missing helmet")
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].message, "Potential missing helmet")
        self.assertEqual(alerts[0].review_status, "Pending")

    def test_helmet_associated_with_another_worker_does_not_suppress_alert(self) -> None:
        detections = [
            detection("Person", 0.94, (0, 0, 100, 200)),
            detection("Person", 0.92, (200, 0, 300, 200)),
            detection("Helmet", 0.95, (225, 5, 260, 35)),
        ]

        assessments = assess_ppe(detections, ppe_model_available=True)
        alerts = build_alerts(
            detections,
            None,
            source="line-1",
            ppe_model_available=True,
        )

        self.assertEqual(
            [assessment.status for assessment in assessments],
            ["Potential missing helmet", "Helmet detected"],
        )
        self.assertEqual(len(alerts), 1)

    def test_duplicate_person_boxes_from_both_models_create_one_alert(self) -> None:
        detections = [
            detection("Person", 0.95, (10, 10, 110, 210), source="pretrained"),
            detection("Person", 0.92, (12, 12, 108, 208), source="fine-tuned PPE"),
        ]

        alerts = build_alerts(
            detections,
            None,
            source="line-1",
            ppe_model_available=True,
        )

        self.assertEqual(len(alerts), 1)

    def test_head_detection_without_associated_helmet_is_flagged(self) -> None:
        detections = [detection("Head", 0.87, (20, 20, 70, 65))]

        assessments = assess_ppe(detections, ppe_model_available=True)

        self.assertEqual(len(assessments), 1)
        self.assertEqual(assessments[0].status, "Potential missing helmet")

    def test_no_ppe_checkpoint_does_not_treat_missing_detection_as_violation(self) -> None:
        detections = [
            detection("person", 0.99, (0, 0, 100, 200), source="pretrained")
        ]

        assessments = assess_ppe(detections, ppe_model_available=False)
        alerts = build_alerts(
            detections,
            None,
            source="line-1",
            ppe_model_available=False,
        )

        self.assertEqual(assessments, [])
        self.assertEqual(alerts, [])

    def test_explicit_no_hardhat_is_not_duplicated_with_person_assessment(self) -> None:
        detections = [
            detection("Person", 0.95, (10, 10, 110, 210)),
            detection("NO-Hardhat", 0.89, (40, 12, 72, 38)),
        ]

        alerts = build_alerts(
            detections,
            None,
            source="line-1",
            ppe_model_available=True,
        )

        self.assertEqual(len(alerts), 1)

    def test_confident_no_hardhat_class_is_not_hidden_by_low_confidence_person(self) -> None:
        detections = [
            detection("Person", 0.3, (10, 10, 110, 210)),
            detection("NO-Hardhat", 0.9, (40, 12, 72, 38)),
        ]

        alerts = build_alerts(
            detections,
            None,
            source="line-1",
            ppe_model_available=True,
        )

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].message, "Potential missing helmet")

    def test_clean_cast_is_displayed_as_acceptable_without_alert(self) -> None:
        prediction = QualityPrediction("ok_front", 0.96)

        alerts = build_alerts([], prediction, source="casting-camera")

        self.assertEqual(quality_disposition(prediction), "Acceptable")
        self.assertEqual(alerts, [])

    def test_defective_cast_is_flagged_for_manual_inspection(self) -> None:
        prediction = QualityPrediction("def_front", 0.82)

        alerts = build_alerts([], prediction, source="casting-camera")

        self.assertEqual(
            quality_disposition(prediction),
            "Manual inspection required",
        )
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].category, "Quality")
        self.assertEqual(alerts[0].severity, "Review")

    def test_low_confidence_person_does_not_create_ppe_alert(self) -> None:
        detections = [detection("Person", 0.49, (0, 0, 100, 200))]

        alerts = build_alerts(
            detections,
            None,
            source="line-1",
            ppe_model_available=True,
        )

        self.assertEqual(alerts, [])

    def test_defective_class_is_queued_even_with_low_confidence(self) -> None:
        prediction = QualityPrediction("def_front", 0.31)

        alerts = build_alerts([], prediction, source="casting-camera")

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].category, "Quality")


if __name__ == "__main__":
    unittest.main()
