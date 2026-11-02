import unittest

from app import build_training_command


class DashboardTrainingCommandTests(unittest.TestCase):
    def test_ppe_training_command_passes_adjustable_yolo_parameters(self) -> None:
        command = build_training_command(
            model_name="PPE YOLO",
            epochs=24,
            batch_size=8,
            device="CUDA:0",
            image_size=832,
            seed=17,
            weights="yolov8n.pt",
        )

        self.assertIn("--confirm-training", command)
        self.assertIn("--epochs", command)
        self.assertEqual(command[command.index("--epochs") + 1], "24")
        self.assertEqual(command[command.index("--batch-size") + 1], "8")
        self.assertEqual(command[command.index("--image-size") + 1], "832")
        self.assertEqual(command[command.index("--seed") + 1], "17")
        self.assertEqual(command[command.index("--weights") + 1], "yolov8n.pt")
        self.assertEqual(command[command.index("--device") + 1], "0")

    def test_quality_training_command_passes_learning_rate_seed_and_device(self) -> None:
        command = build_training_command(
            model_name="MobileNetV3-Small",
            epochs=12,
            batch_size=20,
            device="CPU",
            learning_rate=0.0005,
            seed=123,
        )

        self.assertIn("mobilenet_v3_small", command)
        self.assertEqual(command[command.index("--epochs") + 1], "12")
        self.assertEqual(command[command.index("--batch-size") + 1], "20")
        self.assertEqual(command[command.index("--learning-rate") + 1], "0.0005")
        self.assertEqual(command[command.index("--seed") + 1], "123")
        self.assertEqual(command[command.index("--device") + 1], "cpu")

    def test_unknown_training_model_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown training model"):
            build_training_command(
                model_name="unknown",
                epochs=1,
                batch_size=1,
                device="Automatic",
            )


if __name__ == "__main__":
    unittest.main()
