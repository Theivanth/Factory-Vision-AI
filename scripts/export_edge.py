import argparse

import torch
from torch import nn

from src.quality import CHECKPOINTS, QualityClassifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a trained casting classifier as a CPU TorchScript model."
    )
    parser.add_argument(
        "--model",
        choices=("Custom CNN", "MobileNetV3-Small"),
        required=True,
    )
    args = parser.parse_args()

    checkpoint_path = CHECKPOINTS[args.model]
    classifier = QualityClassifier(checkpoint_path, device="cpu")
    quantized_model = torch.ao.quantization.quantize_dynamic(
        classifier.model.cpu().eval(),
        {nn.Linear},
        dtype=torch.qint8,
    )
    example_input = torch.zeros(1, 3, 224, 224)
    scripted_model = torch.jit.trace(quantized_model, example_input)
    scripted_model = torch.jit.optimize_for_inference(scripted_model)
    destination = checkpoint_path.with_name(f"{checkpoint_path.stem}_edge.torchscript")
    destination.parent.mkdir(parents=True, exist_ok=True)
    scripted_model.save(str(destination))
    print(f"Saved CPU TorchScript model to {destination}")


if __name__ == "__main__":
    main()
