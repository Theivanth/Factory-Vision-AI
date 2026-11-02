import argparse
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS: dict[str, tuple[str, Path, tuple[Path, ...]]] = {
    "ppe": (
        "snehilsanyal/construction-site-safety-image-dataset-roboflow",
        PROJECT_ROOT / "datasets" / "raw" / "ppe",
        tuple(
            PROJECT_ROOT / "datasets" / "raw" / "ppe" / "css-data" / split / kind
            for split in ("train", "valid", "test")
            for kind in ("images", "labels")
        ),
    ),
    "casting": (
        "ravirajsinh45/real-life-industrial-dataset-of-casting-product",
        PROJECT_ROOT / "datasets" / "raw" / "casting",
        tuple(
            PROJECT_ROOT
            / "datasets"
            / "raw"
            / "casting"
            / "casting_data"
            / "casting_data"
            / split
            / class_name
            for split in ("train", "test")
            for class_name in ("def_front", "ok_front")
        ),
    ),
    "hard_hat": (
        "andrewmvd/hard-hat-detection",
        PROJECT_ROOT / "datasets" / "raw" / "hard_hat",
        (PROJECT_ROOT / "datasets" / "raw" / "hard_hat" / "images",),
    ),
}


def download_dataset(name: str) -> None:
    slug, destination, required_directories = DATASETS[name]
    if all(
        directory.is_dir() and any(directory.iterdir())
        for directory in required_directories
    ):
        print(f"{name}: already extracted at {destination}")
        return

    destination.mkdir(parents=True, exist_ok=True)
    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        slug,
        "-p",
        str(destination),
        "--unzip",
    ]
    print(f"Downloading {name} dataset ({slug})...")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download project datasets using configured Kaggle credentials."
    )
    parser.add_argument(
        "--include-hard-hat",
        action="store_true",
        help="Also download the optional ~1.32 GB CC0 helmet dataset.",
    )
    args = parser.parse_args()

    try:
        download_dataset("ppe")
        download_dataset("casting")
        if args.include_hard_hat:
            download_dataset("hard_hat")
    except FileNotFoundError as error:
        raise SystemExit(
            "Kaggle CLI was not found. Install requirements.txt and configure Kaggle API credentials."
        ) from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"Kaggle download failed with exit code {error.returncode}. "
            "Check your Kaggle API credentials and dataset access."
        ) from error


if __name__ == "__main__":
    main()
