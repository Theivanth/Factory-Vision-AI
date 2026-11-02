import argparse
from importlib.util import find_spec
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KERNEL_ID = "theivanth/factoryvisionai"
DATASET_SOURCES = [
    "snehilsanyal/construction-site-safety-image-dataset-roboflow",
    "ravirajsinh45/real-life-industrial-dataset-of-casting-product",
]
NOTEBOOK_NAME = "factoryvisionai.ipynb"
TRAINING_SCRIPT = PROJECT_ROOT / "scripts" / "kaggle_train.py"
IMPORT_SCRIPT = PROJECT_ROOT / "scripts" / "import_kaggle_artifacts.py"


def parse_env_file(env_path: Path) -> dict[str, str]:
    if not env_path.is_file():
        raise FileNotFoundError(
            f"Secrets file not found: {env_path}. Copy .env.example to .env."
        )

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        env_path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            raise ValueError(f"Invalid .env entry at line {line_number}; expected KEY=VALUE.")
        name, value = line.split("=", maxsplit=1)
        name = name.strip()
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid environment variable name at line {line_number}.")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[name] = value
    return values


def build_child_environment(env_path: Path) -> tuple[dict[str, str], str]:
    file_values = parse_env_file(env_path)
    child_env = os.environ.copy()
    child_env.update(file_values)
    token = child_env.get("KAGGLE_API_TOKEN", "").strip()
    username = child_env.get("KAGGLE_USERNAME", "").strip()
    legacy_key = child_env.get("KAGGLE_KEY", "").strip()
    if not token and not (username and legacy_key):
        raise ValueError(
            "Kaggle credentials are not configured. Set KAGGLE_API_TOKEN in the "
            "local .env file, or set both KAGGLE_USERNAME and KAGGLE_KEY."
        )
    kernel_id = child_env.get("KAGGLE_KERNEL_ID", DEFAULT_KERNEL_ID).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", kernel_id):
        raise ValueError(
            "KAGGLE_KERNEL_ID must use owner/kernel-slug format, for example "
            "theivanth/factoryvisionai."
        )
    return child_env, kernel_id


def redact_secrets(text: str, child_env: dict[str, str]) -> str:
    secrets = (
        child_env.get("KAGGLE_API_TOKEN", ""),
        child_env.get("KAGGLE_KEY", ""),
    )
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    return text


def run_kaggle_command(
    arguments: list[str],
    child_env: dict[str, str],
    log_path: Path,
    timeout_seconds: int | None = None,
) -> str:
    completed = subprocess.run(
        ["kaggle", *arguments],
        cwd=PROJECT_ROOT,
        env=child_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
    )
    stdout = redact_secrets(completed.stdout, child_env)
    stderr = redact_secrets(completed.stderr, child_env)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n$ kaggle {' '.join(arguments)}\n")
        log_file.write(stdout)
        log_file.write(stderr)
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Kaggle command failed with exit code {completed.returncode}: "
            f"kaggle {arguments[0]} {arguments[1] if len(arguments) > 1 else ''}. "
            f"See {log_path} for redacted output."
        )
    return stdout + stderr


def ensure_local_import_dependencies() -> None:
    if find_spec("torch") is None:
        raise RuntimeError(
            "Local checkpoint import requires PyTorch, which is not installed in "
            "the active Python environment. Install the project dependencies first "
            "with `python -m pip install -r requirements.txt`, then rerun this script."
        )


def make_notebook(
    training_source: str,
    epochs: int,
    ppe_epochs: int,
) -> dict[str, Any]:
    install_cell = """import importlib.util
import subprocess
import sys

if importlib.util.find_spec("ultralytics") is None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3"],
        check=True,
    )
"""
    training_cell = (
        "import sys\n"
        "__factoryvision_original_argv = sys.argv[:]\n"
        "try:\n"
        f"    sys.argv = ['kaggle_train.py', '--epochs', '{epochs}', "
        f"'--ppe-epochs', '{ppe_epochs}']\n"
        "    exec(compile("
        + json.dumps(training_source)
        + ", 'kaggle_train.py', 'exec'), {'__name__': '__main__'})\n"
        "finally:\n"
        "    sys.argv = __factoryvision_original_argv\n"
    )
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "id": "factoryvision-overview",
                "metadata": {},
                "source": [
                    "# FactoryVisionAI model training\n",
                    "\n",
                    "Runs pretrained YOLO baseline, PPE fine-tuning, and casting CNN / "
                    "MobileNetV3 experiments. Logs and downloadable checkpoints are "
                    "saved under `/kaggle/working/factoryvision_artifacts/`.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "install-runtime",
                "metadata": {},
                "outputs": [],
                "source": install_cell.splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "train-models",
                "metadata": {},
                "outputs": [],
                "source": training_cell.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def make_kernel_metadata(kernel_id: str) -> dict[str, Any]:
    return {
        "id": kernel_id,
        "title": "FactoryVisionAI",
        "code_file": NOTEBOOK_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": DATASET_SOURCES,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def get_kernel_source_file(directory: Path) -> Path:
    notebooks = sorted(directory.glob("*.ipynb"))
    if not notebooks:
        raise FileNotFoundError(
            "Kaggle did not return the existing notebook source; refusing to replace it."
        )
    return notebooks[0]


def backup_existing_notebook(
    kernel_id: str,
    backup_dir: Path,
    child_env: dict[str, str],
    command_log: Path,
) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    run_kaggle_command(
        ["kernels", "pull", kernel_id, "-p", str(backup_dir), "-m"],
        child_env,
        command_log,
        timeout_seconds=180,
    )
    notebook = get_kernel_source_file(backup_dir)
    backup_path = backup_dir / f"original_{notebook.name}"
    if notebook != backup_path:
        shutil.copy2(notebook, backup_path)
    print(f"Backed up the existing Kaggle notebook to {backup_path}")
    return backup_path


def prepare_kernel_folder(
    staging_dir: Path,
    kernel_id: str,
    epochs: int,
    ppe_epochs: int,
) -> None:
    if not TRAINING_SCRIPT.is_file():
        raise FileNotFoundError(f"Kaggle training source is missing: {TRAINING_SCRIPT}")
    training_source = TRAINING_SCRIPT.read_text(encoding="utf-8")
    notebook = make_notebook(training_source, epochs, ppe_epochs)
    (staging_dir / NOTEBOOK_NAME).write_text(
        json.dumps(notebook, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    metadata = make_kernel_metadata(kernel_id)
    (staging_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def wait_for_kernel_completion(
    kernel_id: str,
    child_env: dict[str, str],
    command_log: Path,
    deadline_seconds: int,
    poll_interval: int,
) -> bool:
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        status = run_kaggle_command(
            ["kernels", "status", kernel_id],
            child_env,
            command_log,
            timeout_seconds=120,
        )
        normalized = status.upper()
        if re.search(r"\b(?:KERNELWORKERSTATUS\.)?(?:COMPLETE|SUCCESS)\b", normalized):
            print("Kaggle training completed.")
            return True
        if re.search(
            r"\b(?:KERNELWORKERSTATUS\.)?(?:ERROR|FAILED|CANCELLED|CANCELED)\b",
            normalized,
        ):
            print(f"Kaggle reports that {kernel_id} failed.")
            return False
        print(f"Kaggle status is not complete yet; checking again in {poll_interval}s.")
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Timed out waiting for {kernel_id}. The Kaggle run may still be active; "
        "check its status before starting another run."
    )


def find_model_bundle(download_dir: Path) -> Path:
    bundles = list(download_dir.rglob("factoryvision_kaggle_models.zip"))
    if not bundles:
        raise FileNotFoundError(
            f"Kaggle output did not contain factoryvision_kaggle_models.zip. "
            f"Downloaded files are under {download_dir}."
        )
    return bundles[0]


def deploy_and_import(args: argparse.Namespace) -> None:
    if not args.no_run:
        ensure_local_import_dependencies()
    child_env, kernel_id = build_child_environment(args.env_file)
    if shutil.which("kaggle") is None:
        raise FileNotFoundError("Kaggle CLI was not found on PATH.")
    if not args.env_file.resolve().is_relative_to(PROJECT_ROOT):
        raise ValueError("Keep the Kaggle credential file inside the project directory.")

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (args.output_dir / run_name).resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    command_log = run_root / "deployment.log"
    backup_dir = run_root / "notebook_backup"
    backup_existing_notebook(
        kernel_id,
        backup_dir,
        child_env,
        command_log,
    )

    with tempfile.TemporaryDirectory(
        prefix="factoryvision_kaggle_push_",
        dir=run_root,
    ) as temporary_directory:
        staging_dir = Path(temporary_directory)
        prepare_kernel_folder(
            staging_dir,
            kernel_id,
            args.epochs,
            args.ppe_epochs,
        )
        push_args = [
            "kernels",
            "push",
            "-p",
            str(staging_dir),
            "--accelerator",
            args.accelerator,
            "-t",
            str(args.kernel_timeout_seconds),
        ]
        if args.no_run:
            push_args.append("--no-run")
        run_kaggle_command(
            push_args,
            child_env,
            command_log,
            timeout_seconds=args.wait_timeout_seconds,
        )

    if args.no_run:
        print(
            "Notebook uploaded without running. Re-run without --no-run to train "
            "and import the checkpoints."
        )
        return

    kernel_succeeded = wait_for_kernel_completion(
        kernel_id,
        child_env,
        command_log,
        deadline_seconds=args.wait_timeout_seconds,
        poll_interval=args.poll_interval_seconds,
    )

    download_dir = run_root / "kaggle_output"
    download_dir.mkdir(parents=True, exist_ok=True)
    run_kaggle_command(
        [
            "kernels",
            "output",
            kernel_id,
            "-p",
            str(download_dir),
            "-o",
        ],
        child_env,
        command_log,
        timeout_seconds=1800,
    )
    if not kernel_succeeded:
        raise RuntimeError(
        f"Kaggle training failed. Available logs and partial output are under "
        f"{download_dir}."
        )
    bundle_path = find_model_bundle(download_dir)
    imported = subprocess.run(
        [
            sys.executable,
            str(IMPORT_SCRIPT),
            str(bundle_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if imported.stdout:
        print(imported.stdout, end="" if imported.stdout.endswith("\n") else "\n")
    if imported.returncode != 0:
        raise RuntimeError(
            "Kaggle artifacts were downloaded, but local checkpoint import failed. "
            f"The verified bundle remains at {bundle_path}. "
            f"Importer output: {imported.stderr.strip()}"
        )
    print(f"Kaggle output and logs are saved under {download_dir}")
    print(f"Model bundle: {bundle_path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Back up, run, and download the FactoryVisionAI Kaggle notebook, "
            "then verify/import its best checkpoints."
        )
    )
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "kaggle",
    )
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--ppe-epochs", type=int, default=30)
    parser.add_argument("--kernel-timeout-seconds", type=int, default=21600)
    parser.add_argument("--wait-timeout-seconds", type=int, default=21600)
    parser.add_argument("--poll-interval-seconds", type=int, default=45)
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Upload a new notebook version without starting GPU training.",
    )
    args = parser.parse_args()
    if args.kernel_timeout_seconds < 1:
        parser.error("--kernel-timeout-seconds must be positive.")
    if args.wait_timeout_seconds < 1:
        parser.error("--wait-timeout-seconds must be positive.")
    if args.poll_interval_seconds < 1:
        parser.error("--poll-interval-seconds must be positive.")
    if args.epochs < 1:
        parser.error("--epochs must be positive.")
    if args.ppe_epochs < 1:
        parser.error("--ppe-epochs must be positive.")
    return args


def main() -> None:
    args = parse_arguments()
    deploy_and_import(args)


if __name__ == "__main__":
    main()
