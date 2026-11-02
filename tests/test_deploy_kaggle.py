import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.deploy_kaggle import (
    DEFAULT_KERNEL_ID,
    TRAINING_SCRIPT,
    backup_existing_notebook,
    build_child_environment,
    make_kernel_metadata,
    make_notebook,
    parse_env_file,
    wait_for_kernel_completion,
)


class KaggleDeploymentTests(unittest.TestCase):
    def test_env_parser_supports_comments_quotes_and_equals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_path = Path(temporary_directory) / ".env"
            env_path.write_text(
                "# local credentials\nKAGGLE_API_TOKEN='test=value'\n",
                encoding="utf-8",
            )

            values = parse_env_file(env_path)

        self.assertEqual(values["KAGGLE_API_TOKEN"], "test=value")

    def test_missing_credentials_are_reported_without_echoing_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_path = Path(temporary_directory) / ".env"
            env_path.write_text(
                "KAGGLE_API_TOKEN=\nKAGGLE_KERNEL_ID=theivanth/factoryvisionai\n",
                encoding="utf-8",
            )

            with patch.dict("os.environ", {"KAGGLE_API_TOKEN": ""}):
                with self.assertRaisesRegex(
                    ValueError,
                    "credentials are not configured",
                ):
                    build_child_environment(env_path)

    def test_kernel_id_defaults_to_configured_notebook(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_path = Path(temporary_directory) / ".env"
            env_path.write_text("KAGGLE_API_TOKEN=fake-token\n", encoding="utf-8")

            _, kernel_id = build_child_environment(env_path)

        self.assertEqual(kernel_id, DEFAULT_KERNEL_ID)

    def test_notebook_embeds_training_arguments_and_restores_jupyter_arguments(self) -> None:
        notebook = make_notebook(
            "import sys\n"
            "assert sys.argv[1:] == ['--epochs', '12', '--ppe-epochs', '24']",
            epochs=12,
            ppe_epochs=24,
        )
        training_cell = "".join(notebook["cells"][2]["source"])

        self.assertIn("'--epochs', '12', '--ppe-epochs', '24'", training_cell)
        self.assertIn("sys.argv = __factoryvision_original_argv", training_cell)
        self.assertTrue(all(cell.get("id") for cell in notebook["cells"]))
        compile(training_cell, "generated_notebook_cell.py", "exec")
        original_argv = ["ipykernel_launcher", "-f", "kernel.json"]
        with patch("sys.argv", original_argv.copy()):
            exec(training_cell, {})
            self.assertEqual(__import__("sys").argv, original_argv)

    def test_notebook_contains_current_runtime_metadata_capture(self) -> None:
        notebook = make_notebook(
            TRAINING_SCRIPT.read_text(encoding="utf-8"),
            epochs=15,
            ppe_epochs=30,
        )
        training_cell = "".join(notebook["cells"][2]["source"])

        self.assertIn("runtime_metadata.json", training_cell)
        compile(training_cell, "generated_notebook_cell.py", "exec")

    def test_kernel_metadata_enables_gpu_and_attaches_both_datasets(self) -> None:
        metadata = make_kernel_metadata(DEFAULT_KERNEL_ID)

        self.assertEqual(metadata["id"], DEFAULT_KERNEL_ID)
        self.assertEqual(metadata["enable_gpu"], "true")
        self.assertEqual(len(metadata["dataset_sources"]), 2)
        self.assertEqual(json.loads(json.dumps(metadata)), metadata)

    def test_backup_uses_kaggle_pull_positional_kernel_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            backup_dir = Path(temporary_directory) / "backup"
            command_log = Path(temporary_directory) / "deployment.log"
            child_env = {"KAGGLE_API_TOKEN": "fake-token"}

            def fake_pull(
                arguments: list[str],
                _child_env: dict[str, str],
                _log_path: Path,
                timeout_seconds: int | None = None,
            ) -> str:
                self.assertEqual(
                    arguments,
                    [
                        "kernels",
                        "pull",
                        DEFAULT_KERNEL_ID,
                        "-p",
                        str(backup_dir),
                        "-m",
                    ],
                )
                (backup_dir / "factoryvisionai.ipynb").write_text(
                    "{}",
                    encoding="utf-8",
                )
                return ""

            with patch(
                "scripts.deploy_kaggle.run_kaggle_command",
                side_effect=fake_pull,
            ):
                backup_path = backup_existing_notebook(
                    DEFAULT_KERNEL_ID,
                    backup_dir,
                    child_env,
                    command_log,
                )

            self.assertEqual(backup_path.name, "original_factoryvisionai.ipynb")
            self.assertTrue(backup_path.is_file())

    def test_failed_kernel_status_returns_failure_for_output_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch(
                "scripts.deploy_kaggle.run_kaggle_command",
                return_value='KernelWorkerStatus.ERROR',
            ):
                completed = wait_for_kernel_completion(
                    DEFAULT_KERNEL_ID,
                    {"KAGGLE_API_TOKEN": "fake-token"},
                    Path(temporary_directory) / "deployment.log",
                    deadline_seconds=60,
                    poll_interval=1,
                )

        self.assertFalse(completed)


if __name__ == "__main__":
    unittest.main()
