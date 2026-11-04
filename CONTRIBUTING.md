# Contributing

Thanks for your interest in improving FactoryVisionAI. Contributions should
preserve the project's status as an educational prototype and keep human review
in the loop.

## Before you start

- Check existing issues and changes to avoid duplicating work.
- For a substantial change, open an issue first to discuss the intended behavior.
- Keep changes focused, and do not commit credentials, raw datasets, trained
  checkpoints, Kaggle outputs, or private camera footage.
- Respect the separate licenses and attribution requirements for datasets,
  pretrained weights, and dependencies. The project's MIT license applies only
  to the project source and documentation, not those external materials.

## Development setup

Use Python 3.10 or later and install the project dependencies in a virtual
environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install pytest
```

Some inference and training paths require local checkpoints and datasets. They
are intentionally not included in the repository; consult the README for setup
and data-license information.

## Make and verify changes

- Follow the existing Python style and patterns.
- Add or update tests for behavior changes.
- Run the test suite before submitting:

  ```powershell
  python -m pytest
  ```

- Do not describe model output as certified, production-ready, or a substitute
  for qualified human inspection.

## Pull requests

Provide a concise summary of the change, why it is needed, and the tests or
manual checks you ran. Call out any changes to model behavior, thresholds,
datasets, or user-facing safety language. Keep unrelated refactors out of the
pull request.
