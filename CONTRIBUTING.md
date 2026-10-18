# Contributing to MLXServe

Thanks for your interest! Bug reports, benchmarks on other Macs, fixes and features are all welcome.
Issues and pull requests can be written in English or French.

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
Security issues go through the private process in [SECURITY.md](SECURITY.md), not public issues.

## Before you start

- **Bug**: open an issue with the [bug report form](https://github.com/ROSITO/SoloMLX-server/issues/new?template=bug_report.yml). Your Mac (chip, RAM), the model and the `MLXSERVE_*` settings matter a lot for inference bugs.
- **Feature or larger change**: open a [feature request](https://github.com/ROSITO/SoloMLX-server/issues/new?template=feature_request.yml) first, so we agree on the approach before you spend time on it.
- **Small fix** (typo, docs, obvious bug): a pull request directly is fine.

## Development setup

Requirements: macOS on Apple Silicon, Python 3.12+ (MLX does not run elsewhere; the test suite mostly does, with the stub backend).

```bash
git clone https://github.com/ROSITO/SoloMLX-server.git
cd SoloMLX-server
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,mlx,security]"
# R&D / MoE conversion tests also need: -e ".[rd]" (installs torch)
```

Run the server against a small model to try a change end to end:

```bash
MLXSERVE_DEFAULT_MODEL=mlx-community/Qwen3-4B-Instruct-2507-4bit .venv/bin/mlxserve serve
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

Use `python -m pytest` (not plain `pytest`): some tests import `scripts/`, `tools/` and `training/` from the repository root. Tests that need `mlx` or `torch` skip or fail at import when those extras are missing.

Some tests rewrite files under `bench/`; do not commit those changes unless your PR is about benchmarks:

```bash
git checkout -- bench/
```

## Pull requests

1. Branch from `main` (`fix/...`, `feat/...`, `docs/...`).
2. Keep the change focused: one fix or feature per PR.
3. Add or update tests for behaviour changes. For anything touching inference (generation, caches, memory, tool calls), also check it on a real model and say which one in the PR.
4. Performance claims need numbers: machine, model, prompt size, before/after.
5. Update the README when you add or change a `MLXSERVE_*` setting, an endpoint or a CLI command.
6. Fill in the pull request template.

Commit messages follow a light [Conventional Commits](https://www.conventionalcommits.org/) style, as in the history: `feat: ...`, `fix(scope): ...`, `docs: ...`.

## Code style

- Match the surrounding code: names, comment density, structure.
- Prefer the standard library and what `mlx` / `mlx-lm` already provide over new dependencies.
- Comments explain *why* (a measured number, a constraint), not *what* the code does.

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE) of this project.
