# Repository Guidelines

## Agent-Specific Instructions
Always reply in Chinese unless the user explicitly requests another language.

## Project Structure & Module Organization
`alphago_weak/` is the main package: `board/` has Go rules, `dataset/` parses SGF data, `model/` contains PyTorch neural-network and MCTS code, `gtp/` contains bots, `gui/` contains the interface, and `utils/` has helpers. `zero_report.py` builds from-scratch reports; `zero_ladder.py` handles champion/candidate gating, Elo estimates, replay persistence, and bilingual reports. Tests are in `test/`, fixtures in `test/sgf/`, and images in `img/`. Generated datasets, reports, replay buffers, logs, weights, `.pt`, `.npz`, `.h5`, `.data/`, and `.data-*` stay ignored.

## Build, Test, and Development Commands
- `python -m venv .venv` then `.\.venv\Scripts\Activate.ps1`: create and activate a Windows virtual environment.
- `pip install -e .`: install PyTorch, Typer, pytest, SGF parsing, and the `alphago` console command.
- `pytest`: run the full test suite, including benchmark-backed board tests.
- `pytest test/test_board.py`: run board rules and SGF replay tests only.
- `alphago --help`: list packaged CLI commands.
- `alphago download --type u-go`: download training data into `.data/`.
- `alphago train --init 4096 --epochs 1`: smoke-test preprocessing and training with a short run.
- `alphago self-play --board-size 7 --steps 1 --epochs 1 --simulations 1 --max-moves 8 --record-games 1 --replay-buffer-size 128 --temperature-drop 1`: smoke-test the experimental AlphaGo Zero-style self-play loop.
- `alphago zero-report --force`: regenerate the small 7x7 from-scratch training report under ignored `reports/7x7-zero/`.
- `alphago zero-ladder --generations 10 --candidate-epochs 3 --gate-training-rounds 4 --eval-games 20 --gate-recheck-games 20 --force`: run the current 7x7 champion ladder with longer pre-gate training, borderline rechecks, replay persistence, and entropy/value report metrics.

## Coding Style & Naming Conventions
Use Python with 4-space indentation. Keep modules, functions, and variables in `snake_case`; classes in `CamelCase`; constants uppercase where practical. Preserve domain prefixes such as `Go*`, `GTP*`, and `AlphaGoWeak*`. Prefer type hints for public helpers. No formatter or linter is configured, so group imports as standard library, third-party, then local package imports.

## Testing Guidelines
Tests use `pytest` and `pytest-benchmark`. Name files `test_*.py` and functions `test_*`. Add board regressions to `test/test_board.py`, model encoding tests to `test/test_alphagoweak.py`, and reusable fixtures to `test/sgf/`. Keep fixtures small and deterministic. Run `pytest` before opening a PR.

## Commit & Pull Request Guidelines
History uses short imperative commit messages such as `Add MCTS self-play logic.` or `Edit some docs.` Keep each commit focused. Pull requests should describe behavior changes, list tests run, note training-data or model-weight implications, link issues, and include screenshots when GUI or documentation images change.

## Security & Configuration Tips
Do not commit datasets, reports, replay buffers, trained weights, logs, virtual environments, or generated `.pt`/`.npz`/`.h5` files. Keep experiments under ignored paths such as `.data/`, `.data-*`, `reports/`, or `weights/`. Use `--resume` for interrupted `zero-ladder` runs; it restores champion state, in-progress candidate checkpoints, report state, and persisted replay buffers.
