# Configuration

This directory contains compatibility configuration for existing workflows.

- `requirements.txt`: legacy dependency list; the primary dependency source is the project-level `pyproject.toml`.
- `.python-version`: legacy Python version declaration; the supported version is declared in `pyproject.toml`.

Use `uv sync` to create the project environment and `uv run` to execute commands in it.
