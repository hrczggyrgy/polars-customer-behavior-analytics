"""JSON file I/O utilities."""

import json
from pathlib import Path


def read_json(path: Path | str) -> dict:
    """Read and parse JSON file."""
    return json.loads(Path(path).read_text())


def write_json(path: Path | str, data: dict) -> None:
    """Write dictionary as JSON file."""
    Path(path).write_text(json.dumps(data, indent=2, default=str))
