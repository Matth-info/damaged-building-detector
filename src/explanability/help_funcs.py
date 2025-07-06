from __future__ import annotations

from pathlib import Path

import yaml


def read_config(path: str) -> dict[str, str]:
    """Load YAML file."""
    with Path.open(path) as file:
        return yaml.safe_load(file)
