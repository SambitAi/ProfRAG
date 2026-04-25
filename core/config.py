from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_app_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
