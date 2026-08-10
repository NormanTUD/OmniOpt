"""Load the YAML config that defines all tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    p = Path(path) if path else CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_test_group(config: Dict[str, Any], group: str) -> List[Dict[str, Any]]:
    return list(config.get("groups", {}).get(group, []) or [])
