"""Load the YAML config that defines all tests, with parameter resolution
and tag-based filtering.

The YAML is the single abstract data structure: a list of test definitions,
each tagged. Filters (tags, only, exclude) decide which tests run.
Placeholders in commands are substituted from the `parameters` section,
which is itself populated from CLI flags.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class Parameter:
    name: str
    default: Any
    cli: Optional[str] = None
    type: str = "str"
    description: str = ""

    def parse(self, raw: str) -> Any:
        if self.type == "int":
            return int(raw)
        if self.type == "float":
            return float(raw)
        if self.type == "bool":
            return raw.lower() in ("1", "true", "yes", "on")
        return raw


@dataclass
class Test:
    id: str
    name: str
    cmd: Optional[str] = None
    python_check: Optional[str] = None
    wanted_exit_code: int = 0
    alternative_exit_code: Optional[int] = None
    tags: List[str] = field(default_factory=list)

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags


class TestConfig:
    """Loaded configuration."""

    def __init__(self, data: Dict[str, Any], repo_root: Path):
        self.repo_root = repo_root
        self.parameters: Dict[str, Parameter] = {}
        for name, pdef in (data.get("parameters") or {}).items():
            self.parameters[name] = Parameter(
                name=name,
                default=pdef.get("default"),
                cli=pdef.get("cli"),
                type=pdef.get("type", "str"),
                description=pdef.get("description", ""),
            )
        self.tests: List[Test] = []
        for t in data.get("tests") or []:
            self.tests.append(
                Test(
                    id=t["id"],
                    name=t["name"],
                    cmd=t.get("cmd"),
                    python_check=t.get("python_check"),
                    wanted_exit_code=int(t.get("wanted_exit_code", 0)),
                    alternative_exit_code=(
                        int(t["alternative_exit_code"])
                        if "alternative_exit_code" in t and t["alternative_exit_code"] is not None
                        else None
                    ),
                    tags=list(t.get("tags") or []),
                )
            )
        self.smoke_tests: List[str] = list(data.get("smoke_tests") or [])

    def resolve_parameters(self, cli_args: Dict[str, str]) -> Dict[str, Any]:
        """Resolve parameter values from defaults + CLI overrides."""
        resolved: Dict[str, Any] = {}
        for name, p in self.parameters.items():
            resolved[name] = p.default
        for cli_name, raw in cli_args.items():
            for name, p in self.parameters.items():
                if p.cli and p.cli == cli_name:
                    resolved[name] = p.parse(raw)
        resolved["REPO_ROOT"] = str(self.repo_root)
        return resolved

    def get_test(self, test_id: str) -> Optional[Test]:
        for t in self.tests:
            if t.id == test_id:
                return t
        return None

    def filter(
        self,
        only: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
        any_tag: Optional[Iterable[str]] = None,
    ) -> List[Test]:
        only = list(only) if only else []
        exclude = list(exclude) if exclude else []
        any_tag = list(any_tag) if any_tag else []

        result: List[Test] = []
        for t in self.tests:
            if only and not all(t.has_tag(tag) for tag in only):
                continue
            if any(t.has_tag(tag) for tag in exclude):
                continue
            if any_tag and not any(t.has_tag(tag) for tag in any_tag):
                continue
            result.append(t)
        return result


_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def substitute(cmd: str, params: Dict[str, Any]) -> str:
    """Replace {placeholder} tokens with values from params."""
    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if key not in params:
            raise KeyError(f"Unknown placeholder: {{{key}}}")
        return str(params[key])
    return _PLACEHOLDER_RE.sub(_replace, cmd)


def load_config(path: Optional[str | os.PathLike] = None) -> TestConfig:
    p = Path(path) if path else CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return TestConfig(data, REPO_ROOT)


def list_cli_options(config: TestConfig) -> List[Parameter]:
    return [p for p in config.parameters.values() if p.cli]
