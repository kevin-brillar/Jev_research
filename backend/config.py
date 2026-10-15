"""Config loading: every tunable in this experiment is a file under config/.

Nothing in backend/ hardcodes a prompt, a threshold, a route mapping or an
expected answer. Modules ask this loader for the document they need, so moving a
value between code and config is impossible by construction.

Environment:
    JEV_CONFIG_DIR   override the config root (tests point this at a fixture dir)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def config_dir() -> Path:
    """Root of the config tree (env override wins, so tests never touch the real one)."""
    override = os.getenv("JEV_CONFIG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "config"


def config_path(*parts: str) -> Path:
    return config_dir().joinpath(*parts)


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return loaded


@lru_cache(maxsize=None)
def _cached(path_str: str) -> dict[str, Any]:
    return read_yaml(Path(path_str))


def load(*parts: str) -> dict[str, Any]:
    """Load any config file by relative path, cached per process."""
    return _cached(str(config_path(*parts)))


def policy() -> dict[str, Any]:
    return load("policy.yaml")


def guidance() -> dict[str, Any]:
    return load("guidance.yaml")


def mock_rules() -> dict[str, Any]:
    return load("mock_rules.yaml")


def stub_responses() -> dict[str, Any]:
    return load("stub_responses.yaml")


def question_files() -> list[Path]:
    directory = config_path("questions")
    return sorted(p for p in directory.glob("*.yaml")) if directory.is_dir() else []


def workflow_path(name: str) -> Path:
    path = config_path("workflows", name)
    if path.suffix not in (".yaml", ".yml", ".json"):
        path = path.with_suffix(".yaml")
    return path


def workflow_names() -> list[str]:
    directory = config_path("workflows")
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.yaml"))


def clear_caches() -> None:
    """Drop cached documents (tests switch JEV_CONFIG_DIR between cases)."""
    _cached.cache_clear()
