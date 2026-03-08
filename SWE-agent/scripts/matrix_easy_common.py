#!/usr/bin/env python3
"""Shared helpers for the local matrix_easy run scripts."""

from __future__ import annotations

from pathlib import Path


VARIANTS = [
    "small_coder_only",
    "big_coder_only",
    "big_planner_small_coder",
    "big_planner_big_coder",
    "big_planner_small_coder_small_reviewer",
    "big_planner_small_coder_big_reviewer",
    "all_3_big",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return repo_root().parent


def config_dir() -> Path:
    return repo_root() / "config" / "custom_configs" / "matrix_easy"


def config_path(variant: str) -> Path:
    return config_dir() / f"{variant}.yaml"


def default_sweagent_bin() -> Path:
    candidates = [
        project_root() / ".venv" / "bin" / "sweagent",
        repo_root() / "env" / "bin" / "sweagent",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
