from __future__ import annotations

import os
import sys
from functools import partial
from logging import WARNING, getLogger
from pathlib import Path

import swerex.utils.log as log_swerex
from git import Repo
from packaging import version

from sweagent.utils.log import get_logger

__version__ = "1.1.0"
PYTHON_MINIMUM_VERSION = (3, 11)
SWEREX_MINIMUM_VERSION = "1.2.0"
SWEREX_RECOMMENDED_VERSION = "1.2.1"

# Monkey patch the logger to use our implementation
log_swerex.get_logger = partial(get_logger, emoji="🦖")

# See https://github.com/SWE-agent/SWE-agent/issues/585
getLogger("datasets").setLevel(WARNING)
getLogger("numexpr.utils").setLevel(WARNING)
getLogger("LiteLLM").setLevel(WARNING)

PACKAGE_DIR = Path(__file__).resolve().parent

if sys.version_info < PYTHON_MINIMUM_VERSION:
    msg = (
        f"Python {sys.version_info.major}.{sys.version_info.minor} is not supported. "
        "SWE-agent requires Python 3.11 or higher."
    )
    raise RuntimeError(msg)

assert PACKAGE_DIR.is_dir(), PACKAGE_DIR


def _candidate_bases() -> list[Path]:
    candidates: list[Path] = [PACKAGE_DIR.parent, PACKAGE_DIR.parent.parent]
    cwd = Path.cwd().resolve()
    for path in [cwd, *cwd.parents]:
        candidates.extend([path, path / "SWE-agent", path / "swe-agent"])
    return candidates


def _resolve_repo_root() -> Path:
    env_path = os.getenv("SWE_AGENT_REPO_ROOT")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.is_dir():
            msg = f"SWE_AGENT_REPO_ROOT points to a non-directory path: {path}"
            raise RuntimeError(msg)
        return path

    for base in _candidate_bases():
        if (base / "sweagent").is_dir() and (base / "config").is_dir() and (base / "tools").is_dir():
            return base

    return PACKAGE_DIR.parent


def _resolve_existing_dir(env_var: str, subdir: str) -> Path:
    env_path = os.getenv(env_var)
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.is_dir():
            msg = f"{env_var} points to a non-directory path: {path}"
            raise RuntimeError(msg)
        return path

    for base in _candidate_bases():
        candidate = base / subdir
        if candidate.is_dir():
            return candidate

    msg = (
        f"Could not find required '{subdir}' directory for SWE-agent. "
        f"Set {env_var} explicitly to the correct path."
    )
    raise RuntimeError(msg)


def _resolve_writable_dir(env_var: str, subdir: str) -> Path:
    env_path = os.getenv(env_var)
    if env_path:
        path = Path(env_path).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    for base in _candidate_bases():
        candidate = base / subdir
        if candidate.is_dir():
            return candidate

    fallback = PACKAGE_DIR.parent / subdir
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


CONFIG_DIR = _resolve_existing_dir("SWE_AGENT_CONFIG_DIR", "config")
TOOLS_DIR = _resolve_existing_dir("SWE_AGENT_TOOLS_DIR", "tools")
TRAJECTORY_DIR = _resolve_writable_dir("SWE_AGENT_TRAJECTORY_DIR", "trajectories")
REPO_ROOT = _resolve_repo_root()


def get_agent_commit_hash() -> str:
    """Get the commit hash of the current SWE-agent commit.

    If we cannot get the hash, we return an empty string.
    """
    try:
        repo = Repo(REPO_ROOT, search_parent_directories=False)
    except Exception:
        return "unavailable"
    return repo.head.object.hexsha


def get_rex_commit_hash() -> str:
    import swerex

    try:
        repo = Repo(Path(swerex.__file__).resolve().parent.parent.parent, search_parent_directories=False)
    except Exception:
        return "unavailable"
    return repo.head.object.hexsha


def get_rex_version() -> str:
    from swerex import __version__ as rex_version

    return rex_version


def get_agent_version_info() -> str:
    hash = get_agent_commit_hash()
    rex_hash = get_rex_commit_hash()
    rex_version = get_rex_version()
    return f"This is SWE-agent version {__version__} ({hash=}) with SWE-ReX version {rex_version} ({rex_hash=})."


def impose_rex_lower_bound() -> None:
    rex_version = get_rex_version()
    minimal_rex_version = "1.2.0"
    if version.parse(rex_version) < version.parse(minimal_rex_version):
        msg = (
            f"SWE-ReX version {rex_version} is too old. Please update to at least {minimal_rex_version} by "
            "running `pip install --upgrade swe-rex`."
            "You can also rerun `pip install -e .` in this repository to install the latest version."
        )
        raise RuntimeError(msg)
    if version.parse(rex_version) < version.parse(SWEREX_RECOMMENDED_VERSION):
        msg = (
            f"SWE-ReX version {rex_version} is not recommended. Please update to at least {SWEREX_RECOMMENDED_VERSION} by "
            "running `pip install --upgrade swe-rex`."
            "You can also rerun `pip install -e .` in this repository to install the latest version."
        )
        get_logger("swe-agent", emoji="👋").warning(msg)


impose_rex_lower_bound()
get_logger("swe-agent", emoji="👋").info(get_agent_version_info())


__all__ = [
    "PACKAGE_DIR",
    "CONFIG_DIR",
    "get_agent_commit_hash",
    "get_agent_version_info",
    "__version__",
]
