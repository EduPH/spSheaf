"""Resolve where data, caches and results live.

No script in this repository should contain an absolute path. Everything goes
through this module, which resolves the data root in this order:

1. the ``SPSHEAF_DATA`` environment variable
2. ``data_root`` in ``paths.toml`` at the repository root
3. ``<repo>/data`` if it exists

Layout under the root::

    <root>/raw/<dataset>/       downloaded data, never written to
    <root>/cache/<dataset>/     derived intermediates, safe to delete
    <root>/results/<dataset>/   run outputs, figures, metrics
"""

from __future__ import annotations

import os
from pathlib import Path

try:                                    # Python >= 3.11
    import tomllib
except ModuleNotFoundError:             # pragma: no cover
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

__all__ = ["repo_root", "data_root", "raw", "cache", "results"]

_MARKERS = ("pyproject.toml", ".git")


def repo_root() -> Path:
    """Walk up from this file until a repository marker is found."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if any((parent / m).exists() for m in _MARKERS):
            return parent
    return here.parent.parent.parent


def _from_config() -> Path | None:
    cfg = repo_root() / "paths.toml"
    if not cfg.exists():
        return None
    if tomllib is None:                 # pragma: no cover
        raise RuntimeError(
            "paths.toml exists but no TOML parser is available. "
            "Use Python >= 3.11, or pip install tomli."
        )
    with cfg.open("rb") as fh:
        value = tomllib.load(fh).get("data_root")
    return Path(value).expanduser() if value else None


def data_root() -> Path:
    """The root of the data tree. Raises if it cannot be determined."""
    env = os.environ.get("SPSHEAF_DATA")
    if env:
        return Path(env).expanduser()

    cfg = _from_config()
    if cfg is not None:
        return cfg

    default = repo_root() / "data"
    if default.exists():
        return default

    raise FileNotFoundError(
        "Cannot locate the data root. Either set SPSHEAF_DATA, or copy "
        "paths.example.toml to paths.toml and set data_root in it."
    )


def _sub(kind: str, dataset: str | None, create: bool) -> Path:
    p = data_root() / kind
    if dataset:
        p = p / dataset
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def raw(dataset: str | None = None) -> Path:
    """Read-only input data. Never created implicitly."""
    p = _sub("raw", dataset, create=False)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} does not exist. Raw data is not created by the pipeline; "
            f"see docs/ for where to download {dataset or 'it'}."
        )
    return p


def cache(dataset: str | None = None) -> Path:
    """Derived intermediates. Created on demand; safe to delete."""
    return _sub("cache", dataset, create=True)


def results(dataset: str | None = None) -> Path:
    """Run outputs. Created on demand."""
    return _sub("results", dataset, create=True)
