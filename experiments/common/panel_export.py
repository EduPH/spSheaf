"""Save exactly the arrays a figure panel draws, so the drawing can move out.

The experiment scripts have historically drawn their own results: matplotlib
defaults, frames on spatial panels, titles inside the axes, type at whatever
size ``tight_layout`` produced. The manuscript figures are built in
``figures/`` against a written specification -- 180 mm canvas, 6/7/8 pt type,
no frames on spatial panels, 300 dpi floor -- and composed by ``layout.py``.

The way to reconcile the two is not to restyle the experiment scripts. It is
for an experiment to stop drawing and start *saving what it would have drawn*,
and for ``figures/`` to do all the drawing. This module is that handoff.

One ``.npz`` per figure holding the arrays, plus a ``.json`` sidecar holding the
scalars and the label names, which do not belong in an array file. Nothing here
imports matplotlib, and nothing in ``figures/`` needs to import an experiment.

    from experiments.common.panel_export import export_panels

    export_panels(
        out_dir, "osmfish",
        arrays=dict(grid_coords=grid_coords, boundary_smooth=boundary_smooth),
        meta=dict(ari=0.538, gt_names=list(gt_names)),
    )

Reload with :func:`load_panels`, which returns ``(arrays, meta)``.
"""

from __future__ import annotations

import json
import os

import numpy as np

__all__ = ["export_panels", "load_panels"]


def _jsonable(value):
    """Make numpy scalars and arrays survive a round trip through JSON."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def export_panels(out_dir: str, name: str, arrays: dict, meta: dict | None = None) -> str:
    """Write ``<out_dir>/panels/<name>.npz`` and ``.json``. Returns the npz path.

    ``arrays`` values are anything ``np.asarray`` accepts; ``meta`` carries the
    scalars, parameters and label names the panels need in order to be
    self-describing months later.
    """
    d = os.path.join(out_dir, "panels")
    os.makedirs(d, exist_ok=True)
    npz = os.path.join(d, f"{name}.npz")
    np.savez_compressed(npz, **{k: np.asarray(v) for k, v in arrays.items()})
    with open(os.path.join(d, f"{name}.json"), "w") as fh:
        json.dump(_jsonable(meta or {}), fh, indent=2, sort_keys=True)
    size = os.path.getsize(npz) / 1e6
    print(f"  panel data: {npz} ({size:.1f} MB, {len(arrays)} arrays)")
    return npz


def load_panels(npz_path: str) -> tuple[dict, dict]:
    """Read a panel dump back as ``(arrays, meta)``."""
    with np.load(npz_path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    side = os.path.splitext(npz_path)[0] + ".json"
    meta = {}
    if os.path.exists(side):
        with open(side) as fh:
            meta = json.load(fh)
    return arrays, meta
