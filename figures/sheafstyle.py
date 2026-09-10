"""Figure style for the spSheaf manuscript.

One rule makes the whole scheme work: **every panel is drawn at its exact final
physical size and is never rescaled afterwards.** A 6 pt tick label is only 6 pt
in the printed figure if nothing between here and the page multiplies it. That
is why :func:`save_panel` refuses ``bbox_inches="tight"`` -- it silently changes
the output size, which changes every type size in the figure by the same factor.

Type registers, set to Nature's artwork guide (all figure text 5-7 pt at
final size; panel letters 8 pt bold):

============  ========  =======
element       normal    small
============  ========  =======
tick labels   6 pt      5 pt
other text    7 pt      6 pt
panel labels  8 pt      8 pt
============  ========  =======

These hold only if the figure reaches the page at the width it was drawn at.
A 180 mm figure placed in a 131 mm text block is scaled to 73%, and every
size above is then a quarter smaller than it says.

Canvas widths are Nature's column widths in mm. The default is ``double``
(180 mm), which is the width the manuscript's main figures are typeset at, so
the figure reaches the page at scale 1.0.

Typical use::

    import sheafstyle as ss

    ss.apply()
    fig, ax = ss.panel(88, 60)
    ax.plot(x, y, color=ss.METHOD_COLORS["spSheaf"])
    ss.save_panel(fig, "panels/fig2a.pdf")
"""

from __future__ import annotations

import warnings
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = [
    "MM_PER_IN",
    "WIDTHS_MM",
    "TYPE_PT",
    "METHOD_COLORS",
    "ARM_COLORS",
    "apply",
    "mm",
    "panel",
    "panel_grid",
    "spatial",
    "rasterize",
    "save_panel",
    "check_axes",
]

MM_PER_IN = 25.4

#: Nature column widths, in millimetres.
WIDTHS_MM = {"single": 88.0, "onehalf": 120.0, "double": 180.0}

#: Type sizes in points, by register.
#:
#: Nature's artwork guide puts *all* figure text between 5 and 7 pt at final
#: size, with panel letters at 8 pt bold. The earlier scale (6/8/10, and 4 pt
#: in the small register) sat outside that at both ends: 8 and 10 pt above the
#: maximum, 4 pt below the minimum and genuinely illegible. These are the
#: sizes the figures are actually published at, so they are the ones to hold.
TYPE_PT = {
    "normal": {"tick": 6.0, "text": 7.0, "label": 8.0},
    "small": {"tick": 5.0, "text": 6.0, "label": 8.0},
}

#: Nature's stated floor. Nothing but a mathtext sub/superscript may go below.
MIN_PT = 5.0

#: The set of sizes the verifier will accept in a composed figure.
ALLOWED_PT = sorted({v for reg in TYPE_PT.values() for v in reg.values()})

#: Fixed method -> colour map. Okabe-Ito, colour-vision-deficiency safe.
#: spSheaf keeps the strong blue in every figure it appears in.
METHOD_COLORS = {
    "spSheaf": "#0072B2",
    "SSAM": "#E69F00",
    "SSAM (guided)": "#F0C46A",
    "Points2Regions": "#009E73",
    "FICTURE": "#CC79A7",
    "ground truth": "#333333",
}

#: Fixed colours for the edge-score ablation arms (Table 2 of the manuscript).
#: Coordinate-based scores are blues, subspace-only scores are purples, and the
#: three null controls are greys -- so the inversion in the paper is legible
#: from colour alone.
ARM_COLORS = {
    "sheaf": "#0072B2",
    "sheaf_local": "#56B4E9",
    "identity": "#009E73",
    "neighbour": "#E69F00",
    "procrustes": "#CC79A7",
    "grassmann": "#B266A0",
    "projection": "#9B5A99",
    "shuffled": "#9E9E9E",
    "random": "#BDBDBD",
    "none": "#E0E0E0",
}

_CONTROL_ARMS = ("shuffled", "random", "none")


def apply(register: str = "normal", *, raster_dpi: int = 600) -> None:
    """Install the manuscript rcParams.

    Parameters
    ----------
    register:
        ``"normal"`` (6/7/8 pt) or ``"small"`` (5/6/8 pt).
    raster_dpi:
        Resolution used for artists marked with :func:`rasterize`. Text and
        axes stay vector regardless. 600 leaves headroom above the journal's
        300 dpi floor after any downstream scaling.
    """
    if register not in TYPE_PT:
        raise ValueError(f"register must be one of {sorted(TYPE_PT)}, got {register!r}")
    pt = TYPE_PT[register]

    mpl.rcParams.update(
        {
            # --- type -------------------------------------------------------
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": pt["text"],
            "axes.labelsize": pt["text"],
            "axes.titlesize": pt["text"],
            "xtick.labelsize": pt["tick"],
            "ytick.labelsize": pt["tick"],
            "legend.fontsize": pt["text"],
            "legend.title_fontsize": pt["text"],
            "figure.titlesize": pt["label"],
            # Embed real fonts (Type 42), not Type 3 outlines. Keeps text
            # selectable and editable if a panel is opened in a vector editor.
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            # --- lines and marks -------------------------------------------
            "axes.linewidth": 0.5,
            "grid.linewidth": 0.4,
            "lines.linewidth": 0.9,
            "lines.markersize": 2.5,
            "patch.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            # --- frames -----------------------------------------------------
            # Top and right spines off everywhere; spatial panels drop all four
            # via spatial().
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            # --- output -----------------------------------------------------
            "figure.dpi": 200,          # screen preview only
            "savefig.dpi": raster_dpi,  # applies to rasterized artists
            "savefig.transparent": True,
            "savefig.pad_inches": 0.0,
            "figure.autolayout": False,
            "figure.constrained_layout.use": False,
        }
    )


def mm(*values: float) -> tuple[float, ...] | float:
    """Convert millimetres to inches (matplotlib's unit)."""
    out = tuple(v / MM_PER_IN for v in values)
    return out[0] if len(out) == 1 else out


def panel(width_mm: float, height_mm: float, **kwargs):
    """A single-axes figure whose canvas is exactly ``width_mm`` x ``height_mm``.

    Any keyword arguments are forwarded to :func:`matplotlib.pyplot.subplots`.
    """
    fig, ax = plt.subplots(figsize=(width_mm / MM_PER_IN, height_mm / MM_PER_IN), **kwargs)
    return fig, ax


def panel_grid(width_mm: float, height_mm: float, nrows: int, ncols: int, **kwargs):
    """A multi-axes panel at an exact physical size.

    Use this only for a genuine grid of small multiples that ships as one
    panel. Independent panels of a composed figure should each be their own
    file, placed by ``layout.py`` -- that is what keeps the arrangement
    reproducible.
    """
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width_mm / MM_PER_IN, height_mm / MM_PER_IN),
        **kwargs,
    )
    return fig, axes


def spatial(ax, *, equal: bool = True):
    """Strip an axes down for a spatial plot: no frame, no ticks, equal aspect.

    This is the collaborator's "no axis frames on spatial plots" rule. Applying
    it also marks the axes so :func:`check_axes` can confirm the rule held.
    """
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    if equal:
        ax.set_aspect("equal", adjustable="datalim")
    ax._spsheaf_spatial = True  # read back by check_axes
    return ax


def rasterize(*artists, dpi: int | None = None):
    """Mark artists for rasterization, keeping text and axes vector.

    A whole-section scatter is hundreds of thousands of points; left as vector
    it produces a PDF no typesetter will accept. Rasterizing just the marks
    keeps the file small while every label stays crisp and selectable.
    """
    for a in artists:
        if a is None:
            continue
        a.set_rasterized(True)
        if dpi is not None:
            fig = getattr(a, "figure", None)
            if fig is not None:
                fig.set_dpi(dpi)
    return artists[0] if len(artists) == 1 else artists


def check_axes(fig) -> list[str]:
    """Return a list of style violations found in ``fig``.

    Checks what cannot be recovered from the finished PDF: that axes passed
    through :func:`spatial` really have no visible spines, and that no axes
    carries a visible top or right spine.
    """
    problems: list[str] = []
    for i, ax in enumerate(fig.axes):
        is_spatial = getattr(ax, "_spsheaf_spatial", False)
        visible = [s for s, sp in ax.spines.items() if sp.get_visible()]
        if is_spatial and visible:
            problems.append(f"axes[{i}]: marked spatial but has spines {visible}")
        if not is_spatial:
            bad = [s for s in ("top", "right") if s in visible]
            if bad:
                problems.append(f"axes[{i}]: spines {bad} should be hidden")
    return problems


def save_panel(fig, path, *, strict: bool = True, **kwargs) -> None:
    """Save a panel at exactly the size it was created with.

    ``bbox_inches`` is rejected outright. ``bbox_inches="tight"`` crops the
    canvas to the drawn content, so the saved file is no longer the size you
    declared; place that file at scale 1.0 and the panel is the wrong size,
    place it to fit and every type size is wrong by the scale factor. Set the
    figure size correctly instead, and use ``fig.subplots_adjust`` or
    ``ax.set_position`` to control margins.
    """
    if "bbox_inches" in kwargs:
        raise ValueError(
            "bbox_inches changes the output size and breaks the type spec; "
            "size the figure correctly and adjust margins instead"
        )
    if strict:
        problems = check_axes(fig)
        if problems:
            raise ValueError(
                "style violations in " + str(path) + ":\n  " + "\n  ".join(problems)
            )
    path = str(path)
    fig.savefig(path, pad_inches=0.0, **kwargs)


def legend_handles(names: Sequence[str], colors: dict[str, str] | None = None):
    """Proxy handles for a method or arm legend, in the given order."""
    from matplotlib.patches import Patch

    colors = colors or METHOD_COLORS
    return [Patch(facecolor=colors[n], edgecolor="none", label=n) for n in names]


def arm_style(arm: str) -> dict:
    """Colour and hatch for one ablation arm; controls are hatched."""
    style = {"color": ARM_COLORS.get(arm, "#777777")}
    if arm in _CONTROL_ARMS:
        style["hatch"] = "///"
        style["edgecolor"] = "#666666"
    return style


def _warn_if_unapplied() -> None:
    if mpl.rcParams["pdf.fonttype"] != 42:
        warnings.warn("sheafstyle.apply() has not been called; type spec not in force")
