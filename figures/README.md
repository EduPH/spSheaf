# Figure pipeline

Automates the figure specification — 180 mm canvas, ≥300 dpi, 6/7/8 pt type
(tick labels / other text / panel letters; Nature's artwork guide puts all
figure text at 5–7 pt and panel letters at 8 pt bold, so the 6/8/10 pt first
proposed sits above the journal's maximum), no axis frames on spatial plots —
so it holds by construction instead of by memory.

```
make figures          # build every figure from data
make figures-verify   # assert the spec on the finished PDFs
make figures-test     # unit tests for the pipeline itself
```

## Where things live

```
figures/assets/   legacy rasters the Extended Data composites place (inputs)
figures/out/      everything the pipeline writes (generated, untracked)
```

`assets/` holds run outputs from experiments that have not yet been redrawn
through `sheafstyle`; `ed_composites.py` places them at a declared size with
spec-compliant panel letters, and reports the resolution of each. Replacing
one means writing a figure script that draws it from data and dropping the
raster.

## The sizes only hold at the drawn width

Every figure is 180 mm across, Nature's double-column width, and the type
sizes are Nature's: all text 5-7 pt, panel letters 8 pt bold. Those are final
sizes. Place a 180 mm figure in a 131 mm text block -- which is what
`\includegraphics[width=\linewidth]` does in the manuscript template -- and
everything is scaled to 73%: a 6 pt tick label reaches the page at 4.4 pt.

Place figures at 180 mm, not at `\linewidth`.

## The one rule

**Every panel is drawn at its exact final physical size and is never rescaled.**

A 6 pt tick label is only 6 pt in print if nothing between matplotlib and the
page multiplies it. Export a panel at an arbitrary size, drag it to fit in a
layout tool, and every type size in the spec is wrong by whatever the scale
factor was — silently, with no visible symptom until a copy-editor measures it.

Two consequences:

- `save_panel` **rejects `bbox_inches="tight"`**. It crops the canvas to the
  drawn content, so the file is no longer the size you declared. Size the
  figure correctly and use `subplots_adjust` for margins.
- The canvas is 180 mm, Nature's double-column width, so the figure is placed
  at scale 1.0 at typesetting. (7 inches is 177.8 mm, which would be scaled up
  by 1.2% and land 6 pt type at 6.07 pt. Close enough not to matter, but the
  point of the pipeline is that it doesn't have to be a judgement call.)

## Files

| File | What it does |
|---|---|
| `sheafstyle.py` | rcParams for the type registers, fixed method→colour map, `panel()` at exact size, `spatial()`, `rasterize()`, `save_panel()` |
| `layout.py` | A figure declared as data — canvas plus panel positions in mm — composed at scale 1.0 by `pypdf` |
| `verify_figure.py` | Reads the finished PDF and asserts canvas, type sizes and raster dpi. Exits nonzero |
| `example_fig2.py` | A working Fig. 2 built end to end. Real numbers except panel d |
| `test_figstyle.py` | 18 tests, including that each guarantee fails when violated |

## Writing a panel

```python
import sheafstyle as ss

ss.apply()                      # normal register: 6/7/8 pt
fig, ax = ss.panel(52, 42)      # millimetres, exact
fig.subplots_adjust(left=0.26, right=0.98, top=0.94, bottom=0.32)
ax.bar(names, values, color=[ss.METHOD_COLORS[n] for n in names])
ax.set_ylabel("ARI (MERFISH)")
ss.save_panel(fig, "panels/fig2a.pdf")
```

For a spatial plot:

```python
fig, ax = ss.panel(60, 46)
ss.spatial(ax)                  # no frame, no ticks, equal aspect
sc = ax.scatter(x, y, c=colors, s=0.35, linewidths=0, marker=".")
ss.rasterize(sc)                # marks raster at 600 dpi, text stays vector
ss.save_panel(fig, "panels/fig2d.pdf")
```

`rasterize` matters more than it looks. The osmFISH grid panel is 44,410
points; left as vector it produces a PDF that is slow to open and that some
typesetters reject outright. Rasterizing the marks alone keeps the file small
while every label stays crisp and selectable.

Use `ss.arm_style(name)` for ablation arms — coordinate-based scores are blues,
subspace-only scores purples, and the three null controls come back hatched, so
the inversion between Table 2 and the boundary result reads from colour alone.

## Composing a figure

```python
from layout import Layout, Panel

FIG2 = Layout(
    name="fig2", width_mm=180, height_mm=104,
    panels=[
        Panel("a", "panels/fig2a.pdf", x=5,   y=6, label_dx=-3, label_dy=-5),
        Panel("b", "panels/fig2b.pdf", x=61,  y=6, label_dx=-3, label_dy=-5),
    ],
)
FIG2.compose("fig2.pdf")
```

Coordinates are the designer convention: `x` from the left edge, `y` from the
**top** edge, in mm. `compose` refuses to build a layout whose panels run off
the canvas or overlap by more than 0.5 mm, and `FIG2.summary()` prints the
placement table. Panel letters are drawn by the composer, not by the panels, so
they land identically on every figure.

## Affinity Designer

Still available, and sometimes the right tool for a hand-drawn schematic like
Fig. 1's core-idea panel. Two rules if you go that way:

1. **Import panels at 100%.** Any other scale invalidates the type sizes.
2. **Export at 180 mm** and run `verify_figure.py` on the result. The check is
   the same either way, which is the point — Affinity becomes an optional
   editing step rather than an unreproducible one.

Panels are written with `pdf.fonttype 42`, so text arrives in Affinity as real
editable text rather than outlines.

## What the verifier does and does not check

Checked from the PDF: canvas size to 0.2 mm; every text object's nominal point
size, read from the `Tf` operator and scaled by the text and graphics matrices;
every embedded raster's dpi at its placed size.

Sizes are read from the font operator rather than glyph bounding boxes on
purpose. A rotated tick label measured by its bounding box reports a different
size for every glyph shape — 'i' and 'R' at the same 6 pt measure 2.2 and 5.6 —
which produces a page of false failures.

Not checkable from a PDF: whether a spatial panel has spines. A drawn frame and
a plotted line are the same object once the file is written. That rule is
enforced at draw time by `sheafstyle.check_axes`, which `save_panel` runs
before writing, so a spatial panel with a stray spine fails at build.
