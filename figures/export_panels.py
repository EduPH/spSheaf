"""Export every figure's panels as separate files, for hand composition.

The pipeline normally composes each figure itself, so the arrangement is
reproducible. This script takes the same panels back out again: one file per
panel, each **at exactly the size it occupies in the composed figure**, plus a
manifest giving the position of its top-left corner on the figure's canvas.

Place a panel at 100% and the type lands at 6, 8 and 10 pt as specified.
Scaling a panel to fit is what breaks the specification -- a panel placed at
90% turns every 6 pt label into 5.4 pt -- so the manifest gives positions, not
a target box.

Panel letters are *not* baked into the panel files for the layout-composed
figures; the composer draws them. They are listed in the manifest so they can
be set in the editor, at 10 pt bold.

    python export_panels.py [outdir]

Writes ``<outdir>/<figure>/`` with the panel PDFs, a PNG of each at 600 dpi,
and ``<figure>_placement.csv``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import layout as L

HERE = Path(__file__).parent
DEFAULT_OUT = HERE / "panels_export"

#: Modules whose ``main()`` composes one or more layouts.
LAYOUT_MODULES = ["fig2_manuscript", "ed_benchmark", "ed_osmfish_modes",
                  "ed_composites"]


def capture_layouts(module_name: str) -> list[L.Layout]:
    """Run a figure module, recording every layout it composes.

    Rather than duplicating each module's layout here -- which would go stale
    the first time a figure is rearranged -- the composer is wrapped so the
    real layouts are captured as the module builds them.
    """
    import importlib

    captured: list[L.Layout] = []
    original = L.Layout.compose

    def spy(self, out_path, **kw):
        captured.append(self)
        return original(self, out_path, **kw)

    L.Layout.compose = spy
    try:
        mod = importlib.import_module(module_name)
        importlib.reload(mod)
        mod.main()
    finally:
        L.Layout.compose = original
    return captured


def to_png(pdf: Path, dpi: int = 600) -> Path | None:
    """Render a panel PDF alongside itself as a PNG, if a rasterizer exists."""
    if not shutil.which("pdftoppm"):
        return None
    out = pdf.with_suffix("")
    subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-singlefile",
                    str(pdf), str(out)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out.with_suffix(".png")


def export_layout(lay: L.Layout, outdir: Path, png_dpi: int = 600) -> Path:
    """Copy one layout's panels out, with a manifest of their positions."""
    d = outdir / lay.name
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, p in enumerate(lay.panels):
        src = Path(p.path)
        letter = p.label or f"p{i}"
        dst = d / f"{lay.name}_{letter}{src.suffix}"
        shutil.copyfile(src, dst)
        to_png(dst, png_dpi)
        w, h = lay._panel_size_mm(p)
        rows.append((dst.name, p.label or "", p.x, p.y, w, h))

    manifest = d / f"{lay.name}_placement.csv"
    with manifest.open("w") as fh:
        fh.write("# place each file at 100%, top-left corner at (x_mm, y_mm)\n")
        fh.write(f"# canvas: {lay.width_mm:g} x {lay.height_mm:g} mm\n")
        fh.write("file,label,x_mm,y_mm,width_mm,height_mm\n")
        for name, lbl, x, y, w, h in rows:
            fh.write(f"{name},{lbl},{x:.1f},{y:.1f},{w:.1f},{h:.1f}\n")

    print(f"{lay.name:18s} {lay.width_mm:5.1f} x {lay.height_mm:5.1f} mm, "
          f"{len(rows)} panels -> {d.relative_to(outdir.parent)}")
    for name, lbl, x, y, w, h in rows:
        print(f"    {lbl or '-':>2}  {w:6.1f} x {h:5.1f} mm  "
              f"at ({x:6.1f}, {y:5.1f})  {name}")
    return d


def export_fig1(outdir: Path, png_dpi: int = 600) -> Path:
    """Figure 1 is drawn as one scene, so it exports its own pieces."""
    import fig1_pipeline as f1

    d = outdir / "fig1"
    print(f"{'fig1':18s} {f1.W_MM:5.1f} x {f1.H_MM:5.1f} mm, "
          f"{len(f1.ITEMS)} pieces -> {d.relative_to(outdir.parent)}")
    f1.export_panels(d, png_dpi=png_dpi)
    return d


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    outdir = Path(argv[0]) if argv else DEFAULT_OUT
    outdir.mkdir(parents=True, exist_ok=True)

    export_fig1(outdir)
    skipped = []
    for name in LAYOUT_MODULES:
        try:
            layouts = capture_layouts(name)
        except FileNotFoundError as exc:
            # A figure whose source table or raster is not in this checkout.
            skipped.append((name, Path(str(exc).split(": ")[-1].strip("'")).name))
            continue
        for lay in layouts:
            export_layout(lay, outdir)

    if skipped:
        print("\nskipped, input missing from this checkout:")
        for name, missing in skipped:
            print(f"  {name:20s} needs {missing}")

    print(f"\nall panels under {outdir}")
    print("place every file at 100%; scaling a panel rescales its type.")


if __name__ == "__main__":
    main()
