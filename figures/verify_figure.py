"""Check a finished figure PDF against the manuscript's specification.

Run this on the composed file, not on the panels. It reads the PDF the way a
typesetter would and asserts:

* the canvas is exactly the declared width (default 180 mm, Nature double
  column), so the figure is placed at scale 1.0 and nothing rescales the type;
* every text object is one of the allowed nominal sizes -- 4, 6, 8 or 10 pt;
* every embedded raster is at least 300 dpi at its placed size.

Exit status is 1 if anything fails, so ``make`` stops.

    python verify_figure.py fig2.pdf --width-mm 180

Type sizes are read from the ``Tf`` operator and scaled by the combined text
and graphics matrices, which is the nominal point size a typesetter sees. Do
not use a glyph bounding box for this: rotated tick labels report a different
"size" per glyph shape, so 'i' and 'R' at the same 6 pt measure 2.2 and 5.6.

Two rules cannot be recovered from a PDF and are checked at draw time instead,
by ``sheafstyle.check_axes``: that spatial panels have no spines, and that no
panel shows a top or right spine.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import LAParams, LTFigure, LTImage, LTTextContainer
from pypdf import PdfReader

MM_PER_IN = 25.4
PT_PER_MM = 72.0 / MM_PER_IN

DEFAULT_ALLOWED_PT = (4.0, 6.0, 8.0, 10.0)
DEFAULT_MIN_DPI = 300.0
SIZE_TOL = 0.05  # pt

#: Mathtext sets sub- and superscripts at a fixed fraction of the surrounding
#: size, so `$\alpha_s$` in an 8 pt label legitimately emits a 5.6 pt glyph.
#: Only the sizes that actually carry mathtext are used as bases: axis labels
#: (8 pt) and tick labels (6 pt), giving 5.6 and 4.2. The panel-label size is
#: deliberately excluded -- panel letters are never mathtext, and admitting
#: 10 x 0.7 = 7.0 pt would make a plain 7 pt mistake indistinguishable from a
#: subscript.
SUBSCRIPT_RATIO = 0.7
MATHTEXT_BASE_PT = (6.0, 8.0)
ABSOLUTE_MIN_PT = 4.0


def _mat_scale(cm, tm) -> float:
    """Uniform scale of the combined text/graphics matrix."""
    a = tm[0] * cm[0] + tm[1] * cm[2]
    b = tm[0] * cm[1] + tm[1] * cm[3]
    c = tm[2] * cm[0] + tm[3] * cm[2]
    d = tm[2] * cm[1] + tm[3] * cm[3]
    return math.sqrt(abs(a * d - b * c))


def _type_sizes(path: Path) -> Counter:
    sizes: Counter = Counter()

    def visit(text, cm, tm, font_dict, font_size):
        stripped = (text or "").strip()
        if not stripped or not font_size:
            return
        try:
            size = float(font_size) * _mat_scale(cm, tm)
        except (TypeError, ValueError):
            return
        sizes[round(size, 2)] += len(stripped)

    PdfReader(str(path)).pages[0].extract_text(visitor_text=visit)
    return sizes


def _iter_items(container):
    for item in container:
        yield item
        if isinstance(item, (LTFigure, LTTextContainer)):
            yield from _iter_items(item)


def _images(path: Path) -> list[tuple[str, float, float]]:
    out = []
    page = list(extract_pages(str(path), laparams=LAParams()))[0]
    for item in _iter_items(page):
        if isinstance(item, LTImage) and item.srcsize:
            w_in = item.width / 72.0
            h_in = item.height / 72.0
            if w_in > 0 and h_in > 0:
                out.append((item.name, item.srcsize[0] / w_in, item.srcsize[1] / h_in))
    return out


def verify(
    path: Path,
    *,
    width_mm: float | None = 180.0,
    height_mm: float | None = None,
    allowed_pt: tuple[float, ...] = DEFAULT_ALLOWED_PT,
    min_dpi: float = DEFAULT_MIN_DPI,
    tol_mm: float = 0.2,
) -> tuple[list[str], dict]:
    """Return ``(failures, report)`` for one figure PDF."""
    failures: list[str] = []

    reader = PdfReader(str(path))
    if len(reader.pages) != 1:
        failures.append(f"expected a single page, found {len(reader.pages)}")
    box = reader.pages[0].mediabox
    page_w_mm = float(box.width) / PT_PER_MM
    page_h_mm = float(box.height) / PT_PER_MM

    if width_mm is not None and abs(page_w_mm - width_mm) > tol_mm:
        failures.append(
            f"canvas width {page_w_mm:.2f} mm, expected {width_mm:.2f} mm "
            "(a figure typeset at another width rescales every type size)"
        )
    if height_mm is not None and abs(page_h_mm - height_mm) > tol_mm:
        failures.append(f"canvas height {page_h_mm:.2f} mm, expected {height_mm:.2f} mm")

    derived = tuple(
        a * SUBSCRIPT_RATIO for a in MATHTEXT_BASE_PT
        if a in allowed_pt and a * SUBSCRIPT_RATIO >= ABSOLUTE_MIN_PT - SIZE_TOL
    )
    sizes = _type_sizes(path)
    for size, count in sorted(sizes.items()):
        if any(abs(size - a) <= SIZE_TOL for a in allowed_pt):
            continue
        if any(abs(size - a) <= SIZE_TOL for a in derived):
            continue  # mathtext sub- or superscript of an allowed size
        if size < ABSOLUTE_MIN_PT:
            failures.append(
                f"{count} character(s) at {size:g} pt, below the "
                f"{ABSOLUTE_MIN_PT:g} pt floor"
            )
        else:
            failures.append(
                f"{count} character(s) at {size:g} pt; allowed sizes are "
                + ", ".join(f"{a:g}" for a in allowed_pt)
                + " (or a x%.1f mathtext subscript of one)" % SUBSCRIPT_RATIO
            )

    images = _images(path)
    for name, dpi_x, dpi_y in images:
        worst = min(dpi_x, dpi_y)
        if worst < min_dpi - 1:
            failures.append(
                f"image {name}: {worst:.0f} dpi at placed size, below the "
                f"{min_dpi:.0f} dpi minimum"
            )

    report = {
        "path": str(path),
        "canvas_mm": (round(page_w_mm, 2), round(page_h_mm, 2)),
        "type_sizes_pt": {f"{k:g}": v for k, v in sorted(sizes.items())},
        "images": [(n, round(x), round(y)) for n, x, y in images],
    }
    return failures, report


def _format(report: dict, failures: list[str]) -> str:
    w, h = report["canvas_mm"]
    lines = [report["path"], f"  canvas   {w} x {h} mm"]
    if report["type_sizes_pt"]:
        counts = ", ".join(f"{k} pt x{v}" for k, v in report["type_sizes_pt"].items())
        lines.append(f"  type     {counts}")
    else:
        lines.append("  type     (no text found)")
    if report["images"]:
        for name, dx, dy in report["images"]:
            lines.append(f"  raster   {name}: {dx} x {dy} dpi")
    else:
        lines.append("  raster   none embedded (fully vector)")
    if failures:
        lines.append("  FAIL")
        lines.extend(f"    - {f}" for f in failures)
    else:
        lines.append("  OK")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify a figure PDF against the spec.")
    ap.add_argument("pdfs", nargs="+", type=Path)
    ap.add_argument("--width-mm", type=float, default=180.0,
                    help="expected canvas width in mm; 0 skips the check")
    ap.add_argument("--height-mm", type=float, default=None)
    ap.add_argument("--min-dpi", type=float, default=DEFAULT_MIN_DPI)
    ap.add_argument("--allowed-pt", type=float, nargs="+",
                    default=list(DEFAULT_ALLOWED_PT))
    args = ap.parse_args(argv)

    status = 0
    for pdf in args.pdfs:
        if not pdf.exists():
            print(f"{pdf}: missing")
            status = 1
            continue
        failures, report = verify(
            pdf,
            width_mm=args.width_mm or None,
            height_mm=args.height_mm,
            allowed_pt=tuple(args.allowed_pt),
            min_dpi=args.min_dpi,
        )
        print(_format(report, failures))
        if failures:
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
