"""Declarative figure assembly.

A composed figure is described as data -- a canvas size and a list of panels
with their positions in millimetres -- and built by placing each panel PDF at
**scale 1.0**. Nothing is resized, so the type sizes chosen in ``sheafstyle``
survive into the finished file.

Coordinates use the designer convention: ``x`` from the left edge, ``y`` from
the *top* edge, both in mm. The conversion to PDF's bottom-left origin happens
here so layouts read the way you would draw them.

    from layout import Layout, Panel

    FIG2 = Layout(
        name="fig2",
        width_mm=180,
        height_mm=120,
        panels=[
            Panel("a", "panels/fig2a.pdf", x=0,  y=0),
            Panel("b", "panels/fig2b.pdf", x=92, y=0),
            Panel("c", "panels/fig2c.pdf", x=0,  y=62),
        ],
    )
    FIG2.compose("fig2.pdf")

Panel labels are drawn by the composer, not by the panels, so they land in the
same place on every figure.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from pypdf import PageObject, PdfReader, PdfWriter, Transformation

from sheafstyle import MM_PER_IN, TYPE_PT, WIDTHS_MM

__all__ = ["Panel", "Layout", "PT_PER_MM", "raster_panel", "stack", "row"]

PT_PER_MM = 72.0 / MM_PER_IN  # PDF points per millimetre

#: Where the label sits relative to its panel's top-left corner, in mm.
LABEL_DX_MM = 0.5
LABEL_DY_MM = 1.0
#: Closest a panel letter may sit to the canvas edge.
LABEL_MARGIN_MM = 1.5


def raster_panel(image, out_pdf, width_mm: float,
                 height_mm: float | None = None) -> tuple[Path, float]:
    """Wrap a raster image in a PDF page of exactly ``width_mm`` across.

    Returns ``(path, dpi)``, where ``dpi`` is the image's resolution at that
    physical width. The page is written at the image's native resolution, so
    nothing is resampled; a raster below the print floor stays below it, and
    the returned dpi is what ``verify_figure`` will report.

    This is how a legacy PNG enters the pipeline: its composition, canvas and
    panel labels become spec-compliant and reproducible, while its interior
    stays exactly the pixels it always was.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    img = Image.open(image)
    px_w, px_h = img.size
    if height_mm is None:
        height_mm = width_mm * px_h / px_w
    dpi = px_w / (width_mm / MM_PER_IN)

    fig = plt.figure(figsize=(width_mm / MM_PER_IN, height_mm / MM_PER_IN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, interpolation="none", aspect="auto")
    ax.set_axis_off()
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=dpi, pad_inches=0.0)
    plt.close(fig)
    return out_pdf, dpi


def stack(heights_mm, y0: float, gap: float = 5.0) -> list[float]:
    """Top edges for a vertical stack of panels of the given heights."""
    ys, y = [], y0
    for h in heights_mm:
        ys.append(y)
        y += h + gap
    return ys


def row(widths_mm, total_mm: float, margin: float = 2.0) -> list[float]:
    """Left edges spreading panels evenly across the canvas."""
    n = len(widths_mm)
    free = total_mm - 2 * margin - sum(widths_mm)
    gap = free / (n - 1) if n > 1 else 0.0
    xs, x = [], margin
    for w in widths_mm:
        xs.append(x)
        x += w + gap
    return xs


@dataclass
class Panel:
    """One placed panel.

    Parameters
    ----------
    label:
        Panel letter, e.g. ``"a"``. Pass ``None`` to place the artwork with no
        label (useful for a shared legend or a colour bar).
    path:
        Panel PDF, produced by ``sheafstyle.save_panel``.
    x, y:
        Position of the panel's top-left corner, in mm from the canvas's
        top-left corner.
    label_dx, label_dy:
        Per-panel override of the label offset, in mm.
    """

    label: str | None
    path: str | Path
    x: float
    y: float
    label_dx: float = LABEL_DX_MM
    label_dy: float = LABEL_DY_MM


@dataclass
class Layout:
    """A composed figure."""

    name: str
    width_mm: float = WIDTHS_MM["double"]
    height_mm: float = 100.0
    panels: Sequence[Panel] = field(default_factory=list)
    register: str = "normal"

    # -- geometry --------------------------------------------------------
    @property
    def width_pt(self) -> float:
        return self.width_mm * PT_PER_MM

    @property
    def height_pt(self) -> float:
        return self.height_mm * PT_PER_MM

    def _panel_size_mm(self, panel: Panel) -> tuple[float, float]:
        box = PdfReader(str(panel.path)).pages[0].mediabox
        return (float(box.width) / PT_PER_MM, float(box.height) / PT_PER_MM)

    def bounds(self) -> list[tuple[str, float, float, float, float]]:
        """``(label, x, y, w, h)`` in mm for every panel, for inspection."""
        out = []
        for p in self.panels:
            w, h = self._panel_size_mm(p)
            out.append((p.label or "-", p.x, p.y, w, h))
        return out

    def check(self) -> list[str]:
        """Report panels that fall off the canvas or overlap each other."""
        problems: list[str] = []
        boxes = []
        for p in self.panels:
            w, h = self._panel_size_mm(p)
            if p.x < -1e-6 or p.y < -1e-6:
                problems.append(f"panel {p.label!r}: negative origin ({p.x}, {p.y})")
            if p.x + w > self.width_mm + 1e-6:
                problems.append(
                    f"panel {p.label!r}: overruns right edge by "
                    f"{p.x + w - self.width_mm:.2f} mm"
                )
            if p.y + h > self.height_mm + 1e-6:
                problems.append(
                    f"panel {p.label!r}: overruns bottom edge by "
                    f"{p.y + h - self.height_mm:.2f} mm"
                )
            boxes.append((p.label, p.x, p.y, w, h))

        for i in range(len(boxes)):
            li, xi, yi, wi, hi = boxes[i]
            for j in range(i + 1, len(boxes)):
                lj, xj, yj, wj, hj = boxes[j]
                ox = min(xi + wi, xj + wj) - max(xi, xj)
                oy = min(yi + hi, yj + hj) - max(yi, yj)
                if ox > 0.5 and oy > 0.5:  # 0.5 mm tolerance
                    problems.append(
                        f"panels {li!r} and {lj!r} overlap by "
                        f"{ox:.1f} x {oy:.1f} mm"
                    )
        return problems

    # -- labels ----------------------------------------------------------
    def _label_overlay(self) -> PageObject | None:
        """A transparent page carrying just the panel letters."""
        labelled = [p for p in self.panels if p.label]
        if not labelled:
            return None

        import matplotlib.pyplot as plt

        size_pt = TYPE_PT[self.register]["label"]
        fig = plt.figure(
            figsize=(self.width_mm / MM_PER_IN, self.height_mm / MM_PER_IN)
        )
        for p in labelled:
            # A negative offset on a panel near the edge would push the letter
            # off the canvas, where it is silently lost. Clamp to the margin.
            lx = max(LABEL_MARGIN_MM, p.x + p.label_dx)
            ly = max(LABEL_MARGIN_MM, p.y + p.label_dy)
            # figure coordinates, origin bottom-left
            fx = lx / self.width_mm
            fy = 1.0 - ly / self.height_mm
            fig.text(
                fx,
                fy,
                p.label,
                fontsize=size_pt,
                fontweight="bold",
                ha="left",
                va="top",
            )
        buf = io.BytesIO()
        fig.savefig(buf, format="pdf", transparent=True, pad_inches=0.0)
        plt.close(fig)
        buf.seek(0)
        return PdfReader(buf).pages[0]

    # -- composition -----------------------------------------------------
    def compose(self, out_path: str | Path, *, strict: bool = True) -> Path:
        """Build the figure and write it to ``out_path``."""
        problems = self.check()
        if problems:
            message = f"layout {self.name!r}:\n  " + "\n  ".join(problems)
            if strict:
                raise ValueError(message)
            print("warning: " + message)

        canvas = PageObject.create_blank_page(
            width=self.width_pt, height=self.height_pt
        )

        for p in self.panels:
            src = PdfReader(str(p.path)).pages[0]
            h_mm = float(src.mediabox.height) / PT_PER_MM
            tx = p.x * PT_PER_MM
            # PDF origin is bottom-left; y is measured from the top.
            ty = (self.height_mm - p.y - h_mm) * PT_PER_MM
            canvas.merge_transformed_page(src, Transformation().translate(tx, ty))

        overlay = self._label_overlay()
        if overlay is not None:
            canvas.merge_page(overlay)

        writer = PdfWriter()
        writer.add_page(canvas)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as fh:
            writer.write(fh)
        return out_path

    def summary(self) -> str:
        lines = [
            f"{self.name}: {self.width_mm:g} x {self.height_mm:g} mm "
            f"({self.width_pt:.1f} x {self.height_pt:.1f} pt)"
        ]
        for label, x, y, w, h in self.bounds():
            lines.append(
                f"  {label:>3}  x={x:6.1f}  y={y:6.1f}  {w:5.1f} x {h:5.1f} mm"
            )
        return "\n".join(lines)
