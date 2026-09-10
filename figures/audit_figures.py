"""Audit every image the manuscript includes, against the figure specification.

Resolution is a property of the size a figure is *placed* at, not of the file.
The same 2000 px PNG is 387 dpi in this preprint, whose text block is 131.2 mm
wide, and 282 dpi once the journal typesets it across a 180 mm double column.
Both numbers are reported, and the second is the one that matters for
submission.

    python audit_figures.py <path to the manuscript .tex>

Vector figures are exempt from the resolution check and are verified instead by
``verify_figure.py``, which reads their canvas size and type sizes directly.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MM_PER_IN = 25.4
PT_PER_MM = 72.0 / MM_PER_IN

#: Text-block width of the sn-jnl template, from the compiled log (372 pt).
PREPRINT_WIDTH_MM = 372.0 / PT_PER_MM
#: Nature double-column width, the size the journal typesets a full-width figure.
SUBMISSION_WIDTH_MM = 180.0
MIN_DPI = 300.0

INCLUDE = re.compile(r"\\includegraphics(?:\[([^\]]*)\])?\{([^}]+)\}")
WIDTH_FRAC = re.compile(r"width\s*=\s*([0-9.]*)\s*\\(?:linewidth|textwidth|columnwidth)")
WIDTH_ABS = re.compile(r"width\s*=\s*([0-9.]+)\s*(mm|cm|in|pt)")
GRAPHICSPATH = re.compile(r"\\graphicspath\{(.+?)\}\s*$", re.M)
#: \figfull{...} places a figure at its drawn width, overflowing the narrow
#: text block. It carries no [width=...], so without this the audit reads the
#: macro definition instead and reports every figure as missing.
FIGFULL = re.compile(r"\\figfull\{([^}#]+)\}")

_UNIT_MM = {"mm": 1.0, "cm": 10.0, "in": 25.4, "pt": 25.4 / 72.0}


def graphics_dirs(tex: Path) -> list[Path]:
    """Directories LaTeX will search, from \\graphicspath plus the .tex's own."""
    dirs = [tex.parent]
    text = tex.read_text()
    m = GRAPHICSPATH.search(text)
    if m:
        for d in re.findall(r"\{([^}]*)\}", m.group(1)):
            dirs.append(tex.parent / d)
    return dirs


def resolve(name: str, dirs: list[Path]) -> Path | None:
    for d in dirs:
        for cand in (d / name, d / (name + ".pdf"), d / (name + ".png")):
            if cand.exists():
                return cand
    return None


def placed_width_mm(opts: str | None, base_mm: float) -> float:
    opts = opts or ""
    m = WIDTH_ABS.search(opts)
    if m:
        return float(m.group(1)) * _UNIT_MM[m.group(2)]
    m = WIDTH_FRAC.search(opts)
    if m:
        return base_mm * float(m.group(1) or 1.0)
    return base_mm


def audit(tex: Path, base_mm: float, min_dpi: float = MIN_DPI) -> list[dict]:
    try:
        from PIL import Image
    except ImportError:
        Image = None

    dirs = graphics_dirs(tex)
    rows, seen = [], set()
    text = tex.read_text()
    found = list(INCLUDE.findall(text))
    found += [(f"width={SUBMISSION_WIDTH_MM:g}mm", n)
              for n in FIGFULL.findall(text)]
    for opts, name in found:
        # \newcommand bodies contain \includegraphics{#1}; that is the macro's
        # parameter, not a file
        if "#" in name or name in seen:
            continue
        seen.add(name)
        placed = placed_width_mm(opts, base_mm)
        path = resolve(name, dirs)
        row = {"name": name, "placed_mm": placed, "path": path}
        if path is None:
            row["status"] = "missing"
        elif path.suffix.lower() == ".pdf":
            row["status"] = "vector"
        elif Image is None:
            row["status"] = "unknown (install Pillow)"
        else:
            w, h = Image.open(path).size
            row["px"] = (w, h)
            row["dpi"] = w / (placed / MM_PER_IN)
            row["status"] = "ok" if row["dpi"] >= min_dpi else "under"
        rows.append(row)
    return rows


def report(rows: list[dict], base_mm: float, label: str, min_dpi: float = MIN_DPI) -> str:
    out = [f"placement width {base_mm:.1f} mm ({label})", ""]
    out.append(f"{'figure':32s}{'placed':>9s}{'pixels':>12s}{'dpi':>7s}  status")
    out.append("-" * 72)
    for r in sorted(rows, key=lambda r: (r["status"] != "under", r["name"])):
        px = f"{r['px'][0]}x{r['px'][1]}" if "px" in r else "--"
        dpi = f"{r['dpi']:.0f}" if "dpi" in r else "--"
        flag = {"under": f"below {min_dpi:.0f} dpi", "vector": "vector",
                "missing": "MISSING", "ok": ""}.get(r["status"], r["status"])
        out.append(f"{r['name']:32s}{r['placed_mm']:8.1f}mm{px:>12s}{dpi:>7s}  {flag}")
    under = [r for r in rows if r["status"] == "under"]
    missing = [r for r in rows if r["status"] == "missing"]
    vector = [r for r in rows if r["status"] == "vector"]
    out += [
        "-" * 72,
        f"{len(rows)} figures: {len(vector)} vector, "
        f"{len(rows) - len(vector) - len(missing) - len(under)} raster at or above "
        f"{min_dpi:.0f} dpi, {len(under)} below, {len(missing)} missing",
    ]
    if under:
        out.append("")
        out.append("below the resolution floor at this placement:")
        for r in under:
            need = int(r["placed_mm"] / MM_PER_IN * min_dpi)
            out.append(f"  {r['name']:30s} {r['dpi']:.0f} dpi; needs {need} px wide")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Audit manuscript figures against the spec.")
    ap.add_argument("tex", type=Path)
    ap.add_argument("--preprint-mm", type=float, default=PREPRINT_WIDTH_MM,
                    help="text-block width of the LaTeX template")
    ap.add_argument("--submission-mm", type=float, default=SUBMISSION_WIDTH_MM,
                    help="width the journal typesets a full-width figure at")
    ap.add_argument("--min-dpi", type=float, default=MIN_DPI)
    ap.add_argument("--fail-under", action="store_true",
                    help="exit nonzero if any figure is below the floor at submission width")
    args = ap.parse_args(argv)

    if not args.tex.exists():
        print(f"{args.tex}: not found")
        return 1

    pre = audit(args.tex, args.preprint_mm, args.min_dpi)
    print(report(pre, args.preprint_mm, "this LaTeX template", args.min_dpi))
    print("\n" + "=" * 72 + "\n")
    sub = audit(args.tex, args.submission_mm, args.min_dpi)
    print(report(sub, args.submission_mm, "journal double column", args.min_dpi))

    if args.fail_under and any(r["status"] == "under" for r in sub):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
