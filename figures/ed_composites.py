"""Compose the Extended Data figures that were assembled in LaTeX.

Five Extended Data figures stacked two to five images inside one ``figure``
environment, spaced with ``\\hfill`` and ``minipage`` and lettered with
``\\textbf{a}``. Each becomes a single composed PDF here: one 180 mm canvas,
panel letters at the specification's 10 pt, positions declared as data.

The panels are the existing rasters, placed at their native resolution. That
does not change what is inside them -- a panel below the print floor stays
below it -- and the dpi of each is reported so it is visible rather than
implicit.

    python ed_composites.py
"""

from __future__ import annotations

from pathlib import Path

import sheafstyle as ss
from layout import Layout, Panel, raster_panel, row, stack

HERE = Path(__file__).parent
REPO = HERE.parent
#: Legacy rasters these composites place. They are figure *inputs*, not
#: outputs: each is a run output from an experiment that has not yet been
#: redrawn through sheafstyle. See figures/README.md.
SRC = HERE / "assets"
OUTDIR = HERE / "out"
PANELS = HERE / "panels"
PANELS.mkdir(exist_ok=True)

W = 180.0        # canvas width, Nature double column
GAP = 5.0        # vertical gap between stacked panels
TOP = 6.0        # top margin, leaving room for the panel letter

_dpi_report: list[tuple[str, str, float]] = []


def place(figname: str, image: str, width_mm: float):
    """Wrap one source raster as a panel PDF at its final width."""
    out = PANELS / f"{figname}_{Path(image).stem}.pdf"
    path, dpi = raster_panel(SRC / image, out, width_mm)
    _dpi_report.append((figname, image, dpi))
    from pypdf import PdfReader
    box = PdfReader(str(path)).pages[0].mediabox
    return path, float(box.height) / (72.0 / 25.4)


def build(name: str, items, height_extra: float = 4.0) -> Layout:
    """``items`` is a list of ``(label, image, width_mm, x)`` laid out in a stack."""
    panels, heights = [], []
    for label, image, wmm, x in items:
        path, hmm = place(name, image, wmm)
        panels.append([label, path, x])
        heights.append(hmm)
    ys = stack(heights, TOP, GAP)
    total = ys[-1] + heights[-1] + height_extra
    return Layout(
        name=name, width_mm=W, height_mm=total,
        panels=[Panel(l, p, x, y, label_dx=-4, label_dy=-5)
                for (l, p, x), y in zip(panels, ys)],
    )


def ed_scale() -> Layout:
    """a Stereo-seq whole section; b,c MERSCOPE domains and Allen overlay."""
    name = "ed_scale"
    pa, ha = place(name, "ed_stereoseq.png", 108.0)
    pb, hb = place(name, "brain_domains.png", 86.0)
    pc, hc = place(name, "brain_allen.png", 86.0)
    xs = row([86.0, 86.0], W, margin=2.0)
    y2 = TOP + ha + GAP + 5.0
    total = y2 + max(hb, hc) + 4.0
    return Layout(
        name=name, width_mm=W, height_mm=total,
        panels=[
            Panel("a", pa, (W - 108.0) / 2, TOP, label_dx=-4, label_dy=-5),
            Panel("b", pb, xs[0], y2, label_dx=-4, label_dy=-5),
            Panel("c", pc, xs[1], y2, label_dx=-4, label_dy=-5),
        ],
    )


def ed_merfishslices() -> Layout:
    """Five per-slice domain maps, three above two."""
    name = "ed_merfishslices"
    labels = ["0.04", "0.09", "0.14", "0.19", "0.24"]
    files = [f"ed_merfish_s{l.replace('.', '')}.png" for l in labels]
    wmm = 56.0
    paths, hs = zip(*(place(name, f, wmm) for f in files))
    xs3 = row([wmm] * 3, W, margin=3.0)
    gap = xs3[1] - xs3[0] - wmm            # same gap as the row of three
    span = 2 * wmm + gap
    xs2 = [(W - span) / 2, (W - span) / 2 + wmm + gap]   # centred under it
    y1, y2 = TOP, TOP + max(hs[:3]) + GAP
    total = y2 + max(hs[3:]) + 4.0
    panels = [Panel(labels[i], paths[i], xs3[i], y1, label_dx=-2, label_dy=-5)
              for i in range(3)]
    panels += [Panel(labels[3 + i], paths[3 + i], xs2[i], y2, label_dx=-2, label_dy=-5)
               for i in range(2)]
    return Layout(name=name, width_mm=W, height_mm=total, panels=panels)


def ed_methodval() -> Layout:
    """a controlled construction; b component ablation; c scaling."""
    name = "ed_methodval"
    pa, ha = place(name, "method_sheaf_vs_kernel.png", 176.0)
    pb, hb = place(name, "method_ablation.png", 92.0)
    pc, hc = place(name, "method_scalability.png", 82.0)
    xs = row([92.0, 82.0], W, margin=2.0)
    y2 = TOP + ha + GAP + 5.0
    total = y2 + max(hb, hc) + 4.0
    return Layout(
        name=name, width_mm=W, height_mm=total,
        panels=[
            Panel("a", pa, 2.0, TOP, label_dx=-4, label_dy=-5),
            Panel("b", pb, xs[0], y2, label_dx=-4, label_dy=-5),
            Panel("c", pc, xs[1], y2, label_dx=-4, label_dy=-5),
        ],
    )


def ed_dnaterr() -> Layout:
    """a territories; b chromosome-by-zone heatmap; c cross-nucleus NMI;
    d one nucleus as optical sections and orthogonal slabs, drawn from its
    zone table at the type specification."""
    from ed_nucleus_panels import nucleus_row
    from spsheaf import paths
    name = "ed_dnaterr"
    pa, ha = place(name, "dna_territories.png", 66.0)
    pb, hb = place(name, "dna_zone_chromosome.png", 98.0)
    pc, hc = place(name, "dna_zone_nmi.png", 52.0)
    pd_ = PANELS / "ed_dnaterr_nucleus.pdf"
    hd = nucleus_row(paths.results("takei") / "output_takei" / "nucleus_0_13_zones.csv",
                     pd_, width_mm=W - 12.0)
    xa, xb = 6.0, W - 6.0 - 98.0
    yc = TOP + hb + GAP
    row1 = max(ha, hb + GAP + hc)
    y3 = TOP + row1 + GAP + 2.0
    total = y3 + hd + 3.0
    return Layout(
        name=name, width_mm=W, height_mm=total,
        panels=[
            Panel("a", pa, xa, TOP, label_dx=-4, label_dy=-5),
            Panel("b", pb, xb, TOP, label_dx=-4, label_dy=-5),
            Panel("c", pc, xb, yc, label_dx=-4, label_dy=-5),
            Panel("d", pd_, 6.0, y3, label_dx=-4, label_dy=-3),
        ],
    )


def ed_ifheat() -> Layout:
    """a radial certification; b archetype x mark heatmap, drawn from the
    archetype tables; c one nucleus as one section and one orthogonal slab."""
    from ed_nucleus_panels import ZONE_COLORS, nucleus_row
    from ed_ifheat_panels import radial_certification, zone_mark_heatmap
    from spsheaf import paths
    import pandas as pd
    name = "ed_ifheat"
    ifdir = paths.results("takei") / "output_takei_if"
    pa = PANELS / "ed_ifheat_radial.pdf"
    ha = radial_certification(ifdir / "certification_zone_table.csv",
                              ifdir / "certification_summary.csv",
                              pa, width_mm=120.0)
    pb = PANELS / "ed_ifheat_archetypes.pdf"
    hb = zone_mark_heatmap(ifdir / "zone_mark_heatmap.csv",
                           ifdir / "zone_archetype_sizes.csv", pb, width_mm=150.0)
    # c keeps this nucleus's own zones -- three of its five fall in one pooled
    # archetype, so recolouring by archetype would merge them -- and names each
    # zone's archetype in the legend, which is what links c to b
    SHORT = {"constitutive heterochromatin": "constitutive het.",
             "SINE-rich repressive": "SINE repressive",
             "lamina / nucleolus-associated": "lamina/nucleolar",
             "active / nuclear speckle": "active/speckle"}
    asg = pd.read_csv(ifdir / "zone_archetype_assignment.csv")
    asg = asg[asg.nucleus == 13]
    zone_labels = {int(r.zone): f"zone {int(r.zone)} \u00b7 {SHORT.get(r.archetype, r.archetype)}"
                   for _, r in asg.iterrows()}
    pc = PANELS / "ed_ifheat_nucleus.pdf"
    hc = nucleus_row(ifdir / "nucleus_13_zones.csv", pc, width_mm=W - 12.0,
                     n_sections=1, colors=ZONE_COLORS, scale_um=2,
                     names=zone_labels, slabs=("xz",), legend_w=0.26)
    ya = TOP
    yb = ya + ha + GAP
    yc = yb + hb + GAP
    total = yc + hc + 2.0
    return Layout(
        name=name, width_mm=W, height_mm=total,
        panels=[
            Panel("a", pa, (W - 120.0) / 2, ya, label_dx=-4, label_dy=-5),
            Panel("b", pb, (W - 150.0) / 2, yb, label_dx=-4, label_dy=-5),
            Panel("c", pc, 6.0, yc, label_dx=-4, label_dy=-3),
        ],
    )


#: ed_scale is now drawn from data by ed_scale.py (its Stereo-seq panel is
#: still a placed raster); the builder above is kept for reference.
#: ed_merfishslices moved to its own script (figures/ed_merfishslices.py), which
#: draws from the cache instead of placing August rasters; ed_methodval was
#: dropped from the paper (its claim is now measured on real data, see
#: experiments/merfish/method_validation.py).
BUILDERS = [ed_dnaterr, ed_ifheat]


def main():
    ss.apply()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for builder in BUILDERS:
        lay = builder()
        out = OUTDIR / f"{lay.name}.pdf"
        lay.compose(out)
        print(f"{lay.name:20s} {lay.width_mm:.0f} x {lay.height_mm:.0f} mm  "
              f"{len(lay.panels)} panels -> {out.name}")

    print("\nresolution of each placed raster, at its width in the composed figure:")
    floor = 300.0
    for fig, image, dpi in _dpi_report:
        flag = "" if dpi >= floor else f"  below {floor:.0f}"
        print(f"  {fig:20s} {image:28s} {dpi:6.0f} dpi{flag}")
    under = [r for r in _dpi_report if r[2] < floor]
    print(f"\n{len(_dpi_report) - len(under)} of {len(_dpi_report)} placed rasters "
          f"at or above {floor:.0f} dpi; {len(under)} below.")


if __name__ == "__main__":
    main()
