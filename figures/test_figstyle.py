"""Tests for the figure pipeline: the guarantees must actually fail loudly."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import sheafstyle as ss
from layout import Layout, Panel
from verify_figure import verify


@pytest.fixture(autouse=True)
def _style():
    ss.apply()


# --- sheafstyle -------------------------------------------------------------

def test_panel_has_exact_physical_size(tmp_path):
    fig, ax = ss.panel(88, 60)
    assert fig.get_size_inches() == pytest.approx((88 / 25.4, 60 / 25.4))
    out = tmp_path / "p.pdf"
    ss.save_panel(fig, out)
    from pypdf import PdfReader
    box = PdfReader(str(out)).pages[0].mediabox
    assert float(box.width) / 72 * 25.4 == pytest.approx(88, abs=0.05)
    assert float(box.height) / 72 * 25.4 == pytest.approx(60, abs=0.05)


def test_save_panel_rejects_tight_bbox(tmp_path):
    fig, ax = ss.panel(40, 30)
    with pytest.raises(ValueError, match="bbox_inches"):
        ss.save_panel(fig, tmp_path / "p.pdf", bbox_inches="tight")


def test_spatial_removes_all_spines_and_ticks():
    fig, ax = ss.panel(40, 40)
    ss.spatial(ax)
    assert not any(sp.get_visible() for sp in ax.spines.values())
    assert list(ax.get_xticks()) == [] and list(ax.get_yticks()) == []
    assert ss.check_axes(fig) == []


def test_check_axes_catches_spine_on_spatial_panel():
    fig, ax = ss.panel(40, 40)
    ss.spatial(ax)
    ax.spines["left"].set_visible(True)   # the mistake we want caught
    problems = ss.check_axes(fig)
    assert problems and "spatial" in problems[0]


def test_save_panel_strict_blocks_a_bad_panel(tmp_path):
    fig, ax = ss.panel(40, 40)
    ss.spatial(ax)
    ax.spines["bottom"].set_visible(True)
    with pytest.raises(ValueError, match="style violations"):
        ss.save_panel(fig, tmp_path / "p.pdf")


def test_default_style_hides_top_and_right_spines():
    fig, ax = ss.panel(40, 30)
    ax.plot([0, 1], [0, 1])
    assert ss.check_axes(fig) == []


def test_type_registers_differ():
    assert ss.TYPE_PT["normal"] == {"tick": 6.0, "text": 8.0, "label": 10.0}
    assert ss.TYPE_PT["small"] == {"tick": 4.0, "text": 6.0, "label": 8.0}


def test_method_colors_are_stable():
    # A colour drifting between figures is the failure this map prevents.
    assert ss.METHOD_COLORS["spSheaf"] == "#0072B2"
    assert len({ss.METHOD_COLORS[k] for k in
                ("spSheaf", "SSAM", "Points2Regions", "FICTURE")}) == 4


def test_controls_are_hatched():
    assert "hatch" in ss.arm_style("shuffled")
    assert "hatch" not in ss.arm_style("sheaf")


# --- layout -----------------------------------------------------------------

def _stub(tmp_path, name, w, h):
    fig, ax = ss.panel(w, h)
    ax.plot([0, 1], [0, 1])
    p = tmp_path / name
    ss.save_panel(fig, p)
    return p


def test_layout_composes_to_declared_canvas(tmp_path):
    a = _stub(tmp_path, "a.pdf", 80, 50)
    b = _stub(tmp_path, "b.pdf", 80, 50)
    lay = Layout("t", 180, 60, [Panel("a", a, 2, 5), Panel("b", b, 96, 5)])
    out = lay.compose(tmp_path / "fig.pdf")
    failures, report = verify(out, width_mm=180, height_mm=60)
    assert failures == [], failures
    assert report["canvas_mm"] == (180.0, 60.0)


def test_layout_detects_overrun(tmp_path):
    a = _stub(tmp_path, "a.pdf", 100, 50)
    lay = Layout("t", 180, 60, [Panel("a", a, 120, 5)])
    problems = lay.check()
    assert any("overruns right edge" in p for p in problems)
    with pytest.raises(ValueError, match="overruns"):
        lay.compose(tmp_path / "fig.pdf")


def test_layout_detects_overlap(tmp_path):
    a = _stub(tmp_path, "a.pdf", 80, 50)
    b = _stub(tmp_path, "b.pdf", 80, 50)
    lay = Layout("t", 180, 60, [Panel("a", a, 2, 5), Panel("b", b, 40, 5)])
    assert any("overlap" in p for p in lay.check())


def test_panel_labels_are_at_the_label_size(tmp_path):
    a = _stub(tmp_path, "a.pdf", 80, 50)
    lay = Layout("t", 180, 60, [Panel("a", a, 2, 5)])
    out = lay.compose(tmp_path / "fig.pdf")
    _, report = verify(out, width_mm=180)
    assert "10" in report["type_sizes_pt"]


# --- verifier ---------------------------------------------------------------

def test_verifier_catches_wrong_canvas_width(tmp_path):
    a = _stub(tmp_path, "a.pdf", 80, 50)
    lay = Layout("t", 120, 60, [Panel("a", a, 2, 5)])
    out = lay.compose(tmp_path / "fig.pdf")
    failures, _ = verify(out, width_mm=180)
    assert any("canvas width" in f for f in failures)


def test_verifier_catches_off_spec_type(tmp_path):
    fig, ax = ss.panel(80, 50)
    # 7 pt is not in the spec. It is also exactly 10 x 0.7, so this test
    # pins that the mathtext exemption does not extend to the label size.
    ax.set_xlabel("off spec", fontsize=7)
    p = tmp_path / "a.pdf"
    ss.save_panel(fig, p)
    lay = Layout("t", 180, 60, [Panel(None, p, 2, 5)])
    out = lay.compose(tmp_path / "fig.pdf")
    failures, _ = verify(out, width_mm=180)
    assert any("7 pt" in f for f in failures)


def test_verifier_measures_rotated_labels_correctly(tmp_path):
    # Rotated ticks are where a bounding-box measurement gives nonsense.
    fig, ax = ss.panel(80, 50)
    ax.bar(["alpha", "beta", "gamma"], [1, 2, 3])
    ax.tick_params(axis="x", rotation=40)
    p = tmp_path / "a.pdf"
    ss.save_panel(fig, p)
    lay = Layout("t", 180, 60, [Panel(None, p, 2, 5)])
    out = lay.compose(tmp_path / "fig.pdf")
    failures, report = verify(out, width_mm=180)
    assert failures == [], failures
    assert set(report["type_sizes_pt"]) <= {"4", "6", "8", "10"}


def test_verifier_accepts_mathtext_subscripts(tmp_path):
    # `$\\alpha_s$` in an 8 pt label emits a 5.6 pt glyph. That is typesetting,
    # not a violation.
    fig, ax = ss.panel(80, 50)
    ax.set_xlabel(r"spatial weight $\alpha_s$")
    p = tmp_path / "a.pdf"
    ss.save_panel(fig, p)
    lay = Layout("t", 180, 60, [Panel(None, p, 2, 5)])
    out = lay.compose(tmp_path / "fig.pdf")
    failures, report = verify(out, width_mm=180)
    assert failures == [], failures
    assert "5.6" in report["type_sizes_pt"]


def test_verifier_still_catches_type_below_the_floor(tmp_path):
    fig, ax = ss.panel(80, 50)
    ax.set_xlabel("tiny", fontsize=3)
    p = tmp_path / "a.pdf"
    ss.save_panel(fig, p)
    lay = Layout("t", 180, 60, [Panel(None, p, 2, 5)])
    out = lay.compose(tmp_path / "fig.pdf")
    failures, _ = verify(out, width_mm=180)
    assert any("floor" in f for f in failures)


def test_verifier_catches_low_dpi_raster(tmp_path):
    import numpy as np
    fig, ax = ss.panel(80, 50)
    ss.spatial(ax)
    rng = np.random.default_rng(0)
    sc = ax.scatter(rng.normal(size=500), rng.normal(size=500), s=1)
    sc.set_rasterized(True)
    p = tmp_path / "a.pdf"
    fig.savefig(p, dpi=100, pad_inches=0.0)  # deliberately below spec
    lay = Layout("t", 180, 60, [Panel(None, p, 2, 5)])
    out = lay.compose(tmp_path / "fig.pdf")
    failures, _ = verify(out, width_mm=180, min_dpi=300)
    assert any("dpi" in f for f in failures)


def test_rasterized_scatter_passes_at_600(tmp_path):
    import numpy as np
    fig, ax = ss.panel(80, 50)
    ss.spatial(ax)
    rng = np.random.default_rng(0)
    sc = ax.scatter(rng.normal(size=5000), rng.normal(size=5000), s=1)
    ss.rasterize(sc)
    p = tmp_path / "a.pdf"
    ss.save_panel(fig, p)
    lay = Layout("t", 180, 60, [Panel(None, p, 2, 5)])
    out = lay.compose(tmp_path / "fig.pdf")
    failures, report = verify(out, width_mm=180, min_dpi=300)
    assert failures == [], failures
    assert report["images"], "scatter should have been rasterized"
