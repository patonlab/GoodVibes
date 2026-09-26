"""Image baselines for the reaction-profile gallery: a layout or drawing
regression (a lost series, a moved bar, a wrong connector, a figure that
changes size) fails here, not in review.

Each gallery figure is rendered at a fixed size with its text hidden and
compared with a committed PNG in ``tests/baselines/gallery`` (RMS
tolerance ``TOLERANCE`` on a 0-255 scale). Text is left out because glyph
rasterisation changes between matplotlib and FreeType releases while the
geometry does not; the numbers the text shows are checked in
``test_gallery.py``. Measured when the baselines were made: matplotlib
3.9, 3.10 and 3.11 render them identically (RMS 0, at most 0.67 for the
panels figure); five missing connectors give 3.6, missing error bars
4.1, a lost series 13.

Regenerate after an intended change, and look at the images first:

    GOODVIBES_UPDATE_BASELINES=1 pytest tests/test_gallery_images.py
"""
import importlib.util
import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("yaml")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.testing.compare import compare_images  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "goodvibes" / "examples" / "gallery" / "build_gallery.py"
BASELINES = ROOT / "tests" / "baselines" / "gallery"
DPI = 50
TOLERANCE = 1.5
UPDATE = bool(os.environ.get("GOODVIBES_UPDATE_BASELINES"))


def _gallery():
    spec = importlib.util.spec_from_file_location("build_gallery", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GALLERY


GALLERY = _gallery()


def render_geometry(prof, path) -> None:
    """Save ``prof`` without text: titles, axis and tick labels, value
    labels and the legend are hidden; bars, connectors, markers, error
    bars, conformer dots, barrier arrows and the axes frame stay. No
    ``bbox_inches='tight'``, so the image size is the figure size."""
    fig = prof.figure
    for ax in prof.axes:
        ax.tick_params(which="both", labelleft=False, labelright=False, labelbottom=False)
        ax.title.set_visible(False)
        ax.xaxis.label.set_visible(False)
        ax.yaxis.label.set_visible(False)
        for t in ax.texts:
            if t.get_text():                     # value / barrier labels; keep the bare arrows
                t.set_visible(False)
        if ax.get_legend() is not None:
            ax.get_legend().set_visible(False)
    for t in fig.texts:
        t.set_visible(False)
    with plt.rc_context(prof.rc):
        fig.savefig(path, dpi=DPI)


@pytest.mark.parametrize("name, make", GALLERY, ids=[name for name, _ in GALLERY])
def test_gallery_figure_matches_its_baseline(name, make, tmp_path):
    prof = make()
    actual = tmp_path / f"{name}.png"
    render_geometry(prof, actual)
    prof.close()
    expected = BASELINES / f"{name}.png"
    if UPDATE:
        BASELINES.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(actual, expected)
        pytest.skip(f"baseline written: {expected.relative_to(ROOT)}")
    assert expected.is_file(), (f"no baseline for {name}; create it with "
                                "GOODVIBES_UPDATE_BASELINES=1 pytest tests/test_gallery_images.py")
    diff = compare_images(str(expected), str(actual), tol=TOLERANCE, in_decorator=True)
    assert diff is None, (f"{name} differs from its baseline (RMS {diff['rms']:.2f} > {TOLERANCE}); "
                          f"diff image: {diff['diff']}")


def test_every_baseline_belongs_to_a_gallery_figure():
    names = {name for name, _ in GALLERY}
    stale = sorted(p.stem for p in BASELINES.glob("*.png") if p.stem not in names)
    assert not stale, f"baselines without a gallery figure: {stale}"


def test_a_changed_figure_fails(tmp_path):
    """The comparison is sensitive enough to catch a lost series."""
    name, make = next((n, m) for n, m in GALLERY if n == "column_preset_uncertainty")
    prof = make()
    for coll in prof.ax.collections:
        if (coll.get_gid() or "").startswith("error"):
            coll.set_visible(False)              # the error bars gone
    actual = tmp_path / "changed.png"
    render_geometry(prof, actual)
    prof.close()
    expected = BASELINES / f"{name}.png"
    if not expected.is_file():
        pytest.skip("no baseline yet")
    assert compare_images(str(expected), str(actual), tol=TOLERANCE, in_decorator=True) is not None
