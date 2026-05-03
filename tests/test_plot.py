"""Tests for goodvibes.plot — selectivity strip plots, PES profiles.

Headless: forces the matplotlib 'Agg' backend before importing pyplot
so tests don't need a display. matplotlib is in the test extras and
the [plot] / [full] runtime extras, so it should be available; if not,
the whole module skips via importorskip at collection time.
"""
import os

import pytest

# Skip the whole module cleanly if matplotlib isn't installed (CI runners
# without [plot] extras, etc.). Setting the backend before importing
# pyplot keeps tests working in headless environments.
import matplotlib
matplotlib.use("Agg", force=True)
plt = pytest.importorskip("matplotlib.pyplot")

from goodvibes import plot as gv_plot
from goodvibes.selectivity import SelectivityResult


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _stub_selectivity(labels=("R", "S")):
    """Build a SelectivityResult by hand for the strip-plot tests."""
    files_per_label = {
        "R": ["a/R_1.log", "a/R_2.log", "a/R_3.log"],
        "S": ["a/S_1.log", "a/S_2.log"],
    }
    populations = {"R": 0.7, "S": 0.3}
    raw_boltzmann = {"R": 0.7e-3, "S": 0.3e-3}
    return SelectivityResult(
        temperature=298.15, key="gibbs", labels=list(labels),
        files_per_label=files_per_label,
        populations=populations,
        raw_boltzmann=raw_boltzmann,
        preferred="R", ee=40.0, ddG=0.001,
    )


def _thermo_dict():
    """Five conformer qh-G values in Hartree, mimicking what calc_bbe writes."""
    return {
        "a/R_1.log": -100.0050,
        "a/R_2.log": -100.0040,
        "a/R_3.log": -100.0030,
        "a/S_1.log": -100.0010,
        "a/S_2.log": -100.0005,
    }


# ---------------------------------------------------------------------------
# plot_selectivity_strip
# ---------------------------------------------------------------------------

def test_strip_plot_creates_axes_when_none():
    sel = _stub_selectivity()
    ax = gv_plot.plot_selectivity_strip(sel, _thermo_dict())
    assert ax is not None
    plt.close(ax.figure)


def test_strip_plot_uses_existing_axes():
    sel = _stub_selectivity()
    fig, ax = plt.subplots()
    returned = gv_plot.plot_selectivity_strip(sel, _thermo_dict(), ax=ax)
    assert returned is ax
    plt.close(fig)


def test_strip_plot_xticks_match_label_order():
    """The categorical x-axis must show one tick per species, in the
    order the SelectivityResult declares."""
    sel = _stub_selectivity(labels=("R", "S"))
    ax = gv_plot.plot_selectivity_strip(sel, _thermo_dict())
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["R", "S"]
    plt.close(ax.figure)


def test_strip_plot_scatter_count_matches_files():
    """Total scatter points across all species must equal the total
    file count. ax.collections also includes the LineCollections from
    the lowest/Boltzmann horizontal bars; filter to PathCollections."""
    from matplotlib.collections import PathCollection
    sel = _stub_selectivity()
    ax = gv_plot.plot_selectivity_strip(sel, _thermo_dict())
    n_points = sum(coll.get_offsets().shape[0]
                   for coll in ax.collections
                   if isinstance(coll, PathCollection))
    expected = sum(len(v) for v in sel.files_per_label.values())
    assert n_points == expected
    plt.close(ax.figure)


def test_strip_plot_overlays_optional():
    """show_boltz_mean and show_lowest gate the horizontal-bar overlays
    (per-species LineCollection on the axes)."""
    sel = _stub_selectivity()
    ax = gv_plot.plot_selectivity_strip(
        sel, _thermo_dict(),
        show_boltz_mean=False, show_lowest=False,
    )
    # When both overlays are off, no LineCollection should be present.
    from matplotlib.collections import LineCollection
    line_colls = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert line_colls == []
    plt.close(ax.figure)


def test_strip_plot_kj_mol_units_scales_y_label():
    sel = _stub_selectivity()
    ax = gv_plot.plot_selectivity_strip(sel, _thermo_dict(), units="kJ/mol")
    assert "kJ/mol" in ax.get_ylabel()
    plt.close(ax.figure)


def test_strip_plot_invalid_units_raises():
    sel = _stub_selectivity()
    with pytest.raises(ValueError, match="kcal/mol"):
        gv_plot.plot_selectivity_strip(sel, _thermo_dict(), units="hartree")


def test_strip_plot_callable_lookup():
    """thermo_lookup may be a callable instead of a dict — useful when
    callers want to pull from a list of ThermoResult on-demand."""
    from matplotlib.collections import PathCollection
    sel = _stub_selectivity()
    d = _thermo_dict()
    ax = gv_plot.plot_selectivity_strip(sel, d.__getitem__)
    n_points = sum(c.get_offsets().shape[0] for c in ax.collections
                   if isinstance(c, PathCollection))
    assert n_points == 5
    plt.close(ax.figure)


def test_strip_plot_jitter_deterministic():
    """Same seed → same x-offsets across two runs."""
    from matplotlib.collections import PathCollection
    sel = _stub_selectivity()
    ax1 = gv_plot.plot_selectivity_strip(sel, _thermo_dict(), seed=42)
    ax2 = gv_plot.plot_selectivity_strip(sel, _thermo_dict(), seed=42)
    xs1 = sorted(p[0] for c in ax1.collections
                 if isinstance(c, PathCollection)
                 for p in c.get_offsets())
    xs2 = sorted(p[0] for c in ax2.collections
                 if isinstance(c, PathCollection)
                 for p in c.get_offsets())
    assert xs1 == xs2
    plt.close(ax1.figure)
    plt.close(ax2.figure)


def test_strip_plot_empty_selectivity_raises():
    """A SelectivityResult with no conformers in any species can't be
    plotted — clear error rather than empty plot."""
    sel = SelectivityResult(
        temperature=298.15, key="gibbs", labels=["A", "B"],
        files_per_label={"A": [], "B": []},
        populations={"A": 0.5, "B": 0.5},
        raw_boltzmann={"A": 0.0, "B": 0.0},
        preferred="A", ee=0.0, ddG=0.0,
    )
    with pytest.raises(ValueError, match="no conformers"):
        gv_plot.plot_selectivity_strip(sel, {})


# ---------------------------------------------------------------------------
# plot_pes
# ---------------------------------------------------------------------------

def _two_point_pes_result():
    """Build a 2-point PESResult from synthetic stub bbes."""
    from types import SimpleNamespace
    from goodvibes.pes_loader import PESSpec, build_pes_result
    from goodvibes.pes_model import PESOptions

    def stub(g):
        return SimpleNamespace(
            scf_energy=g - 0.001, zpe=0.005,
            enthalpy=g + 0.005, qh_enthalpy=g + 0.005,
            entropy=1.6e-5, qh_entropy=1.6e-5,
            gibbs_free_energy=g, qh_gibbs_free_energy=g,
            sp_energy=None,
        )

    td = {"a.log": stub(-100.0), "b.log": stub(-50.0)}
    spec = PESSpec(
        pathways={"rxn": ["A", "B"]},
        species={"A": "a", "B": "b"},
        options=PESOptions(units="kcal/mol", decimals=1, gconf=False, QH=False),
    )
    return build_pes_result(spec, td, temperatures=[298.15])


def test_pes_plot_creates_axes():
    result = _two_point_pes_result()
    ax = gv_plot.plot_pes(result)
    assert ax is not None
    plt.close(ax.figure)


def test_pes_plot_xticks_match_point_labels():
    result = _two_point_pes_result()
    ax = gv_plot.plot_pes(result)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["A", "B"]
    plt.close(ax.figure)


def test_pes_plot_show_conformers_requires_lookup():
    result = _two_point_pes_result()
    with pytest.raises(ValueError, match="thermo_lookup"):
        gv_plot.plot_pes(result, show_conformers=True)


def test_pes_plot_y_label_uses_units():
    result = _two_point_pes_result()
    ax = gv_plot.plot_pes(result)
    assert "kcal/mol" in ax.get_ylabel()
    plt.close(ax.figure)


# ---------------------------------------------------------------------------
# Stubs lock in the v5.1 API
# ---------------------------------------------------------------------------

def test_boltzmann_histogram_stub():
    with pytest.raises(NotImplementedError, match="v5.1"):
        gv_plot.plot_boltzmann_histogram([])


def test_temperature_scan_stub():
    with pytest.raises(NotImplementedError, match="v5.1"):
        gv_plot.plot_temperature_scan([])


# ---------------------------------------------------------------------------
# CLI: --strip-plot end-to-end
# ---------------------------------------------------------------------------

def test_cli_strip_plot_writes_image(tmp_path):
    """`goodvibes ... --label R=... --label S=... --strip-plot out.png`
    produces a real image file with non-trivial size."""
    import subprocess
    import sys

    sel_dir = os.path.join(REPO_ROOT, "goodvibes", "examples", "selectivity")
    fixtures = sorted(
        os.path.join(sel_dir, f)
        for f in os.listdir(sel_dir) if f.startswith("DA_") and f.endswith(".out")
    )
    if not fixtures:
        pytest.skip("DA fixtures missing")

    out = tmp_path / "strip.png"
    res = subprocess.run(
        [sys.executable, "-m", "goodvibes", *fixtures,
         "--label", "exo=*_exo_*", "--label", "endo=*_endo_*",
         "--strip-plot", str(out), "--output", "strip_smoke"],
        capture_output=True, text=True, cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": REPO_ROOT},
    )
    assert res.returncode == 0, f"stderr:\n{res.stderr}"
    assert out.exists()
    assert out.stat().st_size > 1000     # non-trivial PNG, not an empty stub


def test_cli_strip_plot_without_labels_fails(tmp_path):
    """`--strip-plot` without `--label` / `--selectivity` is a usage error."""
    import subprocess
    import sys

    sel_dir = os.path.join(REPO_ROOT, "goodvibes", "examples", "selectivity")
    fixture = os.path.join(sel_dir, "DA_exo_12_i.out")
    if not os.path.exists(fixture):
        pytest.skip("DA fixture missing")

    out = tmp_path / "strip.png"
    res = subprocess.run(
        [sys.executable, "-m", "goodvibes", fixture,
         "--strip-plot", str(out), "--output", "no_label"],
        capture_output=True, text=True, cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": REPO_ROOT},
    )
    assert res.returncode != 0
    assert "label" in (res.stdout + res.stderr).lower() or "selectivity" in (res.stdout + res.stderr).lower()


def test_cli_pes_plot_writes_image(tmp_path):
    """`goodvibes ... --pes pathway.yaml --pes-plot out.png` writes a real image."""
    import subprocess
    import sys

    pes_dir = os.path.join(REPO_ROOT, "goodvibes", "examples", "pes")
    yaml = os.path.join(pes_dir, "azabor_PES_v2.yaml")
    if not os.path.exists(yaml):
        pytest.skip("azabor PES fixture missing")

    fixtures = sorted(
        os.path.join(pes_dir, f)
        for f in os.listdir(pes_dir)
        if f.endswith(".log") and "_sp_tzpop." not in f
    )
    out = tmp_path / "pes.png"
    res = subprocess.run(
        [sys.executable, "-m", "goodvibes", *fixtures,
         "--spc", "sp_tzpop", "--pes", yaml,
         "--pes-plot", str(out), "--output", "pes_plot_smoke"],
        capture_output=True, text=True, cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": REPO_ROOT},
    )
    assert res.returncode == 0, f"stderr:\n{res.stderr}"
    assert out.exists()
    assert out.stat().st_size > 1000


def test_cli_pes_plot_without_pes_fails(tmp_path):
    """`--pes-plot` without `--pes` is a usage error."""
    import subprocess
    import sys

    g16_dir = os.path.join(REPO_ROOT, "tests", "g16")
    fixture = os.path.join(g16_dir, "01a_water_hf_freq.log")
    if not os.path.exists(fixture):
        pytest.skip("g16 fixture missing")

    out = tmp_path / "pes.png"
    res = subprocess.run(
        [sys.executable, "-m", "goodvibes", fixture,
         "--pes-plot", str(out), "--output", "no_pes"],
        capture_output=True, text=True, cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": REPO_ROOT},
    )
    assert res.returncode != 0
    assert "--pes" in (res.stdout + res.stderr) or "pathway" in (res.stdout + res.stderr).lower()
