"""plot_profile: merged x axis, series overlays (temperatures, declared
values), panels, edge styling, roles and the ProfileAxes it returns."""
from types import SimpleNamespace

import pytest

import matplotlib
matplotlib.use("Agg", force=True)
plt = pytest.importorskip("matplotlib.pyplot")
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.patches import PathPatch

from conftest import g16path
from goodvibes import compute_thermo
from goodvibes.pes_loader import PESSpec, build_pes_result
from goodvibes.pes_model import ConformerSet, PESOptions, PESResult, Pathway, Point, Series
from goodvibes.plot import ProfileAxes, plot_pes, plot_profile


def _stub(g):
    return SimpleNamespace(
        scf_energy=g - 0.001, zpe=0.005, enthalpy=g + 0.005, qh_enthalpy=g + 0.005,
        entropy=1.6e-5, qh_entropy=1.6e-5, gibbs_free_energy=g, qh_gibbs_free_energy=g,
        sp_energy=None,
    )


def _branches():
    """R → TS_R → P_R and R → TS_S → P_S from stubs (not recomputable)."""
    td = {"r.log": _stub(-100.0), "tsr.log": _stub(-99.97), "pr.log": _stub(-100.02),
          "tss.log": _stub(-99.96), "ps.log": _stub(-100.03)}
    spec = PESSpec(pathways={"R": ["R", "TS_R", "P_R"], "S": ["R", "TS_S", "P_S"]},
                   species={"R": "r", "TS_R": "tsr", "P_R": "pr", "TS_S": "tss", "P_S": "ps"},
                   options=PESOptions(units="kcal/mol", gconf=False, QH=False))
    res = build_pes_result(spec, td)
    for path in res.pathways:
        path.points[1].role = "ts"
        path.points[1].display = path.points[1].label + "‡"
    return res


@pytest.fixture(scope="module")
def real_two_T():
    a = ConformerSet.from_results("A", [compute_thermo(g16path("01a_water_hf_freq.log")),
                                        compute_thermo(g16path("01c_water_hf_freq_isotopes.log"))])
    b = ConformerSet.from_results("B", [compute_thermo(g16path("01b_water_hf_freq_scaled.log"))])
    species = {"A": a, "B": b}
    path = Pathway("rxn", [Point.from_label("A", species), Point.from_label("B", species, role="ts")])
    return PESResult(pathways=[path], options=PESOptions(units="kcal/mol", gconf=False, QH=False),
                     temperatures=[298.15, 400.0])


def _hlines(ax):
    return [c for c in ax.collections if isinstance(c, LineCollection)]


def test_returns_profile_axes_whose_levels_match_the_model():
    res = _branches()
    prof = plot_profile(res)
    assert isinstance(prof, ProfileAxes)
    assert prof.order == ["R", "TS_R", "P_R", "TS_S", "P_S"]
    assert prof.levels == res.levels()
    assert prof.level("S", "TS_S") == pytest.approx(res.levels()["qh_gibbs@298.15K"]["S"]["TS_S"])
    assert [t.get_text() for t in prof.ax.get_xticklabels()] == ["R", "TS_R‡", "P_R", "TS_S‡", "P_S"]
    assert prof.ax.get_ylabel() == "Δqh-G(T) (kcal/mol)"
    assert [t.get_text() for t in prof.ax.get_legend().get_texts()] == ["R", "S"]
    assert len(_hlines(prof.ax)) == 6
    prof.close()


def test_temperature_overlay_draws_one_series_per_temperature(real_two_T):
    prof = plot_profile(real_two_T)
    assert [s.temperature for s in prof.series] == [298.15, 400.0]
    b298 = prof.level("rxn", "B", "qh_gibbs@298.15K")
    b400 = prof.level("rxn", "B", "qh_gibbs@400K")
    assert b298 != b400
    assert len(_hlines(prof.ax)) == 4                      # 2 points × 2 series
    legend = [t.get_text() for t in prof.ax.get_legend().get_texts()]
    assert legend == ["Δqh-G(T) 298.15 K", "Δqh-G(T) 400 K"]
    assert prof.linestyles["qh_gibbs@298.15K"] != prof.linestyles["qh_gibbs@400K"]
    assert prof.ax.get_title() == "rxn"                     # several temperatures: none in the title
    single = plot_profile(real_two_T, temperatures=[400.0])
    assert single.ax.get_title() == "rxn  (T = 400 K)"
    prof.close(); single.close()


def test_stub_result_temperature_overlay_needs_recomputable_entries():
    """Stubs cannot be re-evaluated: the second temperature silently reuses
    the base values (documented pre-4.6 behaviour), so the levels coincide."""
    res = _branches()
    res.temperatures = [298.15, 500.0]
    prof = plot_profile(res)
    assert prof.level("R", "TS_R", "qh_gibbs@298.15K") == prof.level("R", "TS_R", "qh_gibbs@500K")
    prof.close()


def test_declared_series_uses_hollow_markers_and_converts_units():
    res = _branches()
    lit = Series.declared_from("lit", "lit. B3LYP", {"R": {"R": 0.0, "TS_R": 62.76, "P_R": -8.368}},
                               units="kJ/mol", style={"linestyle": "--"})
    prof = plot_profile(res, series=[res.default_series()[0], lit], pathways=["R"])
    assert prof.level("R", "TS_R", "lit") == pytest.approx(15.0)
    assert prof.level("R", "P_R", "lit") == pytest.approx(-2.0)
    markers = [l for l in prof.ax.lines if l.get_marker() == "o"]
    assert len(markers) == 3
    assert prof.linestyles["lit"] == "--"
    legend = [t.get_text() for t in prof.ax.get_legend().get_texts()]
    assert legend == ["Δqh-G(T)", "lit. B3LYP"]
    prof.close()


def test_series_by_id_from_the_result_and_conflicting_arguments():
    res = _branches()
    res.series = [Series(id="g", label="G", quantity="gibbs"), Series(id="e", label="E", quantity="electronic")]
    prof = plot_profile(res)                                   # the result's own series
    assert [s.id for s in prof.series] == ["g", "e"]
    assert prof.ax.get_ylabel() == "relative energy (kcal/mol)"
    only_e = plot_profile(res, series="e")
    assert [s.id for s in only_e.series] == ["e"] and only_e.ax.get_ylabel() == "ΔE (kcal/mol)"
    with pytest.raises(KeyError, match="no series"):
        plot_profile(res, series=["zzz"])
    with pytest.raises(ValueError, match="not both"):
        plot_profile(res, series="e", quantity="gibbs")
    prof.close(); only_e.close()


def test_panels_layout_gives_one_axes_per_pathway():
    res = _branches()
    prof = plot_profile(res, layout="panels")
    assert len(prof.axes) == 2 and prof.layout == "panels"
    assert prof.axes_for("S") is prof.axes[1]
    assert prof.axes[0].get_title(loc="left") == "R"
    assert prof.axes[0].get_legend() is None                # one pathway per axes: no legend
    with pytest.raises(ValueError, match="ax="):
        plot_profile(res, layout="panels", ax=prof.axes[0])
    with pytest.raises(ValueError, match="layout"):
        plot_profile(res, layout="grid")
    prof.close()


def test_edge_kinds_change_the_connectors():
    res = _branches()
    plain = plot_profile(res, pathways=["R"])
    assert len([p for p in plain.ax.patches if isinstance(p, PathPatch)]) == 2
    res.pathways[0] = res.pathways[0].with_edges([("R", "TS_R", "barrierless"), ("TS_R", "P_R", "none")])
    styled = plot_profile(res, pathways=["R"])
    patches = [p for p in styled.ax.patches if isinstance(p, PathPatch)]
    assert len(patches) == 1
    assert patches[0].get_linestyle() in (":", "dotted")
    linear = plot_profile(res, pathways=["R"], style={"connector": "linear"})
    assert len([l for l in linear.ax.lines if l.get_linestyle() in (":", "dotted")]) == 1
    with pytest.raises(ValueError, match="connector"):
        plot_profile(res, style={"connector": "spline"})
    plain.close(); styled.close(); linear.close()


def test_label_points_places_ts_labels_above_and_minima_below():
    res = _branches()
    prof = plot_profile(res, pathways=["R"], label_points=True)
    offsets = {t.get_text(): t.xyann[1] for t in prof.ax.texts}
    ts_value = f"{prof.level('R', 'TS_R'):.1f}"
    assert offsets[ts_value] > 0
    assert all(v < 0 for k, v in offsets.items() if k != ts_value)
    prof.close()


def test_annotate_barrier_reports_the_difference():
    res = _branches()
    prof = plot_profile(res)
    ann = prof.annotate_barrier("S", "R", "TS_S")
    expected = prof.level("S", "TS_S") - prof.level("S", "R")
    assert ann.get_text() == f"{expected:+.1f}"
    with pytest.raises(ValueError, match="no level"):
        prof.annotate_barrier("S", "R", "P_R")
    prof.close()


def test_order_override_and_unknown_point_error():
    res = _branches()
    prof = plot_profile(res, order=["P_S", "TS_S", "R", "TS_R", "P_R"])
    assert prof.x["P_S"] == 0.0 and prof.x["P_R"] == 4.0
    with pytest.raises(ValueError, match="not in the x order"):
        plot_profile(res, order=["R", "TS_R"])
    prof.close()


def test_show_conformers_draws_dots_for_every_pathway(real_two_T):
    prof = plot_profile(real_two_T, temperatures=[298.15], show_conformers=True)
    dots = [c for c in prof.ax.collections if isinstance(c, PathCollection)]
    assert len(dots) == 2                                   # species A has two conformers
    prof.close()


def test_colors_mapping_and_sequence():
    res = _branches()
    prof = plot_profile(res, colors={"R": "red", "S": "green"})
    assert prof.colors == {"R": "red", "S": "green"}
    with pytest.raises(ValueError, match="no color"):
        plot_profile(res, colors={"R": "red"})
    with pytest.raises(ValueError, match="at least 2 colors"):
        plot_profile(res, colors=["red"])
    prof.close()


def test_save_writes_every_path(tmp_path):
    prof = plot_profile(_branches())
    prof.save(tmp_path / "a.png", tmp_path / "a.svg")
    assert (tmp_path / "a.png").stat().st_size > 0 and (tmp_path / "a.svg").stat().st_size > 0
    prof.close()


def test_plot_pes_shim_returns_axes_and_rewrites_its_errors():
    res = _branches()
    ax = plot_pes(res, pathway_index=1, label_points=True)
    assert [t.get_text() for t in ax.get_xticklabels()] == ["R", "TS_S‡", "P_S"]
    with pytest.raises(ValueError, match="plot_pes: quantity 'spc'"):
        plot_pes(res, quantity="spc")
    plt.close(ax.figure)


def test_plot_profile_uses_the_given_axes():
    fig, ax = plt.subplots()
    prof = plot_profile(_branches(), ax=ax)
    assert prof.ax is ax and prof.figure is fig
    plt.close(fig)
