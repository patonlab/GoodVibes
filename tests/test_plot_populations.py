"""plot_boltzmann_histogram and plot_temperature_scan (5.1)."""
import math
import warnings
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

yaml = pytest.importorskip("yaml")

from conftest import g16path  # noqa: E402
from goodvibes import ConformerSet, compute_thermo  # noqa: E402
from goodvibes.constants import KCAL_TO_AU  # noqa: E402
from goodvibes.plot import plot_boltzmann_histogram, plot_temperature_scan  # noqa: E402
from goodvibes.profile import Profile  # noqa: E402

FILES = [g16path(f) for f in ("01a_water_hf_freq.log", "01c_water_hf_freq_isotopes.log",
                              "01b_water_hf_freq_scaled.log")]
KIT = Path(__file__).resolve().parent / "profile_conformance"


@pytest.fixture(scope="module")
def results():
    return [compute_thermo(f) for f in FILES]


def _bars(ax):
    return [(p.get_gid(), p.get_height()) for p in ax.patches]


def test_bar_heights_are_the_populations_most_populated_first(results):
    cset = ConformerSet.from_results("water", results)
    ax = plot_boltzmann_histogram(cset, temperature=350.0)
    heights = [h for _g, h in _bars(ax)]
    assert heights == sorted(heights, reverse=True)
    assert sorted(heights) == pytest.approx(sorted(p * 100 for p in cset.populations(350.0)))
    assert sum(heights) == pytest.approx(100.0)
    assert "T = 350 K" in ax.get_title()
    plt.close(ax.figure)


def test_groups_share_one_distribution_and_the_legend_gives_their_totals(results):
    groups = {"R": results[:2], "S": results[2:]}
    ax = plot_boltzmann_histogram(groups, sort=False)
    gids = [g for g, _h in _bars(ax)]
    assert gids == ["pop-R-01a_water_hf_freq", "pop-R-01c_water_hf_freq_isotopes", "pop-S-01b_water_hf_freq_scaled"]
    assert sum(h for _g, h in _bars(ax)) == pytest.approx(100.0)
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend[0].startswith("R (") and legend[1].startswith("S (")
    plt.close(ax.figure)


def test_top_keeps_the_rest_in_one_bar(results):
    ax = plot_boltzmann_histogram(results, top=1)
    (first, other) = _bars(ax)
    assert other[0] == "pop-other" and first[1] + other[1] == pytest.approx(100.0)
    assert ax.get_xticklabels()[-1].get_text() == "2 other"
    plt.close(ax.figure)


def test_an_energy_only_ensemble_is_weighted_by_the_electronic_energy(tmp_path):
    from goodvibes import compute_batch, read_xyz_frames
    water = "O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n"
    (tmp_path / "e.xyz").write_text(f"3\n-5.0700\n{water}3\n-5.0690\n{water}", encoding="utf-8")
    res = compute_batch(read_xyz_frames(str(tmp_path / "e.xyz")))
    with pytest.raises(ValueError, match="electronic"):
        plot_boltzmann_histogram(res)
    ax = plot_boltzmann_histogram(res, quantity="electronic")
    gap = 0.001 * KCAL_TO_AU
    rt = 0.0019872 * 298.15
    assert _bars(ax)[0][1] == pytest.approx(100 / (1 + math.exp(-gap / rt)), rel=1e-3)
    plt.close(ax.figure)


def test_a_conformer_scan_starts_at_zero_and_follows_the_rollup(results):
    cset = ConformerSet.from_results("water", results[:2])
    temps = [250.0, 300.0, 350.0]
    ax = plot_temperature_scan(cset, temps)
    lines = {ln.get_gid(): ln for ln in ax.get_lines()}
    assert set(lines) == {"scan-qh_gibbs", "scan-qh_enthalpy", "scan-qh_entropy"}
    g = lines["scan-qh_gibbs"].get_ydata()
    assert g[0] == 0.0
    expected = (cset.gconf_corrected(350.0).qh_gibbs - cset.gconf_corrected(250.0).qh_gibbs) * KCAL_TO_AU
    assert g[-1] == pytest.approx(expected)
    assert list(lines["scan-qh_gibbs"].get_xdata()) == temps
    plt.close(ax.figure)
    with pytest.raises(ValueError, match="needs temperatures"):
        plot_temperature_scan(cset)


def test_a_profile_scan_plots_point_levels_against_temperature(results):
    doc = {
        "schema": "reaction-profile/1.0",
        "species": {"A": {}, "B": {}},
        "points": {"R": {"species": "A"}, "TS": {"species": "B", "role": "ts", "display": "TS‡"}},
        "pathways": {"p": ["R", "TS"]},
        "series": [{"id": "G", "quantity": "qh_gibbs"}],
        "goodvibes": {"sources": {"default": {"A": {"files": ["01a_water*", "01c_water*"]},
                                              "B": {"files": "01b_water*"}}}},
    }
    ev = Profile.from_dict(doc).evaluate(results, with_conformers=True)
    ax = plot_temperature_scan(ev, [250.0, 300.0, 350.0])
    (line,) = ax.get_lines()
    assert line.get_gid() == "scan-TS" and line.get_label() == "TS‡"
    hot = ev.evaluate(temperatures=[350.0])
    assert line.get_ydata()[-1] == pytest.approx(hot.series[0].levels["p"]["TS"])
    plt.close(ax.figure)
    with pytest.raises(ValueError, match="no evaluated series with a temperature"):
        plot_temperature_scan(Profile.from_dict(doc))


def test_a_declared_profile_scan_uses_its_series():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prof = Profile.from_dict(yaml.safe_load((KIT / "valid" / "05_selectivity.yaml").read_text(encoding="utf-8")))
    hot = prof.series[0].__class__(**{**prof.series[0].__dict__, "id": "G350", "temperature": 350.0})
    prof.series.append(hot)
    ax = plot_temperature_scan(prof, points=["TS_R", "TS_S"], pathway="R", units="kJ/mol")
    assert [ln.get_gid() for ln in ax.get_lines()] == ["scan-TS_R", "scan-TS_S"]
    assert ax.get_lines()[0].get_ydata()[0] == pytest.approx(15.0 * 4.184)
    assert "kJ/mol" in ax.get_ylabel()
    plt.close(ax.figure)
