"""The quantity registry (goodvibes.quantities) is the single source for what
can be tabulated or plotted, and plot_pes / --pes-plot-quantity draw any of
them. This restores the --gtype E/H/ZPE plotting shipped for issue #57 in
2022 and lost in the 4.x PES rewrite.
"""
import sys

import pytest

from goodvibes.constants import KCAL_TO_AU
from goodvibes.pes_model import ThermoVector
from goodvibes.quantities import QUANTITIES, quantity_ids, resolve_quantity

from test_cli_errors import build_options, gv_logger_cleanup, run_main  # noqa: F401  (fixture re-export)
from test_plot import _two_point_pes_result, _two_pathway_pes_result

TV = ThermoVector(scf_energy=-100.0, zpe=0.05, enthalpy=-99.9, qh_enthalpy=-99.91,
                  entropy=1.0e-4, qh_entropy=0.9e-4, gibbs=-99.93, qh_gibbs=-99.94, sp_energy=-100.5)
TV_NO_SPC = ThermoVector(scf_energy=-100.0, zpe=0.05, enthalpy=-99.9, qh_enthalpy=-99.91,
                         entropy=1.0e-4, qh_entropy=0.9e-4, gibbs=-99.93, qh_gibbs=-99.94)


def test_registry_order_and_labels():
    assert quantity_ids() == ["spc", "electronic", "zpe", "e_zpe", "enthalpy", "qh_enthalpy",
                              "entropy", "qh_entropy", "gibbs", "qh_gibbs"]
    assert QUANTITIES["qh_gibbs"].label == "Δqh-G(T)"
    assert QUANTITIES["entropy"].scale_by_T and QUANTITIES["qh_entropy"].scale_by_T
    assert QUANTITIES["gibbs"].spc_label(True) == "ΔG(T)_SPC"
    assert QUANTITIES["electronic"].spc_label(True) == "ΔE"


@pytest.mark.parametrize("alias, qid", [
    ("scf", "electronic"), ("E", "electronic"), ("energy", "electronic"),
    ("H", "enthalpy"), ("qh-H", "qh_enthalpy"), ("TS", "entropy"), ("T.qh-S", "qh_entropy"),
    ("G", "gibbs"), ("qh_g", "qh_gibbs"), ("QHG", "qh_gibbs"), ("qh_gibbs_free_energy", "qh_gibbs"),
    ("E+ZPE", "e_zpe"), ("e0", "e_zpe"), ("sp", "spc"),
])
def test_aliases(alias, qid):
    assert resolve_quantity(alias).id == qid


def test_unknown_quantity_lists_the_ids():
    with pytest.raises(ValueError, match="qh_gibbs"):
        resolve_quantity("enthalpyy")


def test_thermovector_get_every_quantity():
    T = 300.0
    assert TV.get("electronic") == -100.0
    assert TV.get("zpe") == 0.05
    assert TV.get("e_zpe") == pytest.approx(-100.5 + 0.05)          # SPC-substituted
    assert TV_NO_SPC.get("e_zpe") == pytest.approx(-100.0 + 0.05)   # falls back to scf
    assert TV.get("enthalpy") == -99.9 and TV.get("qh_enthalpy") == -99.91
    assert TV.get("entropy", T) == pytest.approx(T * 1.0e-4)
    assert TV.get("qh_entropy", T) == pytest.approx(T * 0.9e-4)
    assert TV.get("gibbs") == -99.93 and TV.get("qh_gibbs") == -99.94
    assert TV.get("spc") == -100.5 and TV_NO_SPC.get("spc") is None
    with pytest.raises(ValueError, match="temperature"):
        TV.get("entropy")


@pytest.mark.parametrize("quantity, field", [("electronic", "scf_energy"), ("enthalpy", "enthalpy"),
                                              ("gibbs", "gibbs"), ("qh_gibbs", "qh_gibbs")])
def test_plot_pes_quantity_selects_the_values_and_label(quantity, field):
    res = _two_point_pes_result()
    T = res.temperatures[0]
    ax = __import__("goodvibes.plot", fromlist=["plot_pes"]).plot_pes(res, quantity=quantity)
    rels = res.pathways[0].relative(T, gconf=res.options.gconf, QH=res.options.QH,
                                    lowest_only=res.options.lowest_only)
    expected = [getattr(r, field) * KCAL_TO_AU for r in rels]
    drawn = sorted(seg[0][1] for coll in ax.collections for seg in coll.get_segments())
    assert drawn == pytest.approx(sorted(expected))
    assert QUANTITIES[quantity].label in ax.get_ylabel() and "kcal/mol" in ax.get_ylabel()


def test_plot_pes_default_is_qh_gibbs_with_registry_label():
    from goodvibes.plot import plot_pes
    ax = plot_pes(_two_pathway_pes_result())
    assert ax.get_ylabel().startswith("Δqh-G(T)")


def test_plot_pes_spc_quantity_without_spc_raises():
    from goodvibes.plot import plot_pes
    with pytest.raises(ValueError, match="not available"):
        plot_pes(_two_point_pes_result(), quantity="spc")


def test_cli_flag_and_gtype_alias(monkeypatch):
    opts, _ = build_options(monkeypatch, extra=["--pes-plot-quantity", "E"])
    assert opts.pes_plot_quantity == "electronic"
    opts, _ = build_options(monkeypatch, extra=["--gtype", "H"])
    assert opts.pes_plot_quantity == "enthalpy"
    opts, _ = build_options(monkeypatch)
    assert opts.pes_plot_quantity == "qh_gibbs"


def test_cli_unknown_quantity_is_a_usage_error(monkeypatch, capsys):
    from goodvibes import GoodVibes as GV
    monkeypatch.setattr(sys, "argv", ["goodvibes", "tests/g16/01a_water_hf_freq.log", "--gtype", "bogus"])
    with pytest.raises(SystemExit) as exc:
        GV.parse_arguments()
    assert exc.value.code == 2 and "--pes-plot-quantity" in capsys.readouterr().err


def test_cli_pes_plot_quantity_writes_image(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    pytest.importorskip("matplotlib")
    import os
    from conftest import datapath
    ex = datapath("gconf_ee_boltz")
    files = [os.path.join(ex, f) for f in ("aminox_cat_conf65_S.log", "aminox_subs_conf713.log",
                                           "Aminoxylation_TS1_R.log", "Aminoxylation_TS2_S.log")]
    out = tmp_path / "profile_E.png"
    run_main(monkeypatch, tmp_path, files + ["--pes", os.path.join(ex, "gconf_TS.yaml"),
                                             "--pes-plot", str(out), "--gtype", "E"])
    assert out.exists() and out.stat().st_size > 0
