"""Provenance on results: the temperature and options a result was computed
with, where its vibrational scale factor came from (no more silent 1.0),
where its symmetry number came from, and how many imaginary modes the
output reports."""
import json
from contextlib import contextmanager
import warnings
from pathlib import Path

import pytest

from conftest import g16path
from goodvibes import compute_thermo, to_dataframe
from goodvibes.pes_model import ComputedEntry
from goodvibes.thermo import SCALE_FACTOR_SOURCES, ScaleFactorWarning, ThermoOptions
from test_cli_errors import gv_logger_cleanup, run_main  # noqa: F401  (fixture re-export)

IN_DB = g16path("01a_water_hf_freq.log")                    # HF/6-31G(d): in the Truhlar table
NOT_IN_DB = g16path("02_ethane_opt_freq_T398_P2.log")       # B3LYP/6-311+G(d,p): not in it
SINGLE_POINT = g16path("20_benzene_singlepoint.log")
TS = str(Path(__file__).resolve().parents[1] / "goodvibes" / "examples" / "gconf_ee_boltz"
         / "Aminoxylation_TS1_R.log")


@contextmanager
def _no_scale_factor_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", ScaleFactorWarning)
        yield


def test_a_database_level_records_its_factors_and_source():
    r = compute_thermo(IN_DB, temperature=350.0)
    assert r.scale_factor_source == "truhlar"
    assert (r.freq_scale_factor, r.zpe_scale_factor) == pytest.approx((0.922, 0.909))
    assert r.temperature == 350.0
    assert isinstance(r.options, ThermoOptions)
    assert r.options.temperature == 350.0 and r.options.scale_factor_source == "truhlar"
    assert r.symmetry_source == "output" and r.n_imag == 0


def test_a_level_not_in_the_database_warns_instead_of_a_silent_one():
    with pytest.warns(ScaleFactorWarning, match="B3LYP/6-311\\+G\\(d,p\\)"):
        r = compute_thermo(NOT_IN_DB)
    assert r.scale_factor_source == "none-found"
    assert (r.freq_scale_factor, r.zpe_scale_factor) == (1.0, 1.0)


def test_user_factors_do_not_warn():
    with _no_scale_factor_warning():
        r = compute_thermo(NOT_IN_DB, freq_scale_factor=0.98)
        assert r.scale_factor_source == "user"
        assert (r.freq_scale_factor, r.zpe_scale_factor) == (0.98, 0.98)
        r = compute_thermo(NOT_IN_DB, freq_scale_factor=0.98, zpe_scale_factor=0.97)
        assert r.scale_factor_source == "user" and r.zpe_scale_factor == 0.97


def test_a_single_point_has_no_scale_factor_source_and_no_modes():
    with _no_scale_factor_warning():
        r = compute_thermo(SINGLE_POINT)
    assert r.scale_factor_source is None and r.n_imag is None


def test_n_imag_counts_the_modes_the_output_reports():
    r = compute_thermo(TS, invert="auto")
    assert r.n_imag == 1                      # the reaction coordinate, before any inversion
    assert compute_thermo(IN_DB).n_imag == 0


def test_re_evaluation_keeps_the_provenance():
    r = compute_thermo(IN_DB)
    hot = ComputedEntry.from_result(r).bbe(400.0)
    assert hot.scale_factor_source == "truhlar"
    assert hot.options.freq_scale_factor == r.freq_scale_factor
    with pytest.warns(ScaleFactorWarning):
        r = compute_thermo(NOT_IN_DB)
    with _no_scale_factor_warning():           # re-evaluating does not warn again
        hot = ComputedEntry.from_result(r).bbe(400.0)
    assert hot.scale_factor_source == "none-found"


def test_the_dataframe_carries_the_provenance_columns():
    pytest.importorskip("pandas")
    df = to_dataframe([compute_thermo(IN_DB)])
    for col in ("temperature", "freq_scale_factor", "zpe_scale_factor", "scale_factor_source",
                "symmetry_source", "n_imag"):
        assert col in df.columns
    assert "options" not in df.columns
    assert df.loc[0, "scale_factor_source"] in SCALE_FACTOR_SOURCES


def test_ase_input_is_unscaled_by_design():
    pytest.importorskip("ase")
    from ase.build import molecule
    from goodvibes.constants import HARTREE_TO_EV
    from goodvibes.io import QCData
    freqs = [1655.4, 3826.7, 3935.6]
    with _no_scale_factor_warning():
        mlip = QCData.from_atoms(molecule("H2O"), -76.4 * HARTREE_TO_EV, frequencies=freqs,
                                 name="water", method="MACE-MP-0", symm=None)
        r = compute_thermo(qcdata=mlip)
    assert r.scale_factor_source == "mlip-unscaled" and r.freq_scale_factor == 1.0
    assert r.symmetry_source == "assumed"
    dft = QCData.from_atoms(molecule("H2O"), -76.4 * HARTREE_TO_EV, frequencies=freqs,
                            name="water", method="HF/6-31G(d)", symm=None)
    assert compute_thermo(qcdata=dft).scale_factor_source == "truhlar"


def test_symm_records_pymsym_as_the_symmetry_source():
    pytest.importorskip("pymsym")
    assert compute_thermo(IN_DB, symm=True).symmetry_source == "pymsym"


# -- the goodvibes command ---------------------------------------------------------

def test_cli_says_when_no_scale_factor_was_found(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    run_main(monkeypatch, tmp_path, [NOT_IN_DB, "--json", "out.json"])
    text = (tmp_path / "GoodVibes_output.dat").read_text(encoding="utf-8")
    assert "No vibrational scaling factor found for B3LYP/6-311+G(d,p)" in text
    thermo = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))["results"][0]["thermo"]
    assert (thermo["scale_factor_source"], thermo["symmetry_source"], thermo["n_imag"]) == ("none-found", "output", 0)


def test_cli_with_vscal_is_user_and_silent(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    run_main(monkeypatch, tmp_path, [NOT_IN_DB, "--vscal", "0.98", "--json", "out.json"])
    text = (tmp_path / "GoodVibes_output.dat").read_text(encoding="utf-8")
    assert "No vibrational scaling factor found" not in text
    thermo = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))["results"][0]["thermo"]
    assert thermo["scale_factor_source"] == "user"
