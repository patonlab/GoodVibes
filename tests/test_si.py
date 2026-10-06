"""goodvibes.si: the per-structure Supporting Information table and the
goodvibes --si option."""
import csv
import os

import pytest

from conftest import g16path
from goodvibes import compute_batch, compute_thermo
from goodvibes.constants import KCAL_TO_AU
from goodvibes.si import SI_COLUMNS, si_rows, si_xyz, write_si

WATER = g16path("01a_water_hf_freq.log")
TS = g16path("44_ts_sn2_identity_chloride.log")
SP = g16path("20_benzene_singlepoint.log")


@pytest.fixture(scope="module")
def results():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_batch([WATER, TS])


def test_a_row_reports_the_si_quantities(results):
    water, ts = si_rows(results)
    r = results[0]
    assert list(water) == list(SI_COLUMNS)
    assert (water["structure"], water["level_of_theory"], water["temperature"]) == \
        ("01a_water_hf_freq", "HF/6-31G(d)", 298.15)
    assert water["E"] == r.scf_energy and water["qh_G"] == r.qh_gibbs_free_energy
    assert water["TS"] == pytest.approx(298.15 * r.entropy)
    assert (water["n_imag"], water["imaginary"], water["lowest"]) == (0, None, "1859.8, 3930.0, 4038.5")
    assert (water["freq_scale_factor"], water["scale_factor_source"]) == (0.922, "truhlar")
    assert (water["point_group"], water["symmetry_number"], water["symmetry_source"]) == ("C2V", 2, "output")
    assert (ts["n_imag"], ts["imaginary"]) == (1, "-345.3")
    assert water["E_sp"] is None                          # no single point applied


def test_units_and_the_number_of_lowest_modes(results):
    (kcal,) = si_rows(results[:1], units="kcal/mol", n_lowest=1)
    assert kcal["E"] == pytest.approx(results[0].scf_energy * KCAL_TO_AU)
    assert kcal["lowest"] == "1859.8" and kcal["freq_scale_factor"] == 0.922   # not an energy


def test_a_single_point_has_no_thermal_quantities():
    (row,) = si_rows([compute_thermo(SP)])
    assert row["E"] is not None and row["ZPE"] is None and row["qh_G"] is None
    assert row["n_imag"] is None and row["lowest"] is None


def test_the_coordinates_appendix(results):
    text = si_xyz(results)
    lines = text.splitlines()
    assert lines[0] == "3" and lines[1].startswith("01a_water_hf_freq  HF/6-31G(d)  E = -76.010511 hartree")
    assert lines[2].split()[0] == "O" and len(lines) == 2 + 3 + 2 + 6


def test_markdown_latex_csv_and_xyz_files(results, tmp_path):
    md = tmp_path / "si.md"
    assert write_si(results, md) == [str(md)]
    text = md.read_text(encoding="utf-8")
    assert text.startswith("| Structure | Level of theory | T (K) | E (hartree) |")
    assert "E (SP)" not in text                            # an all-empty column is left out
    assert "| 298.15 |" in text and "| 0.922 |" in text
    assert "## Cartesian coordinates (Å)" in text and "```text\n3\n" in text

    tex = tmp_path / "si.tex"
    write_si(results, tex, units="kcal/mol", decimals=2)
    t = tex.read_text(encoding="utf-8")
    assert t.startswith(r"\begin{tabular}") and r"T$\cdot$S (kcal/mol)" in t and r"01a\_water\_hf\_freq" in t
    assert r"\begin{verbatim}" in t

    out = tmp_path / "si.csv"
    written = write_si(results, out)
    assert written == [str(out), str(tmp_path / "si_coordinates.xyz")]
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert list(rows[0]) == list(SI_COLUMNS) and rows[1]["imaginary"] == "-345.3"
    assert (tmp_path / "si_coordinates.xyz").read_text(encoding="utf-8") == si_xyz(results)

    only = tmp_path / "si_only.csv"
    assert write_si(results, only, coordinates=False) == [str(only)]
    assert write_si(results, tmp_path / "all.xyz")[0].endswith("all.xyz")
    with pytest.raises(ValueError, match="unknown format"):
        write_si(results, tmp_path / "si.docx")


def test_goodvibes_si_option(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    from test_cli_errors import run_main
    run_main(monkeypatch, tmp_path, [WATER, TS, "--si", "si.md", "--si-units", "kcal/mol"])
    text = (tmp_path / "si.md").read_text(encoding="utf-8")
    assert "E (kcal/mol)" in text and "44_ts_sn2_identity_chloride" in text
    assert "SI table written to si.md" in (tmp_path / "GoodVibes_output.dat").read_text(encoding="utf-8")
    assert os.path.exists(tmp_path / "si.md")


from test_cli_errors import gv_logger_cleanup  # noqa: E402,F401  (fixture)
