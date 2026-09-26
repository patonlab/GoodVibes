"""The reaction-profile document through the command line: goodvibes
--profile / --with-conformers / --pes <document>, payload 1.1, and the
file-free goodvibes-profile command."""
import json
import math
import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from conftest import g16path
from goodvibes.profile import load_profile, validate_document
from goodvibes.profile_cli import main as gvp

from test_cli_errors import gv_logger_cleanup, run_main  # noqa: F401  (fixture re-export)

KIT = Path(__file__).resolve().parent / "profile_conformance"
WATERS = [g16path("01a_water_hf_freq.log"), g16path("01c_water_hf_freq_isotopes.log"),
          g16path("01b_water_hf_freq_scaled.log")]
V2 = ("pathways:\n  rxn: [A, B]\n"
      "species:\n  A: [01a_water_hf_freq, 01c_water_hf_freq_isotopes]\n  B: [01b_water_hf_freq_scaled]\n")
DOC = {
    "schema": "reaction-profile/1.0",
    "title": "water doc",
    "species": {"A": {}, "B": {}},
    "points": {"start": {"species": "A", "role": "reactant"}, "end": {"species": "B", "role": "ts"},
               "lit-only": {}},
    "pathways": {"rxn": ["start", "end"], "lit": ["start", "lit-only"]},
    "series": [{"id": "G", "quantity": "qh_gibbs", "temperature": 298.15},
               {"id": "lit", "quantity": "gibbs", "source": "declared",
                "levels": {"lit": {"start": 0.0, "lit-only": 3.0}}}],
    "goodvibes": {"sources": {"default": {"A": {"files": ["01a_water_hf_freq", "01c_water_hf_freq_isotopes"]},
                                          "B": {"files": ["01b_water_hf_freq_scaled"]}}},
                  "rollup": {"mode": "boltzmann"}},
}


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text if isinstance(text, str) else yaml.safe_dump(text, allow_unicode=True), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# goodvibes --profile
# ---------------------------------------------------------------------------

def test_profile_from_a_v2_pes_file_matches_the_pes_block(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    pes = _write(tmp_path, "pes.yaml", V2)
    run_main(monkeypatch, tmp_path, WATERS + ["--pes", pes, "--json", "out.json", "--profile", "prof.json",
                                              "--ti", "250,350,50"])
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.1"
    prof = load_profile(tmp_path / "prof.json")
    assert validate_document(prof.to_dict())[0] == []
    assert [s.id for s in prof.series] == ["qh_gibbs@250K", "qh_gibbs@300K", "qh_gibbs@350K"]
    for entry in payload["pes"]["pathways"]:
        T = entry["temperature"]
        assert prof.get_series(f"qh_gibbs@{T:g}K").levels["rxn"]["B"] == pytest.approx(
            entry["points"][1]["relative"]["qh_g"], abs=1e-9)
    assert payload["profile"]["series"] == prof.to_dict()["series"]
    assert "conformers" not in prof.namespace
    assert prof.provenance["invocation"].startswith("goodvibes ")
    text = (tmp_path / "GoodVibes_output.dat").read_text(encoding="utf-8")
    assert "Reaction profile written to prof.json" in text


def test_with_conformers_writes_a_self_contained_yaml(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    pes = _write(tmp_path, "pes.yaml", V2)
    run_main(monkeypatch, tmp_path, WATERS + ["--pes", pes, "--profile", "prof.yaml", "--with-conformers"])
    prof = load_profile(tmp_path / "prof.yaml")
    assert set(prof.namespace["conformers"]["default"]) == {"A", "B"}
    hot = prof.evaluate(temperatures=[400.0])
    from goodvibes import compute_thermo
    b = compute_thermo(WATERS[2], temperature=400.0).qh_gibbs_free_energy
    a = [compute_thermo(f, temperature=400.0).qh_gibbs_free_energy for f in WATERS[:2]]
    # default rollup (gconf) with two conformers: −RT ln Σ exp(−G/RT)
    from goodvibes.constants import GAS_CONSTANT, J_TO_AU, KCAL_TO_AU
    rt = GAS_CONSTANT * 400.0 / J_TO_AU
    g_a = min(a) - rt * math.log(sum(math.exp(-(g - min(a)) / rt) for g in a))
    assert hot.get_series("qh_gibbs@400K").levels["rxn"]["B"] == pytest.approx((b - g_a) * KCAL_TO_AU, abs=1e-6)


def test_a_reaction_profile_document_through_pes(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    pytest.importorskip("matplotlib")
    doc = _write(tmp_path, "doc.yaml", DOC)
    run_main(monkeypatch, tmp_path, WATERS + ["--pes", doc, "--json", "out.json", "--pes-plot", "fig.png"])
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert [s["id"] for s in payload["profile"]["series"]] == ["G", "lit"]
    assert payload["profile"]["goodvibes"]["rollup"]["mode"] == "boltzmann"
    # the table / pes block honour the document's Boltzmann rollup (no --nogconf given) ...
    pes_names = [p["name"] for p in payload["pes"]["pathways"]]
    assert pes_names == ["rxn"]                                     # declared-only pathway skipped
    g_json = payload["pes"]["pathways"][0]["points"][1]["relative"]["qh_g"]
    assert payload["profile"]["series"][0]["levels"]["rxn"]["end"] == pytest.approx(g_json, abs=1e-9)
    text = (tmp_path / "GoodVibes_output.dat").read_text(encoding="utf-8")
    assert "Boltzmann-averaged" in text and "Gconf correction applied" not in text
    assert "declared values only, not tabulated here: lit" in text
    assert (tmp_path / "fig.png").stat().st_size > 0


def test_document_warnings_reach_the_dat_file(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    doc = dict(DOC, colour="red")
    run_main(monkeypatch, tmp_path, WATERS + ["--pes", _write(tmp_path, "doc.yaml", doc)])
    assert "unknown top-level key 'colour'" in (tmp_path / "GoodVibes_output.dat").read_text(encoding="utf-8")


@pytest.mark.parametrize("extra", [["--profile", "p.json"], ["--with-conformers"],
                                   ["--pes", "x.yaml", "--profile", "p.txt"]])
def test_profile_flag_validation(monkeypatch, tmp_path, gv_logger_cleanup, extra):  # noqa: F811
    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, WATERS[:1] + extra)


def test_payload_1_1_and_1_0_both_import(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    pes = _write(tmp_path, "pes.yaml", V2)
    run_main(monkeypatch, tmp_path, WATERS + ["--pes", pes, "--json", "out.json"])
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert "profile" in payload
    run_main(monkeypatch, tmp_path, ["--import", "out.json", "--output", "again"])
    assert "Reading from QCData cache" in (tmp_path / "GoodVibes_again.dat").read_text(encoding="utf-8")
    payload["schema_version"] = "1.0"
    del payload["profile"]
    (tmp_path / "old.json").write_text(json.dumps(payload), encoding="utf-8")
    run_main(monkeypatch, tmp_path, ["--import", "old.json", "--output", "old"])
    assert (tmp_path / "GoodVibes_old.dat").exists()


# ---------------------------------------------------------------------------
# goodvibes-profile
# ---------------------------------------------------------------------------

def test_validate(capsys):
    ok = [str(p) for p in sorted((KIT / "valid").glob("*.*"))]
    assert gvp(["validate", *ok]) == 0
    assert capsys.readouterr().out.count(": valid") == len(ok)
    assert gvp(["validate", str(KIT / "invalid-semantic" / "undefined_point.yaml")]) == 1
    out = capsys.readouterr().out
    assert "INVALID (1 error)" in out and "'B' is not defined" in out
    assert gvp(["validate", str(KIT / "invalid-structural" / "bad_units.yaml")]) == 1
    assert "schema: " in capsys.readouterr().out


def test_validate_strict_and_shorthand_and_legacy(tmp_path, capsys):
    loose = _write(tmp_path, "loose.yaml", {"schema": "reaction-profile/1.0", "points": {"A": {}},
                                            "pathways": {"p": ["A"]}, "colour": 1})
    assert gvp(["validate", loose]) == 0
    assert "warning" in capsys.readouterr().out
    assert gvp(["validate", "--strict", loose]) == 1
    capsys.readouterr()
    root = Path(__file__).resolve().parents[1] / "goodvibes" / "examples"
    assert gvp(["validate", str(root / "pes" / "azabor_PES_v2.yaml"),
                str(root / "gconf_ee_boltz" / "gconf_TS.yaml")]) == 0
    out = capsys.readouterr().out
    assert "v2 PES YAML shorthand" in out and "legacy" in out


def test_plot_writes_every_format(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    out1, out2 = str(tmp_path / "f.png"), str(tmp_path / "f.svg")
    assert gvp(["plot", str(KIT / "valid" / "02_full.yaml"), "-o", out1, "-o", out2,
                "--series", "lit", "--label-points", "--layout", "panels"]) == 0
    assert Path(out1).stat().st_size > 0 and "<svg" in Path(out2).read_text(encoding="utf-8")


def test_plot_from_a_csv_table(tmp_path):
    pytest.importorskip("matplotlib")
    csv = _write(tmp_path, "levels.csv", "point,role,display,Ph,Ph-lit\nR,reactant,R,0,0\nTS1,ts,TS1‡,20.1,18.4\n"
                                         "P,product,P,-12.6,-12.1\n")
    assert gvp(["plot", csv, "-o", str(tmp_path / "t.png"), "--quantity", "E", "--units", "kJ/mol"]) == 0


def test_plot_refuses_files_without_values(tmp_path, capsys):
    root = Path(__file__).resolve().parents[1] / "goodvibes" / "examples"
    with pytest.raises(SystemExit, match="carries no values"):
        gvp(["plot", str(root / "pes" / "azabor_PES_v2.yaml"), "-o", str(tmp_path / "x.png")])
    assert gvp(["plot", str(KIT / "valid" / "04_recipe.yaml"), "-o", str(tmp_path / "x.png")]) == 1
    assert "evaluate the document first" in capsys.readouterr().err


def test_plot_without_matplotlib_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(SystemExit, match="needs matplotlib"):
        gvp(["plot", str(KIT / "valid" / "01_minimal.yaml"), "-o", str(tmp_path / "x.png")])


def test_table_to_stdout_csv_and_markdown(tmp_path, capsys):
    doc = str(KIT / "valid" / "02_full.yaml")
    assert gvp(["table", doc]) == 0
    out = capsys.readouterr().out
    assert "Ph:dft-G-298" in out and "-91.4" in out
    assert gvp(["table", doc, "--long", "-o", str(tmp_path / "t.csv")]) == 0
    assert (tmp_path / "t.csv").read_text(encoding="utf-8").startswith("pathway,point,display,role,series")
    assert gvp(["table", doc, "-o", str(tmp_path / "t.md"), "--decimals", "2"]) == 0
    assert "13.40" in (tmp_path / "t.md").read_text(encoding="utf-8")


def test_convert_csv_legacy_and_json(tmp_path):
    csv = _write(tmp_path, "levels.csv", "point,Ph\nR,0\nP,-3\n")
    assert gvp(["convert", csv, "-o", str(tmp_path / "a.yaml"), "--doc-title", "from csv"]) == 0
    a = load_profile(tmp_path / "a.yaml")
    assert a.title == "from csv" and a.series[0].levels == {"Ph": {"R": 0.0, "P": -3.0}}
    legacy = Path(__file__).resolve().parents[1] / "goodvibes/examples/gconf_ee_boltz/gconf_TS.yaml"
    assert gvp(["convert", str(legacy), "-o", str(tmp_path / "b.json")]) == 0
    b = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
    assert b["schema"] == "reaction-profile/1.0" and "Reaction" in b["pathways"]
    assert gvp(["convert", str(tmp_path / "b.json"), "-o", str(tmp_path / "b.txt")]) == 1


def test_evaluate_needs_embedded_conformers(tmp_path, monkeypatch, gv_logger_cleanup):  # noqa: F811
    with pytest.raises(SystemExit, match="no embedded conformers"):
        gvp(["evaluate", str(KIT / "valid" / "01_minimal.yaml"), "-o", str(tmp_path / "x.json")])
    run_main(monkeypatch, tmp_path, WATERS + ["--pes", _write(tmp_path, "pes.yaml", V2),
                                              "--profile", "p.json", "--with-conformers"])
    assert gvp(["evaluate", str(tmp_path / "p.json"), "-o", str(tmp_path / "hot.json"),
                "--temperatures", "300,400", "--no-conformers"]) == 0
    hot = load_profile(tmp_path / "hot.json")
    assert [s.id for s in hot.series] == ["qh_gibbs@300K", "qh_gibbs@400K"] and "conformers" not in hot.namespace
    pytest.importorskip("matplotlib")
    assert gvp(["plot", str(tmp_path / "p.json"), "--temperatures", "250,500", "-o", str(tmp_path / "t.png")]) == 0
    with pytest.raises(SystemExit, match="positive"):
        gvp(["evaluate", str(tmp_path / "p.json"), "-o", str(tmp_path / "x.json"), "--temperatures", "-5"])


def test_missing_file_is_an_error_not_a_traceback(tmp_path, capsys):
    assert gvp(["table", str(tmp_path / "nope.yaml")]) == 1
    assert "goodvibes-profile: error:" in capsys.readouterr().err


def test_a_reaction_profile_document_with_a_temperature_scan(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    """With --ti an explicit document is tabulated by the model at every
    scan temperature; the legacy text path (which would re-read the file
    with the old parser) is not used."""
    doc = _write(tmp_path, "doc.yaml", DOC)
    run_main(monkeypatch, tmp_path, WATERS + ["--pes", doc, "--ti", "250,350,50", "--json", "out.json"])
    text = (tmp_path / "GoodVibes_output.dat").read_text(encoding="utf-8")
    for T in ("250", "300", "350"):
        assert f"at T = {T} K" in text
    assert "qh-DG(T)" not in text                      # the legacy PES header
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert [p["temperature"] for p in payload["pes"]["pathways"]] == [250.0, 300.0, 350.0]


def test_graph_refuses_a_reaction_profile_document(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    pytest.importorskip("matplotlib")
    doc = _write(tmp_path, "doc.yaml", DOC)
    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, WATERS + ["--pes", doc, "--graph", doc])
    assert "--graph reads only the legacy" in (tmp_path / "GoodVibes_output.dat").read_text(encoding="utf-8")


def test_series_without_temperature_follow_the_run_temperature(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    """A computed series with no temperature is evaluated at the run's
    temperature (--temp), like the tables and the pes block of the same
    payload, not at the document's default."""
    doc = json.loads(json.dumps(DOC))
    doc["series"] = [{"id": "G", "quantity": "qh_gibbs"}]
    run_main(monkeypatch, tmp_path, WATERS + ["--pes", _write(tmp_path, "doc.yaml", doc), "--temp", "350",
                                              "--json", "out.json"])
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    series = payload["profile"]["series"][0]
    assert series["temperature"] == 350.0
    assert payload["pes"]["pathways"][0]["temperature"] == 350.0
    assert series["levels"]["rxn"]["end"] == pytest.approx(
        payload["pes"]["pathways"][0]["points"][1]["relative"]["qh_g"], abs=1e-9)


def test_malformed_yaml_is_reported_not_a_traceback(tmp_path, capsys):
    bad = _write(tmp_path, "bad.yaml", "schema: reaction-profile/1.0\npathways: {p: [A\n")
    good = str(KIT / "valid" / "01_minimal.yaml")
    assert gvp(["validate", bad, good]) == 1
    captured = capsys.readouterr()
    assert "bad.yaml: cannot read" in captured.err and "01_minimal.yaml: valid" in captured.out
    assert gvp(["table", bad]) == 1
    assert "goodvibes-profile: error:" in capsys.readouterr().err


def test_malformed_pes_file_is_a_fatal_error(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    bad = _write(tmp_path, "bad.yaml", "pathways: {p: [A\n")
    with pytest.raises(SystemExit):
        run_main(monkeypatch, tmp_path, WATERS + ["--pes", bad])
    assert "FATAL ERROR: --pes" in (tmp_path / "GoodVibes_output.dat").read_text(encoding="utf-8")
