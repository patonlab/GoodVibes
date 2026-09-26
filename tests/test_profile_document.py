"""goodvibes.profile: the reaction-profile document model.

Parsing (explicit form, v2 PES YAML, legacy text), round trips, CSV
tables, evaluation against thermo data and against embedded conformers,
rows, plotting and the error surface.
"""
import json
import math
import warnings
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from conftest import g16path
from goodvibes import compute_thermo
from goodvibes.constants import KCAL_TO_AU
from goodvibes.pes_model import Series
from goodvibes.profile import (
    SCHEMA_TAG, Profile, ProfileError, ProfileWarning, load_profile, validate_document,
)

KIT = Path(__file__).resolve().parent / "profile_conformance"
ROOT = Path(__file__).resolve().parents[1]
WATER_A = g16path("01a_water_hf_freq.log")
WATER_A2 = g16path("01c_water_hf_freq_isotopes.log")
WATER_B = g16path("01b_water_hf_freq_scaled.log")

WATER_DOC = {
    "schema": SCHEMA_TAG,
    "title": "water",
    "species": {"A": {}, "B": {}},
    "points": {"start": {"species": "A", "role": "reactant"},
               "end": {"species": "B", "role": "product", "display": "B!"},
               "lit-only": {"role": "ts"}},
    "pathways": {"rxn": ["start", "end"], "lit": ["start", "lit-only", "end"]},
    "series": [
        {"id": "G", "quantity": "qh_gibbs", "temperature": 298.15},
        {"id": "lit", "quantity": "gibbs", "temperature": 298.15, "source": "declared",
         "levels": {"lit": {"start": 0.0, "lit-only": 5.0, "end": 1.0}}},
    ],
    "goodvibes": {"sources": {"default": {"A": {"files": ["01a_water_hf_freq", "01c_water_hf_freq_isotopes"]},
                                          "B": {"files": "01b_water*"}}},
                  "rollup": {"mode": "boltzmann"}},
}


@pytest.fixture(scope="module")
def water_results():
    return [compute_thermo(f) for f in (WATER_A, WATER_A2, WATER_B)]


def _expected_boltzmann_dG(results, T=298.15):
    from goodvibes.constants import GAS_CONSTANT, J_TO_AU
    at_T = [compute_thermo(r.file, temperature=T) for r in results]
    ga = [at_T[0].qh_gibbs_free_energy, at_T[1].qh_gibbs_free_energy]
    w = [math.exp(-(g - min(ga)) * J_TO_AU / GAS_CONSTANT / T) for g in ga]
    return (at_T[2].qh_gibbs_free_energy - sum(p * g for p, g in zip(w, ga)) / sum(w)) * KCAL_TO_AU


# ---------------------------------------------------------------------------
# parsing and round trips
# ---------------------------------------------------------------------------

def test_explicit_document_parses_and_round_trips(tmp_path):
    prof = load_profile(KIT / "valid" / "02_full.yaml")
    assert prof.title.startswith("Aza") and prof.units == "kcal/mol" and prof.upgraded_from is None
    assert prof.points["TS1"].role == "ts" and prof.points["TS1"].display == "TS1‡"
    assert prof.points["R"].species_mapping() == {"R1-An": 1, "Aza-Phos": 1}
    assert prof.pathways["Ph"].edges[0].kind == "barrierless"
    assert [e.kind for e in prof.pathways["Ph"].edges] == ["barrierless", "step", "step"]
    assert prof.extensions == {"x-lab-notes": "any x-* key is preserved and never validated"}
    for ext in ("json", "yaml"):
        out = tmp_path / f"p.{ext}"
        prof.dump(out)
        again = load_profile(out)
        assert again.to_dict() == prof.to_dict()
    assert prof.to_dict()["pathways"]["Ph"]["edges"] == [{"from": "R", "to": "Int1", "kind": "barrierless"}]


def test_v2_pes_yaml_is_upgraded_to_the_explicit_form():
    prof = load_profile(ROOT / "goodvibes/examples/pes/azabor_PES_v2.yaml")
    assert prof.upgraded_from == "v2"
    d = prof.to_dict()
    assert d["schema"] == SCHEMA_TAG
    assert d["pathways"]["Ph"]["points"][0] == "R1-An + Aza-Phos"
    assert d["points"]["R1-An + Aza-Phos"]["species"] == {"R1-An": 1, "Aza-Phos": 1}
    assert d["goodvibes"]["sources"]["default"]["AmTS"] == {"files": ["aminationTS-full-unfrz-*"]}
    assert d["style"] == {"decimals": 1}
    assert validate_document(d)[0] == []


def test_legacy_text_is_upgraded():
    prof = load_profile(ROOT / "goodvibes/examples/gconf_ee_boltz/gconf_TS.yaml")
    assert prof.upgraded_from == "legacy"
    assert list(prof.pathways) == ["Reaction"]
    assert prof.pathways["Reaction"].points == ["cat+subs", "TS"]
    assert validate_document(prof.to_dict())[0] == []


def test_directory_sources_round_trip():
    doc = {"pathways": {"p": ["A"]}, "species": {"A": {"dirs": ["TS_*"], "files": "x*"}}}
    prof = Profile.from_dict(doc)
    assert prof.namespace["sources"]["default"]["A"] == {"files": ["x*"], "dirs": ["TS_*"]}
    from goodvibes.profile import _source_to_patterns
    assert _source_to_patterns(prof.namespace["sources"]["default"]["A"]) == ["x*", "@dir:TS_*"]


def test_warnings_for_unknown_keys_and_aliases_and_strict_mode():
    doc = {"schema": SCHEMA_TAG, "points": {"A": {}}, "pathways": {"p": ["A"]}, "colour": "red",
           "series": [{"id": "s", "quantity": "G", "source": "declared", "levels": {"p": {"A": 0}}}]}
    with pytest.warns(ProfileWarning):
        prof = Profile.from_dict(doc)
    assert any("colour" in w for w in prof.warnings)
    assert any("alias" in w for w in prof.warnings)
    assert prof.series[0].quantity == "gibbs"
    with pytest.raises(ProfileError, match="colour"):
        Profile.from_dict(doc, strict=True)


def test_missing_schema_key_on_an_explicit_document_is_assumed_with_a_warning():
    with pytest.warns(ProfileWarning, match="missing"):
        prof = Profile.from_dict({"points": {"A": {}}, "pathways": {"p": {"points": ["A"]}}})
    assert prof.upgraded_from is None


def test_profile_error_lists_every_problem():
    doc = {"schema": SCHEMA_TAG, "units": "calories", "points": {"A": {"role": "saddle"}},
           "pathways": {"p": ["A", "B"]}}
    with pytest.raises(ProfileError) as exc:
        Profile.from_dict(doc)
    assert len(exc.value.errors) == 3
    assert "units" in str(exc.value) and "saddle" in str(exc.value) and "'B'" in str(exc.value)


def test_newer_minor_version_warns_and_other_major_fails():
    doc = {"schema": "reaction-profile/1.7", "points": {"A": {}}, "pathways": {"p": ["A"]}}
    with pytest.warns(ProfileWarning, match="newer"):
        Profile.from_dict(doc)
    doc["schema"] = "reaction-profile/2.0"
    with pytest.raises(ProfileError, match="major"):
        Profile.from_dict(doc)


def test_payload_with_a_profile_block_loads(tmp_path):
    prof = load_profile(KIT / "valid" / "01_minimal.yaml")
    payload = {"schema_version": "1.1", "goodvibes_version": "x", "results": [], "profile": prof.to_dict()}
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_profile(path).levels() == prof.levels()
    del payload["profile"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ProfileError, match="without a `profile` block"):
        load_profile(path)


def test_dump_rejects_unknown_extensions(tmp_path):
    with pytest.raises(ValueError, match="json"):
        load_profile(KIT / "valid" / "01_minimal.yaml").dump(tmp_path / "x.txt")


# ---------------------------------------------------------------------------
# CSV tables
# ---------------------------------------------------------------------------

WIDE = """point,role,display,Ph,Ph-lit
R,reactant,R,0.0,0.0
TS1,ts,TS1‡,20.1,18.4
Int,,,5.0,
P,product,P,-12.6,null
"""


def test_wide_table():
    prof = Profile.from_table(WIDE, units="kJ/mol", quantity="G", temperature=300.0, title="t")
    assert prof.units == "kJ/mol" and prof.title == "t" and prof.default_temperature == 300.0
    assert prof.pathways["Ph"].points == ["R", "TS1", "Int", "P"]
    assert prof.pathways["Ph-lit"].points == ["R", "TS1", "P"]
    s = prof.series[0]
    assert s.declared and s.quantity == "gibbs" and s.temperature == 300.0 and s.label == "ΔG(T) 300 K"
    assert s.levels["Ph-lit"] == {"R": 0.0, "TS1": 18.4, "P": None}
    assert prof.points["TS1"].role == "ts" and prof.points["TS1"].display == "TS1‡"


def test_wide_table_round_trips_through_to_rows(tmp_path):
    prof = Profile.from_table(WIDE)
    out = tmp_path / "t.csv"
    prof.write_table(out, layout="wide")
    again = Profile.from_table(out)
    assert again.levels() == prof.levels()
    two = Profile.from_dict({**prof.to_dict(), "series": prof.to_dict()["series"] + [
        {"id": "E", "quantity": "electronic", "source": "declared", "levels": {"Ph": {"R": 0.0, "P": -20.0}}}]})
    rows = two.to_rows("wide")
    assert "Ph:table" in rows[0] and "Ph:E" in rows[0]
    two.write_table(out, layout="wide")
    assert Profile.from_table(out).levels() == two.levels()


def test_long_table_with_series_units_and_uncertainty():
    text = """pathway,point,value,series,quantity,units,uncertainty,role
Ph,R,0,G,gibbs,kcal/mol,,reactant
Ph,TS,10,G,gibbs,kcal/mol,0.5,ts
Ph,R,0,E,electronic,kJ/mol,,
Ph,TS,41.84,E,electronic,kJ/mol,,
"""
    prof = Profile.from_table(text)
    assert [s.id for s in prof.series] == ["G", "E"]
    assert prof.get_series("E").levels["Ph"]["TS"] == pytest.approx(10.0)
    assert prof.get_series("G").uncertainty == {"Ph": {"TS": 0.5}}
    rows = prof.to_rows("long")
    assert len(rows) == 4
    ts_g = [r for r in rows if (r["point"], r["series"]) == ("TS", "G")]
    assert ts_g[0]["uncertainty"] == 0.5 and ts_g[0]["role"] == "ts"


def test_long_table_uncertainty_is_converted_with_its_value():
    text = """pathway,point,value,units,uncertainty
Ph,R,0,kJ/mol,
Ph,TS,41.84,kJ/mol,4.184
"""
    s = Profile.from_table(text, units="kcal/mol").series[0]
    assert s.levels["Ph"]["TS"] == pytest.approx(10.0)
    assert s.uncertainty["Ph"]["TS"] == pytest.approx(1.0)


def test_table_errors_and_zero_warning():
    with pytest.raises(ProfileError, match="'point' column"):
        Profile.from_table("a,b\n1,2\n", layout="wide")
    with pytest.raises(ProfileError, match="not a number"):
        Profile.from_table("point,Ph\nR,zero\n")
    with pytest.raises(ProfileError, match="no rows"):
        Profile.from_table("")
    with pytest.warns(ProfileWarning, match="at its zero point"):
        Profile.from_table("point,Ph\nR,1.0\nP,2.0\n")


def test_markdown_table(tmp_path):
    out = tmp_path / "t.md"
    Profile.from_table(WIDE).write_table(out)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("| point | role | display | Ph | Ph-lit |") and "| — |" in text


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def test_evaluate_against_thermo_data(water_results):
    prof = Profile.from_dict(WATER_DOC)
    ev = prof.evaluate(water_results, invocation="test")
    g = ev.get_series("G")
    assert not g.declared and set(g.levels) == {"rxn", "lit"}
    assert g.levels["rxn"]["end"] == pytest.approx(_expected_boltzmann_dG(water_results), abs=1e-6)
    assert "lit-only" not in g.levels["lit"]                   # a declared-only point
    assert ev.get_series("lit").levels == prof.get_series("lit").levels
    assert [s.id for s in ev.series] == ["G", "lit"]
    prov = ev.provenance
    assert prov["tool"] == "goodvibes" and prov["invocation"] == "test"
    assert [i["file"] for i in prov["inputs"]] == ["01a_water_hf_freq.log", "01c_water_hf_freq_isotopes.log",
                                                   "01b_water_hf_freq_scaled.log"]
    assert all(len(i["sha1"]) == 40 for i in prov["inputs"])
    assert ev.namespace["rollup"]["mode"] == "boltzmann"
    assert ev.namespace["thermo"]["QS"] == "grimme"
    assert validate_document(ev.to_dict())[0] == []
    # the input document is unchanged
    assert prof.get_series("G").levels is None


def test_evaluate_accepts_a_thermo_data_mapping_and_expands_temperatures(water_results):
    td = {r.file: r.bbe for r in water_results}
    ev = Profile.from_dict(WATER_DOC).evaluate(td, temperatures=[250.0, 350.0])
    ids = [s.id for s in ev.series]
    assert ids == ["G@250K", "G@350K", "lit"]
    assert ev.get_series("G@350K").levels["rxn"]["end"] == pytest.approx(
        _expected_boltzmann_dG(water_results, 350.0), abs=1e-6)
    assert ev.get_series("G@350K").label == "Δqh-G(T) 350 K"


def test_document_without_computed_series_gets_the_default(water_results):
    doc = {k: v for k, v in WATER_DOC.items() if k != "series"}
    ev = Profile.from_dict(doc).evaluate(water_results)
    assert [s.id for s in ev.series] == ["qh_gibbs@298.15K"]
    custom = [Series(id="E", label="ΔE", quantity="electronic", temperature=298.15)]
    assert [s.id for s in Profile.from_dict(doc).evaluate(water_results, default_series=custom).series] == ["E"]


def test_embedded_conformers_re_evaluate_file_free(water_results):
    ev = Profile.from_dict(WATER_DOC).evaluate(water_results, with_conformers=True)
    conf = ev.namespace["conformers"]["default"]
    assert [c["file"] for c in conf["A"]] == ["01a_water_hf_freq.log", "01c_water_hf_freq_isotopes.log"]
    text = json.dumps(ev.to_dict())
    assert "01a_water_hf_freq.log" in text
    again = Profile.from_dict(json.loads(text))
    re_ev = again.evaluate(temperatures=[298.15, 400.0])
    assert re_ev.get_series("G@298.15K").levels["rxn"]["end"] == pytest.approx(
        ev.get_series("G").levels["rxn"]["end"], abs=1e-9)
    assert re_ev.get_series("G@400K").levels["rxn"]["end"] == pytest.approx(
        _expected_boltzmann_dG(water_results, 400.0), abs=1e-6)
    assert "conformers" not in ev.to_dict(include_conformers=False)["goodvibes"]


def test_evaluate_without_data_is_a_clear_error():
    with pytest.raises(ProfileError, match="no species of method"):
        Profile.from_dict(WATER_DOC).evaluate()


def test_unmatched_source_is_an_error(water_results):
    doc = json.loads(json.dumps(WATER_DOC))
    doc["goodvibes"]["sources"]["default"]["B"] = {"files": "nothing*"}
    with pytest.raises(ValueError, match="no files matched"):
        Profile.from_dict(doc).evaluate(water_results)


def test_dedup_in_the_namespace_is_applied(water_results):
    doc = json.loads(json.dumps(WATER_DOC))
    doc["goodvibes"]["dedup"] = {"e_cutoff": 0.05, "ro_cutoff": 0.01}
    same = [compute_thermo(WATER_A), compute_thermo(WATER_A), compute_thermo(WATER_B)]
    same[1] = type(same[1])(**{**same[1].__dict__, "file": WATER_A2})    # the same structure twice
    pes = Profile.from_dict(doc).to_pes_result(same)
    assert len(pes.pathway("rxn").points[0].species[0][1].files) == 1


def test_from_pes_result_for_a_model_built_in_python(water_results):
    from goodvibes import ConformerSet, PESOptions, PESResult, Pathway, Point
    species = {"A": ConformerSet.from_results("A", water_results[:2]),
               "B": ConformerSet.from_results("B", water_results[2:])}
    pes = PESResult([Pathway("rxn", [Point.from_label("A", species), Point.from_label("B", species, role="ts")])],
                    PESOptions(gconf=False), temperatures=[298.15, 350.0])
    prof = Profile.from_pes_result(pes, title="built")
    assert prof.title == "built" and [s.id for s in prof.series] == ["qh_gibbs@298.15K", "qh_gibbs@350K"]
    assert prof.namespace["sources"]["default"]["B"] == {"files": ["01b_water_hf_freq_scaled"]}
    assert prof.points["B"].role == "ts" and prof.namespace["rollup"]["mode"] == "boltzmann"
    assert prof.get_series("qh_gibbs@350K").levels["rxn"]["B"] == pytest.approx(
        _expected_boltzmann_dG(water_results, 350.0), abs=1e-6)
    assert validate_document(prof.to_dict())[0] == []


def test_to_dataframe(water_results):
    pd = pytest.importorskip("pandas")
    df = Profile.from_dict(WATER_DOC).evaluate(water_results).to_dataframe()
    assert isinstance(df, pd.DataFrame) and set(df["series"]) == {"G", "lit"}


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def test_plot_draws_stored_levels_and_annotations():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg", force=True)
    prof = load_profile(KIT / "valid" / "02_full.yaml")
    fig = prof.plot()
    assert fig.ax.get_title() == prof.title
    assert fig.level("Ph", "TS1", "dft-G-298") == 13.4 and fig.level("Ph-lit", "TS1", "lit") == 18.4
    texts = [t.get_text() for t in fig.ax.texts]
    assert "+17.2" in texts and "ΔG_rxn -91.4" in texts
    assert fig.layout == "overlay"
    only = prof.plot(series=["lit"], annotations=False, layout="panels")
    assert [s.id for s in only.series] == ["lit"] and len(only.axes) == 2
    fig.close(); only.close()


def test_plot_of_an_unevaluated_recipe_is_a_clear_error():
    pytest.importorskip("matplotlib")
    with pytest.raises(ProfileError, match="evaluate the document first"):
        load_profile(KIT / "valid" / "04_recipe.yaml").plot()


def test_plot_from_embedded_conformers_without_stored_levels(water_results):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg", force=True)
    ev = Profile.from_dict(WATER_DOC).evaluate(water_results, with_conformers=True)
    d = ev.to_dict()
    del d["series"][0]["levels"]                       # computed live from the conformers
    fig = Profile.from_dict(d).plot()
    assert fig.level("rxn", "end", "G") == pytest.approx(ev.get_series("G").levels["rxn"]["end"])
    fig.close()


def test_warnings_are_quiet_when_filtered():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_profile(KIT / "valid" / "01_minimal.yaml")      # a clean document raises nothing
