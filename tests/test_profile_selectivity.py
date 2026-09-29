"""reaction-profile 1.1 selectivity blocks: branch barriers from series
levels, the Curtin–Hammett checks, versioning and round trips."""
import copy
import json
import math
import warnings
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from goodvibes.constants import GAS_CONSTANT, J_TO_AU, KCAL_TO_AU
from goodvibes.profile import (
    BASE_TAG, SCHEMA_TAG, Profile, ProfileError, SelectivityWarning, load_json_schema, validate_document,
)

KIT = Path(__file__).resolve().parent / "profile_conformance"
DOC = yaml.safe_load((KIT / "valid" / "05_selectivity.yaml").read_text(encoding="utf-8"))
T = 298.15
RT_KCAL = GAS_CONSTANT * T / J_TO_AU * KCAL_TO_AU


def _doc(**changes):
    d = copy.deepcopy(DOC)
    d.update(changes)
    return d


def _expected_ee(gap_kcal):
    r = math.exp(gap_kcal / RT_KCAL)
    return (r - 1) / (r + 1) * 100


def _quiet(prof, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", SelectivityWarning)
        return prof.evaluate_selectivity(*args, **kwargs)


def test_branch_barriers_give_the_populations():
    (r,) = _quiet(Profile.from_dict(DOC))
    assert (r.name, r.series, r.reference, r.labels) == ("er", "G", "Int", ["TS_R", "TS_S"])
    assert r.barriers["TS_R"] * KCAL_TO_AU == pytest.approx(15.0)
    assert r.barriers["TS_S"] * KCAL_TO_AU == pytest.approx(16.2)
    assert r.major == "TS_R" and r.ee_signed == pytest.approx(_expected_ee(1.2))
    assert r.ddG * KCAL_TO_AU == pytest.approx(1.2)
    assert r.ensemble_energies == r.barriers
    assert r.temperature == T and r.quantity == "gibbs"
    assert r.curtin_hammett == "satisfied" and r.warnings == ()


def test_the_ee_sign_follows_the_branch_order():
    d = _doc()
    d["selectivity"][0]["branches"] = ["TS_S", "TS_R"]
    (r,) = _quiet(Profile.from_dict(d))
    assert r.major == "TS_R" and r.ee_signed == pytest.approx(-_expected_ee(1.2))


def test_barriers_are_taken_within_one_pathway_whatever_its_zero():
    d = _doc()
    d["series"][0]["levels"]["S"] = {"Int": 3.0, "TS_S": 19.2, "P_S": -1.0}
    (r,) = _quiet(Profile.from_dict(d))
    assert r.barriers["TS_S"] * KCAL_TO_AU == pytest.approx(16.2)


def test_without_an_interconversion_barrier_curtin_hammett_is_assumed():
    d = _doc()
    del d["selectivity"][0]["interconversion"]
    (r,) = _quiet(Profile.from_dict(d))
    assert r.curtin_hammett == "assumed"


def test_a_slow_interconversion_violates_curtin_hammett():
    d = _doc()
    d["series"][0]["levels"]["swap"]["TS_swap"] = 17.0
    with pytest.warns(SelectivityWarning, match="interconversion barrier"):
        (r,) = Profile.from_dict(d).evaluate_selectivity()
    assert r.curtin_hammett == "violated" and len(r.warnings) == 1


def test_a_deeper_resting_state_violates_curtin_hammett():
    d = _doc()
    d["points"]["Int2"] = {"role": "minimum"}
    d["pathways"]["R"] = ["Int", "Int2", "TS_R", "P_R"]
    d["series"][0]["levels"]["R"] = {"Int": 0.0, "Int2": -3.0, "TS_R": 15.0, "P_R": -5.0}
    with pytest.warns(SelectivityWarning, match="'Int2' .* resting state"):
        (r,) = Profile.from_dict(d).evaluate_selectivity()
    assert r.curtin_hammett == "violated"


def test_a_branch_below_the_reference_and_a_non_ts_branch_are_reported():
    d = _doc()
    d["selectivity"][0]["branches"] = ["P_R", "P_S"]
    del d["selectivity"][0]["interconversion"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        (r,) = Profile.from_dict(d).evaluate_selectivity()
    text = " ".join(str(w.message) for w in caught)
    assert "not a transition state" in text and "below the reference" in text
    assert r.curtin_hammett == "violated" and r.major == "P_R"
    (quiet,) = Profile.from_dict(d).evaluate_selectivity(warn=False)
    assert quiet.warnings == r.warnings


def test_every_series_with_the_levels_is_used_unless_one_is_named():
    d = _doc()
    del d["selectivity"][0]["series"]
    lit = copy.deepcopy(d["series"][0])
    lit.update(id="lit", temperature=253.15)
    lit["levels"]["S"]["TS_S"] = 17.0
    d["series"].append(lit)
    d["series"].append({"id": "E", "quantity": "electronic", "source": "declared",
                        "levels": {"R": {"Int": 0.0, "P_R": -9.0}}})       # no branch levels: skipped
    prof = Profile.from_dict(d)
    results = _quiet(prof)
    assert [(r.series, r.temperature) for r in results] == [("G", T), ("lit", 253.15)]
    (only,) = _quiet(prof, series="lit")
    assert only.ddG * KCAL_TO_AU == pytest.approx(2.0)
    with pytest.raises(ProfileError, match="series 'E' has no levels"):
        prof.evaluate_selectivity(series="E")
    with pytest.raises(ProfileError, match="no selectivity block 'nope'"):
        prof.evaluate_selectivity("nope")


def test_an_unevaluated_document_says_so():
    d = _doc()
    d["series"] = [{"id": "G", "quantity": "gibbs", "temperature": T}]       # computed, no levels
    with pytest.raises(ProfileError, match="evaluate the document first"):
        Profile.from_dict(d).evaluate_selectivity()


# -- versioning and validation -----------------------------------------------------

def test_documents_are_written_at_the_oldest_version_that_expresses_them():
    prof = Profile.from_dict(DOC)
    assert prof.to_dict()["schema"] == SCHEMA_TAG == "reaction-profile/1.1"
    assert prof.to_dict()["selectivity"] == DOC["selectivity"]
    prof.selectivity = []
    assert prof.to_dict()["schema"] == BASE_TAG == "reaction-profile/1.0"
    back = Profile.from_dict(json.loads(json.dumps(Profile.from_dict(DOC).to_dict())))
    assert back.selectivity == DOC["selectivity"]


@pytest.mark.parametrize("change, message", [
    (lambda d: d.update(schema="reaction-profile/1.0"), "is a reaction-profile 1.1 key"),
    (lambda d: d["selectivity"][0].update(reference="Nope"), "'Nope' is not a defined point"),
    (lambda d: d["selectivity"][0].update(branches=["TS_R"]), "at least two point ids"),
    (lambda d: d["selectivity"][0].update(branches=["TS_R", "TS_R"]), "lists a point twice"),
    (lambda d: d["selectivity"][0].update(branches=["TS_R", "Int"]), "is the reference point itself"),
    (lambda d: d["selectivity"][0].update(series="nope"), "unknown series 'nope'"),
    (lambda d: d["selectivity"][0].update(kind="optical"), "must be one of enantio"),
    (lambda d: (d["points"].update(X={}), d["pathways"].update(x=["X"]),
                d["selectivity"][0].update(interconversion="X")), "shares no pathway with the reference"),
    (lambda d: d["selectivity"].append(dict(d["selectivity"][0])), "duplicate selectivity id"),
])
def test_invalid_blocks_are_errors(change, message):
    d = _doc()
    change(d)
    errors, _ = validate_document(d, use_jsonschema=False)
    assert any(message in e for e in errors), errors


def test_the_1_1_schema_only_adds_selectivity_to_1_0():
    from goodvibes.profile import SCHEMA_FILE
    old = json.loads((Path(__file__).resolve().parents[1] / "goodvibes" / "schemas"
                      / "reaction-profile-1.0.schema.json").read_text(encoding="utf-8"))
    new = load_json_schema()
    assert SCHEMA_FILE == "reaction-profile-1.1.schema.json"
    assert old["$id"].endswith("/reaction-profile-1.0.schema.json")          # still published
    assert {k: v for k, v in new["properties"].items() if k != "selectivity"} == \
        {k: v for k, v in old["properties"].items() if k != "selectivity"}
    assert {k for k in new if new.get(k) != old.get(k)} == {"$id", "title", "description", "properties", "allOf"}


# -- computed series and evaluation --------------------------------------------------

def test_a_computed_profile_is_evaluated_per_temperature():
    from conftest import g16path
    from goodvibes import compute_thermo
    files = [g16path(f) for f in ("01a_water_hf_freq.log", "01b_water_hf_freq_scaled.log",
                                  "01c_water_hf_freq_isotopes.log")]
    results = [compute_thermo(f) for f in files]
    doc = {
        "schema": "reaction-profile/1.1",
        "species": {"A": {}, "B": {}, "C": {}},
        "points": {"R": {"species": "A"}, "T1": {"species": "B", "role": "ts"},
                   "T2": {"species": "C", "role": "ts"}},
        "pathways": {"one": ["R", "T1"], "two": ["R", "T2"]},
        "series": [{"id": "G", "quantity": "qh_gibbs", "temperature": T}],
        "selectivity": [{"id": "s", "reference": "R", "branches": ["T1", "T2"], "series": "G"}],
        "goodvibes": {"sources": {"default": {"A": {"files": "01a_water*"}, "B": {"files": "01b_water*"},
                                              "C": {"files": "01c_water*"}}}},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SelectivityWarning)     # water "TSs" are minima: noted, not tested here
        ev = Profile.from_dict(doc).evaluate(results, temperatures=[T, 400.0])
        assert "series" not in ev.selectivity[0]                  # G became G@298.15K and G@400K
        assert validate_document(ev.to_dict())[0] == []
        out = ev.evaluate_selectivity()
    assert [r.series for r in out] == ["G@298.15K", "G@400K"]
    for r in out:
        g = {x.name: x.qh_gibbs_free_energy for x in (compute_thermo(f, temperature=r.temperature) for f in files)}
        assert r.barriers["T1"] == pytest.approx(g["01b_water_hf_freq_scaled"] - g["01a_water_hf_freq"], abs=1e-9)
        assert r.barriers["T2"] == pytest.approx(g["01c_water_hf_freq_isotopes"] - g["01a_water_hf_freq"], abs=1e-9)


def test_a_drawn_figure_keeps_the_block():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg", force=True)
    prof = Profile.from_dict(DOC)
    fig = prof.plot()
    drawn = fig.to_document()
    fig.close()
    assert drawn["schema"] == SCHEMA_TAG and drawn["selectivity"][0]["id"] == "er"
    assert validate_document(drawn)[0] == []


# -- the command-line tools -----------------------------------------------------------

def test_goodvibes_profile_selectivity(tmp_path, capsys):
    from goodvibes.profile_cli import main as gvp
    src = str(KIT / "valid" / "05_selectivity.yaml")
    assert gvp(["selectivity", src]) == 0
    out = capsys.readouterr().out
    assert "er (G, 298.15 K): major TS_R, ee +76.7 % (TS_R vs TS_S)" in out and "Curtin–Hammett satisfied" in out
    assert gvp(["selectivity", src, "-o", str(tmp_path / "sel.csv")]) == 0
    text = (tmp_path / "sel.csv").read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("selectivity,series,temperature,branch,barrier,population")
    capsys.readouterr()
    assert gvp(["selectivity", src, "--json"]) == 0
    (res,) = json.loads(capsys.readouterr().out)
    assert res["units"] == "kcal/mol" and res["barriers"]["TS_S"] == pytest.approx(16.2)


def test_goodvibes_profile_selectivity_errors(capsys):
    from goodvibes.profile_cli import main as gvp
    src = str(KIT / "valid" / "05_selectivity.yaml")
    with pytest.raises(SystemExit, match="needs embedded conformers"):
        gvp(["selectivity", src, "--temperatures", "300"])
    with pytest.raises(SystemExit, match="no `selectivity` blocks"):
        gvp(["selectivity", str(KIT / "valid" / "01_minimal.yaml")])
    assert gvp(["selectivity", src, "--id", "nope"]) == 1
    assert "no selectivity block 'nope'" in capsys.readouterr().err


def test_the_goodvibes_command_prints_and_writes_the_selectivity(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    from conftest import g16path
    from test_cli_errors import run_main
    doc = {
        "schema": "reaction-profile/1.1",
        "species": {"A": {}, "B": {}, "C": {}},
        "points": {"R": {"species": "A"}, "T1": {"species": "B", "role": "ts"},
                   "T2": {"species": "C", "role": "ts"}},
        "pathways": {"one": ["R", "T1"], "two": ["R", "T2"]},
        "series": [{"id": "G", "quantity": "qh_gibbs"}],
        "selectivity": [{"id": "s", "reference": "R", "branches": ["T1", "T2"]}],
        "goodvibes": {"sources": {"default": {"A": {"files": "01a_water*"}, "B": {"files": "01b_water*"},
                                              "C": {"files": "01c_water*"}}}},
    }
    (tmp_path / "doc.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
    files = [g16path(f) for f in ("01a_water_hf_freq.log", "01b_water_hf_freq_scaled.log",
                                  "01c_water_hf_freq_isotopes.log")]
    run_main(monkeypatch, tmp_path, files + ["--pes", "doc.yaml", "--json", "out.json"])
    text = (tmp_path / "GoodVibes_output.dat").read_text(encoding="utf-8")
    assert "Selectivity s (G, 298.15 K): major" in text and "Curtin–Hammett" in text
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert payload["profile"]["schema"] == SCHEMA_TAG
    (res,) = payload["profile_selectivity"]
    assert res["name"] == "s" and set(res["barriers"]) == {"T1", "T2"}


from test_cli_errors import gv_logger_cleanup  # noqa: E402,F401  (fixture)
