"""goodvibes.thermo.by_method: thermochemistry options a method sets for
itself, so a DFT and an MLIP method can use different scaling and qRRHO
settings in one evaluation of a reaction-profile document."""
import copy
import json
import math

import pytest

pytest.importorskip("yaml")

from conftest import g16path
from goodvibes import compute_thermo
from goodvibes.constants import GAS_CONSTANT, J_TO_AU, KCAL_TO_AU
from goodvibes.profile import Profile, ProfileError, validate_document

WATER_A = g16path("01a_water_hf_freq.log")
WATER_A2 = g16path("01c_water_hf_freq_isotopes.log")
WATER_B = g16path("01b_water_hf_freq_scaled.log")
SOURCES = {"A": {"files": ["01a_water_hf_freq", "01c_water_hf_freq_isotopes"]}, "B": {"files": "01b_water*"}}
MLIP = {"freq_scale_factor": 1.0, "QS": "truhlar", "s_freq_cutoff": 50.0}

DOC = {
    "schema": "reaction-profile/1.0",
    "species": {"A": {}, "B": {}},
    "points": {"start": {"species": "A"}, "end": {"species": "B"}},
    "pathways": {"rxn": ["start", "end"]},
    "methods": {"dft": {"level_of_theory": "HF/6-31G(d)"}, "mlip": {"model": "an MLIP"}},
    "series": [
        {"id": "G_dft", "method": "dft", "quantity": "qh_gibbs", "temperature": 298.15},
        {"id": "G_mlip", "method": "mlip", "quantity": "qh_gibbs", "temperature": 298.15},
    ],
    "goodvibes": {"sources": {"dft": SOURCES, "mlip": SOURCES},
                  "rollup": {"mode": "boltzmann"},
                  "thermo": {"by_method": {"mlip": MLIP}}},
}


@pytest.fixture(scope="module")
def results():
    return [compute_thermo(f) for f in (WATER_A, WATER_A2, WATER_B)]


def _dG(T=298.15, **options):
    """Boltzmann ΔG(end − start) with every structure computed directly."""
    a, a2, b = (compute_thermo(f, temperature=T, **options).qh_gibbs_free_energy
                for f in (WATER_A, WATER_A2, WATER_B))
    w = [math.exp(-(g - min(a, a2)) * J_TO_AU / GAS_CONSTANT / T) for g in (a, a2)]
    return (b - (w[0] * a + w[1] * a2) / sum(w)) * KCAL_TO_AU


def _end(prof, sid):
    return prof.get_series(sid).levels["rxn"]["end"]


def test_each_method_is_evaluated_with_its_own_options(results):
    prof = Profile.from_dict(DOC)
    assert prof.method_thermo("mlip") == MLIP and prof.method_thermo("dft") == {}
    ev = prof.evaluate(results)
    assert _end(ev, "G_dft") == pytest.approx(_dG(), abs=1e-6)
    assert _end(ev, "G_mlip") == pytest.approx(_dG(**MLIP), abs=1e-6)
    assert _end(ev, "G_dft") != pytest.approx(_end(ev, "G_mlip"), abs=1e-3)
    # the recipe records the run's options and keeps the per-method ones
    assert ev.namespace["thermo"]["QS"] == "grimme"
    assert ev.namespace["thermo"]["by_method"] == {"mlip": MLIP}
    assert validate_document(ev.to_dict())[0] == []


def test_the_overrides_reach_the_embedded_conformers(results):
    ev = Profile.from_dict(DOC).evaluate(results, with_conformers=True)
    mlip = ev.namespace["conformers"]["mlip"]["A"][0]["options"]
    dft = ev.namespace["conformers"]["dft"]["A"][0]["options"]
    assert (mlip["QS"], mlip["freq_scale_factor"], mlip["zpe_scale_factor"], mlip["scale_factor_source"]) == \
        ("truhlar", 1.0, 1.0, "user")
    assert (dft["QS"], dft["scale_factor_source"]) == ("grimme", "truhlar")
    again = Profile.from_dict(json.loads(json.dumps(ev.to_dict()))).evaluate(temperatures=[400.0])
    assert _end(again, "G_mlip@400K") == pytest.approx(_dG(400.0, **MLIP), abs=1e-6)
    assert _end(again, "G_dft@400K") == pytest.approx(_dG(400.0), abs=1e-6)


def test_a_zpe_factor_alone_keeps_the_looked_up_frequency_factor(results):
    doc = copy.deepcopy(DOC)
    doc["goodvibes"]["thermo"]["by_method"] = {"mlip": {"zpe_scale_factor": 0.95}}
    ev = Profile.from_dict(doc).evaluate(results, with_conformers=True)
    opts = ev.namespace["conformers"]["mlip"]["A"][0]["options"]
    assert (opts["freq_scale_factor"], opts["zpe_scale_factor"], opts["scale_factor_source"]) == \
        (0.922, 0.95, "truhlar")
    assert _end(ev, "G_mlip") == pytest.approx(_dG(zpe_scale_factor=0.95), abs=1e-6)


def test_the_overrides_need_the_parsed_output(results):
    stubs = {}
    for r in results:
        b = copy.copy(r.bbe)
        del b.options                             # a result that cannot be re-evaluated
        stubs[r.file] = b
    doc = copy.deepcopy(DOC)
    doc["series"] = [doc["series"][1]]
    with pytest.raises(ProfileError, match="by_method.mlip: species 'A' cannot be re-evaluated"):
        Profile.from_dict(doc).evaluate(stubs)


@pytest.mark.parametrize("overrides, message", [
    ({"spc": "sp_tz"}, "cannot differ between methods"),
    ({"QS": "rrho"}, "must be 'grimme' or 'truhlar'"),
    ({"s_freq_cutoff": -5}, "must be positive"),
    ({"QH": "yes"}, "must be true or false"),
    ({"temperature": 350.0}, "unknown thermochemistry option"),
])
def test_invalid_overrides_are_errors(overrides, message):
    doc = copy.deepcopy(DOC)
    doc["goodvibes"]["thermo"]["by_method"] = {"mlip": overrides}
    errors, _ = validate_document(doc)
    assert any("goodvibes.thermo.by_method.mlip." in e and message in e for e in errors), errors


def test_a_numeric_method_id_matches_its_series(results):
    doc = copy.deepcopy(DOC)
    doc["goodvibes"]["sources"] = {1: SOURCES, "dft": SOURCES}      # a YAML `1:` key
    doc["goodvibes"]["thermo"]["by_method"] = {1: MLIP}
    doc["series"][1]["method"] = 1
    prof = Profile.from_dict(doc)
    assert prof.method_thermo("1") == MLIP
    assert _end(prof.evaluate(results), "G_mlip") == pytest.approx(_dG(**MLIP), abs=1e-6)


def test_an_unknown_method_is_an_error():
    doc = copy.deepcopy(DOC)
    doc["goodvibes"]["thermo"]["by_method"] = {"xtb": {"QS": "truhlar"}}
    errors, _ = validate_document(doc)
    assert any("by_method.xtb" in e and "not defined" in e for e in errors), errors
