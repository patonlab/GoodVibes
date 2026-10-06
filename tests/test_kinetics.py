"""goodvibes.kinetics: Eyring rates, the energy span model, the step table
and the mikimo export; the Profile hooks and goodvibes-profile kinetics."""
import csv
import json
import math
from pathlib import Path

import pytest

from goodvibes.kinetics import (
    barrier_for_rate, energy_span, eyring_rate, half_life, mikimo_rows, rate_ratio, step_table,
    write_mikimo_csv,
)

KB, H, R = 1.3806488e-23, 6.62606957e-34, 8.3144621
T = 298.15
KCAL = 4184.0


def _eyring(dg_kcal, temp=T):
    return KB * temp / H * math.exp(-dg_kcal * KCAL / (R * temp))


def test_eyring_rate_and_its_inverse():
    assert eyring_rate(20.0) == pytest.approx(_eyring(20.0), rel=1e-6)
    assert eyring_rate(83.68, units="kJ/mol") == pytest.approx(_eyring(20.0), rel=1e-6)
    assert eyring_rate(20.0, kappa=0.5) == pytest.approx(0.5 * _eyring(20.0), rel=1e-6)
    assert barrier_for_rate(eyring_rate(17.3, 350.0), 350.0) == pytest.approx(17.3)
    assert rate_ratio(15.0, 16.0) == pytest.approx(_eyring(15.0) / _eyring(16.0), rel=1e-6)
    assert half_life(2.0) == pytest.approx(math.log(2) / 2.0)
    with pytest.raises(ValueError):
        eyring_rate(10.0, 0.0)
    with pytest.raises(ValueError):
        barrier_for_rate(0.0)


def _brute_tof(levels, ts, dGr, temp=T):
    """The energy-span TOF written out term by term."""
    items = list(levels.items())
    num = math.exp(-dGr * KCAL / (R * temp)) - 1
    den = 0.0
    for i, (pi, ti) in enumerate(items):
        if pi not in ts:
            continue
        for j, (pj, ij) in enumerate(items):
            if pj in ts:
                continue
            d = dGr if i > j else 0.0
            den += math.exp((ti - ij - d) * KCAL / (R * temp))
    return KB * temp / H * num / den


def test_a_single_step_cycle_has_the_barrier_as_its_span():
    es = energy_span({"I": 0.0, "TS": 15.0, "P": -10.0}, ["TS"])
    assert (es.tdts, es.tdi, es.tdts_after_tdi) == ("TS", "I", True)
    assert es.span == pytest.approx(15.0) and es.reaction_energy == pytest.approx(-10.0)
    assert es.tof == pytest.approx(_brute_tof({"I": 0.0, "TS": 15.0}, {"TS"}, -10.0), rel=1e-9)
    assert es.tof_span == pytest.approx(_eyring(15.0), rel=1e-6)
    assert es.control == pytest.approx({"TS": 1.0, "I": 1.0})


def test_a_tdts_before_the_tdi_adds_the_reaction_energy():
    cycle = {"I0": 0.0, "TS1": 15.0, "I1": -10.0, "TS2": 0.0, "P": -5.0}
    es = energy_span(cycle, ["TS1", "TS2"])
    # the next turnover crosses TS1 at 15 - 5 = 10 above the deep I1 at -10
    assert (es.tdts, es.tdi, es.tdts_after_tdi) == ("TS1", "I1", False)
    assert es.span == pytest.approx(15.0 - (-10.0) + (-5.0))
    states = {k: v for k, v in cycle.items() if k != "P"}
    assert es.tof == pytest.approx(_brute_tof(states, {"TS1", "TS2"}, -5.0), rel=1e-9)
    ts_share = es.control["TS1"] + es.control["TS2"]
    int_share = es.control["I0"] + es.control["I1"]
    assert ts_share == pytest.approx(1.0) and int_share == pytest.approx(1.0)
    assert max(es.control, key=lambda k: es.control[k] if k.startswith("TS") else -1) == "TS1"
    assert "TDTS before TDI" in str(es)


def test_an_explicit_reaction_energy_and_an_endergonic_cycle():
    es = energy_span({"I0": 0.0, "TS1": 12.0}, ["TS1"], reaction_energy=-3.0)
    assert es.span == pytest.approx(12.0)
    uphill = energy_span({"I0": 0.0, "TS1": 12.0, "P": 2.0}, ["TS1"])
    assert uphill.tof < 0                                 # no net turnover in this direction
    with pytest.raises(ValueError, match="at least one transition state"):
        energy_span({"I0": 0.0, "I1": -2.0, "P": -3.0}, [])
    with pytest.raises(ValueError, match="at least an intermediate"):
        energy_span({"I0": 0.0, "P": -3.0}, [])


def test_the_step_table():
    rows = step_table({"R": 0.0, "TS1": 10.0, "I": -15.0, "TS2": 0.0, "P": -20.0}, ["TS1", "TS2"])
    assert [(r["from"], r["ts"], r["to"]) for r in rows] == [("R", "TS1", "I"), ("I", "TS2", "P")]
    assert [r["barrier"] for r in rows] == [10.0, 15.0]
    assert rows[1]["barrier_from_lowest"] == 15.0 and rows[1]["step_energy"] == -5.0
    assert rows[1]["k"] == pytest.approx(_eyring(15.0), rel=1e-6)
    deep = step_table({"R": 0.0, "I1": -8.0, "I2": -2.0, "TS": 10.0}, ["TS"])
    assert (deep[0]["from"], deep[0]["barrier"], deep[0]["barrier_from_lowest"]) == ("I2", 12.0, 18.0)
    assert deep[0]["to"] is None and deep[0]["step_energy"] is None


def test_mikimo_rows_rename_points_and_convert_units(tmp_path):
    kj = {"cat": 0.0, "TSa": 41.84, "Pd-int": -20.92, "TSb": 20.92, "product": -41.84}
    header, rows, names = mikimo_rows({"path1": kj}, ["TSa", "TSb"], units="kJ/mol")
    assert header == ["INT0", "TS1", "INT1", "TS2", "Prod"]
    assert names == {"cat": "INT0", "TSa": "TS1", "Pd-int": "INT1", "TSb": "TS2", "product": "Prod"}
    assert rows[0][0] == "path1" and rows[0][1:] == pytest.approx([0.0, 10.0, -5.0, 5.0, -10.0])
    out = tmp_path / "reaction_data.csv"
    write_mikimo_csv(out, {"a": kj, "b": {k: v * 2 for k, v in kj.items()}}, ["TSa", "TSb"], units="kJ/mol")
    lines = list(csv.reader(out.open(encoding="utf-8")))
    assert lines[0] == ["", "INT0", "TS1", "INT1", "TS2", "Prod"]
    assert lines[2][0] == "b" and float(lines[2][2]) == pytest.approx(20.0)
    with pytest.raises(ValueError, match="same sequence"):
        mikimo_rows({"a": kj, "b": {"x": 0.0, "y": 1.0}}, ["TSa", "TSb"])
    with pytest.raises(ValueError, match="is a transition state"):
        mikimo_rows({"a": {"I0": 0.0, "TS1": 10.0}}, ["TS1"])     # mikimo would read the TS as the product


# -- documents and the command line -----------------------------------------------------

KIT = Path(__file__).resolve().parent / "profile_conformance"
yaml = pytest.importorskip("yaml")

CYCLE = {
    "schema": "reaction-profile/1.0",
    "units": "kcal/mol",
    "points": {"I0": {"role": "reactant"}, "TS1": {"role": "ts"}, "I1": {"role": "minimum"},
               "TS2": {"role": "ts"}, "P": {"role": "product"}},
    "pathways": {"cycle": ["I0", "TS1", "I1", "TS2", "P"]},
    "series": [{"id": "G", "quantity": "gibbs", "temperature": 350.0, "source": "declared",
                "levels": {"cycle": {"I0": 0.0, "TS1": 15.0, "I1": -10.0, "TS2": 0.0, "P": -5.0}}}],
}


def test_profile_hooks():
    from goodvibes.profile import Profile, ProfileError
    prof = Profile.from_dict(CYCLE)
    rows = prof.step_table()
    assert [r["ts"] for r in rows] == ["TS1", "TS2"] and rows[0]["pathway"] == "cycle"
    assert rows[0]["temperature"] == 350.0 and rows[0]["k"] == pytest.approx(_eyring(15.0, 350.0), rel=1e-6)
    es = prof.energy_span()
    assert (es.tdts, es.tdi, es.temperature) == ("TS1", "I1", 350.0) and es.span == pytest.approx(20.0)
    with pytest.raises(ProfileError, match="no pathway"):
        prof.step_table("nope")
    with pytest.raises(ProfileError, match="series 'G' has no levels on pathway"):
        Profile.from_dict({**CYCLE, "pathways": {**CYCLE["pathways"], "other": ["I0", "TS1"]}}).step_table("other", "G")


def test_goodvibes_profile_kinetics(tmp_path, capsys):
    from goodvibes.profile_cli import main as gvp
    src = tmp_path / "cycle.yaml"
    src.write_text(yaml.safe_dump(CYCLE), encoding="utf-8")
    assert gvp(["kinetics", str(src), "--span"]) == 0
    out = capsys.readouterr().out
    assert "I0 -> TS1 -> I1: barrier 15.00" in out and "energy span 20.00 kcal/mol (TDTS TS1, TDI I1" in out
    mik = tmp_path / "reaction_data.csv"
    assert gvp(["kinetics", str(src), "--mikimo", str(mik), "--json"]) == 0
    captured = capsys.readouterr()
    assert "I0 = INT0" in captured.err                    # the status line stays out of the JSON
    payload = json.loads(captured.out)
    assert [s["ts"] for s in payload["steps"]] == ["TS1", "TS2"]
    assert mik.read_text(encoding="utf-8").splitlines()[0] == ",INT0,TS1,INT1,TS2,Prod"
    assert gvp(["kinetics", str(src), "-o", str(tmp_path / "steps.csv")]) == 0
    assert (tmp_path / "steps.csv").read_text(encoding="utf-8").startswith("pathway,series,ts,from,to,barrier")


def _two_series(**extra):
    """CYCLE with an electronic-energy series listed before the free-energy one."""
    e = {"id": "E", "quantity": "electronic", "source": "declared",
         "levels": {"cycle": {"I0": 0.0, "TS1": 30.0, "I1": -20.0, "TS2": 5.0, "P": -8.0}}}
    return {**CYCLE, "series": [e, *CYCLE["series"]], **extra}


def test_rates_come_from_a_free_energy_series():
    from goodvibes.profile import Profile, ProfileError, ProfileWarning
    prof = Profile.from_dict(_two_series())
    assert prof.step_table()[0]["series"] == "G"         # not the first series, which holds ΔE
    with pytest.warns(ProfileWarning, match="not a free energy"):
        assert prof.energy_span(series="E").span == pytest.approx(50.0 - 8.0)
    only_e = Profile.from_dict({**CYCLE, "series": _two_series()["series"][:1]})
    with pytest.raises(ProfileError, match="no free-energy series"):
        only_e.step_table()


def test_levels_are_converted_and_must_be_complete(tmp_path):
    from goodvibes.profile import Profile, ProfileError
    prof = Profile.from_dict(CYCLE)
    s = prof.series[0]                          # a series built in code may carry its own units
    s.units, s.levels = "kJ/mol", {"cycle": {k: v * 4.184 for k, v in s.levels["cycle"].items()}}
    assert prof.units == "kcal/mol" and prof.energy_span().span == pytest.approx(20.0)
    assert prof.step_table()[0]["barrier"] == pytest.approx(15.0)
    gap = dict(CYCLE["series"][0], levels={"cycle": {**CYCLE["series"][0]["levels"]["cycle"], "P": None}})
    incomplete = Profile.from_dict({**CYCLE, "series": [gap]})
    with pytest.raises(ProfileError, match="no level for P"):
        incomplete.energy_span()
    with pytest.raises(ProfileError, match="no level for P"):
        incomplete.write_mikimo(tmp_path / "x.csv")


def test_a_mikimo_export_uses_one_series(tmp_path):
    from goodvibes.profile import Profile, ProfileError
    doc = {**CYCLE, "pathways": {**CYCLE["pathways"], "alt": ["I0", "TS1", "I1", "TS2", "P"]}}
    g, = doc["series"]
    other = {"id": "G2", "quantity": "gibbs", "temperature": 298.15, "source": "declared",
             "levels": {"alt": {"I0": 0.0, "TS1": 1.0, "I1": 0.0, "TS2": 1.0, "P": 0.0}}}
    # G covers only 'cycle' and G2 only 'alt': neither can export both, and they are not mixed
    with pytest.raises(ProfileError, match="no series has levels on pathways 'cycle', 'alt'"):
        Profile.from_dict({**doc, "series": [g, other]}).write_mikimo(tmp_path / "x.csv")
    both = dict(g, levels={"cycle": g["levels"]["cycle"], "alt": g["levels"]["cycle"]})
    Profile.from_dict({**doc, "series": [other, both]}).write_mikimo(tmp_path / "y.csv")
    assert [r[0] for r in csv.reader((tmp_path / "y.csv").open(encoding="utf-8"))][1:] == ["cycle", "alt"]
