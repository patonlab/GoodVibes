"""Profile.diff and goodvibes-profile diff."""
import copy
import json
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from goodvibes.profile import Profile  # noqa: E402
from goodvibes.profile_cli import main as gvp  # noqa: E402

KIT = Path(__file__).resolve().parent / "profile_conformance"
SRC = KIT / "valid" / "05_selectivity.yaml"
DOC = yaml.safe_load(SRC.read_text(encoding="utf-8"))


def _prof(d):
    return Profile.from_dict(d)


def _changed(**edits):
    d = copy.deepcopy(DOC)
    for fn in edits.values():
        fn(d)
    return d


def test_a_document_equals_itself_and_its_round_trip():
    a = _prof(DOC)
    assert a.diff(a).identical and not a.diff(a)
    assert a.diff(_prof(json.loads(json.dumps(a.to_dict())))).identical
    assert str(a.diff(a)) == "identical within 0.01 kcal/mol"


def test_levels_are_compared_within_the_tolerance_in_common_units():
    b = _changed(level=lambda d: d["series"][0]["levels"]["S"].update(TS_S=16.25))
    diff = _prof(DOC).diff(_prof(b))
    (d,) = diff.differences
    assert (d.kind, d.where) == ("level", "series.G.levels.S.TS_S")
    assert d.delta == pytest.approx(0.05)
    assert _prof(DOC).diff(_prof(b), tolerance=0.1).identical
    kj = _prof(DOC).diff(_prof(b), units="kJ/mol")
    assert kj.differences[0].delta == pytest.approx(0.05 * 4.184) and kj.units == "kJ/mol"


def test_a_document_in_other_units_is_converted():
    b = copy.deepcopy(DOC)
    b["units"] = "kJ/mol"
    for path in b["series"][0]["levels"].values():
        for p in path:
            path[p] *= 4.184
    diff = _prof(DOC).diff(_prof(b))
    assert [d.where for d in diff.differences] == ["units"]


def test_structure_series_and_blocks():
    def edit(d):
        d["points"]["TS_R"]["role"] = "minimum"
        d["points"]["X"] = {"role": "product"}
        d["pathways"]["x"] = ["Int", "X"]
        d["pathways"]["S"] = ["Int", "TS_S"]
        d["series"][0]["temperature"] = 310.0
        del d["series"][0]["levels"]["S"]["P_S"]
        d["series"].append({"id": "lit", "quantity": "gibbs", "source": "declared",
                            "levels": {"R": {"Int": 0.0, "TS_R": 14.0}}})
        d["selectivity"][0]["kind"] = "diastereo"
        d["annotations"] = [{"pathway": "R", "from": "Int", "to": "TS_R"}]
    diff = _prof(DOC).diff(_prof(_changed(e=edit)))
    got = {(d.kind, d.where) for d in diff.differences}
    assert got == {
        ("point", "points.TS_R.role"), ("point", "points.X"), ("pathway", "pathways.S.points"),
        ("pathway", "pathways.x"), ("series", "series.G.temperature"), ("level", "series.G.levels.S.P_S"),
        ("series", "series.lit"), ("selectivity", "selectivity.er.kind"), ("annotation", "annotations"),
    }
    added = next(d for d in diff.differences if d.where == "series.lit")
    assert added.a is None and str(added).startswith("+ series.lit")
    removed = next(d for d in diff.differences if d.where == "series.G.levels.S.P_S")
    assert removed.b is None and str(removed).startswith("- series.G.levels.S.P_S: -4.00")
    only_lit = _prof(DOC).diff(_prof(_changed(e=edit)), series=["lit"])
    assert "series.G.temperature" not in {d.where for d in only_lit.differences}


def test_the_command_exits_one_when_documents_differ(tmp_path, capsys):
    b = tmp_path / "b.yaml"
    b.write_text(yaml.safe_dump(_changed(level=lambda d: d["series"][0]["levels"]["R"].update(TS_R=15.5))),
                 encoding="utf-8")
    assert gvp(["diff", str(SRC), str(SRC)]) == 0
    assert "identical within 0.01 kcal/mol" in capsys.readouterr().out
    assert gvp(["diff", str(SRC), str(b)]) == 1
    out = capsys.readouterr().out
    assert "~ series.G.levels.R.TS_R: 15.00 -> 15.50 (+0.50)" in out and "1 difference" in out
    assert gvp(["diff", str(SRC), str(b), "--tolerance", "1"]) == 0
    capsys.readouterr()
    assert gvp(["diff", str(SRC), str(b), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["identical"] is False and payload["differences"][0]["delta"] == pytest.approx(0.5)
