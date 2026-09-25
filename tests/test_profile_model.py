"""The 4.6 profile model: ComputedEntry, ConformerSet rollups, point roles,
edges, series and the public Rich table builder (M1 of the direction plan).
"""
import math
from types import SimpleNamespace

import pytest

from conftest import g16path
from goodvibes import compute_thermo
from goodvibes.constants import GAS_CONSTANT, J_TO_AU, KCAL_TO_AU, hartree_factor
from goodvibes.pes_model import (
    ComputedEntry, ConformerSet, Edge, PESOptions, PESResult, Pathway, Point, Series,
    merge_point_order, _bbe_to_vector,
)

WATER_A = g16path("01a_water_hf_freq.log")
WATER_A2 = g16path("01c_water_hf_freq_isotopes.log")
WATER_B = g16path("01b_water_hf_freq_scaled.log")


def _stub(g):
    return SimpleNamespace(
        scf_energy=g - 0.001, zpe=0.005, enthalpy=g + 0.005, qh_enthalpy=g + 0.005,
        entropy=1.6e-5, qh_entropy=1.6e-5, gibbs_free_energy=g, qh_gibbs_free_energy=g,
        sp_energy=None,
    )


@pytest.fixture(scope="module")
def water_results():
    return [compute_thermo(f, temperature=298.15) for f in (WATER_A, WATER_A2)]


@pytest.fixture
def water_set(water_results):
    return ConformerSet.from_results("A", water_results)


@pytest.fixture
def real_result(water_results):
    """A/B pathway from real files: A has two conformers, B one."""
    a = ConformerSet.from_results("A", water_results)
    b = ConformerSet.from_results("B", [compute_thermo(WATER_B, temperature=298.15)])
    species = {"A": a, "B": b}
    path = Pathway(name="rxn", points=[Point.from_label("A", species),
                                       Point.from_label("B", species, role="ts", display="B‡")])
    return PESResult(pathways=[path], options=PESOptions(units="kcal/mol", gconf=False, QH=False),
                     temperatures=[298.15, 400.0])


# ---------------------------------------------------------------------------
# ComputedEntry
# ---------------------------------------------------------------------------

def test_from_options_stores_the_resolved_options(water_results):
    opts = water_results[0].bbe.options
    assert opts.temperature == 298.15
    assert opts.freq_scale_factor is not None      # scale factors resolved, not None
    assert opts.concentration is None              # gas phase follows T


def test_computed_entry_recomputes_at_another_temperature(water_results):
    entry = ComputedEntry.from_result(water_results[0])
    assert entry.base_temperature == 298.15
    direct = compute_thermo(WATER_A, temperature=400.0)
    v = entry.thermo(400.0)
    assert v.qh_gibbs == pytest.approx(direct.qh_gibbs_free_energy, abs=1e-12)
    assert v.entropy == pytest.approx(direct.entropy, abs=1e-15)
    # the base temperature is the original object, never recomputed
    assert entry.bbe() is water_results[0].bbe
    assert entry.thermo() == _bbe_to_vector(water_results[0].bbe)


def test_computed_entry_memoises_per_temperature(water_results):
    entry = ComputedEntry.from_result(water_results[0])
    b1 = entry.bbe(350.0)
    assert entry.bbe(350.0) is b1
    assert len(entry._cache) == 2


def test_computed_entry_from_stub_is_none():
    assert ComputedEntry.from_bbe(_stub(-1.0)) is None
    with pytest.raises(ValueError, match="QCData"):
        ComputedEntry.from_result(SimpleNamespace(bbe=_stub(-1.0), file="x", name="x"))


# ---------------------------------------------------------------------------
# ConformerSet
# ---------------------------------------------------------------------------

def test_conformerset_auto_builds_entries_from_real_bbes(water_results):
    cs = ConformerSet("A", [r.file for r in water_results], [r.bbe for r in water_results])
    assert cs.recomputable and cs.base_temperature == 298.15
    stub = ConformerSet("S", ["a", "b"], [_stub(-1.0), _stub(-2.0)])
    assert not stub.recomputable and stub.base_temperature is None
    # a non-recomputable set returns its base values whatever T is asked for
    assert stub.vectors(500.0) == stub.vectors()


def test_vectors_at_temperature_match_direct_evaluation(water_set):
    direct = [compute_thermo(f, temperature=350.0) for f in (WATER_A, WATER_A2)]
    for v, d in zip(water_set.vectors(350.0), direct):
        assert v.qh_gibbs == pytest.approx(d.qh_gibbs_free_energy, abs=1e-12)


def test_populations_sum_to_one_and_favour_the_lowest(water_set):
    p = water_set.populations(298.15)
    assert sum(p) == pytest.approx(1.0)
    lowest = water_set.lowest_index()
    assert p[lowest] == max(p)


def test_ensemble_free_energy_equals_gconf_qh_gibbs(water_set):
    for T in (298.15, 400.0):
        assert water_set.ensemble_free_energy(T) == pytest.approx(
            water_set.gconf_corrected(T).qh_gibbs, abs=1e-12)
    # and equals −RT ln Σ exp(−Gᵢ/RT) spelled out
    T = 298.15
    gs = [v.qh_gibbs for v in water_set.vectors(T)]
    rt = GAS_CONSTANT * T / J_TO_AU
    g0 = min(gs)
    assert water_set.ensemble_free_energy(T) == pytest.approx(
        g0 - rt * math.log(sum(math.exp(-(g - g0) / rt) for g in gs)))


def test_s_conf_of_two_degenerate_conformers_is_r_ln2():
    cs = ConformerSet("S", ["a", "b"], [_stub(-1.0), _stub(-1.0)])
    assert cs.s_conf(298.15) == pytest.approx(GAS_CONSTANT * math.log(2) / J_TO_AU)
    assert cs.populations(298.15) == pytest.approx([0.5, 0.5])


def test_weight_by_accepts_aliases_and_changes_the_weighting():
    a, b = _stub(-1.0), _stub(-1.001)
    a.scf_energy = -1.010          # a is lowest in E, b is lowest in qh-G
    cs_g = ConformerSet("S", ["a", "b"], [a, b], weight_by="qh-G")
    cs_e = ConformerSet("S", ["a", "b"], [a, b], weight_by="E")
    assert cs_g.weight_by == "qh_gibbs" and cs_e.weight_by == "electronic"
    assert cs_g.lowest_index() == 1 and cs_e.lowest_index() == 0
    assert cs_g.populations(298.15)[1] > 0.5 and cs_e.populations(298.15)[0] > 0.5
    with pytest.raises(ValueError, match="unknown quantity"):
        ConformerSet("S", ["a"], [a], weight_by="banana")


def test_dedup_drops_a_duplicate_conformer():
    r1 = compute_thermo(WATER_A)
    r2 = compute_thermo(WATER_A)             # the same file twice: exact duplicate
    cs = ConformerSet("A", ["a.log", "b.log"], [r1.bbe, r2.bbe])
    kept = cs.dedup()
    assert kept.files == ["a.log"] and kept.recomputable   # the later copy is the flagged duplicate
    assert kept.dedup() is kept


def test_from_results_rejects_empty():
    with pytest.raises(ValueError, match="no results"):
        ConformerSet.from_results("A", [])


# ---------------------------------------------------------------------------
# Point roles / display, Edge, Pathway edges and levels
# ---------------------------------------------------------------------------

def test_point_role_normalisation_and_display():
    cs = ConformerSet("A", ["a"], [_stub(-1.0)])
    assert Point("A", [(1, cs)]).role == "minimum"
    assert Point("A", [(1, cs)], role="TS").is_ts
    assert Point("A", [(1, cs)], role="intermediate").role == "minimum"
    p = Point("A + B", [(1, cs)], display="A‡")
    assert p.display_label == "A‡" and p.id == "A + B"
    assert Point("A", [(1, cs)]).display_label == "A"
    with pytest.raises(ValueError, match="unknown point role"):
        Point("A", [(1, cs)], role="saddle")


def test_edge_kinds_and_default_pathway_edges():
    cs = ConformerSet("A", ["a"], [_stub(-1.0)])
    pts = [Point(l, [(1, cs)]) for l in ("R", "TS", "P")]
    path = Pathway("rxn", pts)
    assert [(e.src, e.dst, e.kind) for e in path.edges] == [("R", "TS", "step"), ("TS", "P", "step")]
    custom = path.with_edges([("R", "TS", "barrierless"), Edge("TS", "P", "none")])
    assert custom.edge_kind("R", "TS") == "barrierless" and custom.edge_kind("TS", "P") == "none"
    assert custom.edge_kind("P", "R") is None
    with pytest.raises(ValueError, match="unknown edge kind"):
        Edge("R", "TS", "teleport")
    with pytest.raises(ValueError, match="not on the pathway"):
        path.with_edges([("R", "X")])


def test_pathway_levels_match_relative(real_result):
    path = real_result.pathways[0]
    for T in real_result.temperatures:
        rels = path.relative(T, gconf=False, QH=False)
        lv = path.levels(T, "qh_gibbs", gconf=False, QH=False)
        assert list(lv) == ["A", "B"]
        assert lv["A"] == 0.0
        assert lv["B"] == rels[1].qh_gibbs
        assert path.levels(T, "entropy", gconf=False, QH=False)["B"] == pytest.approx(T * rels[1].entropy)
    # a quantity that is unavailable is None, never 0
    stub = ConformerSet("S", ["s"], [_stub(-1.0)])
    assert Pathway("p", [Point("S", [(1, stub)])]).levels(298.15, "spc")["S"] is None


def test_pathway_point_lookup(real_result):
    path = real_result.pathways[0]
    assert path.point("B").display_label == "B‡"
    with pytest.raises(KeyError):
        path.point("Z")


# ---------------------------------------------------------------------------
# Series / PESResult
# ---------------------------------------------------------------------------

def test_series_validation():
    with pytest.raises(ValueError, match="needs its levels"):
        Series(id="lit", label="lit", declared=True)
    with pytest.raises(ValueError, match="cannot carry declared levels"):
        Series(id="x", label="x", levels={"rxn": {"A": 0.0}})
    s = Series(id="g", label="G", quantity="G")
    assert s.quantity == "gibbs"


def test_computed_series_evaluates_through_pathway_levels(real_result):
    path = real_result.pathways[0]
    s = Series(id="g400", label="qh-G 400 K", quantity="qh_gibbs", temperature=400.0)
    got = s.evaluate(real_result, path)
    expected = path.levels(400.0, "qh_gibbs", **real_result.options.rollup_kw)["B"] * KCAL_TO_AU
    assert got["B"] == pytest.approx(expected)
    # None temperature → the result's first temperature
    s0 = Series(id="g", label="qh-G", quantity="qh_gibbs")
    assert s0.evaluate(real_result, path)["B"] == pytest.approx(
        path.levels(298.15, "qh_gibbs", **real_result.options.rollup_kw)["B"] * KCAL_TO_AU)


def test_declared_series_is_converted_to_the_result_units(real_result):
    lit = Series.declared_from("lit", "lit.", {"rxn": {"A": 0.0, "B": 41.84}}, units="kJ/mol")
    got = lit.evaluate(real_result, real_result.pathways[0])
    assert got == {"A": 0.0, "B": pytest.approx(41.84 * hartree_factor("kcal/mol") / hartree_factor("kJ/mol"))}
    # a point missing from the declared levels is absent, not zero
    partial = Series.declared_from("p", "p", {"rxn": {"B": 1.0}})
    assert list(partial.evaluate(real_result, real_result.pathways[0])) == ["B"]
    # a pathway the series does not declare evaluates to nothing
    assert Series.declared_from("q", "q", {"other": {"A": 0.0}}).evaluate(real_result, real_result.pathways[0]) == {}


def test_pesresult_default_series_merged_order_and_levels(real_result):
    assert real_result.temperature == 298.15
    ds = real_result.default_series()
    assert [s.temperature for s in ds] == [298.15, 400.0]
    assert ds[0].label == "Δqh-G(T) 298.15 K" and ds[1].id == "qh_gibbs@400K"
    assert real_result.default_series("E", [300.0])[0].label == "ΔE"
    assert real_result.merged_order() == ["A", "B"]
    real_result.order = ["B", "A"]
    assert real_result.merged_order() == ["B", "A"]
    lv = real_result.levels()
    assert set(lv) == {"qh_gibbs@298.15K", "qh_gibbs@400K"}
    assert lv["qh_gibbs@298.15K"]["rxn"]["B"] != lv["qh_gibbs@400K"]["rxn"]["B"]
    assert real_result.recomputable
    assert real_result.pathway("rxn") is real_result.pathways[0] and real_result.pathway(0) is real_result.pathways[0]
    with pytest.raises(KeyError):
        real_result.pathway("nope")


def test_stub_result_is_not_recomputable():
    cs = ConformerSet("A", ["a"], [_stub(-1.0)])
    res = PESResult(pathways=[Pathway("r", [Point("A", [(1, cs)])])], options=PESOptions())
    assert not res.recomputable


@pytest.mark.parametrize("sequences, expected", [
    ([["R", "Int1", "TS1", "Int2", "TS2", "P"], ["R", "TS1", "P"]], ["R", "Int1", "TS1", "Int2", "TS2", "P"]),
    ([["R", "TS1", "P"], ["R", "Int1", "TS1", "Int2", "TS2", "P"]], ["R", "Int1", "TS1", "Int2", "TS2", "P"]),
    ([["R", "TS_R", "P_R"], ["R", "TS_S", "P_S"]], ["R", "TS_R", "P_R", "TS_S", "P_S"]),
    ([["A", "B"], ["A", "B"]], ["A", "B"]),
    ([], []),
])
def test_merge_point_order(sequences, expected):
    assert merge_point_order(sequences) == expected


# ---------------------------------------------------------------------------
# Public Rich table builder
# ---------------------------------------------------------------------------

def test_pes_tables_returns_rich_tables_without_logging_setup(real_result):
    from rich.table import Table
    from goodvibes.output import pes_tables
    tables = pes_tables(real_result)
    assert len(tables) == 1 and isinstance(tables[0], Table)
    assert "RXN: rxn" in tables[0].title and "T = 298.15 K" in tables[0].title
    assert tables[0].row_count == 2
    t400 = pes_tables(real_result, temperature=400.0, conc=1.0)[0]
    assert "T = 400 K" in t400.title and "c = 1 mol/L" in t400.title
