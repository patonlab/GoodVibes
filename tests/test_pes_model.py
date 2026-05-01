"""Tests for goodvibes.pes_model — pure data + arithmetic, no I/O.

Three layers of tests:
  1. ThermoVector arithmetic (add, sub, scalar mul, sp_energy None propagation).
  2. ConformerSet rollups (single, Boltzmann-weighted, gconf-corrected).
  3. Point + Pathway integration (stoichiometry, relative-to-zero).
"""
import math
from types import SimpleNamespace

import pytest

from goodvibes.pes_model import (
    ConformerSet, GAS_CONSTANT, J_TO_AU, KCAL_TO_AU, PESOptions, PESResult,
    Pathway, Point, ThermoVector, parse_point_label,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _bbe(g, h=None, s=None, qh_h=None, qh_s=None, scf=None, zpe=0.005, sp=None):
    """Make a calc_bbe-shaped stub. Defaults reflect typical small-molecule
    magnitudes; only `g` is required."""
    if h is None:
        h = g + 0.005           # H = G + T·S, just need it differ from G
    if s is None:
        s = (h - g) / 298.15    # back out a self-consistent S
    if qh_h is None:
        qh_h = h
    if qh_s is None:
        qh_s = s
    if scf is None:
        scf = g - 0.001
    return SimpleNamespace(
        scf_energy=scf, zpe=zpe,
        enthalpy=h, qh_enthalpy=qh_h,
        entropy=s, qh_entropy=qh_s,
        gibbs_free_energy=g, qh_gibbs_free_energy=g,
        sp_energy=sp,
    )


def _vec(scf=1.0, zpe=0.5, h=2.0, qh_h=2.1, s=0.001, qh_s=0.0011,
         g=1.5, qh_g=1.6, sp=None):
    return ThermoVector(
        scf_energy=scf, zpe=zpe, enthalpy=h, qh_enthalpy=qh_h,
        entropy=s, qh_entropy=qh_s, gibbs=g, qh_gibbs=qh_g, sp_energy=sp,
    )


# ---------------------------------------------------------------------------
# ThermoVector arithmetic
# ---------------------------------------------------------------------------

def test_add_scalar_fields():
    a = _vec(scf=1.0, h=2.0, g=3.0)
    b = _vec(scf=10.0, h=20.0, g=30.0)
    s = a + b
    assert s.scf_energy == 11.0
    assert s.enthalpy == 22.0
    assert s.gibbs == 33.0


def test_sub_scalar_fields():
    a = _vec(scf=10.0, h=20.0)
    b = _vec(scf=1.0, h=2.0)
    d = a - b
    assert d.scf_energy == 9.0
    assert d.enthalpy == 18.0


def test_mul_scalar_left_and_right():
    a = _vec(scf=2.0, h=4.0)
    assert (3 * a).scf_energy == 6.0
    assert (a * 3).enthalpy == 12.0


def test_sp_energy_propagates_when_both_present():
    a = _vec(sp=1.0)
    b = _vec(sp=2.0)
    assert (a + b).sp_energy == 3.0
    assert (a - b).sp_energy == -1.0
    assert (2 * a).sp_energy == 2.0


def test_sp_energy_none_propagates():
    """If either operand has sp_energy=None, the result has sp_energy=None.
    Avoids silent zero-substitution that would mask a missing --spc value."""
    a = _vec(sp=None)
    b = _vec(sp=2.0)
    assert (a + b).sp_energy is None
    assert (a - b).sp_energy is None
    assert (b - a).sp_energy is None


def test_zero_factory_has_sp_None_by_default():
    z = ThermoVector.zero()
    assert z.sp_energy is None
    z2 = ThermoVector.zero(with_sp=True)
    assert z2.sp_energy == 0.0


def test_arithmetic_is_immutable():
    a = _vec(scf=1.0)
    b = _vec(scf=2.0)
    _ = a + b
    assert a.scf_energy == 1.0  # frozen dataclass — original unchanged


# ---------------------------------------------------------------------------
# ConformerSet
# ---------------------------------------------------------------------------

def test_conformer_set_requires_nonempty():
    with pytest.raises(ValueError, match="no conformers"):
        ConformerSet(name="x", files=[], bbes=[])


def test_conformer_set_files_bbes_length_match():
    with pytest.raises(ValueError, match="length mismatch"):
        ConformerSet(name="x", files=["a.log"], bbes=[_bbe(-100), _bbe(-101)])


def test_single_conformer_passes_through():
    """For a single conformer, all three rollups return the same vector."""
    b = _bbe(-100.5, h=-100.0, s=0.05, scf=-100.6)
    cs = ConformerSet(name="x", files=["x.log"], bbes=[b])
    v = cs.lowest_conformer()
    assert v.gibbs == -100.5
    assert cs.boltzmann_weighted(298.15) == v
    assert cs.gconf_corrected(298.15) == v


def test_boltzmann_two_conformers_sum_to_one():
    """Boltzmann probabilities must sum to 1."""
    b1 = _bbe(-100.000)
    b2 = _bbe(-99.999)        # ~6.3 kJ/mol higher
    cs = ConformerSet(name="x", files=["1.log", "2.log"], bbes=[b1, b2])
    # Cross-check by recomputing the probabilities manually.
    g_min = min(b1.qh_gibbs_free_energy, b2.qh_gibbs_free_energy)
    T = 298.15
    w1 = math.exp(-(b1.qh_gibbs_free_energy - g_min) * J_TO_AU / GAS_CONSTANT / T)
    w2 = math.exp(-(b2.qh_gibbs_free_energy - g_min) * J_TO_AU / GAS_CONSTANT / T)
    p1, p2 = w1 / (w1 + w2), w2 / (w1 + w2)
    expected_scf = p1 * b1.scf_energy + p2 * b2.scf_energy
    bw = cs.boltzmann_weighted(T)
    assert math.isclose(bw.scf_energy, expected_scf, rel_tol=1e-12)


def test_boltzmann_degenerate_conformers_average():
    """Two identical conformers → equal weights → arithmetic mean."""
    b1 = _bbe(-100.0, scf=-100.5)
    b2 = _bbe(-100.0, scf=-200.5)
    cs = ConformerSet(name="x", files=["1.log", "2.log"], bbes=[b1, b2])
    bw = cs.boltzmann_weighted(298.15)
    assert math.isclose(bw.scf_energy, -150.5, rel_tol=1e-12)


def test_lowest_conformer_picks_min_qh_gibbs():
    b1 = _bbe(-100.000, scf=-100.5)
    b2 = _bbe(-100.001, scf=-200.5)
    cs = ConformerSet(name="x", files=["1.log", "2.log"], bbes=[b1, b2])
    lo = cs.lowest_conformer()
    assert lo.scf_energy == -200.5         # the one with lower qh-G


def test_gconf_correction_adds_mixing_entropy():
    """gconf-corrected G should be lower than the plain Boltzmann G when
    multiple conformers contribute, by exactly T·(−R Σ pᵢ ln pᵢ)."""
    b1 = _bbe(-100.000)
    b2 = _bbe(-100.000)        # equal → max mixing entropy
    cs = ConformerSet(name="x", files=["1.log", "2.log"], bbes=[b1, b2])
    T = 298.15
    boltz = cs.boltzmann_weighted(T)
    gconf = cs.gconf_corrected(T, QH=True)
    expected_mix_S = -GAS_CONSTANT / J_TO_AU * (
        0.5 * math.log(0.5) + 0.5 * math.log(0.5)
    )
    assert math.isclose(gconf.qh_entropy, boltz.qh_entropy + expected_mix_S, rel_tol=1e-12)
    # G = H − T·S  → gconf G is more negative by T·mixS
    assert math.isclose(gconf.qh_gibbs, boltz.qh_gibbs - T * expected_mix_S, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Point — stoichiometry parsing + thermo
# ---------------------------------------------------------------------------

def test_parse_simple_label():
    assert parse_point_label("A") == [(1, "A")]


def test_parse_sum():
    assert parse_point_label("A + B + C") == [(1, "A"), (1, "B"), (1, "C")]


def test_parse_coefficient():
    assert parse_point_label("2*A") == [(2, "A")]
    assert parse_point_label("2 * A") == [(2, "A")]
    assert parse_point_label("2 * A + B") == [(2, "A"), (1, "B")]
    assert parse_point_label("A + 3*B + C") == [(1, "A"), (3, "B"), (1, "C")]


def test_parse_whitespace_tolerant():
    assert parse_point_label("  A  +   B  ") == [(1, "A"), (1, "B")]


def test_parse_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        parse_point_label("")
    with pytest.raises(ValueError, match="empty"):
        parse_point_label("   ")


def test_parse_zero_coefficient_raises():
    with pytest.raises(ValueError, match="non-positive"):
        parse_point_label("0*A")


def test_parse_species_name_with_punctuation():
    """Species names like 'Cu(III)-S' must round-trip through the parser."""
    assert parse_point_label("Cu(III)-S") == [(1, "Cu(III)-S")]
    assert parse_point_label("Cu(I) + Pdt") == [(1, "Cu(I)"), (1, "Pdt")]


def test_point_thermo_sums_species():
    """A point with two species adds their thermo (with stoichiometry)."""
    a = ConformerSet("A", ["a.log"], [_bbe(-100.0)])
    b = ConformerSet("B", ["b.log"], [_bbe(-50.0)])
    p = Point.from_label("A + B", {"A": a, "B": b})
    th = p.thermo(298.15, gconf=False)
    assert math.isclose(th.qh_gibbs, -150.0, rel_tol=1e-12)


def test_point_thermo_applies_stoichiometry():
    a = ConformerSet("A", ["a.log"], [_bbe(-100.0)])
    p = Point.from_label("3*A", {"A": a})
    th = p.thermo(298.15, gconf=False)
    assert math.isclose(th.qh_gibbs, -300.0, rel_tol=1e-12)


def test_point_unknown_species_raises():
    a = ConformerSet("A", ["a.log"], [_bbe(-100.0)])
    with pytest.raises(KeyError, match="unknown species"):
        Point.from_label("A + Mystery", {"A": a})


# ---------------------------------------------------------------------------
# Pathway
# ---------------------------------------------------------------------------

def test_pathway_default_zero_is_first_point():
    a = ConformerSet("A", ["a.log"], [_bbe(-100.0)])
    b = ConformerSet("B", ["b.log"], [_bbe(-50.0)])
    species = {"A": a, "B": b}
    pa = Point.from_label("A", species)
    pb = Point.from_label("B", species)
    path = Pathway(name="rxn", points=[pa, pb])
    assert path.zero is pa


def test_pathway_relative_subtracts_zero():
    a = ConformerSet("A", ["a.log"], [_bbe(-100.0)])
    b = ConformerSet("B", ["b.log"], [_bbe(-50.0)])
    species = {"A": a, "B": b}
    pa = Point.from_label("A", species)
    pb = Point.from_label("B", species)
    path = Pathway(name="rxn", points=[pa, pb])
    rels = path.relative(298.15, gconf=False)
    assert len(rels) == 2
    assert math.isclose(rels[0].qh_gibbs, 0.0, abs_tol=1e-12)
    assert math.isclose(rels[1].qh_gibbs, 50.0, rel_tol=1e-12)   # -50 − (-100)


def test_pathway_explicit_zero():
    """An explicit `zero` (not the first point) is honored."""
    a = ConformerSet("A", ["a.log"], [_bbe(-100.0)])
    b = ConformerSet("B", ["b.log"], [_bbe(-50.0)])
    species = {"A": a, "B": b}
    pa = Point.from_label("A", species)
    pb = Point.from_label("B", species)
    path = Pathway(name="rxn", points=[pa, pb], zero=pb)
    rels = path.relative(298.15, gconf=False)
    assert math.isclose(rels[0].qh_gibbs, -50.0, rel_tol=1e-12)
    assert math.isclose(rels[1].qh_gibbs, 0.0, abs_tol=1e-12)


def test_pathway_empty_points_raises():
    with pytest.raises(ValueError, match="no points"):
        Pathway(name="empty", points=[])


# ---------------------------------------------------------------------------
# PESOptions / PESResult
# ---------------------------------------------------------------------------

def test_pes_options_kcal_default():
    opts = PESOptions()
    assert math.isclose(opts.to_user_units(1.0), KCAL_TO_AU, rel_tol=1e-12)


def test_pes_options_kj():
    opts = PESOptions(units="kJ/mol")
    assert math.isclose(opts.to_user_units(1.0), J_TO_AU / 1000.0, rel_tol=1e-12)


def test_pes_options_none_passthrough():
    """to_user_units(None) is None — keeps SPC absence visible to formatters."""
    opts = PESOptions()
    assert opts.to_user_units(None) is None


def test_pes_result_holds_pathways_and_options():
    a = ConformerSet("A", ["a.log"], [_bbe(-100.0)])
    species = {"A": a}
    path = Pathway(name="rxn", points=[Point.from_label("A", species)])
    opts = PESOptions(units="kJ/mol", decimals=1)
    result = PESResult(pathways=[path], options=opts, temperatures=[300.0])
    assert result.pathways[0].name == "rxn"
    assert result.options.units == "kJ/mol"
    assert result.temperatures == [300.0]


# ---------------------------------------------------------------------------
# lowest_only mode
# ---------------------------------------------------------------------------

def test_point_thermo_lowest_only_picks_lowest_per_species():
    """A multi-conformer species in lowest_only mode contributes its single
    lowest-qh-G conformer; mixing entropy is not added."""
    a1 = _bbe(-100.000, scf=-100.5)
    a2 = _bbe(-100.001, scf=-200.5)        # lowest qh-G
    a = ConformerSet("A", ["1.log", "2.log"], [a1, a2])
    p = Point.from_label("A", {"A": a})
    th = p.thermo(298.15, lowest_only=True)
    assert th.scf_energy == -200.5         # the lowest conformer's scf


def test_point_thermo_lowest_only_overrides_gconf():
    """lowest_only=True ignores gconf flag entirely."""
    a1 = _bbe(-100.000)
    a2 = _bbe(-100.000)                    # equal G, mixing entropy would apply
    a = ConformerSet("A", ["1.log", "2.log"], [a1, a2])
    p = Point.from_label("A", {"A": a})
    lowest = p.thermo(298.15, gconf=True, lowest_only=True)
    bare = a1                              # the bbe directly
    assert lowest.qh_gibbs == bare.qh_gibbs_free_energy


def test_pathway_relative_lowest_only_threads_through():
    a = ConformerSet("A", ["a.log"], [_bbe(-100.0)])
    bs = ConformerSet("B", ["b1.log", "b2.log"],
                      [_bbe(-50.0), _bbe(-49.5)])
    species = {"A": a, "B": bs}
    pa = Point.from_label("A", species)
    pb = Point.from_label("B", species)
    path = Pathway(name="rxn", points=[pa, pb])
    rels = path.relative(298.15, lowest_only=True)
    # B's lowest qh-G is -50.0 (b1.log); ΔG = -50 − (-100) = 50.
    assert math.isclose(rels[1].qh_gibbs, 50.0, rel_tol=1e-12)


def test_pes_options_has_lowest_only_field():
    opts = PESOptions(lowest_only=True)
    assert opts.lowest_only is True
