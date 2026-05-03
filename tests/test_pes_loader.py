"""Tests for goodvibes.pes_loader — pattern resolution + builder + dispatcher."""
import math
from types import SimpleNamespace

import pytest

from goodvibes.pes_loader import (
    PESSpec, build_pes_result, is_legacy_format, resolve_pattern,
)
from goodvibes.pes_model import PESOptions


def _bbe(g, scf=None):
    if scf is None:
        scf = g - 0.001
    return SimpleNamespace(
        scf_energy=scf, zpe=0.005,
        enthalpy=g + 0.005, qh_enthalpy=g + 0.005,
        entropy=1.6e-5, qh_entropy=1.6e-5,
        gibbs_free_energy=g, qh_gibbs_free_energy=g,
        sp_energy=None,
    )


def _td(*pairs):
    """Build a thermo_data dict from (path, g) pairs."""
    return {p: _bbe(g) for p, g in pairs}


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def test_is_legacy_recognises_pes_marker():
    assert is_legacy_format("--- # PES\n  R: [A, B]")


def test_is_legacy_recognises_species_marker():
    assert is_legacy_format("--- # SPECIES\n  A: A_*")


def test_is_legacy_recognises_format_marker():
    assert is_legacy_format("--- # FORMAT\n  units: kcal/mol")


def test_is_legacy_case_insensitive_marker():
    """Marker names like `# pes` or `# Pes` should still be recognised."""
    assert is_legacy_format("--- # pes\nR: [A]")


def test_is_legacy_rejects_yaml():
    yaml_text = "pathways:\n  R: [A, B]\nspecies:\n  A: {files: A_*}\n"
    assert is_legacy_format(yaml_text) is False


def test_is_legacy_rejects_dashes_without_marker():
    """A file with `---` document separators but no recognised marker is YAML."""
    assert is_legacy_format("---\npathways:\n  R: [A]\n") is False


# ---------------------------------------------------------------------------
# Pattern resolution
# ---------------------------------------------------------------------------

def test_resolve_literal_pattern_matches_exact_stem():
    td = _td(("path/to/foo.log", -100.0), ("path/to/bar.log", -50.0))
    assert resolve_pattern("foo", td) == ["path/to/foo.log"]


def test_resolve_literal_with_extension():
    """Patterns can be written with or without the .log/.out extension —
    both are stripped before matching, since GoodVibes accepts either."""
    td = _td(("a/b/foo.log", -100.0))
    assert resolve_pattern("foo.log", td) == ["a/b/foo.log"]
    assert resolve_pattern("foo", td) == ["a/b/foo.log"]


def test_resolve_glob_pattern():
    td = _td(
        ("dir/conf_1.log", -100.0),
        ("dir/conf_2.log", -100.5),
        ("dir/other.log", -50.0),
    )
    matches = resolve_pattern("conf_*", td)
    assert matches == ["dir/conf_1.log", "dir/conf_2.log"]


def test_resolve_no_match_returns_empty():
    td = _td(("a.log", -100.0))
    assert resolve_pattern("nonexistent", td) == []


# ---------------------------------------------------------------------------
# Directory-based pattern resolution (`@dir:` prefix)
# ---------------------------------------------------------------------------

def test_resolve_dir_pattern_matches_parent_dir_basename():
    """`@dir:X` matches keys whose parent dir's basename is X."""
    td = _td(
        ("R1-An/conf_1.log", -100.0),
        ("R1-An/conf_2.log", -100.5),
        ("THF/thf.log", -50.0),
        ("AmTS/ts1.log", -75.0),
    )
    assert resolve_pattern("@dir:R1-An", td) == [
        "R1-An/conf_1.log", "R1-An/conf_2.log",
    ]
    assert resolve_pattern("@dir:THF", td) == ["THF/thf.log"]


def test_resolve_dir_pattern_supports_glob():
    """fnmatch globs apply to dir names too — `@dir:TS_*` matches any
    parent dir whose basename starts with `TS_`."""
    td = _td(
        ("TS_R/a.log", -100.0),
        ("TS_S/b.log", -99.0),
        ("Reactant/c.log", -200.0),
    )
    matches = resolve_pattern("@dir:TS_*", td)
    assert matches == ["TS_R/a.log", "TS_S/b.log"]


def test_resolve_dir_pattern_does_not_match_descendant_dirs():
    """`@dir:X` is a parent-dir match, not an ancestor match. A file at
    `outer/X/inner/foo.log` has parent `inner`, not `X`, so it doesn't
    match `@dir:X` — this prevents ambiguity in nested layouts."""
    td = _td(
        ("X/foo.log", -100.0),
        ("outer/X/inner/foo.log", -101.0),
    )
    assert resolve_pattern("@dir:X", td) == ["X/foo.log"]


def test_resolve_dir_no_parent_excluded():
    """A bare filename (no parent dir) doesn't match any @dir: pattern."""
    td = _td(
        ("flat.log", -50.0),
        ("sub/in_sub.log", -100.0),
    )
    assert resolve_pattern("@dir:sub", td) == ["sub/in_sub.log"]
    assert resolve_pattern("@dir:flat", td) == []


def test_resolve_dir_empty_thermo_data():
    assert resolve_pattern("@dir:any", {}) == []


def test_resolve_preserves_thermo_data_order():
    """Thermo_data is dict-ordered; resolve preserves that."""
    td = _td(
        ("z.log", -100.0),
        ("a.log", -101.0),
        ("m.log", -102.0),
    )
    assert resolve_pattern("*", td) == ["z.log", "a.log", "m.log"]


# ---------------------------------------------------------------------------
# Builder: PESSpec + thermo_data -> PESResult
# ---------------------------------------------------------------------------

def test_build_minimal_pathway():
    td = _td(("a.log", -100.0), ("b.log", -50.0))
    spec = PESSpec(
        pathways={"rxn": ["A", "B"]},
        species={"A": "a", "B": "b"},
    )
    result = build_pes_result(spec, td)
    assert len(result.pathways) == 1
    path = result.pathways[0]
    assert path.name == "rxn"
    assert len(path.points) == 2
    assert path.zero is path.points[0]    # default zero
    rels = path.relative(298.15, gconf=False)
    assert math.isclose(rels[1].qh_gibbs, 50.0, rel_tol=1e-12)


def test_build_with_explicit_zero():
    td = _td(("a.log", -100.0), ("b.log", -50.0))
    spec = PESSpec(
        pathways={"rxn": ["A", "B"]},
        species={"A": "a", "B": "b"},
        zero={"rxn": "B"},
    )
    result = build_pes_result(spec, td)
    rels = result.pathways[0].relative(298.15, gconf=False)
    # B is zero now, so B has ΔG=0 and A has ΔG=-50 (lower than B)
    assert math.isclose(rels[1].qh_gibbs, 0.0, abs_tol=1e-12)
    assert math.isclose(rels[0].qh_gibbs, -50.0, rel_tol=1e-12)


def test_build_with_stoichiometric_sums():
    td = _td(("x.log", -100.0), ("y.log", -50.0), ("p.log", -200.0))
    spec = PESSpec(
        pathways={"rxn": ["X + Y", "P"]},
        species={"X": "x", "Y": "y", "P": "p"},
    )
    result = build_pes_result(spec, td)
    rels = result.pathways[0].relative(298.15, gconf=False)
    # Reactants: -100 + -50 = -150. Product: -200. Δ = -50.
    assert math.isclose(rels[1].qh_gibbs, -50.0, rel_tol=1e-12)


def test_build_with_coefficients():
    """A point like `2*X` doubles the species' contribution."""
    td = _td(("x.log", -100.0), ("p.log", -250.0))
    spec = PESSpec(
        pathways={"rxn": ["2*X", "P"]},
        species={"X": "x", "P": "p"},
    )
    result = build_pes_result(spec, td)
    rels = result.pathways[0].relative(298.15, gconf=False)
    # Reactants: 2 * -100 = -200. Product: -250. Δ = -50.
    assert math.isclose(rels[1].qh_gibbs, -50.0, rel_tol=1e-12)


def test_build_with_conformer_ensemble():
    """A species pattern matching multiple files becomes a multi-conformer set."""
    td = _td(
        ("conf_1.log", -100.0),
        ("conf_2.log", -99.999),
        ("p.log", -200.0),
    )
    spec = PESSpec(
        pathways={"rxn": ["A", "P"]},
        species={"A": "conf_*", "P": "p"},
    )
    result = build_pes_result(spec, td)
    # Two conformers in A → ConformerSet with both
    a_set = result.pathways[0].points[0].species[0][1]
    assert len(a_set.bbes) == 2


def test_build_unknown_species_raises():
    td = _td(("a.log", -100.0))
    spec = PESSpec(
        pathways={"rxn": ["A", "Mystery"]},
        species={"A": "a"},
    )
    with pytest.raises(KeyError, match="Mystery"):
        build_pes_result(spec, td)


def test_build_pattern_with_no_matches_raises():
    td = _td(("a.log", -100.0))
    spec = PESSpec(
        pathways={"rxn": ["X"]},
        species={"X": "nonexistent_*"},
    )
    with pytest.raises(ValueError, match="no files matched"):
        build_pes_result(spec, td)


def test_build_only_resolves_referenced_species():
    """A SPECIES entry not used by any pathway shouldn't trigger glob resolution."""
    td = _td(("a.log", -100.0))    # `unused_*` would NOT match, but the builder
    spec = PESSpec(                 # should never even try since it's not referenced.
        pathways={"rxn": ["A"]},
        species={"A": "a", "Unused": "unused_*"},
    )
    result = build_pes_result(spec, td)   # no error
    assert len(result.pathways[0].points) == 1


def test_build_uses_options_from_spec():
    td = _td(("a.log", -100.0), ("b.log", -50.0))
    spec = PESSpec(
        pathways={"rxn": ["A", "B"]},
        species={"A": "a", "B": "b"},
        options=PESOptions(units="kJ/mol", decimals=4),
    )
    result = build_pes_result(spec, td)
    assert result.options.units == "kJ/mol"
    assert result.options.decimals == 4


def test_build_passes_temperatures_through():
    td = _td(("a.log", -100.0))
    spec = PESSpec(pathways={"rxn": ["A"]}, species={"A": "a"})
    result = build_pes_result(spec, td, temperatures=[273.15, 298.15, 373.15])
    assert result.temperatures == [273.15, 298.15, 373.15]


# ---------------------------------------------------------------------------
# Directory-based species spec — full stack (YAML parser → builder)
# ---------------------------------------------------------------------------

def test_build_pes_result_with_dir_spec_resolves_subdirs():
    """When the YAML uses `{dir: "X"}`, build_pes_result should pick up
    every thermo_data key whose parent directory's basename is `X`."""
    td = _td(
        ("R/r1.log", -100.0), ("R/r2.log", -100.5),    # → species R
        ("P/p1.log", -200.0),                          # → species P
        ("noise/x.log", -50.0),                        # ignored (no species)
    )
    spec = PESSpec(
        pathways={"rxn": ["R", "P"]},
        species={"R": "@dir:R", "P": "@dir:P"},
    )
    result = build_pes_result(spec, td)
    pathway = result.pathways[0]
    point_R, point_P = pathway.points
    cset_R = point_R.species[0][1]
    cset_P = point_P.species[0][1]
    assert cset_R.files == ["R/r1.log", "R/r2.log"]
    assert cset_P.files == ["P/p1.log"]


def test_build_pes_result_dir_spec_via_yaml():
    """Same scenario, but driving through the YAML parser to ensure
    `{dir: "X"}` round-trips through to the resolver."""
    from goodvibes.pes_yaml import parse_yaml
    td = _td(
        ("R/r1.log", -100.0),
        ("P/p1.log", -200.0),
    )
    text = """
pathways:
  rxn: ["R", "P"]
species:
  R: {dir: "R"}
  P: {dir: "P"}
"""
    spec = parse_yaml(text)
    result = build_pes_result(spec, td)
    pathway = result.pathways[0]
    cset_R = pathway.points[0].species[0][1]
    cset_P = pathway.points[1].species[0][1]
    assert cset_R.files == ["R/r1.log"]
    assert cset_P.files == ["P/p1.log"]


def test_build_pes_result_combined_files_and_dir_via_yaml():
    """A species spec with both `files:` and `dir:` takes the union of
    matches (de-duplicated, original-order preserved)."""
    from goodvibes.pes_yaml import parse_yaml
    td = _td(
        ("explicit.log", -100.0),    # file-glob match
        ("X/in_dir.log", -101.0),    # dir match
        ("X/explicit.log", -102.0),  # would match both — should appear once
    )
    text = """
pathways:
  rxn: ["A"]
species:
  A:
    files: ["explicit.log"]
    dir: "X"
"""
    spec = parse_yaml(text)
    result = build_pes_result(spec, td)
    cset_A = result.pathways[0].points[0].species[0][1]
    # `explicit.log` (file pattern) hits both top-level and X/explicit.log;
    # `dir: X` adds X/in_dir.log; the `dir:` X/explicit.log entry is
    # de-duped by _resolve_species.
    assert cset_A.files == ["explicit.log", "X/explicit.log", "X/in_dir.log"]
