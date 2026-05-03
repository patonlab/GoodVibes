"""Tests for goodvibes.pes_yaml — the proper YAML PES format."""
import pytest

# PyYAML is a test-only dependency; skip the whole module if absent.
yaml = pytest.importorskip("yaml")

from goodvibes.pes_loader import PESSpec
from goodvibes.pes_yaml import parse_yaml


# ---------------------------------------------------------------------------
# Minimal valid YAML
# ---------------------------------------------------------------------------

MINIMAL = """
pathways:
  Reaction: ["A", "B", "C"]

species:
  A: {files: "a_*.log"}
  B: {files: "b_*.log"}
  C: {files: "c_*.log"}
"""


def test_parse_minimal():
    spec = parse_yaml(MINIMAL)
    assert isinstance(spec, PESSpec)
    assert spec.pathways == {"Reaction": ["A", "B", "C"]}
    assert spec.species == {"A": "a_*.log", "B": "b_*.log", "C": "c_*.log"}
    assert spec.zero == {}
    assert spec.options.units == "kcal/mol"


# ---------------------------------------------------------------------------
# Species shorthand forms
# ---------------------------------------------------------------------------

def test_species_bare_string():
    """Shorthand: `Species: 'glob'` instead of `Species: {files: 'glob'}`."""
    text = """
pathways:
  R: ["A"]
species:
  A: "a_*"
"""
    spec = parse_yaml(text)
    assert spec.species == {"A": "a_*"}


def test_species_bare_list():
    text = """
pathways:
  R: ["A"]
species:
  A: ["a1.log", "a2.log"]
"""
    spec = parse_yaml(text)
    assert spec.species == {"A": ["a1.log", "a2.log"]}


def test_species_dict_with_list_files():
    text = """
pathways:
  R: ["A"]
species:
  A: {files: ["a1.log", "a2.log", "a3.log"]}
"""
    spec = parse_yaml(text)
    assert spec.species == {"A": ["a1.log", "a2.log", "a3.log"]}


def test_species_missing_files_key_raises():
    text = """
pathways:
  R: ["A"]
species:
  A: {description: "no files key"}
"""
    with pytest.raises(ValueError, match=r"files.*dir.*dirs"):
        parse_yaml(text)


# ---------------------------------------------------------------------------
# Directory-based species spec ({dir: "X"} / {dirs: [...]})
# ---------------------------------------------------------------------------

def test_species_dir_encoded_with_prefix():
    """`{dir: "X"}` is encoded as `@dir:X` in the PESSpec so the
    downstream resolver can dispatch on the prefix."""
    text = """
pathways:
  R: ["A"]
species:
  A: {dir: "subdir_a"}
"""
    spec = parse_yaml(text)
    assert spec.species == {"A": "@dir:subdir_a"}


def test_species_dirs_list_encoded():
    text = """
pathways:
  R: ["A"]
species:
  A: {dirs: ["d1", "d2"]}
"""
    spec = parse_yaml(text)
    assert spec.species == {"A": ["@dir:d1", "@dir:d2"]}


def test_species_dir_normalizes_trailing_glob():
    """`dir: "X/"`, `dir: "X/*"`, `dir: "X/**"` all normalize to `X`."""
    for raw, expected in [("X", "X"), ("X/", "X"), ("X/*", "X"), ("X/**", "X")]:
        text = f"""
pathways:
  R: ["A"]
species:
  A: {{dir: "{raw}"}}
"""
        spec = parse_yaml(text)
        assert spec.species == {"A": f"@dir:{expected}"}


def test_species_combined_files_and_dir():
    """A species can list both `files:` and `dir:` — the resolver takes the union."""
    text = """
pathways:
  R: ["A"]
species:
  A:
    files: ["explicit.log"]
    dir: "subdir_a"
"""
    spec = parse_yaml(text)
    assert spec.species == {"A": ["explicit.log", "@dir:subdir_a"]}


def test_species_dir_not_string_raises():
    text = """
pathways:
  R: ["A"]
species:
  A: {dir: 42}
"""
    with pytest.raises(ValueError, match=r"`dir:` must be a string"):
        parse_yaml(text)


def test_species_dirs_not_list_raises():
    text = """
pathways:
  R: ["A"]
species:
  A: {dirs: "not-a-list"}
"""
    with pytest.raises(ValueError, match=r"`dirs:` must be a list"):
        parse_yaml(text)


# ---------------------------------------------------------------------------
# Stoichiometric sums + zero overrides
# ---------------------------------------------------------------------------

def test_stoichiometric_points_pass_through_verbatim():
    """`+` / `*` syntax is parsed at PES build time, not here."""
    text = """
pathways:
  R: ["A + B", "2*C", "C + D"]
species:
  A: a
  B: b
  C: c
  D: d
"""
    spec = parse_yaml(text)
    assert spec.pathways == {"R": ["A + B", "2*C", "C + D"]}


def test_zero_per_pathway():
    text = """
pathways:
  R: ["A", "B"]
  S: ["A", "C"]
species:
  A: a
  B: b
  C: c
zero:
  R: B
  S: C
"""
    spec = parse_yaml(text)
    assert spec.zero == {"R": "B", "S": "C"}


def test_zero_referencing_unknown_pathway_raises():
    text = """
pathways:
  R: ["A"]
species:
  A: a
zero:
  Mystery: A
"""
    with pytest.raises(ValueError, match="unknown pathway"):
        parse_yaml(text)


# ---------------------------------------------------------------------------
# FORMAT block
# ---------------------------------------------------------------------------

def test_format_units_and_decimals():
    text = MINIMAL + """
format:
  units: kJ/mol
  decimals: 4
"""
    spec = parse_yaml(text)
    assert spec.options.units == "kJ/mol"
    assert spec.options.decimals == 4


def test_format_dec_alias():
    """Accept `dec:` as an alias for `decimals:` for parity with the legacy format."""
    text = MINIMAL + """
format:
  dec: 3
"""
    spec = parse_yaml(text)
    assert spec.options.decimals == 3


def test_format_extras_preserved():
    text = MINIMAL + """
format:
  dpi: 400
  title: My Reaction
"""
    spec = parse_yaml(text)
    assert spec.format_extras["dpi"] == "400"
    assert spec.format_extras["title"] == "My Reaction"


def test_format_invalid_units_raises():
    text = MINIMAL + "\nformat:\n  units: hartree\n"
    with pytest.raises(ValueError, match="kcal/mol"):
        parse_yaml(text)


def test_format_invalid_decimals_raises():
    text = MINIMAL + "\nformat:\n  decimals: notanumber\n"
    with pytest.raises(ValueError, match="integer"):
        parse_yaml(text)


# ---------------------------------------------------------------------------
# Schema errors
# ---------------------------------------------------------------------------

def test_root_must_be_mapping():
    with pytest.raises(ValueError, match="mapping"):
        parse_yaml("- just\n- a\n- list\n")


def test_missing_pathways_raises():
    text = """
species:
  A: a
"""
    with pytest.raises(ValueError, match="pathways"):
        parse_yaml(text)


def test_missing_species_raises():
    text = """
pathways:
  R: ["A"]
"""
    with pytest.raises(ValueError, match="species"):
        parse_yaml(text)


def test_pathway_value_not_a_list_raises():
    text = """
pathways:
  R: "A, B, C"
species:
  A: a
"""
    with pytest.raises(ValueError, match="list"):
        parse_yaml(text)


def test_empty_pathway_raises():
    text = """
pathways:
  R: []
species:
  A: a
"""
    with pytest.raises(ValueError, match="no points"):
        parse_yaml(text)
