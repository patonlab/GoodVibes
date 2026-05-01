"""Tests for goodvibes.pes_legacy — the line-based `--- # PES` format."""
import os

import pytest

from goodvibes.pes_legacy import parse_legacy
from goodvibes.pes_loader import PESSpec


# ---------------------------------------------------------------------------
# Minimal valid legacy file
# ---------------------------------------------------------------------------

MINIMAL = """\
--- # PES
    Reaction: [A, B, C]

--- # SPECIES
    A: a_*
    B: b_*
    C: c_*
"""


def test_parse_minimal():
    spec = parse_legacy(MINIMAL)
    assert isinstance(spec, PESSpec)
    assert spec.pathways == {"Reaction": ["A", "B", "C"]}
    assert spec.species == {"A": "a_*", "B": "b_*", "C": "c_*"}
    assert spec.zero == {}
    assert spec.options.units == "kcal/mol"      # default
    assert spec.options.decimals == 2            # default


# ---------------------------------------------------------------------------
# Stoichiometric sums in pathway points
# ---------------------------------------------------------------------------

def test_parse_stoichiometric_points():
    text = """\
--- # PES
    Rxn: [A + B, C, D + E]
--- # SPECIES
    A: a
    B: b
    C: c
    D: d
    E: e
"""
    spec = parse_legacy(text)
    assert spec.pathways == {"Rxn": ["A + B", "C", "D + E"]}


# ---------------------------------------------------------------------------
# FORMAT block
# ---------------------------------------------------------------------------

def test_parse_format_units_and_dec():
    text = MINIMAL + """
--- # FORMAT
    units: kJ/mol
    dec: 3
"""
    spec = parse_legacy(text)
    assert spec.options.units == "kJ/mol"
    assert spec.options.decimals == 3


def test_parse_format_zero_applies_to_all_pathways():
    text = """\
--- # PES
    Path1: [A, B]
    Path2: [A, C]
--- # SPECIES
    A: a
    B: b
    C: c
--- # FORMAT
    zero: A
"""
    spec = parse_legacy(text)
    assert spec.zero == {"Path1": "A", "Path2": "A"}


def test_parse_format_extras_preserved():
    """Unknown FORMAT keys (graph styling) must round-trip into format_extras."""
    text = MINIMAL + """
--- # FORMAT
    dpi: 400
    color: black,#26a6a4
    title: Reaction Profile
"""
    spec = parse_legacy(text)
    assert spec.format_extras["dpi"] == "400"
    assert spec.format_extras["color"] == "black,#26a6a4"
    assert spec.format_extras["title"] == "Reaction Profile"


def test_parse_format_invalid_units_raises():
    text = MINIMAL + "\n--- # FORMAT\n    units: hartree\n"
    with pytest.raises(ValueError, match="kcal/mol"):
        parse_legacy(text)


def test_parse_format_invalid_dec_raises():
    text = MINIMAL + "\n--- # FORMAT\n    dec: notanumber\n"
    with pytest.raises(ValueError, match="integer"):
        parse_legacy(text)


# ---------------------------------------------------------------------------
# Comments, whitespace, separator tolerance
# ---------------------------------------------------------------------------

def test_comments_and_blank_lines_ignored():
    text = """\
--- # PES
    # this is a comment
    Reaction: [A, B]

    # another comment
--- # SPECIES
    A: a
    # comment between entries
    B: b
"""
    spec = parse_legacy(text)
    assert spec.pathways == {"Reaction": ["A", "B"]}
    assert spec.species == {"A": "a", "B": "b"}


def test_equals_separator_is_accepted():
    """Legacy treats `:` and `=` interchangeably."""
    text = """\
--- # PES
    Reaction = [A, B]
--- # SPECIES
    A = a_*
    B: b_*
"""
    spec = parse_legacy(text)
    assert spec.pathways == {"Reaction": ["A", "B"]}
    assert spec.species == {"A": "a_*", "B": "b_*"}


def test_leading_indent_tolerated():
    """Section bodies may be indented by any amount (or not at all)."""
    text = """\
--- # PES
Reaction: [A, B]
--- # SPECIES
A: a
B: b
"""
    spec = parse_legacy(text)
    assert spec.pathways == {"Reaction": ["A", "B"]}


def test_folder_line_in_species_ignored():
    """The legacy `folder: PATH` directive is ignored — paths come from thermo_data."""
    text = """\
--- # PES
    R: [A]
--- # SPECIES
    folder: /some/path
    A: a
"""
    spec = parse_legacy(text)
    assert spec.species == {"A": "a"}      # folder line filtered out


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_missing_pes_section_raises():
    text = "--- # SPECIES\n    A: a\n"
    with pytest.raises(ValueError, match="PES"):
        parse_legacy(text)


def test_missing_species_section_raises():
    text = "--- # PES\n    R: [A]\n"
    with pytest.raises(ValueError, match="SPECIES"):
        parse_legacy(text)


def test_pathway_value_not_bracketed_raises():
    text = """\
--- # PES
    R: A, B, C
--- # SPECIES
    A: a
"""
    with pytest.raises(ValueError, match="bracketed list"):
        parse_legacy(text)


def test_empty_pathway_raises():
    text = """\
--- # PES
    R: []
--- # SPECIES
    A: a
"""
    with pytest.raises(ValueError, match="no points"):
        parse_legacy(text)


# ---------------------------------------------------------------------------
# Real fixture: examples/pes/azabor_PES.yaml (the "more interesting" example)
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AZABOR = os.path.join(REPO_ROOT, "goodvibes", "examples", "pes", "azabor_PES.yaml")


@pytest.mark.skipif(not os.path.exists(AZABOR), reason="azabor_PES.yaml fixture missing")
def test_parse_azabor_fixture():
    """The packaged 6-step PES with stoichiometric sums must parse cleanly."""
    with open(AZABOR) as f:
        text = f.read()
    spec = parse_legacy(text)
    assert "Ph" in spec.pathways
    assert len(spec.pathways["Ph"]) == 6
    # Every point in the pathway is a stoichiometric sum
    for point in spec.pathways["Ph"]:
        assert " + " in point or "+" in point
    # All 8 species names are defined
    assert set(spec.species) == {
        "R1-An", "Aza-Phos", "THF", "R1-Comp", "Azir-Comp", "Syn-P",
        "OpenTS", "AmTS",
    }
    # FORMAT keys: dec is parsed; others (color, dpi, title, ...) end up in extras
    assert spec.options.decimals == 1
    assert "dpi" in spec.format_extras
    assert "title" in spec.format_extras
