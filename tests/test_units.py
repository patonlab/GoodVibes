"""Energy units are defined once, in goodvibes.constants, and accepted
everywhere a user chooses display units (PES format, plots, JSON).
Regression: the constants were copied into pes_model.py, pes.py, thermo.py
and hard-coded in output.py, and only kcal/mol and kJ/mol were accepted;
the legacy PES text path silently printed kcal/mol for anything else.
"""
import pytest

from goodvibes import constants
from goodvibes.constants import (HARTREE_TO_EV, J_TO_AU, KCAL_TO_AU, SUPPORTED_UNITS,
                                 canonical_units, hartree_factor)
from goodvibes.pes_model import PESOptions
from goodvibes.pes_yaml import parse_yaml
from goodvibes.pes_legacy import parse_legacy


@pytest.mark.parametrize('spelling, canonical', [
    ('kcal/mol', 'kcal/mol'), ('KCAL/MOL', 'kcal/mol'), ('kcal', 'kcal/mol'),
    ('kJ/mol', 'kJ/mol'), ('kj/mol', 'kJ/mol'), ('kJ', 'kJ/mol'),
    ('eV', 'eV'), ('ev', 'eV'),
    ('hartree', 'hartree'), ('Eh', 'hartree'), ('au', 'hartree'),
])
def test_canonical_units(spelling, canonical):
    assert canonical_units(spelling) == canonical
    assert canonical in SUPPORTED_UNITS


def test_unknown_units_raise_with_the_supported_list():
    with pytest.raises(ValueError, match="kcal/mol, kJ/mol, eV, hartree"):
        canonical_units('furlongs')


def test_factors():
    assert hartree_factor('kcal/mol') == KCAL_TO_AU == 627.509541
    assert hartree_factor('kJ/mol') == pytest.approx(J_TO_AU / 1000.0)
    assert hartree_factor('kJ/mol') == pytest.approx(4.184 * KCAL_TO_AU)
    assert hartree_factor('eV') == HARTREE_TO_EV == pytest.approx(27.2114, abs=1e-4)
    assert hartree_factor('hartree') == 1.0


def test_constants_defined_once():
    import goodvibes.pes_model as pm, goodvibes.pes as pes, goodvibes.thermo as th
    for mod in (pm, pes, th):
        assert mod.GAS_CONSTANT is constants.GAS_CONSTANT
        assert mod.J_TO_AU is constants.J_TO_AU


@pytest.mark.parametrize('units, expected', [('kcal/mol', 627.509541), ('kJ/mol', 2625.4996), ('eV', 27.211386), ('hartree', 1.0)])
def test_pes_options_to_user_units(units, expected):
    assert PESOptions(units=units).to_user_units(1.0) == pytest.approx(expected, rel=1e-6)
    assert PESOptions(units=units).to_user_units(None) is None


def test_pes_options_unknown_units_raise():
    with pytest.raises(ValueError):
        PESOptions(units='cal').to_user_units(1.0)


def test_yaml_and_legacy_formats_accept_ev_and_hartree():
    spec = parse_yaml("pathways:\n  p: [A, B]\nspecies:\n  A: a\n  B: b\nformat:\n  units: ev\n")
    assert spec.options.units == 'eV'
    spec = parse_legacy("--- # PES\n  p: [A, B]\n--- # SPECIES\n  A: a\n  B: b\n--- # FORMAT\n  units: hartree\n")
    assert spec.options.units == 'hartree'
    with pytest.raises(ValueError, match="format.units"):
        parse_yaml("pathways:\n  p: [A, B]\nspecies:\n  A: a\n  B: b\nformat:\n  units: cal\n")
