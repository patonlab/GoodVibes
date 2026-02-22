#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for parsing ORCA 6 output files using goodvibes.io."""

import pytest
from goodvibes.io import getoutData, parse_data, level_of_theory, read_initial
from conftest import orca_path, ORCA_FREQ_FILES


# ---------------------------------------------------------------------------
# parse_data: energy extraction (works for ORCA)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_energy", [
    ('01a_water_hf_freq.out', -76.009114306317),
    ('04_benzene_radical_cation.out', -232.018283530455),
    ('05_methylene_triplet_carbene.out', -39.019780938452),
    ('06_carbon_atom_single_point.out', -37.659292822489),
    ('08_alanine_C1_pcm_water.out', -323.378135366042),
    ('16_o2_superoxide_anion.out', -150.341130103125),
    ('17_iron_complex_quintet.out', -1828.860022847459),
    ('19_acetic_acid_smd_dmso.out', -229.059207744143),
    ('22_hcn_linear_freq_noraman.out', -93.414515903409),
    ('44_ts_sn2_identity_chloride.out', -960.291216714078),
])
def test_parse_data_orca_energy(filename, expected_energy):
    spe, program, *_ = parse_data(orca_path(filename))
    assert program == "Orca"
    assert abs(spe - expected_energy) < 1e-8


# ---------------------------------------------------------------------------
# parse_data: program detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", ORCA_FREQ_FILES)
def test_parse_data_orca_program(filename):
    _, program, *_ = parse_data(orca_path(filename))
    assert program == "Orca"


# ---------------------------------------------------------------------------
# parse_data: charge and multiplicity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_charge, expected_mult", [
    ('01a_water_hf_freq.out', 0, 1),
    ('04_benzene_radical_cation.out', 1, 2),
    ('05_methylene_triplet_carbene.out', 0, 3),
    ('06_carbon_atom_single_point.out', 0, 3),
    ('16_o2_superoxide_anion.out', -1, 2),
    ('17_iron_complex_quintet.out', 0, 5),
    ('44_ts_sn2_identity_chloride.out', -1, 1),
])
def test_parse_data_orca_charge_multiplicity(filename, expected_charge, expected_mult):
    *_, charge, _, mult = parse_data(orca_path(filename))
    assert charge == expected_charge
    assert mult == expected_mult


# ---------------------------------------------------------------------------
# read_initial: termination status
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_progress", [
    ('01a_water_hf_freq.out', 'Normal'),
    ('04_benzene_radical_cation.out', 'Normal'),
    ('17_iron_complex_quintet.out', 'Normal'),
    ('44_ts_sn2_identity_chloride.out', 'Normal'),
    ('51_err_scf_convergence_fe_complex.out', 'Error'),
    ('53_err_wrong_charge_multiplicity.out', 'Incomplete'),
])
def test_read_initial_orca_progress(filename, expected_progress):
    _, _, progress, _, _ = read_initial(orca_path(filename))
    assert progress == expected_progress


# ---------------------------------------------------------------------------
# read_initial: solvation model detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_contains", [
    ('01a_water_hf_freq.out', 'gas phase'),
    ('08_alanine_C1_pcm_water.out', 'CPCM'),
    ('19_acetic_acid_smd_dmso.out', 'SMD'),
    ('29_aniline_cpcm_chloroform.out', 'CPCM'),
])
def test_read_initial_orca_solvation(filename, expected_contains):
    _, solvation_model, _, _, _ = read_initial(orca_path(filename))
    assert expected_contains in solvation_model


# ---------------------------------------------------------------------------
# getoutData: xfail due to cclib ORCA 6 incompatibility
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="cclib cannot parse ORCA 6 output files")
@pytest.mark.parametrize("filename", [
    '01a_water_hf_freq.out',
    '02_ethane_opt_freq_thermo.out',
    '04_benzene_radical_cation.out',
])
def test_getoutData_orca(filename):
    data = getoutData(orca_path(filename))
    assert hasattr(data, 'atom_types')
    assert len(data.atom_types) > 0


# ---------------------------------------------------------------------------
# level_of_theory: xfail (relies on Gaussian archive string format)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="level_of_theory relies on Gaussian archive strings, returns none/none for ORCA")
@pytest.mark.parametrize("filename", ORCA_FREQ_FILES[:3])
def test_level_of_theory_orca(filename):
    lot = level_of_theory(orca_path(filename))
    assert lot != 'none/none'
