#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for parsing ORCA 6 output files using goodvibes.io."""

import pytest
from goodvibes.io import getoutData, parse_data, level_of_theory, read_initial
from conftest import (
    orca_path, ORCA_FREQ_FILES, ORCA_TS_FILES, ORCA_SP_ONLY_FILES,
    ORCA_SOLVATION_FILES, ORCA_ERROR_FILES,
)


# ---------------------------------------------------------------------------
# parse_data: energy extraction (works for ORCA)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_energy", [
    ('01a_water_hf_freq.out', -76.009114306317),
    ('01b_water_hf_freq_scaled.out', -76.009114306317),
    ('01c_water_hf_freq_harmonic.out', -76.009114306317),
    ('03_acetone_linked_opt_freq.out', -193.028138231777),
    ('04_benzene_radical_cation.out', -232.018283530455),
    ('05_methylene_triplet_carbene.out', -39.019780938452),
    ('06_carbon_atom_single_point.out', -37.659292822489),
    ('07_neon_atom_with_freq.out', -128.533272824265),
    ('08_alanine_C1_pcm_water.out', -323.378135366042),
    ('10_formaldehyde_verbose_pop.out', -114.465473088298),
    ('16_o2_superoxide_anion.out', -150.341130103125),
    ('17_iron_complex_quintet.out', -1828.860022847459),
    ('19_acetic_acid_smd_dmso.out', -229.059207744143),
    ('22_hcn_linear_freq_noraman.out', -93.414515903409),
    ('26_pt_complex_genecp_3zone.out', -1563.147942662415),
    ('28_pyridine_smd_acetonitrile_wb97xd3.out', -248.281486182297),
    ('32_cyclohexane_tpss_meta_gga.out', -236.018821783392),
    ('33_methanol_pbe_gga.out', -115.609294526539),
    ('34_butadiene_camb3lyp_rsh.out', -155.940905942644),
    ('35_furan_wb97xv_functional.out', -230.040953801905),
    ('36_imidazole_pbe0d3bj_noraman.out', -226.023935563514),
    ('44_ts_sn2_identity_chloride.out', -960.291216714078),
])
def test_parse_data_orca_energy(filename, expected_energy):
    spe, program, *_ = parse_data(orca_path(filename))
    assert program == "Orca"
    assert abs(spe - expected_energy) < 1e-8


# ---------------------------------------------------------------------------
# parse_data: SP-only files return valid energy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_energy", [
    ('06_carbon_atom_single_point.out', -37.659292822489),
    ('09_caffeine_nmr_giao.out', -680.388157609241),
    ('11_hf_molecule_dlpno_ccsdt_gold_standard.out', -100.338297387399),
    ('13_formaldehyde_tddft_s1.out', -114.270286611863),
    ('17_iron_complex_quintet.out', -1828.860022847459),
    ('20_benzene_singlepoint.out', -232.176166369807),
    ('40_n2o_linear_highT.out', -183.749273785715),
])
def test_parse_data_orca_sp_only(filename, expected_energy):
    spe, program, *_ = parse_data(orca_path(filename))
    assert program == "Orca"
    assert abs(spe - expected_energy) < 1e-8


# ---------------------------------------------------------------------------
# parse_data: program detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", ORCA_FREQ_FILES + ORCA_TS_FILES)
def test_parse_data_orca_program(filename):
    _, program, *_ = parse_data(orca_path(filename))
    assert program == "Orca"


# ---------------------------------------------------------------------------
# parse_data: charge and multiplicity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_charge, expected_mult", [
    ('01a_water_hf_freq.out', 0, 1),
    ('03_acetone_linked_opt_freq.out', 0, 1),
    ('04_benzene_radical_cation.out', 1, 2),
    ('05_methylene_triplet_carbene.out', 0, 3),
    ('06_carbon_atom_single_point.out', 0, 3),
    ('08_alanine_C1_pcm_water.out', 0, 1),
    ('16_o2_superoxide_anion.out', -1, 2),
    ('17_iron_complex_quintet.out', 0, 5),
    ('19_acetic_acid_smd_dmso.out', 0, 1),
    ('31_methylammonium_cpcm_water.out', 1, 1),
    ('44_ts_sn2_identity_chloride.out', -1, 1),
    ('47_ts_e2_elimination_ethylchloride.out', -1, 1),
    ('49_ts_oh_abstraction_methane.out', 0, 2),
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
    ('05_methylene_triplet_carbene.out', 'Normal'),
    ('08_alanine_C1_pcm_water.out', 'Normal'),
    ('10_formaldehyde_verbose_pop.out', 'Normal'),
    ('17_iron_complex_quintet.out', 'Normal'),
    ('22_hcn_linear_freq_noraman.out', 'Normal'),
    ('44_ts_sn2_identity_chloride.out', 'Normal'),
    # Error files — ORCA terminates normally for some error types
    ('52_err_opt_not_converged_maxcycles.out', 'Normal'),
    ('55_err_insufficient_memory.out', 'Error'),
    ('56_err_timed_out.out', 'Normal'),
    ('53_err_wrong_charge_multiplicity.out', 'Incomplete'),
    ('54_err_missing_basis_heavy_atom.out', 'Incomplete'),
    ('57_err_syntax_route_typo.out', 'Incomplete'),
    ('60_err_missing_end_blank_line.out', 'Incomplete'),
    ('61_empty.out', 'Incomplete'),
])
def test_read_initial_orca_progress(filename, expected_progress):
    _, _, progress, _, _ = read_initial(orca_path(filename))
    assert progress == expected_progress


# ---------------------------------------------------------------------------
# read_initial: solvation model detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_contains", [
    ('01a_water_hf_freq.out', 'gas phase'),
    ('05_methylene_triplet_carbene.out', 'gas phase'),
    ('10_formaldehyde_verbose_pop.out', 'gas phase'),
    ('22_hcn_linear_freq_noraman.out', 'gas phase'),
    ('08_alanine_C1_pcm_water.out', 'CPCM'),
    ('19_acetic_acid_smd_dmso.out', 'SMD'),
    ('28_pyridine_smd_acetonitrile_wb97xd3.out', 'SMD'),
    ('29_aniline_cpcm_chloroform.out', 'CPCM'),
    ('30_phenol_smd_thf_pbe0_d3bj.out', 'SMD'),
    ('31_methylammonium_cpcm_water.out', 'CPCM'),
    ('39_oxazole_tpssh_cpcm_dcm.out', 'CPCM'),
    ('42_dmso_linked_cpcm_gasfreq.out', 'CPCM'),
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
    '05_methylene_triplet_carbene.out',
    '22_hcn_linear_freq_noraman.out',
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
