#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for parsing ORCA 6 output files using goodvibes.io."""

import pytest
from goodvibes.io import (parse_data, level_of_theory, read_initial,
                          parse_orca_thermo, parse_qcdata)
from conftest import (orca_path, ORCA_FREQ_FILES, ORCA_TS_FILES,
                      ORCA_SP_ONLY_FILES, ORCA_LINEAR_FILES)


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
    ('17_iron_complex_quintet.out', 'Error'),
    ('22_hcn_linear_freq_noraman.out', 'Normal'),
    ('44_ts_sn2_identity_chloride.out', 'Normal'),
    ('47_ts_e2_elimination_ethylchloride.out', 'Error'),
    # Error files — ORCA terminates normally for some error types
    ('52_err_opt_not_converged_maxcycles.out', 'Error'),
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
# parse_orca_thermo: atom extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_atoms, expected_natoms", [
    ('01a_water_hf_freq.out', ['O', 'H', 'H'], 3),
    ('02_ethane_opt_freq_thermo.out', ['C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'], 8),
    ('04_benzene_radical_cation.out', None, 12),
    ('05_methylene_triplet_carbene.out', ['C', 'H', 'H'], 3),
    ('22_hcn_linear_freq_noraman.out', ['H', 'C', 'N'], 3),
])
def test_parse_orca_thermo_atoms(filename, expected_atoms, expected_natoms):
    qcdata = parse_orca_thermo(orca_path(filename))
    assert len(qcdata.atom_nums) == expected_natoms
    assert len(qcdata.atom_types) == expected_natoms
    if expected_atoms is not None:
        assert qcdata.atom_types == expected_atoms


# ---------------------------------------------------------------------------
# parse_orca_thermo: cartesian coordinates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_natoms", [
    ('01a_water_hf_freq.out', 3),
    ('02_ethane_opt_freq_thermo.out', 8),
    ('05_methylene_triplet_carbene.out', 3),
    ('22_hcn_linear_freq_noraman.out', 3),
])
def test_parse_orca_thermo_cartesians(filename, expected_natoms):
    qcdata = parse_orca_thermo(orca_path(filename))
    assert len(qcdata.cartesians) == expected_natoms
    for coord in qcdata.cartesians:
        assert len(coord) == 3


# ---------------------------------------------------------------------------
# level_of_theory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_level", [
    ('01a_water_hf_freq.out', 'HF/6-31G(d)'),
    ('02_ethane_opt_freq_thermo.out', 'B3LYP/6-311+G(d,p)'),
    ('03_acetone_linked_opt_freq.out', 'B3LYP/6-311G(d,p)'),
    ('04_benzene_radical_cation.out', 'TPSS/6-311+G(d,p)'),
    ('05_methylene_triplet_carbene.out', 'MP2/cc-pVDZ'),
    ('06_carbon_atom_single_point.out', 'HF/cc-pVQZ'),
    ('08_alanine_C1_pcm_water.out', 'M062X/def2-SVP'),
    ('10_formaldehyde_verbose_pop.out', 'wB97X-D3/6-31G(d,p)'),
    ('11_hf_molecule_dlpno_ccsdt_gold_standard.out', 'DLPNO-CCSD(T)/cc-pVTZ'),
    ('12_water_anharmonic_vpt2.out', 'HF/def2-SVP'),
    ('13_formaldehyde_tddft_s1.out', 'BP86/def2-SVP'),
    ('17_iron_complex_quintet.out', 'PBE0/def2-SVP'),
    ('19_acetic_acid_smd_dmso.out', 'B3LYP/6-311+G(d,p)'),
    ('20_benzene_singlepoint.out', 'B3LYP/6-311+G(d,p)'),
    ('22_hcn_linear_freq_noraman.out', 'M062X/6-311+G(d,p)'),
    ('28_pyridine_smd_acetonitrile_wb97xd3.out', 'wB97X-D3/6-311+G(d,p)'),
    ('29_aniline_cpcm_chloroform.out', 'M062X/6-311+G(d,p)'),
    ('31_methylammonium_cpcm_water.out', 'M06/6-311+G(d,p)'),
    ('32_cyclohexane_tpss_meta_gga.out', 'TPSS/def2-TZVP'),
    ('33_methanol_pbe_gga.out', 'PBE/6-311G(d,p)'),
    ('34_butadiene_camb3lyp_rsh.out', 'CAM-B3LYP/6-311+G(d,p)'),
    ('35_furan_wb97xv_functional.out', 'wB97X-V/def2-TZVP'),
    ('36_imidazole_pbe0d3bj_noraman.out', 'PBE0/6-311+G(d,p)'),
    ('39_oxazole_tpssh_cpcm_dcm.out', 'TPSSh/6-311+G(d,p)'),
    ('43_dmabn_bhandhlyp_chargetransfer.out', 'BHandHLYP/6-311+G(d,p)'),
    ('44_ts_sn2_identity_chloride.out', 'B3LYP/6-311+G(d,p)'),
    ('48_ts_nh3_umbrella_inversion.out', 'wB97M-V/6-311+G(d,p)'),
    ('49_ts_oh_abstraction_methane.out', 'wB97X-D3/6-311+G(d,p)'),
])
def test_level_of_theory_orca(filename, expected_level):
    lot = level_of_theory(orca_path(filename))
    assert lot == expected_level


# ---------------------------------------------------------------------------
# parse_orca_thermo: SCF energy extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_energy", [
    ('01a_water_hf_freq.out', -76.009114306317),
    ('03_acetone_linked_opt_freq.out', -193.028138231777),
    ('04_benzene_radical_cation.out', -232.018283530455),
    ('05_methylene_triplet_carbene.out', -39.019780938452),
    ('06_carbon_atom_single_point.out', -37.659292822489),
    ('07_neon_atom_with_freq.out', -128.533272824265),
    ('08_alanine_C1_pcm_water.out', -323.378135366042),
    ('16_o2_superoxide_anion.out', -150.341130103125),
    ('22_hcn_linear_freq_noraman.out', -93.414515903409),
    ('32_cyclohexane_tpss_meta_gga.out', -236.018821783392),
    ('44_ts_sn2_identity_chloride.out', -960.291216714078),
])
def test_parse_orca_thermo_scf_energy(filename, expected_energy):
    qcdata = parse_orca_thermo(orca_path(filename))
    assert abs(qcdata.scf_energy - expected_energy) < 1e-8


# ---------------------------------------------------------------------------
# parse_orca_thermo: frequency count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_nfreqs", [
    ('01a_water_hf_freq.out', 3),            # 3 atoms, nonlinear: 3*3-6=3
    ('03_acetone_linked_opt_freq.out', 24),   # 10 atoms: 3*10-6=24
    ('05_methylene_triplet_carbene.out', 3),  # 3 atoms: 3*3-6=3
    ('07_neon_atom_with_freq.out', 0),        # 1 atom: no vibrational modes
    ('16_o2_superoxide_anion.out', 1),        # 2 atoms, linear: 3*2-5=1
    ('22_hcn_linear_freq_noraman.out', 4),    # 3 atoms, linear: 3*3-5=4
    ('32_cyclohexane_tpss_meta_gga.out', 48), # 18 atoms: 3*18-6=48
])
def test_parse_orca_thermo_frequencies(filename, expected_nfreqs):
    qcdata = parse_orca_thermo(orca_path(filename))
    assert len(qcdata.frequency_wn) == expected_nfreqs


# ---------------------------------------------------------------------------
# parse_orca_thermo: imaginary frequencies for TS files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", ORCA_TS_FILES)
def test_parse_orca_thermo_ts_imaginary(filename):
    qcdata = parse_orca_thermo(orca_path(filename))
    assert len(qcdata.im_frequency_wn) >= 1
    for f in qcdata.im_frequency_wn:
        assert f < 0.0


# ---------------------------------------------------------------------------
# parse_orca_thermo: zero-point correction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_zpe", [
    ('01a_water_hf_freq.out', 0.02234428),
    ('03_acetone_linked_opt_freq.out', 0.07868568),
    ('05_methylene_triplet_carbene.out', 0.01776186),
    ('16_o2_superoxide_anion.out', 0.00264481),
    ('22_hcn_linear_freq_noraman.out', 0.01662938),
])
def test_parse_orca_thermo_zpe(filename, expected_zpe):
    qcdata = parse_orca_thermo(orca_path(filename))
    assert abs(qcdata.zero_point_corr - expected_zpe) < 1e-7


# ---------------------------------------------------------------------------
# parse_orca_thermo: linearity detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", ORCA_LINEAR_FILES)
def test_parse_orca_thermo_linear(filename):
    qcdata = parse_orca_thermo(orca_path(filename))
    assert qcdata.linear_mol is True


@pytest.mark.parametrize("filename", [
    '01a_water_hf_freq.out',
    '03_acetone_linked_opt_freq.out',
    '05_methylene_triplet_carbene.out',
])
def test_parse_orca_thermo_nonlinear(filename):
    qcdata = parse_orca_thermo(orca_path(filename))
    assert qcdata.linear_mol is False


# ---------------------------------------------------------------------------
# parse_orca_thermo: job type detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_jobtype", [
    ('01a_water_hf_freq.out', 'Freq'),
    ('03_acetone_linked_opt_freq.out', 'GSFreq'),
    ('06_carbon_atom_single_point.out', 'SP'),
    ('22_hcn_linear_freq_noraman.out', 'GSFreq'),
    ('44_ts_sn2_identity_chloride.out', 'TSFreq'),
])
def test_parse_orca_thermo_job_type(filename, expected_jobtype):
    qcdata = parse_orca_thermo(orca_path(filename))
    assert qcdata.job_type == expected_jobtype


# ---------------------------------------------------------------------------
# parse_orca_thermo: multiplicity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_mult", [
    ('01a_water_hf_freq.out', 1),
    ('04_benzene_radical_cation.out', 2),
    ('05_methylene_triplet_carbene.out', 3),
    ('16_o2_superoxide_anion.out', 2),
])
def test_parse_orca_thermo_multiplicity(filename, expected_mult):
    qcdata = parse_orca_thermo(orca_path(filename))
    assert qcdata.multiplicity == expected_mult


# ---------------------------------------------------------------------------
# parse_orca_thermo: solvation detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_contains", [
    ('08_alanine_C1_pcm_water.out', 'CPCM'),
    ('19_acetic_acid_smd_dmso.out', 'SMD'),
    ('29_aniline_cpcm_chloroform.out', 'CPCM'),
])
def test_parse_orca_thermo_solvation(filename, expected_contains):
    qcdata = parse_orca_thermo(orca_path(filename))
    assert expected_contains in qcdata.solvation_model


# ---------------------------------------------------------------------------
# parse_orca_thermo: SP-only files (no frequencies)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", ORCA_SP_ONLY_FILES)
def test_parse_orca_thermo_sp_only(filename):
    qcdata = parse_orca_thermo(orca_path(filename))
    assert len(qcdata.frequency_wn) == 0
    assert qcdata.zero_point_corr is None


# ---------------------------------------------------------------------------
# parse_orca_thermo: program metadata
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", ORCA_FREQ_FILES[:5])
def test_parse_orca_thermo_program(filename):
    qcdata = parse_orca_thermo(orca_path(filename))
    assert qcdata.program == 'Orca'
    assert 'ORCA version' in qcdata.version_program


# ---------------------------------------------------------------------------
# parse_qcdata: dispatcher routes to ORCA parser
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", [
    '01a_water_hf_freq.out',
    '03_acetone_linked_opt_freq.out',
    '06_carbon_atom_single_point.out',
    '44_ts_sn2_identity_chloride.out',
])
def test_parse_qcdata_dispatches_orca(filename):
    qcdata = parse_qcdata(orca_path(filename))
    assert qcdata.program == 'Orca'
    assert qcdata.scf_energy is not None
