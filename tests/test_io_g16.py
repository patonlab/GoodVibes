#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for parsing Gaussian 16 output files using goodvibes.io."""

import pytest
from goodvibes.io import getoutData, parse_data, level_of_theory, read_initial, gaussian_jobtype
from conftest import g16path, G16_FREQ_FILES, G16_SP_ONLY_FILES, G16_ERROR_FILES


# ---------------------------------------------------------------------------
# getoutData: atom extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_atoms, expected_natoms", [
    ('01a_water_hf_freq.log', ['O', 'H', 'H'], 3),
    ('01b_water_hf_freq_scaled.log', ['O', 'H', 'H'], 3),
    ('01c_water_hf_freq_isotopes.log', ['O', 'H', 'H'], 3),
    ('02_ethane_opt_freq_T398_P2.log', ['C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'], 8),
    ('04_benzene_radical_cation.log', None, 12),
    ('05_methylene_triplet_carbene.log', ['C', 'H', 'H'], 3),
    ('07_neon_atom_with_freq.log', ['Ne'], 1),
    ('10_formaldehyde_verbose_pop.log', ['C', 'O', 'H', 'H'], 4),
    ('12_water_anharmonic_vpt2.log', ['O', 'H', 'H'], 3),
    ('13_formaldehyde_tddft_s1.log', ['C', 'O', 'H', 'H'], 4),
    ('14_water_dimer_counterpoise_bsse.log', ['O', 'H', 'H', 'O', 'H', 'H'], 6),
    ('15_methanol_oniom_qmmm.log', ['C', 'H', 'H', 'H', 'O', 'H'], 6),
    ('16_o2_superoxide_anion.log', ['O', 'O'], 2),
    ('21_naphthalene_pm7_semiempirical.log', None, 16),
    ('22_hcn_linear_freq_noraman.log', ['H', 'C', 'N'], 3),
    ('24_iodobenzene_genecp_sdd.log', None, 12),
    ('25_pd_complex_genecp_def2.log', None, 19),
    ('26_pt_complex_genecp_3zone.log', None, 11),
    ('27_custom_functional_iop_b20lyp.log', ['C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'], 8),
    ('33_methanol_pbepbe_gga.log', ['C', 'O', 'H', 'H', 'H', 'H'], 6),
    ('34_butadiene_camb3lyp_rsh.log', None, 10),
    ('35_furan_mn15_functional.log', ['O', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H'], 9),
    ('36_imidazole_apfd_noraman.log', None, 9),
    ('37_planar_cyclohexane_2nd_order_saddle.log', None, 18),
    ('38_naphthalene_scsmp2.log', None, 16),
    ('39_oxazole_tpssh_cpcm_dcm.log', ['O', 'C', 'N', 'C', 'C', 'H', 'H', 'H'], 8),
    ('41_thiophene_freq_noraman_nmr.log', ['S', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H'], 9),
    ('43_dmabn_bhandhlyp_chargetransfer.log', None, 19),
    ('50_ts_cyclopropane_ring_opening.log', ['C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H'], 9),
])
def test_getoutData_atoms(filename, expected_atoms, expected_natoms):
    data = getoutData(g16path(filename))
    assert len(data.atom_nums) == expected_natoms
    assert len(data.atom_types) == expected_natoms
    if expected_atoms is not None:
        assert data.atom_types == expected_atoms


# ---------------------------------------------------------------------------
# getoutData: frequency extraction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_nfreqs", [
    ('01a_water_hf_freq.log', 3),           # 3 atoms, nonlinear: 3*3-6=3
    ('01b_water_hf_freq_scaled.log', 3),
    ('01c_water_hf_freq_isotopes.log', 3),
    ('02_ethane_opt_freq_T398_P2.log', 18), # 8 atoms: 3*8-6=18
    ('03_acetone_linked_opt_freq.log', 24), # 10 atoms: 3*10-6=24
    ('04_benzene_radical_cation.log', 30),  # 12 atoms: 3*12-6=30
    ('05_methylene_triplet_carbene.log', 3),
    ('10_formaldehyde_verbose_pop.log', 6), # 4 atoms: 3*4-6=6
    ('12_water_anharmonic_vpt2.log', 3),    # 3 atoms: harmonic freqs only
    ('13_formaldehyde_tddft_s1.log', 6),    # 4 atoms: 3*4-6=6
    ('15_methanol_oniom_qmmm.log', 12),     # 6 atoms: 3*6-6=12
    ('16_o2_superoxide_anion.log', 1),      # 2 atoms, linear: 3*2-5=1
    ('21_naphthalene_pm7_semiempirical.log', 42), # 16 atoms: 3*16-6=42 (PM7)
    ('22_hcn_linear_freq_noraman.log', 4),  # 3 atoms, linear: 3*3-5=4
    ('24_iodobenzene_genecp_sdd.log', 30),  # 12 atoms: 3*12-6=30
    ('25_pd_complex_genecp_def2.log', 51),  # 19 atoms: 3*19-6=51
    ('26_pt_complex_genecp_3zone.log', 27), # 11 atoms: 3*11-6=27
    ('27_custom_functional_iop_b20lyp.log', 18), # 8 atoms: 3*8-6=18
    ('32_cyclohexane_tpss_meta_gga.log', 48), # 18 atoms: 3*18-6=48
    ('33_methanol_pbepbe_gga.log', 12),     # 6 atoms: 3*6-6=12
    ('34_butadiene_camb3lyp_rsh.log', 24),  # 10 atoms: 3*10-6=24
    ('35_furan_mn15_functional.log', 21),   # 9 atoms: 3*9-6=21
    ('36_imidazole_apfd_noraman.log', 21),  # 9 atoms: 3*9-6=21
    ('37_planar_cyclohexane_2nd_order_saddle.log', 48), # 18 atoms: 3*18-6=48
    ('38_naphthalene_scsmp2.log', 42),      # 16 atoms: 3*16-6=42
    ('39_oxazole_tpssh_cpcm_dcm.log', 18),  # 8 atoms: 3*8-6=18
    ('41_thiophene_freq_noraman_nmr.log', 21), # 9 atoms: 3*9-6=21
    ('43_dmabn_bhandhlyp_chargetransfer.log', 51), # 19 atoms: 3*19-6=51
])
def test_getoutData_frequencies(filename, expected_nfreqs):
    data = getoutData(g16path(filename))
    assert len(data.FREQS) == expected_nfreqs


# ---------------------------------------------------------------------------
# getoutData: cartesian coordinates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_natoms", [
    ('01a_water_hf_freq.log', 3),
    ('02_ethane_opt_freq_T398_P2.log', 8),
    ('15_methanol_oniom_qmmm.log', 6),
    ('17_iron_complex_quintet.log', 11),
    ('25_pd_complex_genecp_def2.log', 19),
    ('32_cyclohexane_tpss_meta_gga.log', 18),
    ('34_butadiene_camb3lyp_rsh.log', 10),
    ('43_dmabn_bhandhlyp_chargetransfer.log', 19),
])
def test_getoutData_cartesians(filename, expected_natoms):
    data = getoutData(g16path(filename))
    assert len(data.cartesians) == expected_natoms
    for coord in data.cartesians:
        assert len(coord) == 3


# ---------------------------------------------------------------------------
# getoutData: files without frequencies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", G16_SP_ONLY_FILES + ['50_ts_cyclopropane_ring_opening.log'])
def test_getoutData_no_freq(filename):
    data = getoutData(g16path(filename))
    assert not hasattr(data, 'FREQS') or len(data.FREQS) == 0


# ---------------------------------------------------------------------------
# parse_data: SCF energy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_energy", [
    ('01a_water_hf_freq.log', -76.0105109111),
    ('01b_water_hf_freq_scaled.log', -76.0105109111),
    ('01c_water_hf_freq_isotopes.log', -76.0105109111),
    ('02_ethane_opt_freq_T398_P2.log', -79.8565443694),
    ('03_acetone_linked_opt_freq.log', -193.145434374),
    ('04_benzene_radical_cation.log', -232.018282639),
    ('06_carbon_atom_single_point.log', -37.6882982133),
    ('08_alanine_C1_pcm_water.log', -323.372758893),
    ('12_water_anharmonic_vpt2.log', -76.4086180472),
    ('13_formaldehyde_tddft_s1.log', -114.211828974),
    ('14_water_dimer_counterpoise_bsse.log', -152.908379026304),
    ('15_methanol_oniom_qmmm.log', -40.576093511027),
    ('16_o2_superoxide_anion.log', -150.404290823),
    ('17_iron_complex_quintet.log', -1828.86923243),
    ('18_propane_linked_composite_dh.log', -119.0154712666),
    ('20_benzene_singlepoint.log', -232.330087387),
    ('21_naphthalene_pm7_semiempirical.log', 0.5005315171),
    ('22_hcn_linear_freq_noraman.log', -93.4144433480),
    ('24_iodobenzene_genecp_sdd.log', -243.143383234),
    ('25_pd_complex_genecp_def2.log', -969.787908989),
    ('26_pt_complex_genecp_3zone.log', -1563.59861652),
    ('27_custom_functional_iop_b20lyp.log', -79.7757347692),
    ('33_methanol_pbepbe_gga.log', -115.609030933),
    ('34_butadiene_camb3lyp_rsh.log', -155.940737936),
    ('35_furan_mn15_functional.log', -229.846453816),
    ('36_imidazole_apfd_noraman.log', -226.091932029),
    ('37_planar_cyclohexane_2nd_order_saddle.log', -235.839174194),
    ('39_oxazole_tpssh_cpcm_dcm.log', -246.156816878),
    ('41_thiophene_freq_noraman_nmr.log', -553.045047175),
    ('42_dmso_linked_cpcm_gasfreq.log', -553.190794184),
    ('43_dmabn_bhandhlyp_chargetransfer.log', -456.988390233),
    ('44_ts_sn2_identity_chloride.log', -960.457792359),
])
def test_parse_data_energy(filename, expected_energy):
    spe, *_ = parse_data(g16path(filename))
    assert abs(spe - expected_energy) < 1e-6


# ---------------------------------------------------------------------------
# parse_data: program detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", G16_FREQ_FILES + G16_SP_ONLY_FILES)
def test_parse_data_program_gaussian(filename):
    _, program, *_ = parse_data(g16path(filename))
    assert program == "Gaussian"


# ---------------------------------------------------------------------------
# parse_data: charge and multiplicity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_charge, expected_mult", [
    ('01a_water_hf_freq.log', 0, 1),
    ('04_benzene_radical_cation.log', 1, 2),
    ('05_methylene_triplet_carbene.log', 0, 3),
    ('06_carbon_atom_single_point.log', 0, 3),
    ('14_water_dimer_counterpoise_bsse.log', 0, 1),
    ('15_methanol_oniom_qmmm.log', 0, 1),
    ('16_o2_superoxide_anion.log', -1, 2),
    ('17_iron_complex_quintet.log', 0, 5),
    ('21_naphthalene_pm7_semiempirical.log', 0, 1),
    ('44_ts_sn2_identity_chloride.log', -1, 1),
])
def test_parse_data_charge_multiplicity(filename, expected_charge, expected_mult):
    spe, program, version, solvation, file, charge, disp, mult = parse_data(g16path(filename))
    assert charge == expected_charge
    assert mult == expected_mult


# ---------------------------------------------------------------------------
# read_initial: termination status
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_progress", [
    ('01a_water_hf_freq.log', 'Normal'),
    ('02_ethane_opt_freq_T398_P2.log', 'Normal'),
    ('08_alanine_C1_pcm_water.log', 'Normal'),
    ('44_ts_sn2_identity_chloride.log', 'Normal'),
    ('51_err_scf_convergence_fe_complex.log', 'Normal'),
    ('52_err_opt_not_converged_maxcycles.log', 'Error'),
    ('53_err_wrong_charge_multiplicity.log', 'Error'),
    ('54_err_missing_basis_heavy_atom.log', 'Error'),
    ('55_err_insufficient_memory.log', 'Error'),
    ('56_err_timed_out.log', 'Incomplete'),
    ('57_err_syntax_route_typo.log', 'Error'),
    ('58_err_linear_bend_formBX.log', 'Normal'),
    ('59_err_basis_linear_dependency.log', 'Normal'),
    ('60_err_missing_end_blank_line.log', 'Error'),
    ('50_ts_cyclopropane_ring_opening.log', 'Incomplete'),
    ('61_empty.log', 'Incomplete'),
])
def test_read_initial_progress(filename, expected_progress):
    _, _, progress, _, _ = read_initial(g16path(filename))
    assert progress == expected_progress


# ---------------------------------------------------------------------------
# read_initial: solvation model detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_contains", [
    ('01a_water_hf_freq.log', 'gas phase'),
    ('08_alanine_C1_pcm_water.log', 'scrf'),
    ('19_acetic_acid_smd_dmso.log', 'smd'),
    ('28_pyridine_smd_acetonitrile_wb97xd.log', 'smd'),
    ('29_aniline_cpcm_chloroform.log', 'cpcm'),
    ('31_methylammonium_cpcm_water.log', 'cpcm'),
    ('39_oxazole_tpssh_cpcm_dcm.log', 'cpcm'),
])
def test_read_initial_solvation(filename, expected_contains):
    _, solvation_model, _, _, _ = read_initial(g16path(filename))
    assert expected_contains in solvation_model.lower()


# ---------------------------------------------------------------------------
# level_of_theory
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_level", [
    ('01a_water_hf_freq.log', 'HF/6-31G(d)'),
    ('01b_water_hf_freq_scaled.log', 'HF/6-31G(d)'),
    ('01c_water_hf_freq_isotopes.log', 'HF/6-31G(d)'),
    ('02_ethane_opt_freq_T398_P2.log', 'B3LYP/6-311+G(d,p)'),
    ('03_acetone_linked_opt_freq.log', 'B3LYP/6-311G(d,p)'),
    ('04_benzene_radical_cation.log', 'TPSSTPSS/6-311+G(d,p)'),
    ('08_alanine_C1_pcm_water.log', 'M062X/def2SVP'),
    ('10_formaldehyde_verbose_pop.log', 'wB97XD/6-31G(d,p)'),
    ('12_water_anharmonic_vpt2.log', 'B3LYP/6-31G(d)'),
    ('13_formaldehyde_tddft_s1.log', 'B97D TD-FC/def2SVP'),
    ('14_water_dimer_counterpoise_bsse.log', 'B3LYP/6-311+G(d,p)'),
    ('15_methanol_oniom_qmmm.log', 'ONIOM(B3LYP/6-31G(d):PM6)/Mixed'),
    ('18_propane_linked_composite_dh.log', 'B3LYP/6-31G(d)'),
    ('20_benzene_singlepoint.log', 'B3LYP/6-311+G(d,p)'),
    ('21_naphthalene_pm7_semiempirical.log', 'PM7/ZDO'),
    ('22_hcn_linear_freq_noraman.log', 'M062X/6-311+G(d,p)'),
    ('24_iodobenzene_genecp_sdd.log', 'B3LYP/GenECP'),
    ('25_pd_complex_genecp_def2.log', 'PBE1PBE/GenECP'),
    ('26_pt_complex_genecp_3zone.log', 'B3LYP/GenECP'),
    ('27_custom_functional_iop_b20lyp.log', 'BLYP/6-311+G(d,p)'),
    ('28_pyridine_smd_acetonitrile_wb97xd.log', 'wB97XD/6-311+G(d,p)'),
    ('32_cyclohexane_tpss_meta_gga.log', 'TPSSTPSS/def2TZVP'),
    ('33_methanol_pbepbe_gga.log', 'PBEPBE/6-311G(d,p)'),
    ('34_butadiene_camb3lyp_rsh.log', 'CAM-B3LYP/6-311+G(d,p)'),
    ('35_furan_mn15_functional.log', 'MN15/def2TZVP'),
    ('36_imidazole_apfd_noraman.log', 'APFD/6-311+G(d,p)'),
    ('37_planar_cyclohexane_2nd_order_saddle.log', 'B3LYP/6-31G(d)'),
    ('38_naphthalene_scsmp2.log', 'MP2-FC/CC-pVTZ'),
    ('39_oxazole_tpssh_cpcm_dcm.log', 'TPSSh/6-311+G(d,p)'),
    ('41_thiophene_freq_noraman_nmr.log', 'mPW1PW91/6-311+G(2d,p)'),
    ('42_dmso_linked_cpcm_gasfreq.log', 'B3LYP/6-311+G(d,p)'),
    ('43_dmabn_bhandhlyp_chargetransfer.log', 'BHandHLYP/6-311+G(d,p)'),
    ('44_ts_sn2_identity_chloride.log', 'B3LYP/6-311+G(d,p)'),
])
def test_level_of_theory(filename, expected_level):
    lot = level_of_theory(g16path(filename))
    assert lot == expected_level


# ---------------------------------------------------------------------------
# gaussian_jobtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_jobtype", [
    ('01a_water_hf_freq.log', 'Freq'),
    ('01b_water_hf_freq_scaled.log', 'Freq'),
    ('01c_water_hf_freq_isotopes.log', 'Freq'),
    ('02_ethane_opt_freq_T398_P2.log', 'GSFreq'),
    ('03_acetone_linked_opt_freq.log', 'GSFreq'),
    ('06_carbon_atom_single_point.log', 'SP'),
    ('12_water_anharmonic_vpt2.log', 'Freq'),
    ('13_formaldehyde_tddft_s1.log', 'GSFreq'),
    ('14_water_dimer_counterpoise_bsse.log', 'SP'),
    ('15_methanol_oniom_qmmm.log', 'GSFreq'),
    ('20_benzene_singlepoint.log', 'SP'),
    ('21_naphthalene_pm7_semiempirical.log', 'GSFreq'),
    ('24_iodobenzene_genecp_sdd.log', 'GSFreq'),
    ('33_methanol_pbepbe_gga.log', 'GSFreq'),
    ('34_butadiene_camb3lyp_rsh.log', 'GSFreq'),
    ('35_furan_mn15_functional.log', 'GSFreq'),
    ('36_imidazole_apfd_noraman.log', 'GSFreq'),
    ('38_naphthalene_scsmp2.log', 'Freq'),
    ('39_oxazole_tpssh_cpcm_dcm.log', 'GSFreq'),
    ('41_thiophene_freq_noraman_nmr.log', 'GSFreq'),
    ('43_dmabn_bhandhlyp_chargetransfer.log', 'GSFreq'),
    ('44_ts_sn2_identity_chloride.log', 'TSFreq'),
    ('45_ts_diels_alder_butadiene_ethylene.log', 'TSFreq'),
])
def test_gaussian_jobtype(filename, expected_jobtype):
    jt = gaussian_jobtype(g16path(filename))
    assert jt == expected_jobtype
