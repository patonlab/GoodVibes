#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for parsing Gaussian 16 output files using goodvibes.io."""

import pytest
from goodvibes.io import (parse_data, level_of_theory, read_initial,
                          gaussian_jobtype, parse_gaussian_thermo, parse_qcdata)
from conftest import (g16path, G16_FREQ_FILES, G16_SP_ONLY_FILES,
                      G16_LINEAR_FILES, G16_LINKED_FILES)


# ---------------------------------------------------------------------------
# parse_gaussian_thermo: atom extraction
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
    ('21_naphthalene_pm7_semiempirical.log', None, 18),
    ('22_hcn_linear_freq_noraman.log', ['H', 'C', 'N'], 3),
    ('24_iodobenzene_genecp_sdd.log', None, 12),
    ('33_methanol_pbepbe_gga.log', ['C', 'O', 'H', 'H', 'H', 'H'], 6),
    ('34_butadiene_camb3lyp_rsh.log', None, 10),
    ('35_furan_mn15_functional.log', ['O', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H'], 9),
    ('36_imidazole_apfd_noraman.log', None, 9),
    ('37_planar_cyclohexane_3rd_order_saddle.log', None, 18),
    ('39_oxazole_tpssh_cpcm_dcm.log', ['O', 'C', 'N', 'C', 'C', 'H', 'H', 'H'], 8),
    ('41_thiophene_freq_noraman_nmr.log', ['S', 'C', 'C', 'C', 'C', 'H', 'H', 'H', 'H'], 9),
    ('43_dmabn_bhandhlyp_chargetransfer.log', None, 21),
])
def test_parse_gaussian_thermo_atoms(filename, expected_atoms, expected_natoms):
    qcdata = parse_gaussian_thermo(g16path(filename))
    assert len(qcdata.atom_nums) == expected_natoms
    assert len(qcdata.atom_types) == expected_natoms
    if expected_atoms is not None:
        assert qcdata.atom_types == expected_atoms


# ---------------------------------------------------------------------------
# parse_gaussian_thermo: cartesian coordinates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_natoms", [
    ('01a_water_hf_freq.log', 3),
    ('02_ethane_opt_freq_T398_P2.log', 8),
    ('15_methanol_oniom_qmmm.log', 6),
    ('32_cyclohexane_tpss_meta_gga.log', 18),
    ('34_butadiene_camb3lyp_rsh.log', 10),
    ('43_dmabn_bhandhlyp_chargetransfer.log', 21),
])
def test_parse_gaussian_thermo_cartesians(filename, expected_natoms):
    qcdata = parse_gaussian_thermo(g16path(filename))
    assert len(qcdata.cartesians) == expected_natoms
    for coord in qcdata.cartesians:
        assert len(coord) == 3


# ---------------------------------------------------------------------------
# parse_data: SCF energy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_energy", [
    ('01a_water_hf_freq.log', -76.0105109111),
    ('01b_water_hf_freq_scaled.log', -76.0105109111),
    ('01c_water_hf_freq_isotopes.log', -76.0105109111),
    ('02_ethane_opt_freq_T398_P2.log', -79.8565443694),
    ('03_acetone_linked_opt_freq.log', -193.213023257),
    ('04_benzene_radical_cation.log', -232.018282639),
    ('06_carbon_atom_single_point.log', -37.6882982133),
    ('08_alanine_C1_pcm_water.log', -323.372758893),
    ('12_water_anharmonic_vpt2.log', -76.4087262936),
    ('13_formaldehyde_tddft_s1.log', -114.214128137),
    ('14_water_dimer_counterpoise_bsse.log', -152.908379026304),
    ('15_methanol_oniom_qmmm.log', -40.576093511027),
    ('16_o2_superoxide_anion.log', -150.404290823),
    ('18_propane_linked_composite_dh.log', -119.02630007243),
    ('20_benzene_singlepoint.log', -232.330087387),
    ('21_naphthalene_pm7_semiempirical.log', 0.0633337850263),
    ('22_hcn_linear_freq_noraman.log', -93.4144433480),
    ('24_iodobenzene_genecp_sdd.log', -243.143383234),
    ('33_methanol_pbepbe_gga.log', -115.611164634),
    ('34_butadiene_camb3lyp_rsh.log', -155.940737936),
    ('35_furan_mn15_functional.log', -229.846453816),
    ('36_imidazole_apfd_noraman.log', -226.091932029),
    ('37_planar_cyclohexane_3rd_order_saddle.log', -235.839174194),
    ('39_oxazole_tpssh_cpcm_dcm.log', -246.156816878),
    ('41_thiophene_freq_noraman_nmr.log', -553.045047175),
    ('42_dmso_linked_cpcm_gasfreq.log', -553.282038258),
    ('43_dmabn_bhandhlyp_chargetransfer.log', -458.292489277),
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
    ('51_err_scf_convergence_fe_complex.log', 'Error'),
    ('52_err_opt_not_converged_maxcycles.log', 'Error'),
    ('53_err_wrong_charge_multiplicity.log', 'Error'),
    ('54_err_missing_basis_heavy_atom.log', 'Error'),
    ('55_err_insufficient_memory.log', 'Error'),
    ('56_err_timed_out.log', 'Incomplete'),
    ('57_err_syntax_route_typo.log', 'Error'),
    ('60_err_missing_end_blank_line.log', 'Error'),
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


def test_read_initial_bare_scrf_no_crash(tmp_path):
    """A Gaussian route with bare `scrf` (no body) must not raise IndexError
    when read_initial parses the keyword block. Gaussian defaults the bare
    keyword to PCM/water."""
    log = tmp_path / 'bare_scrf.log'
    log.write_text(
        ' Entering Gaussian System, Link 0=g16\n'
        ' Gaussian 16:  ES64L-G16RevC.01 25-Dec-2019\n'
        ' ----------------------------------------------------------------------\n'
        ' # b3lyp/6-31g(d) opt freq scrf\n'
        ' ----------------------------------------------------------------------\n'
        '\n'
        ' Charge =  0 Multiplicity = 1\n'
        ' Normal termination of Gaussian 16.\n'
    )
    _, solvation_model, progress, _, _ = read_initial(str(log))
    assert 'scrf' in solvation_model.lower()
    assert progress == 'Normal'


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
    ('28_pyridine_smd_acetonitrile_wb97xd.log', 'wB97XD/6-311+G(d,p)'),
    ('32_cyclohexane_tpss_meta_gga.log', 'TPSSTPSS/def2TZVP'),
    ('33_methanol_pbepbe_gga.log', 'PBEPBE/6-311G(d,p)'),
    ('34_butadiene_camb3lyp_rsh.log', 'CAM-B3LYP/6-311+G(d,p)'),
    ('35_furan_mn15_functional.log', 'MN15/def2TZVP'),
    ('36_imidazole_apfd_noraman.log', 'APFD/6-311+G(d,p)'),
    ('37_planar_cyclohexane_3rd_order_saddle.log', 'B3LYP/6-31G(d)'),
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
    ('39_oxazole_tpssh_cpcm_dcm.log', 'GSFreq'),
    ('41_thiophene_freq_noraman_nmr.log', 'GSFreq'),
    ('43_dmabn_bhandhlyp_chargetransfer.log', 'GSFreq'),
    ('44_ts_sn2_identity_chloride.log', 'TSFreq'),
    ('45_ts_diels_alder_butadiene_ethylene.log', 'TSFreq'),
])
def test_gaussian_jobtype(filename, expected_jobtype):
    jt = gaussian_jobtype(g16path(filename))
    assert jt == expected_jobtype


# ---------------------------------------------------------------------------
# parse_gaussian_thermo: SCF energy extraction
# ---------------------------------------------------------------------------
# Expected values match calc_bbe.scf_energy (same line-by-line parse logic).

@pytest.mark.parametrize("filename, expected_energy", [
    ('01a_water_hf_freq.log', -76.010511),
    ('01b_water_hf_freq_scaled.log', -76.010511),
    ('01c_water_hf_freq_isotopes.log', -76.010511),
    ('02_ethane_opt_freq_T398_P2.log', -79.856544),
    ('03_acetone_linked_opt_freq.log', -193.213023),
    ('04_benzene_radical_cation.log', -232.018283),
    ('05_methylene_triplet_carbene.log', -39.019781),
    ('08_alanine_C1_pcm_water.log', -323.372759),
    ('10_formaldehyde_verbose_pop.log', -114.462989),
    ('12_water_anharmonic_vpt2.log', -76.408726),
    ('13_formaldehyde_tddft_s1.log', -114.214128),
    ('15_methanol_oniom_qmmm.log', -40.576094),
    ('16_o2_superoxide_anion.log', -150.404291),
    ('21_naphthalene_pm7_semiempirical.log', 0.063334),
    ('22_hcn_linear_freq_noraman.log', -93.414443),
    ('24_iodobenzene_genecp_sdd.log', -243.143383),
    ('33_methanol_pbepbe_gga.log', -115.611165),
    ('34_butadiene_camb3lyp_rsh.log', -155.940738),
    ('35_furan_mn15_functional.log', -229.846454),
    ('36_imidazole_apfd_noraman.log', -226.091932),
    ('43_dmabn_bhandhlyp_chargetransfer.log', -458.292489),
    ('44_ts_sn2_identity_chloride.log', -960.457792),
])
def test_parse_gaussian_thermo_scf_energy(filename, expected_energy):
    qcdata = parse_gaussian_thermo(g16path(filename))
    assert expected_energy == round(qcdata.scf_energy, 6)


# ---------------------------------------------------------------------------
# parse_gaussian_thermo: frequency count
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_nfreqs", [
    ('01a_water_hf_freq.log', 3),
    ('01b_water_hf_freq_scaled.log', 3),
    ('01c_water_hf_freq_isotopes.log', 3),
    ('02_ethane_opt_freq_T398_P2.log', 18),
    ('03_acetone_linked_opt_freq.log', 24),
    ('04_benzene_radical_cation.log', 30),
    ('05_methylene_triplet_carbene.log', 3),
    ('08_alanine_C1_pcm_water.log', 33),
    ('10_formaldehyde_verbose_pop.log', 6),
    ('13_formaldehyde_tddft_s1.log', 6),
    ('15_methanol_oniom_qmmm.log', 12),
    ('16_o2_superoxide_anion.log', 1),
    ('22_hcn_linear_freq_noraman.log', 4),
    ('24_iodobenzene_genecp_sdd.log', 30),
    ('32_cyclohexane_tpss_meta_gga.log', 48),
    ('33_methanol_pbepbe_gga.log', 12),
    ('34_butadiene_camb3lyp_rsh.log', 24),
    ('35_furan_mn15_functional.log', 21),
    ('36_imidazole_apfd_noraman.log', 21),
    ('39_oxazole_tpssh_cpcm_dcm.log', 18),
    ('41_thiophene_freq_noraman_nmr.log', 21),
    ('43_dmabn_bhandhlyp_chargetransfer.log', 56),
])
def test_parse_gaussian_thermo_frequencies(filename, expected_nfreqs):
    qcdata = parse_gaussian_thermo(g16path(filename))
    assert len(qcdata.frequency_wn) == expected_nfreqs


# ---------------------------------------------------------------------------
# parse_gaussian_thermo: imaginary frequencies for TS files
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_nim", [
    ('44_ts_sn2_identity_chloride.log', 1),
    ('45_ts_diels_alder_butadiene_ethylene.log', 1),
    ('46_ts_h3_hydrogen_abstraction.log', 1),
    ('47_ts_e2_elimination_ethylchloride.log', 1),
    ('48_ts_nh3_umbrella_inversion.log', 1),
    ('37_planar_cyclohexane_3rd_order_saddle.log', 3),
])
def test_parse_gaussian_thermo_imaginary_freqs(filename, expected_nim):
    qcdata = parse_gaussian_thermo(g16path(filename))
    assert len(qcdata.im_frequency_wn) == expected_nim
    for f in qcdata.im_frequency_wn:
        assert f < 0.0


# ---------------------------------------------------------------------------
# parse_gaussian_thermo: zero-point correction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_zpe", [
    ('01a_water_hf_freq.log', 0.022391),
    ('02_ethane_opt_freq_T398_P2.log', 0.074312),
    ('03_acetone_linked_opt_freq.log', 0.083123),
    ('05_methylene_triplet_carbene.log', 0.017752),
    ('10_formaldehyde_verbose_pop.log', 0.026990),
    ('15_methanol_oniom_qmmm.log', 0.048069),
    ('16_o2_superoxide_anion.log', 0.002650),
    ('22_hcn_linear_freq_noraman.log', 0.016660),
    ('33_methanol_pbepbe_gga.log', 0.049754),
    ('44_ts_sn2_identity_chloride.log', 0.036855),
])
def test_parse_gaussian_thermo_zpe(filename, expected_zpe):
    qcdata = parse_gaussian_thermo(g16path(filename))
    assert abs(qcdata.zero_point_corr - expected_zpe) < 1e-5


# ---------------------------------------------------------------------------
# parse_gaussian_thermo: linearity detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", G16_LINEAR_FILES)
def test_parse_gaussian_thermo_linear(filename):
    qcdata = parse_gaussian_thermo(g16path(filename))
    assert qcdata.linear_mol is True


@pytest.mark.parametrize("filename", [
    '01a_water_hf_freq.log',
    '02_ethane_opt_freq_T398_P2.log',
    '05_methylene_triplet_carbene.log',
    '33_methanol_pbepbe_gga.log',
])
def test_parse_gaussian_thermo_nonlinear(filename):
    qcdata = parse_gaussian_thermo(g16path(filename))
    assert qcdata.linear_mol is False


# ---------------------------------------------------------------------------
# parse_gaussian_thermo: linked jobs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", G16_LINKED_FILES)
def test_parse_gaussian_thermo_linked_jobs(filename):
    """Linked jobs should still produce valid parsed data."""
    qcdata = parse_gaussian_thermo(g16path(filename))
    assert qcdata.scf_energy is not None
    assert len(qcdata.frequency_wn) > 0
    assert qcdata.zero_point_corr is not None


# ---------------------------------------------------------------------------
# parse_gaussian_thermo: job type matches gaussian_jobtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_jobtype", [
    ('01a_water_hf_freq.log', 'Freq'),
    ('02_ethane_opt_freq_T398_P2.log', 'GSFreq'),
    ('03_acetone_linked_opt_freq.log', 'GSFreq'),
    ('06_carbon_atom_single_point.log', 'SP'),
    ('20_benzene_singlepoint.log', 'SP'),
    ('44_ts_sn2_identity_chloride.log', 'TSFreq'),
])
def test_parse_gaussian_thermo_job_type(filename, expected_jobtype):
    qcdata = parse_gaussian_thermo(g16path(filename))
    assert qcdata.job_type == expected_jobtype


# ---------------------------------------------------------------------------
# parse_gaussian_thermo: program and provenance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", G16_FREQ_FILES[:5])
def test_parse_gaussian_thermo_program(filename):
    qcdata = parse_gaussian_thermo(g16path(filename))
    assert qcdata.program == 'Gaussian'
    assert 'Gaussian' in qcdata.version_program


# ---------------------------------------------------------------------------
# parse_gaussian_thermo: multiplicity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_mult", [
    ('01a_water_hf_freq.log', 1),
    ('04_benzene_radical_cation.log', 2),
    ('05_methylene_triplet_carbene.log', 3),
    ('16_o2_superoxide_anion.log', 2),
])
def test_parse_gaussian_thermo_multiplicity(filename, expected_mult):
    qcdata = parse_gaussian_thermo(g16path(filename))
    assert qcdata.multiplicity == expected_mult


# ---------------------------------------------------------------------------
# parse_gaussian_thermo: SP-only files (no frequencies)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", G16_SP_ONLY_FILES)
def test_parse_gaussian_thermo_sp_only(filename):
    qcdata = parse_gaussian_thermo(g16path(filename))
    assert len(qcdata.frequency_wn) == 0
    assert qcdata.zero_point_corr is None


# ---------------------------------------------------------------------------
# parse_qcdata: dispatcher routes to Gaussian parser
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", [
    '01a_water_hf_freq.log',
    '02_ethane_opt_freq_T398_P2.log',
    '06_carbon_atom_single_point.log',
    '44_ts_sn2_identity_chloride.log',
])
def test_parse_qcdata_dispatches_gaussian(filename):
    qcdata = parse_qcdata(g16path(filename))
    assert qcdata.program == 'Gaussian'
    assert qcdata.scf_energy is not None
