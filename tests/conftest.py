#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
G16_DIR = os.path.join(TESTS_DIR, 'g16')
ORCA_DIR = os.path.join(TESTS_DIR, 'orca6')

# Legacy path helper for existing test_goodvibes.py
try:
    import goodvibes
    BASEPATH = os.path.join(goodvibes.__path__[0])
except ImportError:
    BASEPATH = os.path.normpath(os.path.join(TESTS_DIR, '..', 'goodvibes'))


def datapath(path):
    return os.path.join(BASEPATH, 'examples', path)


def g16path(filename):
    """Return absolute path to a G16 test file."""
    return os.path.join(G16_DIR, filename)


def orca_path(filename):
    """Return absolute path to an ORCA 6 test file."""
    return os.path.join(ORCA_DIR, filename)


# ---------- Categorized G16 file lists ----------

G16_FREQ_FILES = [
    '01a_water_hf_freq.log',
    '01b_water_hf_freq_scaled.log',
    '01c_water_hf_freq_isotopes.log',
    '02_ethane_opt_freq_T398_P2.log',
    '03_acetone_linked_opt_freq.log',
    '04_benzene_radical_cation.log',
    '05_methylene_triplet_carbene.log',
    '08_alanine_C1_pcm_water.log',
    '10_formaldehyde_verbose_pop.log',
    '12_water_anharmonic_vpt2.log',
    '13_formaldehyde_tddft_s1.log',
    '15_methanol_oniom_qmmm.log',
    '16_o2_superoxide_anion.log',
    '21_naphthalene_pm7_semiempirical.log',
    '22_hcn_linear_freq_noraman.log',
    '23_cs2_linear_anharmonic_noraman.log',
    '24_iodobenzene_genecp_sdd.log',
    '28_pyridine_smd_acetonitrile_wb97xd.log',
    '29_aniline_cpcm_chloroform.log',
    '32_cyclohexane_tpss_meta_gga.log',
    '33_methanol_pbepbe_gga.log',
    '34_butadiene_camb3lyp_rsh.log',
    '35_furan_mn15_functional.log',
    '36_imidazole_apfd_noraman.log',
    '37_planar_cyclohexane_3rd_order_saddle.log',
    '39_oxazole_tpssh_cpcm_dcm.log',
    '40_n2o_linear_highT_highP.log',
    '41_thiophene_freq_noraman_nmr.log',
    '43_dmabn_bhandhlyp_chargetransfer.log',
]

G16_TS_FILES = [
    '44_ts_sn2_identity_chloride.log',
    '45_ts_diels_alder_butadiene_ethylene.log',
    '46_ts_h3_hydrogen_abstraction.log',
    '47_ts_e2_elimination_ethylchloride.log',
    '48_ts_nh3_umbrella_inversion.log',
]

G16_SP_ONLY_FILES = [
    '06_carbon_atom_single_point.log',
    '09_caffeine_nmr_giao.log',
    '11_hf_molecule_ccsdt_gold_standard.log',
    '14_water_dimer_counterpoise_bsse.log',
    '20_benzene_singlepoint.log',
]

G16_LINEAR_FILES = [
    '22_hcn_linear_freq_noraman.log',
    '23_cs2_linear_anharmonic_noraman.log',
    '40_n2o_linear_highT_highP.log',
]

G16_LINKED_FILES = [
    '03_acetone_linked_opt_freq.log',
    '18_propane_linked_composite_dh.log',
    '42_dmso_linked_cpcm_gasfreq.log',
]

G16_SOLVATION_FILES = [
    '08_alanine_C1_pcm_water.log',
    '19_acetic_acid_smd_dmso.log',
    '28_pyridine_smd_acetonitrile_wb97xd.log',
    '29_aniline_cpcm_chloroform.log',
    '30_phenol_smd_thf_pbe0_d3bj.log',
    '31_methylammonium_cpcm_water.log',
    '39_oxazole_tpssh_cpcm_dcm.log',
]

G16_ERROR_FILES = [
    '51_err_scf_convergence_fe_complex.log',
    '52_err_opt_not_converged_maxcycles.log',
    '53_err_wrong_charge_multiplicity.log',
    '54_err_missing_basis_heavy_atom.log',
    '55_err_insufficient_memory.log',
    '56_err_timed_out.log',
    '57_err_syntax_route_typo.log',
    '60_err_missing_end_blank_line.log',
]

# ---------- Categorized ORCA file lists ----------

ORCA_FREQ_FILES = [
    '01a_water_hf_freq.out',
    '02_ethane_opt_freq_thermo.out',
    '01b_water_hf_freq_scaled.out',
    '01c_water_hf_freq_harmonic.out',
    '01d_water_hf_freq_qhcutoff.out',
    '03_acetone_linked_opt_freq.out',
    '04_benzene_radical_cation.out',
    '05_methylene_triplet_carbene.out',
    '07_neon_atom_with_freq.out',
    '08_alanine_C1_pcm_water.out',
    '10_formaldehyde_verbose_pop.out',
    '15_methanol_qmqm2_xtb.out',
    '16_o2_superoxide_anion.out',
    '18_propane_linked_composite_dh.out',
    '19_acetic_acid_smd_dmso.out',
    '21_naphthalene_xtb2_semiempirical.out',
    '22_hcn_linear_freq_noraman.out',
    '23_cs2_linear_anharmonic_noraman.out',
    '24_iodobenzene_genecp_sdd.out',
    '26_pt_complex_genecp_3zone.out',
    '28_pyridine_smd_acetonitrile_wb97xd3.out',
    '29_aniline_cpcm_chloroform.out',
    '30_phenol_smd_thf_pbe0_d3bj.out',
    '31_methylammonium_cpcm_water.out',
    '32_cyclohexane_tpss_meta_gga.out',
    '33_methanol_pbe_gga.out',
    '34_butadiene_camb3lyp_rsh.out',
    '35_furan_wb97xv_functional.out',
    '36_imidazole_pbe0d3bj_noraman.out',
    '37_planar_cyclohexane_3rd_order_saddle.out',
    '38_naphthalene_scsmp2.out',
    '39_oxazole_tpssh_cpcm_dcm.out',
    '42_dmso_linked_cpcm_gasfreq.out',
    '43_dmabn_bhandhlyp_chargetransfer.out',
]

ORCA_TS_FILES = [
    '44_ts_sn2_identity_chloride.out',
    '45_ts_diels_alder_butadiene_ethylene.out',
    '46_ts_neb_cope_rearrangement.out',
    '48_ts_nh3_umbrella_inversion.out',
    '49_ts_oh_abstraction_methane.out',
]

ORCA_SP_ONLY_FILES = [
    '06_carbon_atom_single_point.out',
    '09_caffeine_nmr_giao.out',
    '11_hf_molecule_dlpno_ccsdt_gold_standard.out',
    '13_formaldehyde_tddft_s1.out',
    '20_benzene_singlepoint.out',
    '40_n2o_linear_highT.out',
    '41_thiophene_nmr_giao.out',
]

ORCA_SOLVATION_FILES = [
    '08_alanine_C1_pcm_water.out',
    '19_acetic_acid_smd_dmso.out',
    '28_pyridine_smd_acetonitrile_wb97xd3.out',
    '29_aniline_cpcm_chloroform.out',
    '30_phenol_smd_thf_pbe0_d3bj.out',
    '31_methylammonium_cpcm_water.out',
    '39_oxazole_tpssh_cpcm_dcm.out',
    '42_dmso_linked_cpcm_gasfreq.out',
]

ORCA_LINEAR_FILES = [
    '16_o2_superoxide_anion.out',
    '22_hcn_linear_freq_noraman.out',
    '23_cs2_linear_anharmonic_noraman.out',
]

ORCA_ERROR_FILES = [
    '14_water_dimer_gcp.out',
    '17_iron_complex_quintet.out',
    '47_ts_e2_elimination_ethylchloride.out',
    '51_err_scf_convergence_fe_complex.out',
    '52_err_opt_not_converged_maxcycles.out',
    '53_err_wrong_charge_multiplicity.out',
    '54_err_missing_basis_heavy_atom.out',
    '55_err_insufficient_memory.out',
    '56_err_timed_out.out',
    '57_err_syntax_route_typo.out',
    '58_err_linear_bend_formBX.out',
    '59_err_basis_linear_dependency.out',
    '60_err_missing_end_blank_line.out',
]
