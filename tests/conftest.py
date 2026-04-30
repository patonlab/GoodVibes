#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
G16_DIR = os.path.join(TESTS_DIR, 'g16')
ORCA_DIR = os.path.join(TESTS_DIR, 'orca6')
ORCA5_DIR = os.path.join(TESTS_DIR, 'orca5')
XTB_DIR = os.path.join(TESTS_DIR, 'xtb')
ASE_DIR = os.path.join(TESTS_DIR, 'ase')
QCHEM_DIR = os.path.join(TESTS_DIR, 'qchem6')

# Legacy path helper for existing test_goodvibes.py
try:
    import goodvibes
    BASEPATH = os.path.join(goodvibes.__path__[0])
except ImportError:
    BASEPATH = os.path.normpath(os.path.join(TESTS_DIR, '..', 'goodvibes'))


def datapath(path):
    """
    Resolve a filesystem path inside the package's examples directory.
    
    Parameters:
        path (str): Relative path or filename within the examples directory.
    
    Returns:
        str: Absolute or normalized filesystem path to the requested examples resource.
    """
    return os.path.join(BASEPATH, 'examples', path)


def g16path(filename):
    """
    Get the filesystem path to a G16 test fixture.
    
    Parameters:
        filename (str): Name of the G16 fixture file (including extension).
    
    Returns:
        str: Filesystem path to the specified file within the G16 tests directory.
    """
    return os.path.join(G16_DIR, filename)


def orca_path(filename):
    """
    Get the filesystem path for an ORCA 6 test fixture.
    
    Parameters:
    	filename (str): Filename relative to the ORCA 6 tests directory.
    
    Returns:
    	path (str): Full filesystem path to the specified ORCA 6 test file.
    """
    return os.path.join(ORCA_DIR, filename)


def orca5_path(filename):
    """
    Get the absolute filesystem path for an ORCA 5 test fixture.
    
    Returns:
        path (str): Absolute path to the fixture file named by `filename`.
    """
    return os.path.join(ORCA5_DIR, filename)


def xtb_path(filename):
    """
    Get the absolute filesystem path to an xTB test fixture.
    
    Parameters:
        filename (str): Name of the fixture file located in the xTB tests directory.
    
    Returns:
        str: Absolute path to the specified xTB fixture file.
    """
    return os.path.join(XTB_DIR, filename)


def ase_path(filename):
    """
    Get the absolute filesystem path for an ASE extxyz test fixture.
    
    Parameters:
        filename (str): Name of the extxyz file relative to the ASE tests directory.
    
    Returns:
        str: Absolute path to the specified ASE extxyz fixture.
    """
    return os.path.join(ASE_DIR, filename)


def qchem_path(filename):
    """
    Return the absolute filesystem path to a Q-Chem 6 test fixture.
    
    Parameters:
        filename (str): Filename relative to the Q-Chem 6 tests directory.
    
    Returns:
        path (str): Absolute path to the requested test file.
    """
    return os.path.join(QCHEM_DIR, filename)


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
    '46_ts_neb_claisen_rearrangement.out',
    '47_ts_e2_elimination_ethylchloride.out',
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

# ---------- Categorized xtb file lists ----------

XTB_FREQ_FILES = [
    '01_water.out',
    '02_ethane.out',
    '03_acetone.out',
    '04_benzene_radical_cation.out',
    '05_methylene_triplet_carbene.out',
    '07_neon_atom_with_freq.out',
    '08_alanine.out',
    '09_caffeine.out',
    '10_formaldehyde.out',
    '11_hf_molecule.out',
    '12_water_dimer.out',
    '13_methanol.out',
    '14_o2_superoxide_anion.out',
    '15_iron_complex_quintet.out',
    '16_propane.out',
    '19_naphthalene.out',
    '20_hcn_linear.out',
    '21_cs2_linear.out',
    '22_iodobenzene.out',
    '23_pd_complex.out',
    '24_pt_complex.out',
    '25_ethane.out',
    '26_pyridine_acetonitrile.out',
    '27_aniline_chloroform.out',
    '28_phenol_thf.out',
    '30_cyclohexane.out',
    '31_methanol.out',
    '32_butadiene.out',
    '33_furan.out',
    '34_imidazole.out',
    '35_planar_cyclohexane_3rd_order_saddle.out',
    '36_naphthalene.out',
    '37_oxazole_dcm.out',
    '38_n2o_linear.out',
    '39_thiophene.out',
    '40_dmso.out',
    '41_dmabn.out',
]

XTB_SP_ONLY_FILES = [
    '06_carbon_atom_single_point.out',
    '17_acetic_acid_dmso.out',
    '18_benzene_singlepoint.out',
    '29_methylammonium_water.out',
]

XTB_SOLVATION_FILES = [
    '17_acetic_acid_dmso.out',          # CPCM-X
    '26_pyridine_acetonitrile.out',     # GBSA
    '27_aniline_chloroform.out',        # GBSA
    '28_phenol_thf.out',                # GBSA
    '29_methylammonium_water.out',      # GBSA, SP only
    '37_oxazole_dcm.out',               # GBSA
]

XTB_LINEAR_FILES = [
    '14_o2_superoxide_anion.out',
    '20_hcn_linear.out',
    '21_cs2_linear.out',
    '38_n2o_linear.out',
]

XTB_GFN1_FILES = [
    '32_butadiene.out',
]

XTB_SADDLE_FILES = [
    '35_planar_cyclohexane_3rd_order_saddle.out',
]

XTB_ERROR_FILES = [
    '42_empty.out',
]

# ---------- Q-Chem 6 file lists ----------
# Mirror of the g16 fixtures translated to Q-Chem 6 inputs and run on acme.
# Two files have no Q-Chem analogue and are intentionally absent from the
# qchem6 directory: 15_methanol_oniom_qmmm (different QM/MM machinery in
# Q-Chem) and 21_naphthalene_pm7 (Q-Chem doesn't ship PM7).

QCHEM_FREQ_FILES = [
    '01a_water_hf_freq.out',
    '01b_water_hf_freq_scaled.out',
    '01c_water_hf_freq_isotopes.out',
    '02_ethane_opt_freq_T398_P2.out',
    '03_acetone_linked_opt_freq.out',
    '04_benzene_radical_cation.out',
    '05_methylene_triplet_carbene.out',
    '08_alanine_C1_pcm_water.out',
    '10_formaldehyde_verbose_pop.out',
    '12_water_anharmonic_vpt2.out',
    '13_formaldehyde_tddft_s1.out',
    '16_o2_superoxide_anion.out',
    '18_propane_linked_composite_dh.out',
    '19_acetic_acid_smd_dmso.out',
    '22_hcn_linear_freq_noraman.out',
    '23_cs2_linear_anharmonic_noraman.out',
    '24_iodobenzene_genecp_sdd.out',
    '28_pyridine_smd_acetonitrile_wb97xd.out',
    '29_aniline_cpcm_chloroform.out',
    '30_phenol_smd_thf_pbe0_d3bj.out',
    '31_methylammonium_cpcm_water.out',
    '32_cyclohexane_tpss_meta_gga.out',
    '33_methanol_pbepbe_gga.out',
    '34_butadiene_camb3lyp_rsh.out',
    '35_furan_mn15_functional.out',
    '36_imidazole_apfd_noraman.out',
    '39_oxazole_tpssh_cpcm_dcm.out',
    '40_n2o_linear_highT_highP.out',
    '41_thiophene_freq_noraman_nmr.out',
    '42_dmso_linked_cpcm_gasfreq.out',
    '43_dmabn_bhandhlyp_chargetransfer.out',
]

QCHEM_TS_FILES = [
    '37_planar_cyclohexane_3rd_order_saddle.out',  # 2nd-order saddle (multiple imaginary)
    '44_ts_sn2_identity_chloride.out',
    '45_ts_diels_alder_butadiene_ethylene.out',
    '46_ts_h3_hydrogen_abstraction.out',
    '47_ts_e2_elimination_ethylchloride.out',
    '48_ts_nh3_umbrella_inversion.out',
]

QCHEM_SP_ONLY_FILES = [
    '06_carbon_atom_single_point.out',
    '07_neon_atom_with_freq.out',          # atom; freq job emits no modes
    '09_caffeine_nmr_giao.out',
    '11_hf_molecule_ccsdt_gold_standard.out',
    '14_water_dimer_counterpoise_bsse.out',
    '20_benzene_singlepoint.out',
]

QCHEM_LINEAR_FILES = [
    '22_hcn_linear_freq_noraman.out',
    '23_cs2_linear_anharmonic_noraman.out',
    '40_n2o_linear_highT_highP.out',
]

QCHEM_SOLVATION_FILES = [
    '08_alanine_C1_pcm_water.out',
    '19_acetic_acid_smd_dmso.out',
    '28_pyridine_smd_acetonitrile_wb97xd.out',
    '29_aniline_cpcm_chloroform.out',
    '30_phenol_smd_thf_pbe0_d3bj.out',
    '31_methylammonium_cpcm_water.out',
    '39_oxazole_tpssh_cpcm_dcm.out',
    '42_dmso_linked_cpcm_gasfreq.out',
]

QCHEM_LINKED_FILES = [
    '03_acetone_linked_opt_freq.out',
    '18_propane_linked_composite_dh.out',  # composite: B3LYP opt+freq, B2PLYP SP
    '42_dmso_linked_cpcm_gasfreq.out',
]

QCHEM_ERROR_FILES = [
    '51_err_scf_convergence_fe_complex.out',
    '52_err_opt_not_converged_maxcycles.out',
    '53_err_wrong_charge_multiplicity.out',
    '54_err_missing_basis_heavy_atom.out',
    '55_err_insufficient_memory.out',
    '56_err_timed_out.out',
    '57_err_syntax_route_typo.out',
    '60_err_missing_end_blank_line.out',
]


# ---------- ASE extxyz file lists ----------
# Generated from the matching g16 logs by tests/ase/_generate.py.
# Each fixture has a g16 counterpart so cross-validation in test_thermo_ase.py
# checks that the same physical inputs give the same H/S/G regardless of source format.

ASE_FREQ_FILES = [
    '01_water.extxyz',
    '05_methylene_triplet.extxyz',
    '08_alanine_pcm_water.extxyz',
    '10_formaldehyde.extxyz',
    '22_hcn_linear.extxyz',
]

ASE_TS_FILES = [
    '44_ts_sn2.extxyz',
]

ASE_LINEAR_FILES = [
    '22_hcn_linear.extxyz',
]

ASE_SOLVATION_FILES = [
    '08_alanine_pcm_water.extxyz',
]

# (extxyz, matching g16 log) pairs used by cross-validation tests.
ASE_G16_PAIRS = [
    ('01_water.extxyz',           '01a_water_hf_freq.log'),
    ('05_methylene_triplet.extxyz','05_methylene_triplet_carbene.log'),
    ('08_alanine_pcm_water.extxyz','08_alanine_C1_pcm_water.log'),
    ('10_formaldehyde.extxyz',    '10_formaldehyde_verbose_pop.log'),
    ('22_hcn_linear.extxyz',      '22_hcn_linear_freq_noraman.log'),
    ('44_ts_sn2.extxyz',          '44_ts_sn2_identity_chloride.log'),
]


# ---------- ORCA 5 file lists (lightweight regression coverage) ----------
# The ORCA 5 suite is intentionally a regression layer rather than a full
# mirror of orca6. ORCA5_FILES is the set used by the broad "parse_qcdata
# doesn't crash" sweep — all non-error, non-broken files.
#
# Notes on previously-excluded files now included:
#   - 07_neon_atom_with_freq.out: the rerun outputs from the cluster still
#     contain stray binary bytes around the XCFun banner (likely a SLURM
#     buffering artefact, not an ORCA bug). The parser now opens with
#     errors='replace' so this is handled silently.
#   - 42_dmso_linked_cpcm_gasfreq.out: a real ORCA 5 rerun. Note this file
#     reports 2 imaginary frequencies where the matching ORCA 6 file
#     reports none — the SCF differs by ~0.07 Ha, suggesting the ORCA 5
#     opt didn't settle as cleanly. Included for parser-robustness coverage.
#   - 46_ts_neb_claisen_rearrangement.out: NEB-TS now succeeds, producing
#     a proper TS with 1 imaginary mode that agrees with ORCA 6 to 5e-6 Ha.

ORCA5_FILES = [
    '01a_water_hf_freq.out',
    '01b_water_hf_freq_scaled.out',
    '01c_water_hf_freq_harmonic.out',
    '01d_water_hf_freq_qhcutoff.out',
    '02_ethane_opt_freq_thermo.out',
    '03_acetone_linked_opt_freq.out',
    '04_benzene_radical_cation.out',
    '05_methylene_triplet_carbene.out',
    '06_carbon_atom_single_point.out',
    '07_neon_atom_with_freq.out',
    '08_alanine_C1_pcm_water.out',
    '09_caffeine_nmr_giao.out',
    '10_formaldehyde_verbose_pop.out',
    '11_hf_molecule_dlpno_ccsdt_gold_standard.out',
    '12_water_anharmonic_vpt2.out',
    '13_formaldehyde_tddft_s1.out',
    '15_methanol_qmqm2_xtb.out',
    '16_o2_superoxide_anion.out',
    '18_propane_linked_composite_dh.out',
    '19_acetic_acid_smd_dmso.out',
    '20_benzene_singlepoint.out',
    '21_naphthalene_xtb2_semiempirical.out',
    '22_hcn_linear_freq_noraman.out',
    '23_cs2_linear_anharmonic_noraman.out',
    '24_iodobenzene_genecp_sdd.out',
    '25_pd_complex_genecp_def2.out',
    '26_pt_complex_genecp_3zone.out',
    '27_custom_functional_camb3lyp.out',
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
    '40_n2o_linear_highT.out',
    '41_thiophene_nmr_giao.out',
    '42_dmso_linked_cpcm_gasfreq.out',
    '43_dmabn_bhandhlyp_chargetransfer.out',
    '44_ts_sn2_identity_chloride.out',
    '45_ts_diels_alder_butadiene_ethylene.out',
    '46_ts_neb_claisen_rearrangement.out',
    '47_ts_e2_elimination_ethylchloride.out',
    '48_ts_nh3_umbrella_inversion.out',
    '49_ts_oh_abstraction_methane.out',
    '50_ts_scants_oh_ch4_abstraction.out',
]

ORCA5_ERROR_FILES = [
    '14_water_dimer_gcp.out',
    '17_iron_complex_quintet.out',
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
