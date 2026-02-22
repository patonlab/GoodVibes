#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for thermochemistry calculations on Gaussian 16 output files."""

import math
import pytest
from goodvibes.thermo import (
    calc_bbe, calc_translational_energy, calc_rotational_energy,
    calc_vibrational_energy, calc_electronic_entropy, calc_damp,
)
from conftest import g16path, G16_SOLVATION_FILES, G16_ERROR_FILES

GAS_CONSTANT = 8.3144621
ATMOS = 101.325
T_DEFAULT = 298.15
CONC_DEFAULT = ATMOS / (GAS_CONSTANT * T_DEFAULT)


def _calc(filename, QS='grimme', QH=False, temp=T_DEFAULT, conc=None, scale=1.0):
    """Helper to run calc_bbe with common defaults."""
    if conc is None:
        conc = ATMOS / (GAS_CONSTANT * temp)
    return calc_bbe(g16path(filename), QS, QH, 100.0, 100.0,
                    temp, conc, scale, None, False, False, 0)


# ===========================================================================
# calc_bbe: SCF energy extraction
# ===========================================================================

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
])
def test_calc_bbe_scf_energy(filename, expected_energy):
    bbe = _calc(filename)
    assert expected_energy == round(bbe.scf_energy, 6)


# ===========================================================================
# calc_bbe: ZPE validation against Gaussian ground truth
# ===========================================================================
# Expected ZPE values are from the "Zero-point correction=" line in each log
# file. Files 01b/01c use scale=0.95 (matching Gaussian's Freq=(Scale=0.95)).

@pytest.mark.parametrize("filename, expected_zpe, scale", [
    ('01a_water_hf_freq.log', 0.022391, 1.0),
    ('01b_water_hf_freq_scaled.log', 0.021271, 0.95),
    ('01c_water_hf_freq_isotopes.log', 0.015484, 0.95),
    ('02_ethane_opt_freq_T398_P2.log', 0.074312, 1.0),
    ('03_acetone_linked_opt_freq.log', 0.083123, 1.0),
    ('04_benzene_radical_cation.log', 0.096143, 1.0),
    ('05_methylene_triplet_carbene.log', 0.017752, 1.0),
    ('07_neon_atom_with_freq.log', 0.000000, 1.0),
    ('08_alanine_C1_pcm_water.log', 0.108829, 1.0),
    ('10_formaldehyde_verbose_pop.log', 0.026990, 1.0),
    ('13_formaldehyde_tddft_s1.log', 0.022788, 1.0),
    ('15_methanol_oniom_qmmm.log', 0.048069, 1.0),
    ('16_o2_superoxide_anion.log', 0.002650, 1.0),
    ('18_propane_linked_composite_dh.log', 0.104121, 1.0),
    ('19_acetic_acid_smd_dmso.log', 0.061204, 1.0),
    ('21_naphthalene_pm7_semiempirical.log', 0.142153, 1.0),
    ('22_hcn_linear_freq_noraman.log', 0.016660, 1.0),
    ('24_iodobenzene_genecp_sdd.log', 0.089780, 1.0),
    ('28_pyridine_smd_acetonitrile_wb97xd.log', 0.089313, 1.0),
    ('29_aniline_cpcm_chloroform.log', 0.117799, 1.0),
    ('30_phenol_smd_thf_pbe0_d3bj.log', 0.105021, 1.0),
    ('31_methylammonium_cpcm_water.log', 0.078862, 1.0),
    ('32_cyclohexane_tpss_meta_gga.log', 0.167579, 1.0),
    ('33_methanol_pbepbe_gga.log', 0.049754, 1.0),
    ('34_butadiene_camb3lyp_rsh.log', 0.085607, 1.0),
    ('35_furan_mn15_functional.log', 0.070614, 1.0),
    ('36_imidazole_apfd_noraman.log', 0.071548, 1.0),
    ('37_planar_cyclohexane_2nd_order_saddle.log', 0.171417, 1.0),
    ('39_oxazole_tpssh_cpcm_dcm.log', 0.058178, 1.0),
    ('40_n2o_linear_highT_highP.log', 0.011275, 1.0),
    ('41_thiophene_freq_noraman_nmr.log', 0.067045, 1.0),
    ('42_dmso_linked_cpcm_gasfreq.log', 0.078872, 1.0),
    ('43_dmabn_bhandhlyp_chargetransfer.log', 0.177537, 1.0),
    ('44_ts_sn2_identity_chloride.log', 0.036855, 1.0),
    ('45_ts_diels_alder_butadiene_ethylene.log', 0.142126, 1.0),
    ('46_ts_h3_hydrogen_abstraction.log', 0.008961, 1.0),
    ('47_ts_e2_elimination_ethylchloride.log', 0.072351, 1.0),
    ('48_ts_nh3_umbrella_inversion.log', 0.033320, 1.0),
    ('12_water_anharmonic_vpt2.log', 0.021762, 1.0),
    ('23_cs2_linear_anharmonic_noraman.log', 0.006942, 1.0),
])
def test_calc_bbe_zpe_vs_gaussian(filename, expected_zpe, scale):
    """Validate calc_bbe ZPE against Gaussian's 'Zero-point correction=' value."""
    bbe = _calc(filename, scale=scale)
    assert abs(bbe.zpe - expected_zpe) < 1e-5


# ===========================================================================
# calc_bbe: enthalpy validation against Gaussian ground truth
# ===========================================================================
# Expected values are from the "Sum of electronic and thermal Enthalpies=" line
# in each log file.  Files 01b/01c use scale=0.95 (matching Gaussian's
# Freq=(Scale=0.95)).  File 02 uses T=398.15 K (non-default temperature).
# Files 12/23 are anharmonic (VPT2) — the parser skips duplicate frequency
# blocks from the VPT2 recap section.
# File 40 is a CCSD job — the parser reads the CCSD energy from the archive.

# Temperature / pressure overrides for files run at non-default conditions.
_TEMP_PRESSURE = {
    '02_ethane_opt_freq_T398_P2.log': (398.15, 2.0),
    '40_n2o_linear_highT_highP.log': (1000.0, 100.0),
}


@pytest.mark.parametrize("filename, expected_enthalpy, scale", [
    ('01a_water_hf_freq.log', -75.984342, 1.0),
    ('01b_water_hf_freq_scaled.log', -75.985462, 0.95),
    ('01c_water_hf_freq_isotopes.log', -75.991239, 0.95),
    ('02_ethane_opt_freq_T398_P2.log', -79.775639, 1.0),
    ('03_acetone_linked_opt_freq.log', -193.123482, 1.0),
    ('04_benzene_radical_cation.log', -231.915905, 1.0),
    ('05_methylene_triplet_carbene.log', -38.998232, 1.0),
    ('07_neon_atom_with_freq.log', -128.530912, 1.0),
    ('08_alanine_C1_pcm_water.log', -323.256198, 1.0),
    ('10_formaldehyde_verbose_pop.log', -114.432190, 1.0),
    ('13_formaldehyde_tddft_s1.log', -114.187368, 1.0),
    ('15_methanol_oniom_qmmm.log', -40.523659, 1.0),
    ('16_o2_superoxide_anion.log', -150.398316, 1.0),
    ('18_propane_linked_composite_dh.log', -119.034669, 1.0),
    ('19_acetic_acid_smd_dmso.log', -229.114966, 1.0),
    ('21_naphthalene_pm7_semiempirical.log', 0.213464, 1.0),
    ('22_hcn_linear_freq_noraman.log', -93.394316, 1.0),
    ('24_iodobenzene_genecp_sdd.log', -243.046767, 1.0),
    ('28_pyridine_smd_acetonitrile_wb97xd.log', -248.172568, 1.0),
    ('29_aniline_cpcm_chloroform.log', -287.435067, 1.0),
    ('30_phenol_smd_thf_pbe0_d3bj.log', -307.103825, 1.0),
    ('31_methylammonium_cpcm_water.log', -96.185969, 1.0),
    ('32_cyclohexane_tpss_meta_gga.log', -235.844194, 1.0),
    ('33_methanol_pbepbe_gga.log', -115.557140, 1.0),
    ('34_butadiene_camb3lyp_rsh.log', -155.849545, 1.0),
    ('35_furan_mn15_functional.log', -229.771205, 1.0),
    ('36_imidazole_apfd_noraman.log', -226.015703, 1.0),
    ('37_planar_cyclohexane_2nd_order_saddle.log', -235.662617, 1.0),
    ('39_oxazole_tpssh_cpcm_dcm.log', -246.094129, 1.0),
    ('41_thiophene_freq_noraman_nmr.log', -552.973020, 1.0),
    ('42_dmso_linked_cpcm_gasfreq.log', -553.196506, 1.0),
    ('43_dmabn_bhandhlyp_chargetransfer.log', -458.104514, 1.0),
    ('44_ts_sn2_identity_chloride.log', -960.415291, 1.0),
    ('45_ts_diels_alder_butadiene_ethylene.log', -234.262894, 1.0),
    ('46_ts_h3_hydrogen_abstraction.log', -1.631659, 1.0),
    ('47_ts_e2_elimination_ethylchloride.log', -615.256024, 1.0),
    ('48_ts_nh3_umbrella_inversion.log', -56.370273, 1.0),
    ('12_water_anharmonic_vpt2.log', -76.383185, 1.0),
    ('23_cs2_linear_anharmonic_noraman.log', -834.555189, 1.0),
    ('40_n2o_linear_highT_highP.log', -184.344167, 1.0),
])
def test_calc_bbe_enthalpy_vs_gaussian(filename, expected_enthalpy, scale):
    """Validate calc_bbe enthalpy against Gaussian's
    'Sum of electronic and thermal Enthalpies=' value."""
    temp, pressure = _TEMP_PRESSURE.get(filename, (T_DEFAULT, 1.0))
    conc = (pressure * ATMOS) / (GAS_CONSTANT * temp)
    bbe = calc_bbe(g16path(filename), 'grimme', False, 100.0, 100.0,
                   temp, conc, scale, None, False, False, 0)
    assert abs(bbe.enthalpy - expected_enthalpy) < 1e-5


# ===========================================================================
# calc_bbe: Gibbs free energy validation against Gaussian ground truth
# ===========================================================================
# Expected values are from the "Sum of electronic and thermal Free Energies="
# line in each log file.  Because calc_bbe uses Grimme quasi-harmonic entropy
# (default s_freq_cutoff=100 cm⁻¹), the Gibbs free energy should match
# Gaussian's standard harmonic value for molecules without very low-frequency
# modes.  Files with low-frequency vibrations may show small differences due
# to the quasi-harmonic treatment, but all tested files here agree within
# 1e-5 Hartree.

@pytest.mark.parametrize("filename, expected_gibbs, scale", [
    ('01a_water_hf_freq.log', -76.005752, 1.0),
    ('01b_water_hf_freq_scaled.log', -76.006871, 0.95),
    ('01c_water_hf_freq_isotopes.log', -76.013724, 0.95),
    ('02_ethane_opt_freq_T398_P2.log', -79.811771, 1.0),
    ('03_acetone_linked_opt_freq.log', -193.158615, 1.0),
    ('04_benzene_radical_cation.log', -231.949639, 1.0),
    ('05_methylene_triplet_carbene.log', -39.020460, 1.0),
    ('07_neon_atom_with_freq.log', -128.547504, 1.0),
    ('08_alanine_C1_pcm_water.log', -323.294481, 1.0),
    ('10_formaldehyde_verbose_pop.log', -114.457002, 1.0),
    ('13_formaldehyde_tddft_s1.log', -114.213205, 1.0),
    ('15_methanol_oniom_qmmm.log', -40.550918, 1.0),
    ('16_o2_superoxide_anion.log', -150.421430, 1.0),
    ('18_propane_linked_composite_dh.log', -119.065772, 1.0),
    ('19_acetic_acid_smd_dmso.log', -229.147625, 1.0),
    ('21_naphthalene_pm7_semiempirical.log', 0.173932, 1.0),
    ('22_hcn_linear_freq_noraman.log', -93.417134, 1.0),
    ('24_iodobenzene_genecp_sdd.log', -243.084711, 1.0),
    ('28_pyridine_smd_acetonitrile_wb97xd.log', -248.204492, 1.0),
    ('29_aniline_cpcm_chloroform.log', -287.470848, 1.0),
    ('30_phenol_smd_thf_pbe0_d3bj.log', -307.139246, 1.0),
    ('31_methylammonium_cpcm_water.log', -96.213477, 1.0),
    ('32_cyclohexane_tpss_meta_gga.log', -235.879243, 1.0),
    ('33_methanol_pbepbe_gga.log', -115.584165, 1.0),
    ('34_butadiene_camb3lyp_rsh.log', -155.880943, 1.0),
    ('35_furan_mn15_functional.log', -229.801433, 1.0),
    ('36_imidazole_apfd_noraman.log', -226.046633, 1.0),
    ('37_planar_cyclohexane_2nd_order_saddle.log', -235.695663, 1.0),
    ('39_oxazole_tpssh_cpcm_dcm.log', -246.124815, 1.0),
    ('41_thiophene_freq_noraman_nmr.log', -553.004545, 1.0),
    ('42_dmso_linked_cpcm_gasfreq.log', -553.231609, 1.0),
    ('43_dmabn_bhandhlyp_chargetransfer.log', -458.149414, 1.0),
    ('44_ts_sn2_identity_chloride.log', -960.447302, 1.0),
    ('45_ts_diels_alder_butadiene_ethylene.log', -234.299599, 1.0),
    ('46_ts_h3_hydrogen_abstraction.log', -1.649526, 1.0),
    ('47_ts_e2_elimination_ethylchloride.log', -615.295437, 1.0),
    ('48_ts_nh3_umbrella_inversion.log', -56.392409, 1.0),
    ('12_water_anharmonic_vpt2.log', -76.404596, 1.0),
    ('23_cs2_linear_anharmonic_noraman.log', -834.582140, 1.0),
    ('40_n2o_linear_highT_highP.log', -184.434385, 1.0),
])
def test_calc_bbe_gibbs_vs_gaussian(filename, expected_gibbs, scale):
    """Validate calc_bbe Gibbs free energy against Gaussian's
    'Sum of electronic and thermal Free Energies=' value."""
    temp, pressure = _TEMP_PRESSURE.get(filename, (T_DEFAULT, 1.0))
    conc = (pressure * ATMOS) / (GAS_CONSTANT * temp)
    bbe = calc_bbe(g16path(filename), 'grimme', False, 100.0, 100.0,
                   temp, conc, scale, None, False, False, 0)
    assert abs(bbe.gibbs_free_energy - expected_gibbs) < 1e-5


# ===========================================================================
# calc_bbe: Grimme quasi-harmonic entropy at 298.15 K
# ===========================================================================

@pytest.mark.parametrize("filename, E, ZPE, H, TS, TqhS, G, qhG", [
    ('01a_water_hf_freq.log', -76.010511, 0.022391, -75.984342, 0.021409, 0.021409, -76.005752, -76.005752),
    ('01b_water_hf_freq_scaled.log', -76.010511, 0.022391, -75.984342, 0.021409, 0.021409, -76.005752, -76.005752),
    ('01c_water_hf_freq_isotopes.log', -76.010511, 0.016299, -75.990427, 0.022482, 0.022482, -76.012908, -76.012908),
    ('02_ethane_opt_freq_T398_P2.log', -79.856544, 0.074312, -79.777802, 0.025856, 0.025858, -79.803658, -79.80366),
    ('03_acetone_linked_opt_freq.log', -193.213023, 0.083123, -193.123482,
     0.035133, 0.033946, -193.158615, -193.157428),
    ('04_benzene_radical_cation.log', -232.018283, 0.096143, -231.915906, 0.033733, 0.033741, -231.949639, -231.949647),
    ('05_methylene_triplet_carbene.log', -39.019781, 0.017752, -38.998232, 0.022228, 0.022228, -39.02046, -39.02046),
    ('08_alanine_C1_pcm_water.log', -323.372759, 0.108829, -323.256198, 0.038283, 0.037853, -323.294481, -323.294051),
    ('10_formaldehyde_verbose_pop.log', -114.462989, 0.02699, -114.43219, 0.024813, 0.024813, -114.457002, -114.457002),
    ('16_o2_superoxide_anion.log', -150.404291, 0.00265, -150.398316, 0.023114, 0.023114, -150.42143, -150.42143),
    ('22_hcn_linear_freq_noraman.log', -93.414443, 0.01666, -93.394316, 0.022818, 0.022819, -93.417134, -93.417134),
    ('28_pyridine_smd_acetonitrile_wb97xd.log', -248.267075, 0.089313, -248.172568,
     0.031925, 0.031927, -248.204492, -248.204495),
    ('29_aniline_cpcm_chloroform.log', -287.559548, 0.117799, -287.435067,
     0.035782, 0.035789, -287.470848, -287.470856),
    ('32_cyclohexane_tpss_meta_gga.log', -236.018587, 0.167579, -235.844194,
     0.035049, 0.035058, -235.879243, -235.879251),
    ('34_butadiene_camb3lyp_rsh.log', -155.940738, 0.085607, -155.849545, 0.031398, 0.031393, -155.880943, -155.880938),
    ('35_furan_mn15_functional.log', -229.846454, 0.070614, -229.771205, 0.030228, 0.030228, -229.801433, -229.801434),
    ('36_imidazole_apfd_noraman.log', -226.091932, 0.071548, -226.015703, 0.030931, 0.030932, -226.046633, -226.046634),
    ('39_oxazole_tpssh_cpcm_dcm.log', -246.156817, 0.058178, -246.094129, 0.030686, 0.030687, -246.124815, -246.124816),
    ('40_n2o_linear_highT_highP.log', -184.371844, 0.011275, -184.356964, 0.024894, 0.024894, -184.381857, -184.381858),
    ('41_thiophene_freq_noraman_nmr.log', -553.045047, 0.067045, -552.97302,
     0.031525, 0.031526, -553.004545, -553.004547),
])
def test_calc_bbe_grimme_298(filename, E, ZPE, H, TS, TqhS, G, qhG):
    bbe = _calc(filename, QS='grimme')
    assert E == round(bbe.scf_energy, 6)
    if hasattr(bbe, 'gibbs_free_energy'):
        assert ZPE == round(bbe.zpe, 6)
        assert H == round(bbe.enthalpy, 6)
        assert TS == round(T_DEFAULT * bbe.entropy, 6)
        assert TqhS == round(T_DEFAULT * bbe.qh_entropy, 6)
        assert G == round(bbe.gibbs_free_energy, 6)
        assert qhG == round(bbe.qh_gibbs_free_energy, 6)


# ===========================================================================
# calc_bbe: Truhlar quasi-harmonic entropy at 298.15 K
# ===========================================================================

@pytest.mark.parametrize("filename, E, ZPE, H, TS, TqhS, G, qhG", [
    ('01a_water_hf_freq.log', -76.010511, 0.022391, -75.984342, 0.021409, 0.021409, -76.005752, -76.005752),
    ('02_ethane_opt_freq_T398_P2.log', -79.856544, 0.074312, -79.777802, 0.025856, 0.025856, -79.803658, -79.803658),
    ('03_acetone_linked_opt_freq.log', -193.213023, 0.083123, -193.123482, 0.035133, 0.033379, -193.158615, -193.15686),
    ('04_benzene_radical_cation.log', -232.018283, 0.096143, -231.915906, 0.033733, 0.033733, -231.949639, -231.949639),
    ('05_methylene_triplet_carbene.log', -39.019781, 0.017752, -38.998232, 0.022228, 0.022228, -39.02046, -39.02046),
    ('08_alanine_C1_pcm_water.log', -323.372759, 0.108829, -323.256198, 0.038283, 0.037836, -323.294481, -323.294034),
    ('10_formaldehyde_verbose_pop.log', -114.462989, 0.02699, -114.43219, 0.024813, 0.024813, -114.457002, -114.457002),
    ('22_hcn_linear_freq_noraman.log', -93.414443, 0.01666, -93.394316, 0.022818, 0.022818, -93.417134, -93.417134),
])
def test_calc_bbe_truhlar_298(filename, E, ZPE, H, TS, TqhS, G, qhG):
    bbe = _calc(filename, QS='truhlar')
    assert E == round(bbe.scf_energy, 6)
    if hasattr(bbe, 'gibbs_free_energy'):
        assert ZPE == round(bbe.zpe, 6)
        assert H == round(bbe.enthalpy, 6)
        assert TS == round(T_DEFAULT * bbe.entropy, 6)
        assert TqhS == round(T_DEFAULT * bbe.qh_entropy, 6)
        assert G == round(bbe.gibbs_free_energy, 6)
        assert qhG == round(bbe.qh_gibbs_free_energy, 6)


# ===========================================================================
# calc_bbe: Quasi-harmonic enthalpy (QH=True, Head-Gordon correction)
# ===========================================================================

@pytest.mark.parametrize("filename, E, ZPE, H, qhH, TS, TqhS, G, qhG", [
    ('01a_water_hf_freq.log', -76.010511, 0.022391, -75.984342, -75.984343,
     0.021409, 0.021409, -76.005752, -76.005752),
    ('02_ethane_opt_freq_T398_P2.log', -79.856544, 0.074312, -79.777802, -79.777811,
     0.025856, 0.025858, -79.803658, -79.803669),
    ('03_acetone_linked_opt_freq.log', -193.213023, 0.083123, -193.123482, -193.124073,
     0.035133, 0.033946, -193.158615, -193.158019),
    ('05_methylene_triplet_carbene.log', -39.019781, 0.017752, -38.998232, -38.998232,
     0.022228, 0.022228, -39.02046, -39.02046),
    ('22_hcn_linear_freq_noraman.log', -93.414443, 0.01666, -93.394316, -93.394316,
     0.022818, 0.022819, -93.417134, -93.417135),
])
def test_calc_bbe_qh_enthalpy(filename, E, ZPE, H, qhH, TS, TqhS, G, qhG):
    bbe = _calc(filename, QH=True)
    assert E == round(bbe.scf_energy, 6)
    if hasattr(bbe, 'gibbs_free_energy'):
        assert ZPE == round(bbe.zpe, 6)
        assert H == round(bbe.enthalpy, 6)
        assert qhH == round(bbe.qh_enthalpy, 6)
        assert TS == round(T_DEFAULT * bbe.entropy, 6)
        assert TqhS == round(T_DEFAULT * bbe.qh_entropy, 6)
        assert G == round(bbe.gibbs_free_energy, 6)
        assert qhG == round(bbe.qh_gibbs_free_energy, 6)


# ===========================================================================
# calc_bbe: temperature variations
# ===========================================================================

@pytest.mark.parametrize("temp, expected_H, expected_TS, expected_G", [
    (100.0, -75.986854, 0.005796, -75.992650),
    (200.0, -75.985587, 0.013349, -75.998936),
    (400.0, -75.983043, 0.030222, -76.013265),
    (500.0, -75.981746, 0.039224, -76.020970),
])
def test_calc_bbe_temperature(temp, expected_H, expected_TS, expected_G):
    bbe = _calc('01a_water_hf_freq.log', temp=temp)
    assert expected_H == round(bbe.enthalpy, 6)
    assert expected_TS == round(temp * bbe.entropy, 6)
    assert expected_G == round(bbe.gibbs_free_energy, 6)


# Ground-truth validation at non-standard temperature / pressure.
# Expected values are from the Gaussian "Sum of electronic and thermal
# Enthalpies=" and "Sum of electronic and thermal Free Energies=" lines
# printed at the temperature and pressure specified in the input file.

@pytest.mark.parametrize("filename, temp, pressure, expected_H, expected_G", [
    ('02_ethane_opt_freq_T398_P2.log', 398.15, 2.0, -79.775639, -79.811771),
    ('40_n2o_linear_highT_highP.log', 1000.0, 100.0, -184.344167, -184.434385),
])
def test_calc_bbe_nonstandard_temp_pressure(filename, temp, pressure,
                                            expected_H, expected_G):
    """Validate calc_bbe at non-standard T/P against Gaussian ground truth."""
    conc = (pressure * ATMOS) / (GAS_CONSTANT * temp)
    bbe = calc_bbe(g16path(filename), 'grimme', False, 100.0, 100.0,
                   temp, conc, 1.0, None, False, False, 0)
    assert abs(bbe.enthalpy - expected_H) < 1e-5
    assert abs(bbe.gibbs_free_energy - expected_G) < 1e-5


# ===========================================================================
# calc_bbe: frequency scaling factor
# ===========================================================================

@pytest.mark.parametrize("scale, expected_zpe, expected_G", [
    (0.909, 0.020353, -76.007789),
    (0.950, 0.021271, -76.006871),
    (1.000, 0.022391, -76.005752),
])
def test_calc_bbe_freq_scaling(scale, expected_zpe, expected_G):
    bbe = _calc('01a_water_hf_freq.log', scale=scale)
    assert expected_zpe == round(bbe.zpe, 6)
    assert expected_G == round(bbe.gibbs_free_energy, 6)


# Ground-truth validation for files computed with non-standard frequency
# scaling.  Expected ZPE, H, and G are from the Gaussian "Zero-point
# correction=", "Sum of electronic and thermal Enthalpies=", and
# "Sum of electronic and thermal Free Energies=" lines.  Both 01b and 01c
# use Freq=(Scale=0.95).

@pytest.mark.parametrize("filename, scale, expected_zpe, expected_H, expected_G", [
    ('01b_water_hf_freq_scaled.log', 0.95, 0.021271, -75.985462, -76.006871),
    ('01c_water_hf_freq_isotopes.log', 0.95, 0.015484, -75.991239, -76.013724),
])
def test_calc_bbe_freq_scaling_vs_gaussian(filename, scale, expected_zpe,
                                           expected_H, expected_G):
    """Validate calc_bbe with non-standard scale factor against Gaussian
    ground truth."""
    bbe = _calc(filename, scale=scale)
    assert abs(bbe.zpe - expected_zpe) < 1e-5
    assert abs(bbe.enthalpy - expected_H) < 1e-5
    assert abs(bbe.gibbs_free_energy - expected_G) < 1e-5


# ===========================================================================
# calc_bbe: transition states (imaginary frequencies)
# ===========================================================================

@pytest.mark.parametrize("filename, E, ZPE, G, qhG, num_imag", [
    ('44_ts_sn2_identity_chloride.log', -960.457792, 0.036855, -960.447302, -960.447302, 1),
    ('45_ts_diels_alder_butadiene_ethylene.log', -234.412283, 0.142126, -234.299599, -234.299576, 1),
    ('46_ts_h3_hydrogen_abstraction.log', -1.644035, 0.008961, -1.649526, -1.649526, 1),
    ('47_ts_e2_elimination_ethylchloride.log', -615.336047, 0.072351, -615.295437, -615.294275, 1),
    ('48_ts_nh3_umbrella_inversion.log', -56.407377, 0.03332, -56.392409, -56.392409, 1),
])
def test_calc_bbe_transition_states(filename, E, ZPE, G, qhG, num_imag):
    bbe = _calc(filename)
    assert E == round(bbe.scf_energy, 6)
    assert ZPE == round(bbe.zpe, 6)
    assert G == round(bbe.gibbs_free_energy, 6)
    assert qhG == round(bbe.qh_gibbs_free_energy, 6)
    assert len(bbe.im_frequency_wn) == num_imag
    assert all(f < 0 for f in bbe.im_frequency_wn)


# ===========================================================================
# calc_bbe: solvation model files run without error
# ===========================================================================

@pytest.mark.parametrize("filename", G16_SOLVATION_FILES)
def test_calc_bbe_solvation(filename):
    bbe = _calc(filename)
    assert hasattr(bbe, 'gibbs_free_energy')


# ===========================================================================
# calc_bbe: linear molecules
# ===========================================================================

@pytest.mark.parametrize("filename, expected_nfreqs", [
    ('22_hcn_linear_freq_noraman.log', 4),   # 3N-5 = 4
    ('23_cs2_linear_anharmonic_noraman.log', 4),  # 3N-5 = 4 (linear)
    ('40_n2o_linear_highT_highP.log', 4),    # 3N-5 = 4
])
def test_calc_bbe_linear(filename, expected_nfreqs):
    bbe = _calc(filename)
    assert hasattr(bbe, 'gibbs_free_energy')
    assert len(bbe.frequency_wn) == expected_nfreqs


# ===========================================================================
# calc_bbe: linked Gaussian jobs
# ===========================================================================

@pytest.mark.parametrize("filename, expected_nfreqs", [
    ('03_acetone_linked_opt_freq.log', 24),
    ('18_propane_linked_composite_dh.log', 27),
    ('42_dmso_linked_cpcm_gasfreq.log', 24),
])
def test_calc_bbe_linked_jobs(filename, expected_nfreqs):
    bbe = _calc(filename)
    assert len(bbe.frequency_wn) == expected_nfreqs
    assert hasattr(bbe, 'gibbs_free_energy')


# ===========================================================================
# calc_bbe: single-point only (no thermochemistry)
# ===========================================================================

@pytest.mark.parametrize("filename, expected_energy", [
    ('06_carbon_atom_single_point.log', -37.688298),
    ('09_caffeine_nmr_giao.log', -680.38766),
    ('11_hf_molecule_ccsdt_gold_standard.log', -100.338356),
    ('14_water_dimer_counterpoise_bsse.log', -152.908379),
    ('20_benzene_singlepoint.log', -232.330087),
])
def test_calc_bbe_sp_only(filename, expected_energy):
    bbe = _calc(filename)
    assert expected_energy == round(bbe.scf_energy, 6)
    assert not hasattr(bbe, 'gibbs_free_energy')


# ===========================================================================
# calc_bbe: error files handled gracefully
# ===========================================================================

@pytest.mark.parametrize("filename", G16_ERROR_FILES)
def test_calc_bbe_error_files(filename):
    """Error files should not cause unhandled crashes."""
    try:
        _calc(filename)
    except (AttributeError, ValueError, IndexError):
        pass  # Expected for severely malformed files
    except SystemExit:
        pass  # Some files trigger sys.exit


# ===========================================================================
# Individual thermo function unit tests
# ===========================================================================

def test_calc_translational_energy():
    """E_trans = 3/2 RT."""
    energy = calc_translational_energy(298.15)
    expected = 1.5 * GAS_CONSTANT * 298.15
    assert abs(energy - expected) < 0.01


def test_calc_rotational_energy_nonlinear():
    """E_rot = 3/2 RT for nonlinear molecules."""
    energy = calc_rotational_energy(298.15)
    expected = 1.5 * GAS_CONSTANT * 298.15
    assert abs(energy - expected) < 0.01


def test_calc_rotational_energy_linear():
    """E_rot = RT for linear molecules."""
    energy = calc_rotational_energy(298.15, linear=True)
    expected = GAS_CONSTANT * 298.15
    assert abs(energy - expected) < 0.01


def test_calc_rotational_energy_atom():
    """Atoms have zero rotational energy."""
    energy = calc_rotational_energy(298.15, monatomic=True)
    assert energy == 0.0


def test_calc_electronic_entropy_singlet():
    """S_elec = R * ln(1) = 0 for singlet."""
    entropy = calc_electronic_entropy(1)
    assert entropy == 0.0


def test_calc_electronic_entropy_doublet():
    """S_elec = R * ln(2) for doublet."""
    entropy = calc_electronic_entropy(2)
    expected = GAS_CONSTANT * math.log(2)
    assert abs(entropy - expected) < 1e-6


def test_calc_electronic_entropy_quintet():
    """S_elec = R * ln(5) for quintet."""
    entropy = calc_electronic_entropy(5)
    expected = GAS_CONSTANT * math.log(5)
    assert abs(entropy - expected) < 1e-6


def test_calc_damp_above_cutoff():
    """Frequencies well above cutoff should give damp close to 1.0."""
    damp = calc_damp([500.0], 100.0)
    assert damp[0] > 0.99


def test_calc_damp_at_cutoff():
    """At cutoff frequency, damp should be 0.5."""
    damp = calc_damp([100.0], 100.0)
    assert abs(damp[0] - 0.5) < 0.01


def test_calc_damp_below_cutoff():
    """Frequencies well below cutoff should give damp close to 0.0."""
    damp = calc_damp([10.0], 100.0)
    assert damp[0] < 0.01


# --- calc_vibrational_energy tests ---

PLANCK = 6.62606957e-34
BOLTZ = 1.3806488e-23
SPEED_C = 2.99792458e10


def _vib_factor(freq, scale, temperature):
    """Compute hv/kT for a single frequency."""
    return (PLANCK * freq * scale * SPEED_C) / (BOLTZ * temperature)


def test_calc_vibrational_energy_single_freq():
    """Single frequency gives correct ZPE + thermal contribution."""
    freq = 1000.0
    T = 298.15
    f = _vib_factor(freq, 1.0, T)
    expected = f * GAS_CONSTANT * T * (0.5 + 1.0 / (math.exp(f) - 1.0))
    result = calc_vibrational_energy(T, [freq])
    assert abs(result - expected) < 1e-6


def test_calc_vibrational_energy_multiple_freqs():
    """Sum over multiple frequencies."""
    freqs = [500.0, 1500.0, 3000.0]
    T = 298.15
    expected = 0.0
    for freq in freqs:
        f = _vib_factor(freq, 1.0, T)
        expected += f * GAS_CONSTANT * T * (0.5 + 1.0 / (math.exp(f) - 1.0))
    result = calc_vibrational_energy(T, freqs)
    assert abs(result - expected) < 1e-6


def test_calc_vibrational_energy_scaling():
    """Frequency scale factor is applied correctly."""
    freq = 1000.0
    T = 298.15
    scale = 0.965
    f = _vib_factor(freq, scale, T)
    expected = f * GAS_CONSTANT * T * (0.5 + 1.0 / (math.exp(f) - 1.0))
    result = calc_vibrational_energy(T, [freq], freq_scale_factor=scale)
    assert abs(result - expected) < 1e-6


def test_calc_vibrational_energy_empty_freqs():
    """Empty frequency list returns zero energy."""
    result = calc_vibrational_energy(298.15, [])
    assert result == 0.0


def test_calc_vibrational_energy_zero_temperature():
    """Temperature of zero raises ValueError."""
    with pytest.raises(ValueError, match="Temperature must be positive"):
        calc_vibrational_energy(0.0, [1000.0])


def test_calc_vibrational_energy_negative_temperature():
    """Negative temperature raises ValueError."""
    with pytest.raises(ValueError, match="Temperature must be positive"):
        calc_vibrational_energy(-10.0, [1000.0])


def test_calc_vibrational_energy_very_low_temperature():
    """Very low temperature raises ValueError for overflow protection."""
    with pytest.raises(ValueError, match="Temperature may be too low"):
        calc_vibrational_energy(0.01, [3000.0])
