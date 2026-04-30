#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for thermochemistry calculations on Q-Chem 6 output files.

Structured to mirror tests/test_thermo_g16.py:
  - SCF energy / ZPE / enthalpy / Gibbs ground-truth tables
  - Grimme and Truhlar quasi-harmonic 298 K tables
  - QH=True (Head-Gordon enthalpy correction) table
  - Temperature scan, non-standard T/P, freq scaling
  - Transition states, solvation, linear, linked, single-point, error files

Plus two extra layers specific to Q-Chem:
  - Cross-validation against the matching tests/g16/ fixture
  - Direct comparison against Q-Chem's printed STANDARD THERMODYNAMIC
    QUANTITIES block
"""

import math
import pytest

from goodvibes.thermo import calc_bbe
from conftest import (qchem_path, g16path,
                      QCHEM_FREQ_FILES, QCHEM_TS_FILES,
                      QCHEM_SOLVATION_FILES, QCHEM_ERROR_FILES)


GAS_CONSTANT = 8.3144621
ATMOS = 101.325
T_DEFAULT = 298.15
CONC_DEFAULT = ATMOS / (GAS_CONSTANT * T_DEFAULT)
HA_TO_KCAL = 627.509541
R_CAL_MOL_K = 1.98720425864083  # cal/mol/K


# Files with non-default T/P (matching the g16 twins).
_TEMP_PRESSURE = {
    '02_ethane_opt_freq_T398_P2.out': (398.15, 2.0),
    '40_n2o_linear_highT_highP.out': (1000.0, 100.0),
}


def _calc(filename, QS='grimme', QH=False, temp=T_DEFAULT, conc=None, scale=1.0):
    if conc is None:
        conc = ATMOS / (GAS_CONSTANT * temp)
    return calc_bbe(qchem_path(filename), QS, QH, 100.0, 100.0,
                    temp, conc, scale, None, None, None, 0)


# ===========================================================================
# calc_bbe: SCF energy extraction
# ===========================================================================

@pytest.mark.parametrize("filename, expected_energy", [
    ('01a_water_hf_freq.out', -76.010511),
    ('01b_water_hf_freq_scaled.out', -76.010511),
    ('01c_water_hf_freq_isotopes.out', -76.010511),
    ('02_ethane_opt_freq_T398_P2.out', -79.856532),
    ('03_acetone_linked_opt_freq.out', -193.213019),
    ('04_benzene_radical_cation.out', -232.014198),
    ('05_methylene_triplet_carbene.out', -39.019781),
    ('08_alanine_C1_pcm_water.out', -323.376152),
    ('10_formaldehyde_verbose_pop.out', -114.461164),
    ('12_water_anharmonic_vpt2.out', -76.397315),
    ('13_formaldehyde_tddft_s1.out', -114.317315),
    ('16_o2_superoxide_anion.out', -150.404302),
    ('22_hcn_linear_freq_noraman.out', -93.414078),
    ('24_iodobenzene_genecp_sdd.out', -243.143513),
    ('33_methanol_pbepbe_gga.out', -115.611194),
    ('34_butadiene_camb3lyp_rsh.out', -155.940747),
    ('35_furan_mn15_functional.out', -229.847745),
    ('36_imidazole_apfd_noraman.out', -226.294181),
    ('43_dmabn_bhandhlyp_chargetransfer.out', -458.292545),
])
def test_calc_bbe_scf_energy(filename, expected_energy):
    bbe = _calc(filename)
    assert expected_energy == round(bbe.scf_energy, 6)


# ===========================================================================
# calc_bbe: ZPE
# ===========================================================================
# 01b/01c are scaled at 0.95 to match the g16 freq-scale fixtures. (Q-Chem
# does not bake the scale factor into the output frequencies, so goodvibes
# applies it at thermo time.)

@pytest.mark.parametrize("filename, expected_zpe, scale", [
    ('01a_water_hf_freq.out', 0.022391, 1.0),
    ('01b_water_hf_freq_scaled.out', 0.021271, 0.95),
    ('01c_water_hf_freq_isotopes.out', 0.021271, 0.95),
    ('02_ethane_opt_freq_T398_P2.out', 0.074311, 1.0),
    ('03_acetone_linked_opt_freq.out', 0.083125, 1.0),
    ('04_benzene_radical_cation.out', 0.095144, 1.0),
    ('05_methylene_triplet_carbene.out', 0.017765, 1.0),
    ('08_alanine_C1_pcm_water.out', 0.108525, 1.0),
    ('10_formaldehyde_verbose_pop.out', 0.027099, 1.0),
    ('12_water_anharmonic_vpt2.out', 0.021773, 1.0),
    ('13_formaldehyde_tddft_s1.out', 0.022963, 1.0),
    ('16_o2_superoxide_anion.out', 0.002653, 1.0),
    ('18_propane_linked_composite_dh.out', 0.104162, 1.0),
    ('19_acetic_acid_smd_dmso.out', 0.061138, 1.0),
    ('22_hcn_linear_freq_noraman.out', 0.016653, 1.0),
    ('23_cs2_linear_anharmonic_noraman.out', 0.006953, 1.0),
    ('24_iodobenzene_genecp_sdd.out', 0.089795, 1.0),
    ('28_pyridine_smd_acetonitrile_wb97xd.out', 0.089582, 1.0),
    ('29_aniline_cpcm_chloroform.out', 0.116594, 1.0),
    ('30_phenol_smd_thf_pbe0_d3bj.out', 0.105101, 1.0),
    ('31_methylammonium_cpcm_water.out', 0.078612, 1.0),
    ('32_cyclohexane_tpss_meta_gga.out', 0.167534, 1.0),
    ('33_methanol_pbepbe_gga.out', 0.049801, 1.0),
    ('34_butadiene_camb3lyp_rsh.out', 0.085606, 1.0),
    ('35_furan_mn15_functional.out', 0.070611, 1.0),
    ('36_imidazole_apfd_noraman.out', 0.070926, 1.0),
    ('37_planar_cyclohexane_3rd_order_saddle.out', 0.171655, 1.0),
    ('39_oxazole_tpssh_cpcm_dcm.out', 0.058180, 1.0),
    ('40_n2o_linear_highT_highP.out', 0.011273, 1.0),
    ('41_thiophene_freq_noraman_nmr.out', 0.067030, 1.0),
    ('42_dmso_linked_cpcm_gasfreq.out', 0.076727, 1.0),
    ('43_dmabn_bhandhlyp_chargetransfer.out', 0.177569, 1.0),
    ('44_ts_sn2_identity_chloride.out', 0.036803, 1.0),
    ('45_ts_diels_alder_butadiene_ethylene.out', 0.142195, 1.0),
    ('46_ts_h3_hydrogen_abstraction.out', 0.008967, 1.0),
    ('47_ts_e2_elimination_ethylchloride.out', 0.072487, 1.0),
    ('48_ts_nh3_umbrella_inversion.out', 0.033313, 1.0),
])
def test_calc_bbe_zpe(filename, expected_zpe, scale):
    bbe = _calc(filename, scale=scale)
    assert abs(bbe.zpe - expected_zpe) < 1e-5


# ===========================================================================
# calc_bbe: enthalpy
# ===========================================================================

@pytest.mark.parametrize("filename, expected_enthalpy, scale", [
    ('01a_water_hf_freq.out', -75.984342, 1.0),
    ('01b_water_hf_freq_scaled.out', -75.985461, 0.95),
    ('01c_water_hf_freq_isotopes.out', -75.985461, 0.95),
    ('02_ethane_opt_freq_T398_P2.out', -79.775630, 1.0),
    ('03_acetone_linked_opt_freq.out', -193.123457, 1.0),
    ('04_benzene_radical_cation.out', -231.913250, 1.0),
    ('05_methylene_triplet_carbene.out', -38.998219, 1.0),
    ('08_alanine_C1_pcm_water.out', -323.259873, 1.0),
    ('10_formaldehyde_verbose_pop.out', -114.430257, 1.0),
    ('12_water_anharmonic_vpt2.out', -76.371763, 1.0),
    ('13_formaldehyde_tddft_s1.out', -114.290379, 1.0),
    ('16_o2_superoxide_anion.out', -150.398324, 1.0),
    ('18_propane_linked_composite_dh.out', -118.761993, 1.0),
    ('19_acetic_acid_smd_dmso.out', -229.115470, 1.0),
    ('22_hcn_linear_freq_noraman.out', -93.393957, 1.0),
    ('23_cs2_linear_anharmonic_noraman.out', -834.520938, 1.0),
    ('24_iodobenzene_genecp_sdd.out', -243.046876, 1.0),
    ('28_pyridine_smd_acetonitrile_wb97xd.out', -248.169353, 1.0),
    ('29_aniline_cpcm_chloroform.out', -287.436848, 1.0),
    ('30_phenol_smd_thf_pbe0_d3bj.out', -307.104335, 1.0),
    ('31_methylammonium_cpcm_water.out', -96.198114, 1.0),
    ('32_cyclohexane_tpss_meta_gga.out', -235.840293, 1.0),
    ('33_methanol_pbepbe_gga.out', -115.557144, 1.0),
    ('34_butadiene_camb3lyp_rsh.out', -155.849549, 1.0),
    ('35_furan_mn15_functional.out', -229.772499, 1.0),
    ('36_imidazole_apfd_noraman.out', -226.218547, 1.0),
    ('37_planar_cyclohexane_3rd_order_saddle.out', -235.662542, 1.0),
    ('39_oxazole_tpssh_cpcm_dcm.out', -246.091688, 1.0),
    ('40_n2o_linear_highT_highP.out', -184.344167, 1.0),
    ('41_thiophene_freq_noraman_nmr.out', -552.973071, 1.0),
    ('42_dmso_linked_cpcm_gasfreq.out', -553.108478, 1.0),
    ('43_dmabn_bhandhlyp_chargetransfer.out', -458.104541, 1.0),
    ('44_ts_sn2_identity_chloride.out', -960.415289, 1.0),
    ('45_ts_diels_alder_butadiene_ethylene.out', -234.262014, 1.0),
    ('46_ts_h3_hydrogen_abstraction.out', -1.631654, 1.0),
    ('47_ts_e2_elimination_ethylchloride.out', -615.255958, 1.0),
    ('48_ts_nh3_umbrella_inversion.out', -56.370279, 1.0),
])
def test_calc_bbe_enthalpy(filename, expected_enthalpy, scale):
    temp, pressure = _TEMP_PRESSURE.get(filename, (T_DEFAULT, 1.0))
    conc = (pressure * ATMOS) / (GAS_CONSTANT * temp)
    bbe = _calc(filename, temp=temp, conc=conc, scale=scale)
    assert abs(bbe.enthalpy - expected_enthalpy) < 1e-5


# ===========================================================================
# calc_bbe: Gibbs free energy
# ===========================================================================

@pytest.mark.parametrize("filename, expected_gibbs, scale", [
    ('01a_water_hf_freq.out', -76.005752, 1.0),
    ('01b_water_hf_freq_scaled.out', -76.006871, 0.95),
    ('01c_water_hf_freq_isotopes.out', -76.006871, 0.95),
    ('02_ethane_opt_freq_T398_P2.out', -79.814013, 1.0),
    ('03_acetone_linked_opt_freq.out', -193.158749, 1.0),
    ('04_benzene_radical_cation.out', -231.946929, 1.0),
    ('05_methylene_triplet_carbene.out', -39.020445, 1.0),
    ('08_alanine_C1_pcm_water.out', -323.298302, 1.0),
    ('10_formaldehyde_verbose_pop.out', -114.455067, 1.0),
    ('12_water_anharmonic_vpt2.out', -76.393174, 1.0),
    ('13_formaldehyde_tddft_s1.out', -114.316217, 1.0),
    ('16_o2_superoxide_anion.out', -150.421438, 1.0),
    ('18_propane_linked_composite_dh.out', -118.792343, 1.0),
    ('19_acetic_acid_smd_dmso.out', -229.146185, 1.0),
    ('22_hcn_linear_freq_noraman.out', -93.416776, 1.0),
    ('23_cs2_linear_anharmonic_noraman.out', -834.547891, 1.0),
    ('24_iodobenzene_genecp_sdd.out', -243.084838, 1.0),
    ('28_pyridine_smd_acetonitrile_wb97xd.out', -248.201904, 1.0),
    ('29_aniline_cpcm_chloroform.out', -287.472279, 1.0),
    ('30_phenol_smd_thf_pbe0_d3bj.out', -307.139717, 1.0),
    ('31_methylammonium_cpcm_water.out', -96.225667, 1.0),
    ('32_cyclohexane_tpss_meta_gga.out', -235.875357, 1.0),
    ('33_methanol_pbepbe_gga.out', -115.584124, 1.0),
    ('34_butadiene_camb3lyp_rsh.out', -155.880961, 1.0),
    ('35_furan_mn15_functional.out', -229.802726, 1.0),
    ('36_imidazole_apfd_noraman.out', -226.249526, 1.0),
    ('37_planar_cyclohexane_3rd_order_saddle.out', -235.695575, 1.0),
    ('39_oxazole_tpssh_cpcm_dcm.out', -246.122370, 1.0),
    ('40_n2o_linear_highT_highP.out', -184.434386, 1.0),
    ('41_thiophene_freq_noraman_nmr.out', -553.004596, 1.0),
    ('42_dmso_linked_cpcm_gasfreq.out', -553.140734, 1.0),
    ('43_dmabn_bhandhlyp_chargetransfer.out', -458.150055, 1.0),
    ('44_ts_sn2_identity_chloride.out', -960.449028, 1.0),
    ('45_ts_diels_alder_butadiene_ethylene.out', -234.298711, 1.0),
    ('46_ts_h3_hydrogen_abstraction.out', -1.649520, 1.0),
    ('47_ts_e2_elimination_ethylchloride.out', -615.294823, 1.0),
    ('48_ts_nh3_umbrella_inversion.out', -56.392415, 1.0),
])
def test_calc_bbe_gibbs(filename, expected_gibbs, scale):
    temp, pressure = _TEMP_PRESSURE.get(filename, (T_DEFAULT, 1.0))
    conc = (pressure * ATMOS) / (GAS_CONSTANT * temp)
    bbe = _calc(filename, temp=temp, conc=conc, scale=scale)
    assert abs(bbe.gibbs_free_energy - expected_gibbs) < 1e-5


# ===========================================================================
# calc_bbe: Grimme quasi-harmonic entropy at 298.15 K
# ===========================================================================

@pytest.mark.parametrize("filename, E, ZPE, H, TS, TqhS, G, qhG", [
    ('01a_water_hf_freq.out', -76.010511, 0.022391, -75.984342, 0.021409, 0.021409, -76.005752, -76.005752),
    ('04_benzene_radical_cation.out', -232.014198, 0.095144, -231.91325, 0.03368, 0.033686, -231.946929, -231.946936),
    ('05_methylene_triplet_carbene.out', -39.019781, 0.017765, -38.998219, 0.022226, 0.022226, -39.020445, -39.020445),
    ('08_alanine_C1_pcm_water.out', -323.376152, 0.108525, -323.259873, 0.038429, 0.037913, -323.298302, -323.297786),
    ('10_formaldehyde_verbose_pop.out', -114.461164, 0.027099, -114.430257, 0.024810, 0.024810, -114.455067, -114.455067),
    ('16_o2_superoxide_anion.out', -150.404302, 0.002653, -150.398324, 0.023113, 0.023113, -150.421438, -150.421438),
    ('22_hcn_linear_freq_noraman.out', -93.414078, 0.016653, -93.393957, 0.022819, 0.022819, -93.416776, -93.416776),
    ('28_pyridine_smd_acetonitrile_wb97xd.out', -248.264110, 0.089582, -248.169353, 0.032551, 0.032554, -248.201904, -248.201907),
    ('29_aniline_cpcm_chloroform.out', -287.559893, 0.116594, -287.436848, 0.035431, 0.035437, -287.472279, -287.472285),
    ('32_cyclohexane_tpss_meta_gga.out', -236.014649, 0.167534, -235.840293, 0.035064, 0.035073, -235.875357, -235.875366),
    ('34_butadiene_camb3lyp_rsh.out', -155.940747, 0.085606, -155.849549, 0.031412, 0.031407, -155.880961, -155.880956),
    ('35_furan_mn15_functional.out', -229.847745, 0.070611, -229.772499, 0.030228, 0.030228, -229.802726, -229.802727),
    ('36_imidazole_apfd_noraman.out', -226.294181, 0.070926, -226.218547, 0.030978, 0.030979, -226.249526, -226.249527),
    ('39_oxazole_tpssh_cpcm_dcm.out', -246.154375, 0.058180, -246.091688, 0.030683, 0.030683, -246.122370, -246.122371),
    ('41_thiophene_freq_noraman_nmr.out', -553.045085, 0.067030, -552.973071, 0.031525, 0.031527, -553.004596, -553.004598),
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
    ('01a_water_hf_freq.out', -76.010511, 0.022391, -75.984342, 0.021409, 0.021409, -76.005752, -76.005752),
    ('03_acetone_linked_opt_freq.out', -193.213019, 0.083125, -193.123457, 0.035292, 0.033430, -193.158749, -193.156887),
    ('04_benzene_radical_cation.out', -232.014198, 0.095144, -231.91325, 0.033680, 0.033680, -231.946929, -231.946929),
    ('05_methylene_triplet_carbene.out', -39.019781, 0.017765, -38.998219, 0.022226, 0.022226, -39.020445, -39.020445),
    ('08_alanine_C1_pcm_water.out', -323.376152, 0.108525, -323.259873, 0.038429, 0.037851, -323.298302, -323.297724),
    ('10_formaldehyde_verbose_pop.out', -114.461164, 0.027099, -114.430257, 0.024810, 0.024810, -114.455067, -114.455067),
    ('22_hcn_linear_freq_noraman.out', -93.414078, 0.016653, -93.393957, 0.022819, 0.022819, -93.416776, -93.416776),
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
    ('01a_water_hf_freq.out', -76.010511, 0.022391, -75.984342, -75.984342,
     0.021409, 0.021409, -76.005752, -76.005752),
    ('03_acetone_linked_opt_freq.out', -193.213019, 0.083125, -193.123457, -193.124068,
     0.035292, 0.034038, -193.158749, -193.158107),
    ('05_methylene_triplet_carbene.out', -39.019781, 0.017765, -38.998219, -38.998219,
     0.022226, 0.022226, -39.020445, -39.020445),
    ('22_hcn_linear_freq_noraman.out', -93.414078, 0.016653, -93.393957, -93.393958,
     0.022819, 0.022819, -93.416776, -93.416777),
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
    bbe = _calc('01a_water_hf_freq.out', temp=temp)
    assert expected_H == round(bbe.enthalpy, 6)
    assert expected_TS == round(temp * bbe.entropy, 6)
    assert expected_G == round(bbe.gibbs_free_energy, 6)


# Non-standard T/P fixtures: Q-Chem job runs at the specified T and P.
# Expected H/G come from calc_bbe at the matching T/P (regression layer).

@pytest.mark.parametrize("filename, temp, pressure, expected_H, expected_G", [
    ('02_ethane_opt_freq_T398_P2.out', 398.15, 2.0, -79.775630, -79.814013),
    ('40_n2o_linear_highT_highP.out', 1000.0, 100.0, -184.344167, -184.434386),
])
def test_calc_bbe_nonstandard_temp_pressure(filename, temp, pressure,
                                            expected_H, expected_G):
    conc = (pressure * ATMOS) / (GAS_CONSTANT * temp)
    bbe = _calc(filename, temp=temp, conc=conc)
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
    bbe = _calc('01a_water_hf_freq.out', scale=scale)
    assert expected_zpe == round(bbe.zpe, 6)
    assert expected_G == round(bbe.gibbs_free_energy, 6)


# ===========================================================================
# calc_bbe: transition states (imaginary frequencies)
# ===========================================================================

@pytest.mark.parametrize("filename, E, ZPE, G, qhG, num_imag", [
    ('44_ts_sn2_identity_chloride.out', -960.457751, 0.036803, -960.449028, -960.449027, 1),
    ('45_ts_diels_alder_butadiene_ethylene.out', -234.411469, 0.142195, -234.298711, -234.298690, 1),
    ('46_ts_h3_hydrogen_abstraction.out', -1.644035, 0.008967, -1.649520, -1.649520, 1),
    ('47_ts_e2_elimination_ethylchloride.out', -615.336054, 0.072487, -615.294823, -615.293999, 1),
    ('48_ts_nh3_umbrella_inversion.out', -56.407376, 0.033313, -56.392415, -56.392415, 1),
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

@pytest.mark.parametrize("filename", QCHEM_SOLVATION_FILES)
def test_calc_bbe_solvation(filename):
    bbe = _calc(filename)
    assert hasattr(bbe, 'gibbs_free_energy')


# ===========================================================================
# calc_bbe: linear molecules
# ===========================================================================

@pytest.mark.parametrize("filename, expected_nfreqs", [
    ('22_hcn_linear_freq_noraman.out', 4),    # 3N-5 = 4
    ('23_cs2_linear_anharmonic_noraman.out', 4),
    ('40_n2o_linear_highT_highP.out', 4),
])
def test_calc_bbe_linear(filename, expected_nfreqs):
    temp, pressure = _TEMP_PRESSURE.get(filename, (T_DEFAULT, 1.0))
    conc = (pressure * ATMOS) / (GAS_CONSTANT * temp)
    bbe = _calc(filename, temp=temp, conc=conc)
    assert hasattr(bbe, 'gibbs_free_energy')
    assert len(bbe.frequency_wn) == expected_nfreqs


# ===========================================================================
# calc_bbe: linked Q-Chem jobs (opt+freq, separate single-point)
# ===========================================================================

@pytest.mark.parametrize("filename, expected_nfreqs", [
    ('03_acetone_linked_opt_freq.out', 24),
    ('18_propane_linked_composite_dh.out', 27),
    ('42_dmso_linked_cpcm_gasfreq.out', 21),
])
def test_calc_bbe_linked_jobs(filename, expected_nfreqs):
    bbe = _calc(filename)
    assert len(bbe.frequency_wn) == expected_nfreqs
    assert hasattr(bbe, 'gibbs_free_energy')


# ===========================================================================
# calc_bbe: single-point only (no thermochemistry)
# ===========================================================================

@pytest.mark.parametrize("filename, expected_energy", [
    ('06_carbon_atom_single_point.out', -37.688298),
    ('09_caffeine_nmr_giao.out', -680.387782),
    ('11_hf_molecule_ccsdt_gold_standard.out', -100.338356),
    ('14_water_dimer_counterpoise_bsse.out', -152.909092),
    ('20_benzene_singlepoint.out', -232.330116),
])
def test_calc_bbe_sp_only(filename, expected_energy):
    bbe = _calc(filename)
    assert expected_energy == round(bbe.scf_energy, 6)
    assert not hasattr(bbe, 'gibbs_free_energy')


# ===========================================================================
# calc_bbe: error files handled gracefully
# ===========================================================================

@pytest.mark.parametrize("filename", QCHEM_ERROR_FILES)
def test_calc_bbe_error_files(filename):
    """Error files should not cause unhandled crashes."""
    try:
        _calc(filename)
    except (AttributeError, ValueError, IndexError):
        pass
    except SystemExit:
        pass


# ===========================================================================
# Don't-crash sweep across all freq + TS fixtures
# ===========================================================================

@pytest.mark.parametrize("filename", QCHEM_FREQ_FILES + QCHEM_TS_FILES)
def test_calc_bbe_completes_for_all_freq_files(filename):
    """Every freq/TS fixture should produce a Gibbs free energy without
    crashing — the parser populates all the QCData fields calc_bbe needs."""
    temp, pressure = _TEMP_PRESSURE.get(filename, (T_DEFAULT, 1.0))
    conc = (pressure * ATMOS) / (GAS_CONSTANT * temp)
    bbe = _calc(filename, temp=temp, conc=conc)
    assert bbe.qh_gibbs_free_energy is not None
    assert bbe.zpe is not None


# ===========================================================================
# Cross-validation against matching g16 fixtures (rough numerical agreement)
# ===========================================================================

# Pairs where the Q-Chem and g16 inputs use exactly the same method/basis;
# expect ZPE and H to agree to ~5e-4 Hartree (SCF/grid defaults differ
# slightly between programs). Several g16 fixtures use functionals that
# Q-Chem maps slightly differently (TPSSTPSS→tpss, PBEPBE→pbe, APFD subbed
# with B3LYP+D3BJ) so we only cross-validate the cleanest pairs.

@pytest.mark.parametrize("qcname, g16name, tol_h", [
    ('01a_water_hf_freq.out', '01a_water_hf_freq.log', 1e-3),
    ('05_methylene_triplet_carbene.out', '05_methylene_triplet_carbene.log', 5e-3),  # MP2 vs UMP2 differ slightly
    ('10_formaldehyde_verbose_pop.out', '10_formaldehyde_verbose_pop.log', 5e-3),
    ('22_hcn_linear_freq_noraman.out', '22_hcn_linear_freq_noraman.log', 5e-3),
    ('44_ts_sn2_identity_chloride.out', '44_ts_sn2_identity_chloride.log', 5e-3),
])
def test_qchem_vs_g16_thermo_agreement(qcname, g16name, tol_h):
    """Same molecule, same level of theory in Q-Chem and Gaussian: ZPE
    and H should agree to within `tol_h` Hartree (SCF noise + grid
    differences between programs)."""
    a = calc_bbe(qchem_path(qcname), 'grimme', False, 100.0, 100.0,
                 T_DEFAULT, CONC_DEFAULT, 1.0, None, None, None, 0)
    b = calc_bbe(g16path(g16name), 'grimme', False, 100.0, 100.0,
                 T_DEFAULT, CONC_DEFAULT, 1.0, None, None, None, 0)
    assert abs(a.zpe - b.zpe) < tol_h
    assert abs(a.enthalpy - b.enthalpy) < tol_h


# ===========================================================================
# Ground-truth comparison: calc_bbe vs Q-Chem's printed thermo block
# ===========================================================================
# Q-Chem prints in its STANDARD THERMODYNAMIC QUANTITIES block:
#   "Zero point vibrational energy:  N kcal/mol"
#   "Total Enthalpy:                  N kcal/mol"  (= H_corr; SCF separate)
#   "Total Entropy:                   N cal/mol.K"
#   "QRRHO-Total Entropy:             N cal/mol.K" (Grimme mRRHO, α=4, ω=100)
# Q-Chem does NOT add electronic entropy R*ln(2S+1); goodvibes does. We
# subtract it from goodvibes' value for fair comparison.

def _qchem_thermo_block(path):
    """Extract printed ZPE / H / S / qh-S (kcal/mol or cal/mol/K) from the
    LAST harmonic STANDARD THERMODYNAMIC QUANTITIES block."""
    with open(path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    last = -1
    for i, line in enumerate(lines):
        if 'STANDARD THERMODYNAMIC QUANTITIES AT' in line:
            last = i
    if last < 0:
        return None
    out = {}
    for line in lines[last:]:
        s = line.strip()
        if s.startswith('Zero point vibrational energy:'):
            out['zpe'] = float(s.split(':')[1].split()[0])
        elif s.startswith('Total Enthalpy:'):
            out['h'] = float(s.split(':')[1].split()[0])
        elif s.startswith('Total Entropy:'):
            out['s'] = float(s.split(':')[1].split()[0])
        elif s.startswith('QRRHO-Total Entropy:'):
            out['qhs'] = float(s.split(':')[1].split()[0])
        if 'Thank you very much' in line:
            break
    return out


# Files where calc_bbe should agree with Q-Chem's printed values to <0.01
# kcal/mol on H/ZPE and <0.05 cal/mol/K on S/qhS. Excludes:
#   - 12 and 23 (VPT2 anharmonic): Q-Chem reports anharmonic ZPE in the
#     thermo block; goodvibes uses harmonic. The difference IS the
#     anharmonic correction (-0.69 kcal/mol for water, -0.05 for CS2).
QCHEM_THERMO_PARITY_FILES = [
    f for f in (QCHEM_FREQ_FILES + QCHEM_TS_FILES)
    if f not in ('12_water_anharmonic_vpt2.out',
                 '23_cs2_linear_anharmonic_noraman.out')
]


@pytest.mark.parametrize("filename", QCHEM_THERMO_PARITY_FILES)
def test_calc_bbe_matches_qchem_printed_thermo(filename):
    qref = _qchem_thermo_block(qchem_path(filename))
    if not qref or 'h' not in qref:
        pytest.skip("no Q-Chem thermo block in file")
    # Q-Chem always prints its thermo block at 298.15 K / 1 atm regardless
    # of the analogous Gaussian job's T/P — compare at default conditions.
    bbe = _calc(filename)
    s_elec = R_CAL_MOL_K * math.log(bbe.multiplicity)
    zpe_gv = bbe.zpe * HA_TO_KCAL
    h_corr_gv = (bbe.enthalpy - bbe.scf_energy) * HA_TO_KCAL
    s_gv = bbe.entropy * HA_TO_KCAL * 1000 - s_elec
    qhs_gv = bbe.qh_entropy * HA_TO_KCAL * 1000 - s_elec
    assert abs(zpe_gv - qref['zpe']) < 0.01, f"ZPE: {zpe_gv:.4f} vs {qref['zpe']:.4f}"
    assert abs(h_corr_gv - qref['h']) < 0.01, f"H: {h_corr_gv:.4f} vs {qref['h']:.4f}"
    assert abs(s_gv - qref['s']) < 0.05, f"S: {s_gv:.4f} vs {qref['s']:.4f}"
    if 'qhs' in qref:
        assert abs(qhs_gv - qref['qhs']) < 0.05, f"qhS: {qhs_gv:.4f} vs {qref['qhs']:.4f}"
