#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for thermochemistry calculations on ORCA 6 output files.

All calc_bbe tests are marked xfail because:
  - cclib (used by getoutData) cannot parse ORCA 6 output
  - thermo.py lacks an ORCA-specific frequency/rotational constant parsing block

Ground-truth values are extracted from the ORCA output files' thermochemistry
sections ("Zero point energy", "Total Enthalpy", "Final entropy term",
"Final Gibbs free energy").  ORCA's "Final Gibbs free energy" uses quasi-RRHO
with a default reference frequency of 100 cm⁻¹, matching GoodVibes' default.
For files with multiple thermochemistry sections (opt+freq jobs), the LAST
section's values are used (corresponding to the final optimized geometry).
"""

import pytest
from goodvibes.thermo import calc_bbe
from conftest import (
    orca_path, ORCA_FREQ_FILES, ORCA_TS_FILES, ORCA_SP_ONLY_FILES,
    ORCA_SOLVATION_FILES, ORCA_ERROR_FILES,
)

GAS_CONSTANT = 8.3144621
ATMOS = 101.325
T_DEFAULT = 298.15
CONC_DEFAULT = ATMOS / (GAS_CONSTANT * T_DEFAULT)

XFAIL_REASON = (
    "calc_bbe relies on cclib (getoutData) which cannot parse ORCA 6 output; "
    "thermo.py lacks ORCA frequency/rotational constant parsing"
)


def _calc(filename, QS='grimme', QH=False, s_freq_cutoff=100.0,
          temp=T_DEFAULT, conc=None, scale=1.0):
    """Helper to run calc_bbe with common defaults."""
    if conc is None:
        conc = ATMOS / (GAS_CONSTANT * temp)
    return calc_bbe(orca_path(filename), QS, QH, s_freq_cutoff, 100.0,
                    temp, conc, scale, 'none', False, False, 0)


# ===========================================================================
# calc_bbe: ZPE validation against ORCA ground truth
# ===========================================================================
# Expected ZPE values are from the "Zero point energy" line in each ORCA
# output file.  File 01b uses SCALFREQ 1.035 in ORCA input; calc_bbe needs
# scale=1.035 to match.  All other files use scale=1.0.

@pytest.mark.xfail(reason=XFAIL_REASON)
@pytest.mark.parametrize("filename, expected_zpe, scale", [
    ('01a_water_hf_freq.out', 0.02234428, 1.0),
    ('01c_water_hf_freq_harmonic.out', 0.02234428, 1.0),
    ('03_acetone_linked_opt_freq.out', 0.07868568, 1.0),
    ('04_benzene_radical_cation.out', 0.09505855, 1.0),
    ('05_methylene_triplet_carbene.out', 0.01776186, 1.0),
    ('07_neon_atom_with_freq.out', 0.00000000, 1.0),
    ('08_alanine_C1_pcm_water.out', 0.10859018, 1.0),
    ('10_formaldehyde_verbose_pop.out', 0.02706211, 1.0),
    ('15_methanol_qmqm2_xtb.out', 0.04985553, 1.0),
    ('16_o2_superoxide_anion.out', 0.00264481, 1.0),
    ('18_propane_linked_composite_dh.out', 0.10351579, 1.0),
    ('19_acetic_acid_smd_dmso.out', 0.06091323, 1.0),
    ('21_naphthalene_xtb2_semiempirical.out', 0.10047860, 1.0),
    ('22_hcn_linear_freq_noraman.out', 0.01662938, 1.0),
    ('23_cs2_linear_anharmonic_noraman.out', 0.00693025, 1.0),
    ('24_iodobenzene_genecp_sdd.out', 0.08969456, 1.0),
    ('26_pt_complex_genecp_3zone.out', 0.05876966, 1.0),
    ('28_pyridine_smd_acetonitrile_wb97xd3.out', 0.08944227, 1.0),
    ('29_aniline_cpcm_chloroform.out', 0.11130469, 1.0),
    ('30_phenol_smd_thf_pbe0_d3bj.out', 0.10398287, 1.0),
    ('31_methylammonium_cpcm_water.out', 0.07843093, 1.0),
    ('32_cyclohexane_tpss_meta_gga.out', 0.16749826, 1.0),
    ('33_methanol_pbe_gga.out', 0.04910236, 1.0),
    ('34_butadiene_camb3lyp_rsh.out', 0.08558470, 1.0),
    ('35_furan_wb97xv_functional.out', 0.07067844, 1.0),
    ('36_imidazole_pbe0d3bj_noraman.out', 0.07170801, 1.0),
    ('38_naphthalene_scsmp2.out', 0.19058317, 1.0),
    ('39_oxazole_tpssh_cpcm_dcm.out', 0.05812599, 1.0),
    ('42_dmso_linked_cpcm_gasfreq.out', 0.07877533, 1.0),
    ('43_dmabn_bhandhlyp_chargetransfer.out', 0.15116358, 1.0),
])
def test_calc_bbe_orca_zpe_vs_orca(filename, expected_zpe, scale):
    """Validate calc_bbe ZPE against ORCA's 'Zero point energy' value."""
    bbe = _calc(filename, scale=scale)
    assert abs(bbe.zpe - expected_zpe) < 1e-5


# ===========================================================================
# calc_bbe: enthalpy validation against ORCA ground truth
# ===========================================================================
# Expected values are from the "Total Enthalpy" line in each ORCA output.
# For files with multiple thermo sections, the last section is used.

@pytest.mark.xfail(reason=XFAIL_REASON)
@pytest.mark.parametrize("filename, expected_enthalpy", [
    ('01a_water_hf_freq.out', -75.98299220),
    ('01c_water_hf_freq_harmonic.out', -75.98299220),
    ('03_acetone_linked_opt_freq.out', -192.94041854),
    ('04_benzene_radical_cation.out', -231.91741024),
    ('05_methylene_triplet_carbene.out', -38.99822252),
    ('07_neon_atom_with_freq.out', -128.53091234),
    ('08_alanine_C1_pcm_water.out', -323.26179581),
    ('10_formaldehyde_verbose_pop.out', -114.43460122),
    ('15_methanol_qmqm2_xtb.out', -44.51671796),
    ('16_o2_superoxide_anion.out', -150.33516104),
    ('18_propane_linked_composite_dh.out', -118.92947046),
    ('19_acetic_acid_smd_dmso.out', -228.99355445),
    ('21_naphthalene_xtb2_semiempirical.out', -23.95298356),
    ('22_hcn_linear_freq_noraman.out', -93.39441825),
    ('23_cs2_linear_anharmonic_noraman.out', -834.41011160),
    ('24_iodobenzene_genecp_sdd.out', -529.19051650),
    ('26_pt_complex_genecp_3zone.out', -1563.07795808),
    ('28_pyridine_smd_acetonitrile_wb97xd3.out', -248.18685164),
    ('29_aniline_cpcm_chloroform.out', -249.23544956),
    ('30_phenol_smd_thf_pbe0_d3bj.out', -307.09970845),
    ('31_methylammonium_cpcm_water.out', -96.19877931),
    ('32_cyclohexane_tpss_meta_gga.out', -235.84449310),
    ('33_methanol_pbe_gga.out', -115.55630149),
    ('34_butadiene_camb3lyp_rsh.out', -155.84973310),
    ('35_furan_wb97xv_functional.out', -229.96566203),
    ('36_imidazole_pbe0d3bj_noraman.out', -225.94755282),
    ('38_naphthalene_scsmp2.out', -382.56382353),
    ('39_oxazole_tpssh_cpcm_dcm.out', -246.09659411),
    ('42_dmso_linked_cpcm_gasfreq.out', -553.04404630),
    ('43_dmabn_bhandhlyp_chargetransfer.out', -456.82749801),
])
def test_calc_bbe_orca_enthalpy_vs_orca(filename, expected_enthalpy):
    """Validate calc_bbe enthalpy against ORCA's 'Total Enthalpy' value."""
    bbe = _calc(filename)
    assert abs(bbe.enthalpy - expected_enthalpy) < 1e-5


# ===========================================================================
# calc_bbe: Gibbs free energy validation against ORCA ground truth
# ===========================================================================
# Expected values are from the "Final Gibbs free energy" line.  ORCA uses
# quasi-RRHO with reference frequency 100 cm⁻¹ by default, matching
# GoodVibes' default Grimme quasi-harmonic treatment.

@pytest.mark.xfail(reason=XFAIL_REASON)
@pytest.mark.parametrize("filename, expected_gibbs", [
    ('01a_water_hf_freq.out', -76.00440191),
    ('01c_water_hf_freq_harmonic.out', -76.00440191),
    ('03_acetone_linked_opt_freq.out', -192.98045622),
    ('04_benzene_radical_cation.out', -231.95046217),
    ('05_methylene_triplet_carbene.out', -39.02045013),
    ('07_neon_atom_with_freq.out', -128.54751680),
    ('08_alanine_C1_pcm_water.out', -323.29970533),
    ('10_formaldehyde_verbose_pop.out', -114.45941587),
    ('15_methanol_qmqm2_xtb.out', -44.54363334),
    ('16_o2_superoxide_anion.out', -150.35827549),
    ('18_propane_linked_composite_dh.out', -118.95880295),
    ('19_acetic_acid_smd_dmso.out', -229.02434361),
    ('21_naphthalene_xtb2_semiempirical.out', -23.99995730),
    ('22_hcn_linear_freq_noraman.out', -93.41723879),
    ('23_cs2_linear_anharmonic_noraman.out', -834.43707404),
    ('24_iodobenzene_genecp_sdd.out', -529.22842010),
    ('26_pt_complex_genecp_3zone.out', -1563.12668319),
    ('28_pyridine_smd_acetonitrile_wb97xd3.out', -248.21877747),
    ('29_aniline_cpcm_chloroform.out', -249.26811728),
    ('30_phenol_smd_thf_pbe0_d3bj.out', -307.13460541),
    ('31_methylammonium_cpcm_water.out', -96.22650621),
    ('32_cyclohexane_tpss_meta_gga.out', -235.88024096),
    ('33_methanol_pbe_gga.out', -115.58273881),
    ('34_butadiene_camb3lyp_rsh.out', -155.88113341),
    ('35_furan_wb97xv_functional.out', -229.99586819),
    ('36_imidazole_pbe0d3bj_noraman.out', -225.97847566),
    ('38_naphthalene_scsmp2.out', -382.59849449),
    ('39_oxazole_tpssh_cpcm_dcm.out', -246.12728485),
    ('42_dmso_linked_cpcm_gasfreq.out', -553.07898285),
    ('43_dmabn_bhandhlyp_chargetransfer.out', -456.87092406),
])
def test_calc_bbe_orca_gibbs_vs_orca(filename, expected_gibbs):
    """Validate calc_bbe Gibbs free energy against ORCA's
    'Final Gibbs free energy' value."""
    bbe = _calc(filename)
    assert abs(bbe.gibbs_free_energy - expected_gibbs) < 1e-5


# ===========================================================================
# calc_bbe: entropy validation against ORCA ground truth
# ===========================================================================
# Expected values are from the "Final entropy term" line in each ORCA output.
# This is T*S in Hartree at 298.15 K.

@pytest.mark.xfail(reason=XFAIL_REASON)
@pytest.mark.parametrize("filename, expected_TS", [
    ('01a_water_hf_freq.out', 0.02140971),
    ('04_benzene_radical_cation.out', 0.03305194),
    ('05_methylene_triplet_carbene.out', 0.02222761),
    ('07_neon_atom_with_freq.out', 0.01660446),
    ('10_formaldehyde_verbose_pop.out', 0.02481465),
    ('16_o2_superoxide_anion.out', 0.02311445),
    ('22_hcn_linear_freq_noraman.out', 0.02282053),
    ('32_cyclohexane_tpss_meta_gga.out', 0.03574786),
    ('33_methanol_pbe_gga.out', 0.02643733),
    ('34_butadiene_camb3lyp_rsh.out', 0.03140031),
    ('35_furan_wb97xv_functional.out', 0.03020616),
    ('36_imidazole_pbe0d3bj_noraman.out', 0.03092284),
    ('43_dmabn_bhandhlyp_chargetransfer.out', 0.04342605),
])
def test_calc_bbe_orca_entropy_vs_orca(filename, expected_TS):
    """Validate calc_bbe entropy against ORCA's 'Final entropy term' (T*S)."""
    bbe = _calc(filename)
    assert abs(T_DEFAULT * bbe.qh_entropy - expected_TS) < 1e-5


# ===========================================================================
# calc_bbe: transition states with ground-truth values
# ===========================================================================
# For files with multiple thermo sections (opt+freq), the LAST section is
# used.  Each TS should have exactly 1 imaginary frequency.

@pytest.mark.xfail(reason=XFAIL_REASON)
@pytest.mark.parametrize("filename, expected_zpe, expected_H, expected_G", [
    ('44_ts_sn2_identity_chloride.out', 0.03682585, -960.24874107, -960.28182960),
    ('45_ts_diels_alder_butadiene_ethylene.out', 0.14245543, -234.26023290, -234.29687965),
    ('46_ts_neb_cope_rearrangement.out', 0.14427177, -233.85997087, -233.89652812),
    ('47_ts_e2_elimination_ethylchloride.out', 0.07450085, -615.03764523, -615.07236231),
    ('48_ts_nh3_umbrella_inversion.out', 0.03291185, -56.50170779, -56.52450163),
    ('49_ts_oh_abstraction_methane.out', 0.05332669, -116.21228132, -116.24675878),
])
def test_calc_bbe_orca_transition_states(filename, expected_zpe, expected_H,
                                         expected_G):
    """Validate ORCA TS thermochemistry and imaginary frequency detection."""
    bbe = _calc(filename)
    assert abs(bbe.zpe - expected_zpe) < 1e-5
    assert abs(bbe.enthalpy - expected_H) < 1e-5
    assert abs(bbe.gibbs_free_energy - expected_G) < 1e-5
    assert len(bbe.im_frequency_wn) == 1
    assert all(f < 0 for f in bbe.im_frequency_wn)


# ===========================================================================
# calc_bbe: frequency scaling factor (ORCA SCALFREQ 1.035)
# ===========================================================================
# File 01b uses SCALFREQ 1.035 in the ORCA input.  calc_bbe should receive
# unscaled frequencies from the parser and apply scale=1.035 to match.

@pytest.mark.xfail(reason=XFAIL_REASON)
def test_calc_bbe_orca_scaling():
    """Validate calc_bbe with scale=1.035 against ORCA SCALFREQ 1.035."""
    bbe = _calc('01b_water_hf_freq_scaled.out', scale=1.035)
    assert abs(bbe.zpe - 0.02312633) < 1e-5
    assert abs(bbe.enthalpy - (-75.98221042)) < 1e-5
    assert abs(bbe.gibbs_free_energy - (-76.00361983)) < 1e-5


# ===========================================================================
# calc_bbe: non-standard quasi-RRHO cutoff (200 cm⁻¹)
# ===========================================================================
# File 01d uses QRRHORefFreq 200 in the ORCA input (vs default 100 cm⁻¹).
# For water (all frequencies >> 200 cm⁻¹), the effect is negligible.

@pytest.mark.xfail(reason=XFAIL_REASON)
def test_calc_bbe_orca_nonstandard_cutoff():
    """Validate calc_bbe with s_freq_cutoff=200 against ORCA QRRHORefFreq 200."""
    bbe = _calc('01d_water_hf_freq_qhcutoff.out', s_freq_cutoff=200.0)
    assert abs(bbe.zpe - 0.02234428) < 1e-5
    assert abs(bbe.enthalpy - (-75.98299220)) < 1e-5
    assert abs(bbe.gibbs_free_energy - (-76.00440190)) < 1e-5


# ===========================================================================
# calc_bbe: non-standard temperature and pressure
# ===========================================================================
# File 02 has 9 thermochemistry sections (T=290/295/300 K × P=1/2/3 atm).
# IMPORTANT: This file uses PrintThermoChem, so ORCA's "Total Enthalpy" and
# "Final Gibbs free energy" are thermal corrections only (no electronic
# energy).  When calc_bbe works for ORCA, comparison should use
# (bbe.enthalpy - bbe.scf_energy) for enthalpy and
# (bbe.gibbs_free_energy - bbe.scf_energy) for Gibbs.

@pytest.mark.xfail(reason=XFAIL_REASON)
@pytest.mark.parametrize("temp, pressure, expected_zpe, expected_H_thermal, "
                         "expected_TS, expected_G_thermal", [
    (290.00, 1.0, 0.07423948, 0.07851915, 0.02500860, 0.05351056),
    (290.00, 2.0, 0.07423948, 0.07851915, 0.02437203, 0.05414712),
    (290.00, 3.0, 0.07423948, 0.07851915, 0.02399966, 0.05451949),
    (295.00, 1.0, 0.07423948, 0.07861430, 0.02553568, 0.05307862),
    (295.00, 2.0, 0.07423948, 0.07861430, 0.02488814, 0.05372616),
    (295.00, 3.0, 0.07423948, 0.07861430, 0.02450935, 0.05410495),
    (300.00, 1.0, 0.07423948, 0.07871054, 0.02606549, 0.05264506),
    (300.00, 2.0, 0.07423948, 0.07871054, 0.02540697, 0.05330358),
    (300.00, 3.0, 0.07423948, 0.07871054, 0.02502176, 0.05368878),
])
def test_calc_bbe_orca_nonstandard_temp_pressure(temp, pressure, expected_zpe,
                                                  expected_H_thermal,
                                                  expected_TS,
                                                  expected_G_thermal):
    """Validate calc_bbe thermal corrections at non-standard T/P against ORCA.

    Because PrintThermoChem omits electronic energy, we compare thermal
    corrections: (bbe.enthalpy - bbe.scf_energy) and
    (bbe.gibbs_free_energy - bbe.scf_energy).
    """
    conc = (pressure * ATMOS) / (GAS_CONSTANT * temp)
    bbe = calc_bbe(orca_path('02_ethane_opt_freq_thermo.out'), 'grimme', False,
                   100.0, 100.0, temp, conc, 1.0, 'none', False, False, 0)
    assert abs(bbe.zpe - expected_zpe) < 1e-5
    H_thermal = bbe.enthalpy - bbe.scf_energy
    assert abs(H_thermal - expected_H_thermal) < 1e-5
    G_thermal = bbe.gibbs_free_energy - bbe.scf_energy
    assert abs(G_thermal - expected_G_thermal) < 1e-5


# ===========================================================================
# calc_bbe: solvation model files
# ===========================================================================

@pytest.mark.xfail(reason=XFAIL_REASON)
@pytest.mark.parametrize("filename", ORCA_SOLVATION_FILES)
def test_calc_bbe_orca_solvation(filename):
    """Solvated ORCA files should produce Gibbs free energy."""
    bbe = _calc(filename)
    assert hasattr(bbe, 'gibbs_free_energy')


# ===========================================================================
# calc_bbe: single-point only (no thermochemistry)
# ===========================================================================

@pytest.mark.xfail(reason=XFAIL_REASON)
@pytest.mark.parametrize("filename, expected_energy", [
    ('06_carbon_atom_single_point.out', -37.659293),
    ('09_caffeine_nmr_giao.out', -680.388158),
    ('11_hf_molecule_dlpno_ccsdt_gold_standard.out', -100.338297),
    ('13_formaldehyde_tddft_s1.out', -114.270287),
    ('17_iron_complex_quintet.out', -1828.860023),
    ('20_benzene_singlepoint.out', -232.176166),
    ('40_n2o_linear_highT.out', -183.749274),
])
def test_calc_bbe_orca_sp_only(filename, expected_energy):
    """SP-only files should have scf_energy but no gibbs_free_energy."""
    bbe = _calc(filename)
    assert expected_energy == round(bbe.scf_energy, 6)
    assert not hasattr(bbe, 'gibbs_free_energy')


# ===========================================================================
# calc_bbe: error files handled gracefully
# ===========================================================================

@pytest.mark.parametrize("filename", ORCA_ERROR_FILES)
def test_calc_bbe_orca_error_files(filename):
    """Error files should not cause unhandled crashes."""
    try:
        bbe = _calc(filename)
    except (AttributeError, ValueError, IndexError):
        pass  # Expected for severely malformed files
    except SystemExit:
        pass  # Some files trigger sys.exit


# ===========================================================================
# calc_bbe: 2nd-order saddle point (multiple thermo sections)
# ===========================================================================
# File 37 has two thermo sections from opt+freq.  The LAST section (final
# geometry) should be used.

@pytest.mark.xfail(reason=XFAIL_REASON)
def test_calc_bbe_orca_second_order_saddle():
    """File 37 (planar cyclohexane) is a 2nd-order saddle with 2 imaginary
    frequencies.  Validate last thermo section values."""
    bbe = _calc('37_planar_cyclohexane_2nd_order_saddle.out')
    assert abs(bbe.zpe - 0.17135959) < 1e-5
    assert abs(bbe.enthalpy - (-235.48389423)) < 1e-5
    assert abs(bbe.gibbs_free_energy - (-235.51694625)) < 1e-5
    assert len(bbe.im_frequency_wn) == 2
