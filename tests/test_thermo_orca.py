#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for thermochemistry calculations on ORCA 6 output files.

Ground-truth values are extracted from the ORCA output files' thermochemistry
sections ("Zero point energy", "Total Enthalpy", "Final entropy term",
"Final Gibbs free energy").  ORCA's "Final Gibbs free energy" uses quasi-RRHO
with a default reference frequency of 100 cm-1, matching GoodVibes' default.
For files with multiple thermochemistry sections (opt+freq jobs), the LAST
section's values are used (corresponding to the final optimized geometry).
"""

import pytest
from goodvibes.thermo import calc_bbe
from conftest import orca_path, ORCA_SOLVATION_FILES, ORCA_ERROR_FILES

GAS_CONSTANT = 8.3144621
ATMOS = 101.325
T_DEFAULT = 298.15
CONC_DEFAULT = ATMOS / (GAS_CONSTANT * T_DEFAULT)

def _calc(filename, QS='grimme', QH=False, s_freq_cutoff=100.0,
          temp=T_DEFAULT, conc=None, scale=1.0):
    """Helper to run calc_bbe with common defaults.

    Uses inertia='conf' (Bav from rotational constants) to match ORCA's
    default quasi-RRHO treatment, which computes Bav per conformer rather
    than using Grimme's global value of 1e-44 kg m^2.
    """
    if conc is None:
        conc = ATMOS / (GAS_CONSTANT * temp)
    return calc_bbe(orca_path(filename), QS, QH, s_freq_cutoff, 100.0,
                    temp, conc, scale, None, None, None, 0,
                    inertia='conf')


# ===========================================================================
# calc_bbe: ZPE validation against ORCA ground truth
# ===========================================================================

@pytest.mark.parametrize("filename, expected_zpe, scale", [
    ('01a_water_hf_freq.out', 0.02234428, 1.0),
    ('01b_water_hf_freq_scaled.out', 0.02312633, 1.035),
    ('01c_water_hf_freq_harmonic.out', 0.02234428, 1.0),
    ('01d_water_hf_freq_qhcutoff.out', 0.02234428, 1.0),
    ('02_ethane_opt_freq_thermo.out', 0.07424074, 1.0),
    ('03_acetone_linked_opt_freq.out', 0.08308455, 1.0),
    ('04_benzene_radical_cation.out', 0.09505855, 1.0),
    ('05_methylene_triplet_carbene.out', 0.01776186, 1.0),
    ('07_neon_atom_with_freq.out', 0.00000000, 1.0),
    ('08_alanine_C1_pcm_water.out', 0.10873070, 1.0),
    ('10_formaldehyde_verbose_pop.out', 0.02706211, 1.0),
    ('15_methanol_qmqm2_xtb.out', 0.04985553, 1.0),
    ('16_o2_superoxide_anion.out', 0.00264481, 1.0),
    ('18_propane_linked_composite_dh.out', 0.10351579, 1.0),
    ('19_acetic_acid_smd_dmso.out', 0.06110813, 1.0),
    ('21_naphthalene_xtb2_semiempirical.out', 0.14284037, 1.0),
    ('22_hcn_linear_freq_noraman.out', 0.01662938, 1.0),
    ('23_cs2_linear_anharmonic_noraman.out', 0.00693025, 1.0),
    ('24_iodobenzene_genecp_sdd.out', 0.08969456, 1.0),
    ('26_pt_complex_genecp_3zone.out', 0.45830142, 1.0),
    ('28_pyridine_smd_acetonitrile_wb97xd3.out', 0.08944227, 1.0),
    ('29_aniline_cpcm_chloroform.out', 0.11650285, 1.0),
    ('30_phenol_smd_thf_pbe0_d3bj.out', 0.10501223, 1.0),
    ('31_methylammonium_cpcm_water.out', 0.07843093, 1.0),
    ('32_cyclohexane_tpss_meta_gga.out', 0.16749826, 1.0),
    ('33_methanol_pbe_gga.out', 0.04910236, 1.0),
    ('34_butadiene_camb3lyp_rsh.out', 0.08558470, 1.0),
    ('35_furan_wb97xv_functional.out', 0.07067844, 1.0),
    ('36_imidazole_pbe0d3bj_noraman.out', 0.07170801, 1.0),
    ('38_naphthalene_scsmp2.out', 0.14727608, 1.0),
    ('39_oxazole_tpssh_cpcm_dcm.out', 0.05812599, 1.0),
    ('42_dmso_linked_cpcm_gasfreq.out', 0.07877533, 1.0),
    ('43_dmabn_bhandhlyp_chargetransfer.out', 0.17747277, 1.0),
])
def test_calc_bbe_orca_zpe_vs_orca(filename, expected_zpe, scale):
    """Validate calc_bbe ZPE against ORCA's 'Zero point energy' value."""
    bbe = _calc(filename, scale=scale)
    assert abs(bbe.zpe - expected_zpe) < 5e-6


# ===========================================================================
# calc_bbe: enthalpy validation against ORCA ground truth
# ===========================================================================

@pytest.mark.parametrize("filename, expected_enthalpy, scale", [
    ('01a_water_hf_freq.out', -75.98299220, 1.0),
    ('01b_water_hf_freq_scaled.out', -75.98221042, 1.035),
    ('01c_water_hf_freq_harmonic.out', -75.98299220, 1.0),
    ('01d_water_hf_freq_qhcutoff.out', -75.98299220, 1.0),
    ('03_acetone_linked_opt_freq.out', -193.00605124, 1.0),
    ('04_benzene_radical_cation.out', -231.91741024, 1.0),
    ('05_methylene_triplet_carbene.out', -38.99822252, 1.0),
    ('07_neon_atom_with_freq.out', -128.53091234, 1.0),
    ('08_alanine_C1_pcm_water.out', -323.26179048, 1.0),
    ('10_formaldehyde_verbose_pop.out', -114.43460122, 1.0),
    ('15_methanol_qmqm2_xtb.out', -44.51671796, 1.0),
    ('16_o2_superoxide_anion.out', -150.33516104, 1.0),
    # Composite job (issue #101): GoodVibes now reports the B3LYP opt/freq
    # electronic energy that ORCA's thermochemistry is built on, not the
    # trailing RI-B2PLYP single point. This golden is ORCA's own printed
    # "Total Enthalpy"; the linked SP is applied only with --spc link.
    ('18_propane_linked_composite_dh.out', -118.92947046, 1.0),
    ('19_acetic_acid_smd_dmso.out', -228.99665302, 1.0),
    ('21_naphthalene_xtb2_semiempirical.out', -25.32342490, 1.0),
    ('22_hcn_linear_freq_noraman.out', -93.39441825, 1.0),
    ('23_cs2_linear_anharmonic_noraman.out', -834.41011160, 1.0),
    ('24_iodobenzene_genecp_sdd.out', -529.19051650, 1.0),
    ('26_pt_complex_genecp_3zone.out', -1962.77099230, 1.0),
    ('28_pyridine_smd_acetonitrile_wb97xd3.out', -248.18685164, 1.0),
    ('29_aniline_cpcm_chloroform.out', -287.43857423, 1.0),
    ('30_phenol_smd_thf_pbe0_d3bj.out', -307.10510604, 1.0),
    ('31_methylammonium_cpcm_water.out', -96.19877931, 1.0),
    ('32_cyclohexane_tpss_meta_gga.out', -235.84449310, 1.0),
    ('33_methanol_pbe_gga.out', -115.55630149, 1.0),
    ('34_butadiene_camb3lyp_rsh.out', -155.84973310, 1.0),
    ('35_furan_wb97xv_functional.out', -229.96566203, 1.0),
    ('36_imidazole_pbe0d3bj_noraman.out', -225.94755282, 1.0),
    ('38_naphthalene_scsmp2.out', -384.48026364, 1.0),
    ('39_oxazole_tpssh_cpcm_dcm.out', -246.09659411, 1.0),
    ('42_dmso_linked_cpcm_gasfreq.out', -553.04404630, 1.0),
    ('43_dmabn_bhandhlyp_chargetransfer.out', -458.10505407, 1.0),
])
def test_calc_bbe_orca_enthalpy_vs_orca(filename, expected_enthalpy, scale):
    """Validate calc_bbe enthalpy against ORCA's 'Total Enthalpy' value."""
    bbe = _calc(filename, scale=scale)
    assert abs(bbe.enthalpy - expected_enthalpy) < 5e-6


# ===========================================================================
# calc_bbe: Gibbs free energy validation against ORCA ground truth
# ===========================================================================

@pytest.mark.parametrize("filename, expected_gibbs, scale", [
    ('01a_water_hf_freq.out', -76.00440191, 1.0),
    ('01b_water_hf_freq_scaled.out', -76.00361983, 1.035),
    ('01c_water_hf_freq_harmonic.out', -76.00440191, 1.0),
    ('01d_water_hf_freq_qhcutoff.out', -76.00440190, 1.0),
    ('03_acetone_linked_opt_freq.out', -193.04036425, 1.0),
    ('04_benzene_radical_cation.out', -231.95046217, 1.0),
    ('05_methylene_triplet_carbene.out', -39.02045013, 1.0),
    ('07_neon_atom_with_freq.out', -128.54751680, 1.0),
    ('08_alanine_C1_pcm_water.out', -323.29956237, 1.0),
    ('10_formaldehyde_verbose_pop.out', -114.45941587, 1.0),
    ('15_methanol_qmqm2_xtb.out', -44.54363334, 1.0),
    ('16_o2_superoxide_anion.out', -150.35827549, 1.0),
    ('18_propane_linked_composite_dh.out', -118.95880295, 1.0),  # issue #101: ORCA Final Gibbs (B3LYP level)
    ('19_acetic_acid_smd_dmso.out', -229.02899765, 1.0),
    ('21_naphthalene_xtb2_semiempirical.out', -25.36172065, 1.0),
    ('22_hcn_linear_freq_noraman.out', -93.41723879, 1.0),
    ('23_cs2_linear_anharmonic_noraman.out', -834.43707404, 1.0),
    ('24_iodobenzene_genecp_sdd.out', -529.22842010, 1.0),
    ('26_pt_complex_genecp_3zone.out', -1962.86411156, 1.0),
    ('28_pyridine_smd_acetonitrile_wb97xd3.out', -248.21877747, 1.0),
    ('29_aniline_cpcm_chloroform.out', -287.47404545, 1.0),
    ('30_phenol_smd_thf_pbe0_d3bj.out', -307.14051748, 1.0),
    ('31_methylammonium_cpcm_water.out', -96.22650621, 1.0),
    ('32_cyclohexane_tpss_meta_gga.out', -235.88024096, 1.0),
    ('33_methanol_pbe_gga.out', -115.58273881, 1.0),
    ('34_butadiene_camb3lyp_rsh.out', -155.88113341, 1.0),
    ('35_furan_wb97xv_functional.out', -229.99586819, 1.0),
    ('36_imidazole_pbe0d3bj_noraman.out', -225.97847566, 1.0),
    ('38_naphthalene_scsmp2.out', -384.52000251, 1.0),
    ('39_oxazole_tpssh_cpcm_dcm.out', -246.12728485, 1.0),
    ('42_dmso_linked_cpcm_gasfreq.out', -553.07898285, 1.0),
    ('43_dmabn_bhandhlyp_chargetransfer.out', -458.14938726, 1.0),
])
def test_calc_bbe_orca_gibbs_vs_orca(filename, expected_gibbs, scale):
    """Validate calc_bbe quasi-harmonic Gibbs free energy against ORCA's
    'Final Gibbs free energy' value (ORCA uses quasi-RRHO by default)."""
    bbe = _calc(filename, scale=scale)
    assert abs(bbe.qh_gibbs_free_energy - expected_gibbs) < 5e-6


# ===========================================================================
# calc_bbe: entropy validation against ORCA ground truth
# ===========================================================================

@pytest.mark.parametrize("filename, expected_TS, scale", [
    ('01a_water_hf_freq.out', 0.02140971, 1.0),
    ('01b_water_hf_freq_scaled.out', 0.02140941, 1.035),
    ('01c_water_hf_freq_harmonic.out', 0.02140971, 1.0),
    ('01d_water_hf_freq_qhcutoff.out', 0.02140970, 1.0),
    ('04_benzene_radical_cation.out', 0.03305194, 1.0),
    ('05_methylene_triplet_carbene.out', 0.02222761, 1.0),
    ('07_neon_atom_with_freq.out', 0.01660446, 1.0),
    ('10_formaldehyde_verbose_pop.out', 0.02481465, 1.0),
    ('16_o2_superoxide_anion.out', 0.02311445, 1.0),
    ('22_hcn_linear_freq_noraman.out', 0.02282053, 1.0),
    ('32_cyclohexane_tpss_meta_gga.out', 0.03574786, 1.0),
    ('33_methanol_pbe_gga.out', 0.02643733, 1.0),
    ('34_butadiene_camb3lyp_rsh.out', 0.03140031, 1.0),
    ('35_furan_wb97xv_functional.out', 0.03020616, 1.0),
    ('36_imidazole_pbe0d3bj_noraman.out', 0.03092284, 1.0),
    ('43_dmabn_bhandhlyp_chargetransfer.out', 0.04433318, 1.0),
])
def test_calc_bbe_orca_entropy_vs_orca(filename, expected_TS, scale):
    """Validate calc_bbe entropy against ORCA's 'Final entropy term' (T*S)."""
    bbe = _calc(filename, scale=scale)
    assert abs(T_DEFAULT * bbe.qh_entropy - expected_TS) < 5e-6


# ===========================================================================
# calc_bbe: transition states with ground-truth values
# ===========================================================================

@pytest.mark.parametrize("filename, expected_zpe, expected_H, expected_G", [
    ('44_ts_sn2_identity_chloride.out', 0.03682585, -960.24874107, -960.28182960),
    ('45_ts_diels_alder_butadiene_ethylene.out', 0.14245543,
                 -234.26023290, -234.29687965),
    ('46_ts_neb_claisen_rearrangement.out', 0.12054950,
                 -269.72988246, -269.76537289),
    ('47_ts_e2_elimination_ethylchloride.out', 0.07229701,
                 -615.09225957, -615.13028218),
    ('48_ts_nh3_umbrella_inversion.out', 0.03291185, -56.50170779, -56.52450163),
    ('49_ts_oh_abstraction_methane.out', 0.05332669,
                 -116.21228132, -116.24675878),
])
def test_calc_bbe_orca_transition_states(filename, expected_zpe, expected_H,
                                         expected_G):
    """Validate ORCA TS thermochemistry and imaginary frequency detection."""
    bbe = _calc(filename)
    assert abs(bbe.zpe - expected_zpe) < 5e-6
    assert abs(bbe.enthalpy - expected_H) < 5e-6
    assert abs(bbe.qh_gibbs_free_energy - expected_G) < 5e-6
    assert len(bbe.im_frequency_wn) == 1
    assert all(f < 0 for f in bbe.im_frequency_wn)


# ===========================================================================
# calc_bbe: non-standard quasi-RRHO cutoff (200 cm-1)
# ===========================================================================

def test_calc_bbe_orca_nonstandard_cutoff():
    """Validate calc_bbe with s_freq_cutoff=200 against ORCA QRRHORefFreq 200."""
    bbe = _calc('01d_water_hf_freq_qhcutoff.out', s_freq_cutoff=200.0)
    assert abs(bbe.zpe - 0.02234428) < 5e-6
    assert abs(bbe.enthalpy - (-75.98299220)) < 5e-6
    assert abs(bbe.qh_gibbs_free_energy - (-76.00440190)) < 5e-6


# ===========================================================================
# calc_bbe: non-standard temperature and pressure
# ===========================================================================

@pytest.mark.parametrize(
    "temp, expected_enthalpy, expected_gibbs",
    [
        (77.0,  -79.71630302, -79.72142169),
        (298.0, -79.71285067, -79.73870486),
        (330.0, -79.71221055, -79.74151351),
        (450.0, -79.70936467, -79.75260969),
    ],
)
def test_calc_bbe_orca_nonstandard_temp_pressure(
    temp, expected_enthalpy, expected_gibbs,
):
    """Validate calc_bbe at non-standard temperatures against ORCA ground truth.

    Ground-truth values from 02_ethane_opt_freq_thermo.out which reports
    thermochemistry at 77, 298, 330, and 450 K (all at 1 atm).
    """
    conc = ATMOS / (GAS_CONSTANT * temp)
    bbe = calc_bbe(orca_path('02_ethane_opt_freq_thermo.out'), 'grimme', False,
                   100.0, 100.0, temp, conc, 1.0, None, None, None, 0,
                   inertia='conf')
    assert abs(bbe.zpe - 0.07424074) < 5e-6
    assert abs(bbe.enthalpy - expected_enthalpy) < 5e-6
    assert abs(bbe.qh_gibbs_free_energy - expected_gibbs) < 5e-6


# ===========================================================================
# calc_bbe: solvation model files
# ===========================================================================

@pytest.mark.parametrize("filename", ORCA_SOLVATION_FILES)
def test_calc_bbe_orca_solvation(filename):
    """Solvated ORCA files should produce Gibbs free energy."""
    bbe = _calc(filename)
    assert hasattr(bbe, 'gibbs_free_energy')


# ===========================================================================
# calc_bbe: single-point only (no thermochemistry)
# ===========================================================================

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
        _calc(filename)
    except (AttributeError, ValueError, IndexError):
        pass  # Expected for severely malformed files
    except SystemExit:
        pass  # Some files trigger sys.exit


# ===========================================================================
# calc_bbe: 3rd-order saddle point (multiple thermo sections)
# ===========================================================================

def test_calc_bbe_orca_third_order_saddle():
    """File 37 (planar cyclohexane) is a 3rd-order saddle with 3 imaginary
    frequencies.  Validate last thermo section values."""
    bbe = _calc('37_planar_cyclohexane_3rd_order_saddle.out')
    assert abs(bbe.zpe - 0.17135959) < 5e-6
    assert abs(bbe.enthalpy - (-235.48389423)) < 5e-6
    assert abs(bbe.qh_gibbs_free_energy - (-235.51694625)) < 5e-6
    assert len(bbe.im_frequency_wn) == 3


