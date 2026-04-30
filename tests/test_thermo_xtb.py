#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end thermochemistry tests on xtb output files.

Two complementary checks:

* **E, ZPE, H** are compared directly against xtb's printed values
  (``| TOTAL ENERGY``, ``| TOTAL ENTHALPY``, ``:: zero point energy``).
  These quantities are reproduced to within 1–5 µEh, matching the
  G16 / ORCA test suites.

* **Quasi-harmonic G** is locked in as a *regression* against the values
  GoodVibes itself currently produces.  It is not compared to xtb's
  ``TOTAL FREE ENERGY`` because:
  (a) For non-singlets, xtb omits the R·ln(mult) electronic-spin entropy
      that GoodVibes includes — discrepancies of ~RT·ln(mult) (660 µEh
      for doublets, 1.5 mEh for quintets) are by-design.
  (b) Even for singlets, a ~7–9 µEh systematic offset arises from minor
      physical-constant precision differences (kB, h, NA) between xtb's
      thermo and GoodVibes', which exceeds the 5 µEh ORCA tolerance.

GoodVibes is run with xtb's defaults (50 cm⁻¹ rotor cutoff for qRRHO,
1 atm reference, ``inertia='conf'`` so Bav is computed per conformer).
"""

import re
import pytest
from goodvibes.thermo import calc_bbe
from conftest import xtb_path

GAS_CONSTANT = 8.3144621
ATMOS = 101.325
T_DEFAULT = 298.15
CONC_DEFAULT = ATMOS / (GAS_CONSTANT * T_DEFAULT)

XTB_S_CUTOFF = 50.0  # xtb's default rotor cutoff


def _calc(filename):
    return calc_bbe(xtb_path(filename), 'grimme', False, XTB_S_CUTOFF, 100.0,
                    T_DEFAULT, CONC_DEFAULT, 1.0, None, None, None, 0,
                    inertia='conf')


def _xtb_truth(filename):
    """Extract (E, H, G, ZPE) from xtb's printed thermo summary."""
    with open(xtb_path(filename), encoding='utf-8', errors='replace') as f:
        data = f.read()
    m_e = re.findall(r'^\s+\| TOTAL ENERGY\s+(-?\d+\.\d+)', data, re.M)
    m_h = re.findall(r'^\s+\| TOTAL ENTHALPY\s+(-?\d+\.\d+)', data, re.M)
    m_g = re.findall(r'^\s+\| TOTAL FREE ENERGY\s+(-?\d+\.\d+)', data, re.M)
    m_z = re.findall(r':: zero point energy\s+(-?\d+\.\d+)', data)
    return float(m_e[-1]), float(m_h[-1]), float(m_g[-1]), float(m_z[-1])


# Singlet OPT+freq files (the water dimer is excluded — its near-zero
# imaginary mode at -10 cm⁻¹ falls inside xtb's -20 cm⁻¹ imaginary cutoff,
# so xtb counts it as a near-zero rotor mode while GoodVibes treats it as
# imaginary; the resulting ZPE/H disagreement is ~1 mEh).
SINGLET_FREQ_FILES = [
    '01_water.out', '02_ethane.out', '03_acetone.out',
    '08_alanine.out', '09_caffeine.out', '10_formaldehyde.out',
    '11_hf_molecule.out', '13_methanol.out', '16_propane.out',
    '19_naphthalene.out', '20_hcn_linear.out', '21_cs2_linear.out',
    '22_iodobenzene.out', '23_pd_complex.out', '24_pt_complex.out',
    '25_ethane.out', '26_pyridine_acetonitrile.out',
    '27_aniline_chloroform.out', '28_phenol_thf.out',
    '30_cyclohexane.out', '31_methanol.out', '32_butadiene.out',
    '33_furan.out', '34_imidazole.out', '36_naphthalene.out',
    '37_oxazole_dcm.out', '38_n2o_linear.out', '39_thiophene.out',
    '40_dmso.out', '41_dmabn.out',
]

NONSINGLET_FREQ_FILES = [
    '04_benzene_radical_cation.out',     # mult = 2
    '05_methylene_triplet_carbene.out',  # mult = 3
    '14_o2_superoxide_anion.out',        # mult = 2
    '15_iron_complex_quintet.out',       # mult = 5
]

ALL_FREQ_FILES = SINGLET_FREQ_FILES + NONSINGLET_FREQ_FILES


# ===========================================================================
# Comparison against xtb's printed values
# ===========================================================================

@pytest.mark.parametrize("filename", ALL_FREQ_FILES)
def test_calc_bbe_xtb_scf_energy(filename):
    """E reproduces xtb's TOTAL ENERGY exactly (1 µEh; we just parse it)."""
    bbe = _calc(filename)
    e_xtb, _, _, _ = _xtb_truth(filename)
    assert abs(bbe.scf_energy - e_xtb) < 1e-6


@pytest.mark.parametrize("filename", ALL_FREQ_FILES)
def test_calc_bbe_xtb_zpe(filename):
    """ZPE reproduces xtb's printed value to 1 µEh."""
    bbe = _calc(filename)
    _, _, _, z_xtb = _xtb_truth(filename)
    assert abs(bbe.zpe - z_xtb) < 1e-6


@pytest.mark.parametrize("filename", ALL_FREQ_FILES)
def test_calc_bbe_xtb_enthalpy(filename):
    """Enthalpy reproduces xtb's TOTAL ENTHALPY to 5 µEh (ORCA-style)."""
    bbe = _calc(filename)
    _, h_xtb, _, _ = _xtb_truth(filename)
    assert abs(bbe.enthalpy - h_xtb) < 5e-6


# ===========================================================================
# Regression test: quasi-harmonic Gibbs free energy
# ---------------------------------------------------------------------------
# Ground-truth values are GoodVibes' own current outputs.  Tolerance 1 µEh
# (G16-strict) ensures any future change that shifts G is caught.
# ===========================================================================

@pytest.mark.parametrize("filename, expected_qhG", [
    ('01_water.out', -5.0680492800),
    ('02_ethane.out', -7.2835546086),
    ('03_acetone.out', -13.4119703635),
    ('08_alanine.out', -20.9693044572),
    ('09_caffeine.out', -42.0121514517),
    ('10_formaldehyde.out', -7.1710812398),
    ('11_hf_molecule.out', -5.2307332842),
    ('13_methanol.out', -8.1991110840),
    ('16_propane.out', -10.4135852023),
    ('19_naphthalene.out', -25.3617289689),
    ('20_hcn_linear.out', -5.5071516783),
    ('21_cs2_linear.out', -8.6272192197),
    ('22_iodobenzene.out', -19.1317298989),
    ('23_pd_complex.out', -24.1311258064),
    ('24_pt_complex.out', -58.3892104017),
    ('25_ethane.out', -7.2835546086),
    ('26_pyridine_acetonitrile.out', -16.0995373564),
    ('27_aniline_chloroform.out', -19.2405261289),
    ('28_phenol_thf.out', -19.8949032067),
    ('30_cyclohexane.out', -18.8453295199),
    ('31_methanol.out', -8.1971311618),
    ('32_butadiene.out', -11.5706910003),
    ('33_furan.out', -14.6023750283),
    ('34_imidazole.out', -14.2547978544),
    ('36_naphthalene.out', -23.9996716522),
    ('37_oxazole_dcm.out', -14.9011446815),
    ('38_n2o_linear.out', -9.8543861016),
    ('39_thiophene.out', -13.7799476000),
    ('40_dmso.out', -14.5725400249),
    ('41_dmabn.out', -30.0406106872),
    ('04_benzene_radical_cation.out', -15.3000984439),
    ('05_methylene_triplet_carbene.out', -2.9398433702),
    ('14_o2_superoxide_anion.out', -8.1845024238),
    ('15_iron_complex_quintet.out', -33.8729856005),
])
def test_calc_bbe_xtb_qh_gibbs_regression(filename, expected_qhG):
    bbe = _calc(filename)
    assert abs(bbe.qh_gibbs_free_energy - expected_qhG) < 1e-6


# ===========================================================================
# --hess-only saddle point: parser fallback for missing mass / rotational data
# ---------------------------------------------------------------------------
# File 35 is xtb --hess (analytical Hessian only).  xtb skips the THERMODYNAMIC
# block in --hess mode, so the printed "molecular mass" and "rotational
# constants" lines are absent — parse_xtb_thermo recomputes them from the
# geometry that was already loaded, otherwise calc_bbe would crash with a
# math domain error in calc_translational_entropy (m=0 → log(0)).
# ===========================================================================

def test_calc_bbe_xtb_hess_only_saddle():
    """3rd-order saddle from xtb --hess — exercises the geometry-derived mass
    and rotational-constant fallback in parse_xtb_thermo."""
    bbe = _calc('35_planar_cyclohexane_3rd_order_saddle.out')
    e_xtb, h_xtb, _, z_xtb = _xtb_truth(
        '35_planar_cyclohexane_3rd_order_saddle.out')
    assert abs(bbe.scf_energy - e_xtb) < 1e-6
    assert abs(bbe.zpe - z_xtb) < 1e-6
    # H requires rotational temperatures; matching xtb's printed value to
    # 5 µEh implies the geometry-derived rotational fallback is correct.
    assert abs(bbe.enthalpy - h_xtb) < 5e-6
    # 3rd-order saddle: 3 imaginary modes
    assert len(bbe.im_frequency_wn) == 3
    assert all(f < 0 for f in bbe.im_frequency_wn)
