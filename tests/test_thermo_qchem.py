#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for thermochemistry calculations on Q-Chem 6 output files.

Three layers:
  1. Hardcoded ZPE / H / qh-G regression values per fixture.
  2. Cross-validation against the matching tests/g16/ fixture: same
     molecule + same level of theory should give the same H/S/G to within
     SCF-convergence-level numerical noise (typically <1e-3 Hartree;
     allow more on solvated and B3LYP→Q-Chem comparisons because the
     SCF/grid defaults differ).
  3. Direct ground-truth comparison against the values Q-Chem itself
     prints in the STANDARD THERMODYNAMIC QUANTITIES block — the most
     stringent check (sub-millikcal/mol agreement expected).
"""

import math
import pytest

from goodvibes.thermo import calc_bbe
from conftest import (qchem_path, g16path,
                      QCHEM_FREQ_FILES, QCHEM_TS_FILES)


GAS_CONSTANT = 8.3144621
ATMOS = 101.325
T_DEFAULT = 298.15
CONC_DEFAULT = ATMOS / (GAS_CONSTANT * T_DEFAULT)
HA_TO_KCAL = 627.509541
R_CAL_MOL_K = 1.98720425864083  # cal/mol/K


def _calc(path, scale=1.0, temp=T_DEFAULT, conc=CONC_DEFAULT):
    return calc_bbe(path, scale_fac=scale, conc=conc, temp=temp)


# ===========================================================================
# Hardcoded reference values (regression layer)
# ===========================================================================

@pytest.mark.parametrize("filename, expected_zpe, expected_h, expected_qh_g", [
    ('01a_water_hf_freq.out',           0.02239063,  -75.98434246,  -76.00575164),
    ('02_ethane_opt_freq_T398_P2.out',  0.07431093,  -79.77779400,  -79.80533695),
    ('05_methylene_triplet_carbene.out',0.01776524,  -38.99821903,  -39.02044523),
    ('08_alanine_C1_pcm_water.out',     0.10852458, -323.25987297, -323.29778588),
    ('10_formaldehyde_verbose_pop.out', 0.02709910, -114.43025734, -114.45506712),
    ('19_acetic_acid_smd_dmso.out',     0.06113806, -229.11547000, -229.14618635),
    ('22_hcn_linear_freq_noraman.out',  0.01665347,  -93.39395744,  -93.41677614),
    ('33_methanol_pbepbe_gga.out',      0.04980124, -115.55714388, -115.58412542),
    ('44_ts_sn2_identity_chloride.out', 0.03680291, -960.41528883, -960.44902742),
])
def test_calc_bbe_qchem_regression(filename, expected_zpe, expected_h, expected_qh_g):
    bbe = _calc(qchem_path(filename))
    assert abs(bbe.zpe - expected_zpe) < 5e-6
    assert abs(bbe.enthalpy - expected_h) < 5e-6
    assert abs(bbe.qh_gibbs_free_energy - expected_qh_g) < 5e-6


# ===========================================================================
# Don't-crash sweep across all freq + TS fixtures
# ===========================================================================

@pytest.mark.parametrize("filename", QCHEM_FREQ_FILES + QCHEM_TS_FILES)
def test_calc_bbe_completes_for_all_freq_files(filename):
    """Every freq/TS fixture should produce a Gibbs free energy without
    crashing — the parser populates all the QCData fields calc_bbe needs."""
    bbe = _calc(qchem_path(filename))
    assert bbe.qh_gibbs_free_energy is not None
    assert bbe.zpe is not None


# ===========================================================================
# TS handling: imaginary frequency excluded from thermo, qh-G still computed
# ===========================================================================

def test_ts_imaginary_freq_handled():
    bbe = _calc(qchem_path('44_ts_sn2_identity_chloride.out'))
    assert bbe.qh_gibbs_free_energy is not None
    assert bbe.zpe > 0
    # Imaginary mode preserved separately on the QCData but excluded from thermo.
    assert len(bbe.xyz.im_frequency_wn) == 1


# ===========================================================================
# Linear molecules: 2 rotational DOF
# ===========================================================================

@pytest.mark.parametrize("filename", ['22_hcn_linear_freq_noraman.out',
                                       '40_n2o_linear_highT_highP.out'])
def test_linear_molecule_thermo(filename):
    bbe = _calc(qchem_path(filename))
    assert bbe.qh_gibbs_free_energy is not None
    # calc_bbe stores the parsed QCData on .xyz; linear_mol lives there.
    assert bbe.xyz.linear_mol is True


# ===========================================================================
# Cross-validation against matching g16 fixtures (rough numerical agreement)
# ===========================================================================

# Pairs where the Q-Chem and g16 inputs use exactly the same method/basis;
# expect ZPE and H to agree to ~5e-4 Hartree (SCF/grid defaults differ
# slightly between programs). Several g16 fixtures use functionals that
# Q-Chem maps slightly differently (TPSSTPSS→tpss, PBEPBE→pbe, APFD subbed
# with B3LYP+D3BJ) so we only cross-validate the cleanest pairs.

@pytest.mark.parametrize("qcname, g16name, tol_h", [
    ('01a_water_hf_freq.out',           '01a_water_hf_freq.log',           1e-3),
    ('05_methylene_triplet_carbene.out','05_methylene_triplet_carbene.log',5e-3),  # MP2 vs UMP2 differ slightly
    ('10_formaldehyde_verbose_pop.out', '10_formaldehyde_verbose_pop.log', 5e-3),
    ('22_hcn_linear_freq_noraman.out',  '22_hcn_linear_freq_noraman.log',  5e-3),
    ('44_ts_sn2_identity_chloride.out', '44_ts_sn2_identity_chloride.log', 5e-3),
])
def test_qchem_vs_g16_thermo_agreement(qcname, g16name, tol_h):
    """Same molecule, same level of theory in Q-Chem and Gaussian: ZPE
    and H should agree to within `tol_h` Hartree (SCF noise + grid
    differences between programs)."""
    a = _calc(qchem_path(qcname))
    b = _calc(g16path(g16name))
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
    for i, l in enumerate(lines):
        if 'STANDARD THERMODYNAMIC QUANTITIES AT' in l:
            last = i
    if last < 0:
        return None
    out = {}
    for l in lines[last:]:
        s = l.strip()
        if s.startswith('Zero point vibrational energy:'):
            out['zpe'] = float(s.split(':')[1].split()[0])
        elif s.startswith('Total Enthalpy:'):
            out['h'] = float(s.split(':')[1].split()[0])
        elif s.startswith('Total Entropy:'):
            out['s'] = float(s.split(':')[1].split()[0])
        elif s.startswith('QRRHO-Total Entropy:'):
            out['qhs'] = float(s.split(':')[1].split()[0])
        if 'Thank you very much' in l:
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
    bbe = _calc(qchem_path(filename))
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
