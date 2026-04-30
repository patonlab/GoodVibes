#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Lightweight regression coverage for ORCA 5 output parsing and thermo.

The ORCA 6 suite (test_thermo_orca.py, test_io_orca.py) deeply exercises
parser correctness — this file's job is narrower: catch format regressions
between ORCA 5 and 6.  Three layers:

1.  parse_qcdata sweep — every non-error ORCA 5 file must parse without
    an unhandled exception and report program == "Orca".
2.  Version detection — parse_data must surface "5" in the version string.
3.  Spot-check thermo — ZPE/H/G against ORCA 5's printed values for a
    representative set covering HF, DFT, MP2, semi-empirical, solvated,
    linear, heavy-atom, and TS jobs.

Tolerance matches test_thermo_orca.py (5e-6 Eh) — see that file's header
for the precision rationale.
"""

import pytest

from goodvibes.io import parse_data, parse_qcdata
from goodvibes.thermo import calc_bbe
from conftest import ORCA5_FILES, ORCA5_ERROR_FILES, orca5_path

GAS_CONSTANT = 8.3144621
ATMOS = 101.325
T_DEFAULT = 298.15


def _calc(filename, scale=1.0):
    """Run calc_bbe with the same defaults as test_thermo_orca.py."""
    conc = ATMOS / (GAS_CONSTANT * T_DEFAULT)
    return calc_bbe(orca5_path(filename), 'grimme', False, 100.0, 100.0,
                    T_DEFAULT, conc, scale, None, None, None, 0,
                    inertia='conf')


# ---------------------------------------------------------------------------
# parse_qcdata broad sweep — must succeed and identify program as Orca
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", ORCA5_FILES)
def test_parse_qcdata_orca5_no_crash(filename):
    qc = parse_qcdata(orca5_path(filename))
    assert qc is not None
    assert qc.program == "Orca"


# ---------------------------------------------------------------------------
# Version detection — parse_data must surface ORCA 5 in the version string
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", [
    '01a_water_hf_freq.out',
    '08_alanine_C1_pcm_water.out',
    '26_pt_complex_genecp_3zone.out',
    '44_ts_sn2_identity_chloride.out',
])
def test_parse_data_orca5_version(filename):
    _, _, version, *_ = parse_data(orca5_path(filename))
    assert "5" in version
    assert "ORCA" in version.upper()


# ---------------------------------------------------------------------------
# Error files — should not raise unhandled exceptions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", ORCA5_ERROR_FILES)
def test_calc_bbe_orca5_error_files(filename):
    try:
        _calc(filename)
    except (AttributeError, ValueError, IndexError, ZeroDivisionError):
        pass
    except SystemExit:
        pass


# ---------------------------------------------------------------------------
# Spot-check thermochemistry against ORCA 5 ground truth
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_zpe, expected_H, expected_G", [
    ('01a_water_hf_freq.out',                    0.02234417,    -75.98299232,    -76.00440203),
    ('04_benzene_radical_cation.out',            0.09503119,   -231.91743139,   -231.95049545),
    ('08_alanine_C1_pcm_water.out',              0.10872264,   -323.26174381,   -323.29950799),
    ('19_acetic_acid_smd_dmso.out',              0.06114068,   -228.99654812,   -229.02890173),
    ('22_hcn_linear_freq_noraman.out',           0.01663743,    -93.39440570,    -93.41722698),
    ('26_pt_complex_genecp_3zone.out',           0.45836734,  -1962.77095479,  -1962.86339641),
    ('29_aniline_cpcm_chloroform.out',           0.11660787,   -287.43842265,   -287.47387938),
    ('32_cyclohexane_tpss_meta_gga.out',         0.16750488,   -235.84448868,   -235.87957707),
    ('38_naphthalene_scsmp2.out',                0.14728005,   -384.48025346,   -384.52000863),
    ('44_ts_sn2_identity_chloride.out',          0.03682768,   -960.24874252,   -960.28183049),
    ('45_ts_diels_alder_butadiene_ethylene.out', 0.14244409,   -234.26022322,   -234.29691417),
])
def test_calc_bbe_orca5_thermo_vs_orca5(filename, expected_zpe, expected_H,
                                         expected_G):
    """Validate calc_bbe ZPE/H/G against ORCA 5's printed values."""
    bbe = _calc(filename)
    assert abs(bbe.zpe - expected_zpe) < 5e-6
    assert abs(bbe.enthalpy - expected_H) < 5e-6
    assert abs(bbe.qh_gibbs_free_energy - expected_G) < 5e-6
