#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for thermochemistry calculations on ORCA 6 output files.

All tests are marked xfail because calc_bbe relies on getoutData (cclib)
which cannot parse ORCA 6 output, and thermo.py lacks an ORCA-specific
frequency/rotational constant parsing block.
"""

import pytest
from goodvibes.thermo import calc_bbe
from conftest import orca_path, ORCA_FREQ_FILES, ORCA_TS_FILES

GAS_CONSTANT = 8.3144621
ATMOS = 101.325
T_DEFAULT = 298.15
CONC_DEFAULT = ATMOS / (GAS_CONSTANT * T_DEFAULT)

XFAIL_REASON = (
    "calc_bbe relies on cclib (getoutData) which cannot parse ORCA 6 output; "
    "thermo.py lacks ORCA frequency/rotational constant parsing"
)


@pytest.mark.xfail(reason=XFAIL_REASON)
@pytest.mark.parametrize("filename", [
    '01a_water_hf_freq.out',
    '02_ethane_opt_freq_thermo.out',
    '04_benzene_radical_cation.out',
    '05_methylene_triplet_carbene.out',
    '08_alanine_C1_pcm_water.out',
])
def test_calc_bbe_orca_basic(filename):
    bbe = calc_bbe(orca_path(filename), 'grimme', False, 100.0, 100.0,
                   T_DEFAULT, CONC_DEFAULT, 1.0, 'none', False, False, 0)
    assert hasattr(bbe, 'gibbs_free_energy')


@pytest.mark.xfail(reason=XFAIL_REASON)
@pytest.mark.parametrize("filename", ORCA_TS_FILES)
def test_calc_bbe_orca_transition_state(filename):
    bbe = calc_bbe(orca_path(filename), 'grimme', False, 100.0, 100.0,
                   T_DEFAULT, CONC_DEFAULT, 1.0, 'none', False, False, 0)
    assert hasattr(bbe, 'gibbs_free_energy')
    assert len(bbe.im_frequency_wn) == 1


@pytest.mark.xfail(reason=XFAIL_REASON)
@pytest.mark.parametrize("filename", [
    '08_alanine_C1_pcm_water.out',
    '19_acetic_acid_smd_dmso.out',
    '29_aniline_cpcm_chloroform.out',
])
def test_calc_bbe_orca_solvation(filename):
    bbe = calc_bbe(orca_path(filename), 'grimme', False, 100.0, 100.0,
                   T_DEFAULT, CONC_DEFAULT, 1.0, 'none', False, False, 0)
    assert hasattr(bbe, 'gibbs_free_energy')
