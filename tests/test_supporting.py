#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for supporting modules: vib_scale_factors and media."""

import pytest
from goodvibes.vib_scale_factors import scaling_data, scaling_refs, scaling_data_dict, scaling_data_dict_mod
from goodvibes.media import solvents


# ---------------------------------------------------------------------------
# vib_scale_factors: reference index validation
# ---------------------------------------------------------------------------

def test_scaling_refs_indices_valid():
    """All reference indices in scaling_data must be valid indices into scaling_refs."""
    refs = list(scaling_data['zpe_ref']) + list(scaling_data['harm_ref']) + list(scaling_data['fund_ref'])
    assert max(refs, default=-1) < len(scaling_refs)


# ---------------------------------------------------------------------------
# vib_scale_factors: dictionary lookup
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level_basis, expected_zpe_fac", [
    ('B3LYP/6-31G(D)', 0.977),
    ('HF/6-31G(D)', 0.909),
])
def test_scaling_data_dict_lookup(level_basis, expected_zpe_fac):
    entry = scaling_data_dict[level_basis.upper()]
    assert abs(float(entry.zpe_fac) - expected_zpe_fac) < 0.001


def test_scaling_data_dict_mod_strips_hyphens():
    """scaling_data_dict_mod removes hyphens for fuzzy matching."""
    # Verify that a key exists that would have hyphens removed
    assert len(scaling_data_dict_mod) > 0
    # Check that a known entry without hyphens is accessible
    for key in scaling_data_dict_mod:
        assert '-' not in key
        break  # Just check the first key


# ---------------------------------------------------------------------------
# media: solvents dictionary
# ---------------------------------------------------------------------------

def test_solvents_common_entries():
    assert 'h2o' in solvents
    assert 'water' in solvents
    assert 'dmso' in solvents
    assert 'thf' in solvents
    assert 'methanol' in solvents
    assert 'toluene' in solvents


def test_solvents_values_positive():
    for name, (mw, density) in solvents.items():
        assert mw > 0, f"Solvent {name}: MW must be positive"
        assert density > 0, f"Solvent {name}: density must be positive"


def test_water_properties():
    mw, density = solvents['h2o']
    assert abs(mw - 18.02) < 0.01
    assert abs(density - 0.998) < 0.002
