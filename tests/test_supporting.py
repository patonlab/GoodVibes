#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for supporting modules: vib_scale_factors and media."""

import pytest
from goodvibes.vib_scale_factors import (
    scaling_data_dict, scaling_refs, ScalingData,
    canonicalize_level, FUNCTIONAL_ALIASES,
)
from goodvibes.media import solvents


# ---------------------------------------------------------------------------
# vib_scale_factors: reference index validation
# ---------------------------------------------------------------------------

def test_scaling_refs_indices_valid():
    """All reference indices in scaling_data_dict must be valid indices into scaling_refs."""
    for key, entry in scaling_data_dict.items():
        assert entry.zpe_ref < len(scaling_refs), f"{key}: zpe_ref {entry.zpe_ref} out of range"
        assert entry.harm_ref < len(scaling_refs), f"{key}: harm_ref {entry.harm_ref} out of range"
        assert entry.fund_ref < len(scaling_refs), f"{key}: fund_ref {entry.fund_ref} out of range"


# ---------------------------------------------------------------------------
# vib_scale_factors: dictionary lookup
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level_basis, expected_zpe_fac", [
    ('B3LYP/6-31G(D)', 0.977),
    ('HF/6-31G(D)', 0.909),
])
def test_scaling_data_dict_lookup(level_basis, expected_zpe_fac):
    key = canonicalize_level(level_basis)
    entry = scaling_data_dict[key]
    assert abs(float(entry.zpe_fac) - expected_zpe_fac) < 0.001


# ---------------------------------------------------------------------------
# vib_scale_factors: canonicalization and aliasing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alias, canonical", [
    ("PBE1PBE", "PBE0"),
    ("M06-2X", "M062X"),
    ("M06-L", "M06L"),
    ("M06-HF", "M06HF"),
    ("M05-2X", "M052X"),
    ("wB97X-D", "WB97XD"),
    ("B97-3", "B973"),
    ("CAMB3LYP", "CAM-B3LYP"),
    ("HSE06", "HSEH1PBE"),
    ("PBEPBE", "PBE"),
])
def test_functional_aliases(alias, canonical):
    """Aliases resolve to the canonical functional name."""
    assert canonicalize_level(alias + "/MG3S") == canonical + "/MG3S"


def test_canonicalize_idempotent():
    """Canonicalizing an already-canonical key returns the same key."""
    for key in scaling_data_dict:
        assert canonicalize_level(key) == key


def test_alias_lookup_finds_entry():
    """Looking up via an alias finds the same data as the canonical name."""
    # PBE1PBE/MG3S should resolve to PBE0/MG3S (zpe_fac=0.975)
    key = canonicalize_level("PBE1PBE/MG3S")
    assert key in scaling_data_dict
    assert abs(scaling_data_dict[key].zpe_fac - 0.975) < 0.001


def test_canonicalize_case_insensitive():
    """Canonicalization is case-insensitive."""
    assert canonicalize_level("b3lyp/6-31g(d)") == canonicalize_level("B3LYP/6-31G(D)")


def test_canonicalize_no_basis():
    """Semi-empirical methods without basis set are handled."""
    assert canonicalize_level("AM1") == "AM1"
    assert canonicalize_level("pm3") == "PM3"


def test_entry_count():
    """Verify we loaded a reasonable number of entries."""
    assert len(scaling_data_dict) >= 200


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
