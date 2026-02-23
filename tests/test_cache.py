#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for QCData JSON caching (serialization round-trip, precision, integration)."""

import json
import os

import pytest

from conftest import g16path, orca_path, G16_FREQ_FILES, G16_TS_FILES, ORCA_FREQ_FILES, ORCA_TS_FILES
from goodvibes.io import parse_qcdata, qcdata_to_dict, dict_to_qcdata, save_cache, load_cache, CACHE_VERSION
from goodvibes.thermo import calc_bbe
from dataclasses import asdict


# --- Round-trip serialization tests ---

@pytest.mark.parametrize("filename", G16_FREQ_FILES[:5] + G16_TS_FILES[:2])
def test_qcdata_roundtrip_g16(filename):
    """Parse a G16 file, serialize to dict, deserialize back, verify all fields match."""
    original = parse_qcdata(g16path(filename))
    d = qcdata_to_dict(original)
    restored = dict_to_qcdata(d)
    assert asdict(restored) == asdict(original)


@pytest.mark.parametrize("filename", ORCA_FREQ_FILES[:5] + ORCA_TS_FILES[:2])
def test_qcdata_roundtrip_orca(filename):
    """Parse an ORCA file, serialize to dict, deserialize back, verify all fields match."""
    original = parse_qcdata(orca_path(filename))
    d = qcdata_to_dict(original)
    restored = dict_to_qcdata(d)
    assert asdict(restored) == asdict(original)


# --- Float precision test ---

def test_scf_energy_precision_g16():
    """Verify SCF energy survives JSON round-trip with exact equality."""
    original = parse_qcdata(g16path('01a_water_hf_freq.log'))
    d = qcdata_to_dict(original)
    json_str = json.dumps(d)
    d2 = json.loads(json_str)
    restored = dict_to_qcdata(d2)
    assert restored.scf_energy == original.scf_energy


def test_scf_energy_precision_orca():
    """Verify SCF energy survives JSON round-trip with exact equality."""
    original = parse_qcdata(orca_path('01a_water_hf_freq.out'))
    d = qcdata_to_dict(original)
    json_str = json.dumps(d)
    d2 = json.loads(json_str)
    restored = dict_to_qcdata(d2)
    assert restored.scf_energy == original.scf_energy


# --- Cache save/load tests ---

def test_cache_save_load(tmp_path):
    """Save a multi-file cache, load it back, verify contents."""
    files = {
        'water': parse_qcdata(g16path('01a_water_hf_freq.log')),
        'ethane': parse_qcdata(g16path('02_ethane_opt_freq_T398_P2.log')),
    }
    cache_dict = {key: qcdata_to_dict(qcdata) for key, qcdata in files.items()}
    cache_path = str(tmp_path / 'test_cache.json')
    save_cache(cache_dict, cache_path)

    # Verify file exists and is valid JSON
    assert os.path.exists(cache_path)
    with open(cache_path) as f:
        raw = json.load(f)
    assert raw['_cache_version'] == CACHE_VERSION
    assert 'water' in raw['entries']
    assert 'ethane' in raw['entries']

    # Load and verify
    loaded = load_cache(cache_path)
    assert set(loaded.keys()) == {'water', 'ethane'}
    assert loaded['water'].scf_energy == files['water'].scf_energy
    assert loaded['ethane'].frequency_wn == files['ethane'].frequency_wn


def test_cache_version_mismatch(tmp_path):
    """Loading a cache with wrong version raises ValueError."""
    cache_path = str(tmp_path / 'bad_cache.json')
    with open(cache_path, 'w') as f:
        json.dump({'_cache_version': 999, 'entries': {}}, f)
    with pytest.raises(ValueError, match="Cache version mismatch"):
        load_cache(cache_path)


# --- Integration test: cached calc_bbe matches direct ---

def test_calc_bbe_cached_matches_direct_g16():
    """calc_bbe with pre-parsed QCData gives identical results to parsing from file."""
    filepath = g16path('01a_water_hf_freq.log')

    # Direct (parses from file)
    bbe_direct = calc_bbe(filepath, 'grimme', False, 100.0, 100.0, 298.15,
                          0.0408740470708, 1.0, None, None, None)

    # Via cache round-trip
    qcdata = parse_qcdata(filepath)
    d = qcdata_to_dict(qcdata)
    json_str = json.dumps(d)
    d2 = json.loads(json_str)
    restored = dict_to_qcdata(d2)
    bbe_cached = calc_bbe(filepath, 'grimme', False, 100.0, 100.0, 298.15,
                          0.0408740470708, 1.0, None, None, None, qcdata=restored)

    assert bbe_direct.scf_energy == bbe_cached.scf_energy
    assert bbe_direct.enthalpy == bbe_cached.enthalpy
    assert bbe_direct.gibbs_free_energy == bbe_cached.gibbs_free_energy
    assert bbe_direct.qh_gibbs_free_energy == bbe_cached.qh_gibbs_free_energy


def test_calc_bbe_cached_matches_direct_orca():
    """calc_bbe with pre-parsed QCData gives identical results to parsing from file."""
    filepath = orca_path('01a_water_hf_freq.out')

    # Direct (parses from file)
    bbe_direct = calc_bbe(filepath, 'grimme', False, 100.0, 100.0, 298.15,
                          0.0408740470708, 1.0, None, None, None)

    # Via cache round-trip
    qcdata = parse_qcdata(filepath)
    d = qcdata_to_dict(qcdata)
    json_str = json.dumps(d)
    d2 = json.loads(json_str)
    restored = dict_to_qcdata(d2)
    bbe_cached = calc_bbe(filepath, 'grimme', False, 100.0, 100.0, 298.15,
                          0.0408740470708, 1.0, None, None, None, qcdata=restored)

    assert bbe_direct.scf_energy == bbe_cached.scf_energy
    assert bbe_direct.enthalpy == bbe_cached.enthalpy
    assert bbe_direct.gibbs_free_energy == bbe_cached.gibbs_free_energy
    assert bbe_direct.qh_gibbs_free_energy == bbe_cached.qh_gibbs_free_energy


def test_calc_bbe_cached_different_temperature():
    """Cached QCData produces correct results at a different temperature."""
    filepath = g16path('01a_water_hf_freq.log')

    # Parse once, cache it
    qcdata = parse_qcdata(filepath)
    d = qcdata_to_dict(qcdata)
    json_str = json.dumps(d)
    restored = dict_to_qcdata(json.loads(json_str))

    # Run at 400K from cache
    bbe_cached_400 = calc_bbe(filepath, 'grimme', False, 100.0, 100.0, 400.0,
                              0.0408740470708, 1.0, None, None, None, qcdata=restored)

    # Run at 400K directly
    bbe_direct_400 = calc_bbe(filepath, 'grimme', False, 100.0, 100.0, 400.0,
                              0.0408740470708, 1.0, None, None, None)

    assert bbe_cached_400.enthalpy == bbe_direct_400.enthalpy
    assert bbe_cached_400.gibbs_free_energy == bbe_direct_400.gibbs_free_energy


def test_calc_bbe_cache_only_no_file_on_disk(tmp_path):
    """calc_bbe works from cached QCData even when the original file does not exist."""
    # Parse from real file, then cache the data
    filepath = g16path('01a_water_hf_freq.log')
    qcdata = parse_qcdata(filepath)
    d = qcdata_to_dict(qcdata)
    restored = dict_to_qcdata(json.loads(json.dumps(d)))

    # Use a fake file path that does not exist
    fake_path = str(tmp_path / 'nonexistent.log')

    bbe = calc_bbe(fake_path, 'grimme', False, 100.0, 100.0, 298.15,
                   0.0408740470708, 1.0, None, None, None, qcdata=restored)

    # Should produce valid thermochemistry
    assert bbe.scf_energy == qcdata.scf_energy
    assert hasattr(bbe, 'gibbs_free_energy')
    assert hasattr(bbe, 'enthalpy')

    # Compare against direct parsing
    bbe_direct = calc_bbe(filepath, 'grimme', False, 100.0, 100.0, 298.15,
                          0.0408740470708, 1.0, None, None, None)
    assert bbe.enthalpy == bbe_direct.enthalpy
    assert bbe.gibbs_free_energy == bbe_direct.gibbs_free_energy
