#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for QCData JSON caching (serialization round-trip, precision, integration)."""

import json
import os

import pytest

from conftest import (g16path, orca_path, ase_path,
                      G16_FREQ_FILES, G16_TS_FILES,
                      ORCA_FREQ_FILES, ORCA_TS_FILES,
                      ASE_FREQ_FILES, ASE_TS_FILES)
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


@pytest.mark.parametrize("filename", ASE_FREQ_FILES + ASE_TS_FILES)
def test_qcdata_roundtrip_ase(filename):
    """ASE-sourced QCData round-trips through the cache schema unchanged
    (the cache layer is source-format-agnostic)."""
    original = parse_qcdata(ase_path(filename))
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


# --- SPC caching: --spc results live on QCData so --import skips the SPC re-parse ---

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ETHANE = os.path.join(REPO_ROOT, "goodvibes", "examples", "ethane.out")
ETHANE_TZ = os.path.join(REPO_ROOT, "goodvibes", "examples", "ethane_TZ.out")


def test_spc_parse_populates_qcdata_sp_fields():
    """When --spc is used, the parsed SPC numbers are written back onto
    the QCData container so a subsequent qcdata_to_dict round-trip
    carries them through."""
    qcdata = parse_qcdata(ETHANE)
    assert qcdata.sp_energy is None         # untouched before calc_bbe
    bbe = calc_bbe(ETHANE, 'grimme', False, 100.0, 100.0, 298.15,
                   0.0408740470708, 1.0, None, spc='TZ', qcdata=qcdata)
    # calc_bbe parsed the SPC and persisted the result into qcdata.
    assert qcdata.sp_energy == bbe.sp_energy
    assert qcdata.sp_suffix == 'TZ'
    assert qcdata.sp_file.endswith('ethane_TZ.out')
    # The cached SPC energy is the higher-level number, not the parent SCF.
    assert qcdata.sp_energy != qcdata.scf_energy


def test_spc_cache_hit_skips_disk_lookup(monkeypatch):
    """A second calc_bbe call with the same --spc reuses the cached values
    on QCData and never calls find_spc_file()."""
    # Prime the cache via a real first parse.
    qcdata = parse_qcdata(ETHANE)
    calc_bbe(ETHANE, 'grimme', False, 100.0, 100.0, 298.15,
            0.0408740470708, 1.0, None, spc='TZ', qcdata=qcdata)
    cached_sp = qcdata.sp_energy
    assert cached_sp is not None

    # Now block disk SPC lookups; the second call must succeed entirely
    # off the cache.
    import goodvibes.thermo as gv_thermo
    calls = {"n": 0}
    def boom(*a, **kw):
        calls["n"] += 1
        raise AssertionError("find_spc_file should not be called on cache hit")
    monkeypatch.setattr(gv_thermo, 'find_spc_file', boom)

    bbe2 = calc_bbe(ETHANE, 'grimme', False, 100.0, 100.0, 298.15,
                    0.0408740470708, 1.0, None, spc='TZ', qcdata=qcdata)
    assert calls["n"] == 0
    assert bbe2.sp_energy == cached_sp


def test_spc_cache_miss_on_different_suffix_refreshes(monkeypatch):
    """If the user passes a --spc suffix that doesn't match the cached
    sp_suffix, the cache is bypassed and the SPC file is re-resolved
    from disk; on a successful re-parse the cache is updated."""
    qcdata = parse_qcdata(ETHANE)
    # Prime cache with a fake suffix that doesn't match the real fixture.
    qcdata.sp_energy = -42.0
    qcdata.sp_suffix = 'OTHER'
    qcdata.sp_file = '/nope/elsewhere.out'

    calls = {"n": 0}
    real_find = __import__('goodvibes.io', fromlist=['find_spc_file']).find_spc_file
    def counting(*a, **kw):
        calls["n"] += 1
        return real_find(*a, **kw)
    import goodvibes.thermo as gv_thermo
    monkeypatch.setattr(gv_thermo, 'find_spc_file', counting)

    bbe = calc_bbe(ETHANE, 'grimme', False, 100.0, 100.0, 298.15,
                   0.0408740470708, 1.0, None, spc='TZ', qcdata=qcdata)
    assert calls["n"] == 1                  # disk lookup did fire
    assert bbe.sp_energy != -42.0           # stale value not used
    # Cache refreshed with the new suffix.
    assert qcdata.sp_suffix == 'TZ'
    assert qcdata.sp_energy == bbe.sp_energy


def test_spc_roundtrip_through_json(tmp_path):
    """Full round-trip: parse + --spc, qcdata_to_dict, JSON, dict_to_qcdata —
    sp_* fields all survive."""
    qcdata = parse_qcdata(ETHANE)
    calc_bbe(ETHANE, 'grimme', False, 100.0, 100.0, 298.15,
            0.0408740470708, 1.0, None, spc='TZ', qcdata=qcdata)
    sp_before = qcdata.sp_energy

    d = qcdata_to_dict(qcdata)
    restored = dict_to_qcdata(json.loads(json.dumps(d)))

    assert restored.sp_energy == sp_before
    assert restored.sp_suffix == 'TZ'
    assert restored.sp_file == qcdata.sp_file
    assert restored.sp_version_program == qcdata.sp_version_program


def test_spc_import_skips_reparse_when_files_gone(tmp_path):
    """End-to-end: cache an SPC parse, then run again without the SPC
    file present on disk — the cached sp_energy still drives the
    thermo. This is the headline workflow that motivated this feature."""
    # First pass: parse parent + SPC, capture qcdata.
    qcdata = parse_qcdata(ETHANE)
    bbe1 = calc_bbe(ETHANE, 'grimme', False, 100.0, 100.0, 298.15,
                    0.0408740470708, 1.0, None, spc='TZ', qcdata=qcdata)
    sp1 = bbe1.sp_energy
    assert sp1 is not None and sp1 != '!'

    # Second pass: copy the parent file into a tmpdir but NOT the SPC
    # file; rerun --spc TZ pointing at the tmpdir copy. Without the
    # cache this would be sp_energy = '!'; with the cache it succeeds.
    import shutil
    parent_copy = tmp_path / "ethane.out"
    shutil.copy(ETHANE, parent_copy)
    # Re-target qcdata.file at the tmpdir parent so find_spc_file would
    # search the (SPC-less) tmpdir if the cache check failed.
    qcdata.file = str(parent_copy)

    bbe2 = calc_bbe(str(parent_copy), 'grimme', False, 100.0, 100.0, 298.15,
                    0.0408740470708, 1.0, None, spc='TZ', qcdata=qcdata)
    assert bbe2.sp_energy == sp1
    assert bbe2.gibbs_free_energy == bbe1.gibbs_free_energy


# --- find_spc_file: exact-suffix matching ---

def test_find_spc_file_requires_exact_suffix(tmp_path):
    """--spc TZVP must match optimization_TZVP.log exactly. It must NOT
    fall through to optimization_def2_TZVP.log via a wildcard. Users who
    want the def2_TZVP file pass the full suffix."""
    from goodvibes.io import find_spc_file
    parent = tmp_path / "optimization.log"
    parent.write_text("")
    decoy = tmp_path / "optimization_def2_TZVP.log"
    decoy.write_text("")

    assert find_spc_file(str(tmp_path / "optimization"), "TZVP") is None

    exact = tmp_path / "optimization_TZVP.log"
    exact.write_text("")
    assert find_spc_file(str(tmp_path / "optimization"), "TZVP") == str(exact)

    # The full suffix correctly resolves the def2_TZVP file.
    assert find_spc_file(str(tmp_path / "optimization"), "def2_TZVP") == str(decoy)


def test_find_spc_file_prefers_log_over_out(tmp_path):
    """When both .log and .out variants exist, .log wins (matches the
    historical search order)."""
    from goodvibes.io import find_spc_file
    (tmp_path / "job_TZ.log").write_text("")
    (tmp_path / "job_TZ.out").write_text("")
    assert find_spc_file(str(tmp_path / "job"), "TZ") == str(tmp_path / "job_TZ.log")


# --- --spc CPU accounting ---

def test_spc_cpu_does_not_overwrite_parent_cpu():
    """When --spc is used, the parent CPU stays on bbe.cpu and the SPC
    CPU lands on bbe.sp_cpu. print_cpu_time then sums both. Regression
    for a bug where assigning the SPC CPU to bbe.cpu silently dropped
    the parent's CPU from the TOTAL CPU line."""
    qcdata = parse_qcdata(ETHANE)
    parent_cpu = list(qcdata.cpu)            # captured before calc_bbe

    bbe = calc_bbe(ETHANE, 'grimme', False, 100.0, 100.0, 298.15,
                   0.0408740470708, 1.0, None, spc='TZ', qcdata=qcdata)

    assert bbe.cpu == parent_cpu             # parent CPU preserved
    assert bbe.sp_cpu is not None            # SPC CPU captured separately
    assert bbe.sp_cpu != parent_cpu          # and is distinct from the parent


def test_sp_cpu_sums_multistep_gaussian():
    """Gaussian linked SPC jobs (opt+SP, composite methods, etc.) emit one
    `Job cpu time` line per step. sp_cpu() must SUM them, matching the
    main parser's behavior. Regression for a bug where sp_cpu() returned
    only the LAST line, silently dropping the bulk of the CPU time on
    multi-step SPC files. Uses 02_ethane_opt_freq_T398_P2.log which has
    two Job-cpu-time lines (18.4 s + 26.5 s = 44.9 s)."""
    from goodvibes.io import sp_cpu, parse_qcdata
    from conftest import g16path
    multistep = g16path('02_ethane_opt_freq_T398_P2.log')

    cpu = sp_cpu(multistep)
    # The main parser sums all Job-cpu-time lines into qcdata.cpu; sp_cpu
    # must produce the same sum so --spc CPU totals match the standalone
    # parse.
    qcdata_cpu = parse_qcdata(multistep).cpu
    assert cpu == qcdata_cpu

    # And concretely: 18.4 s + 26.5 s = 44.9 s = 44900 ms (the parser
    # encodes seconds as msecs with the secs slot at 0).
    assert cpu == [0, 0, 0, 0, 44900]


def test_sp_cpu_applies_orca_nprocs_scaling():
    """ORCA prints wall time only (TOTAL RUN TIME); the main parser
    scales by parallel-MPI process count to estimate effective CPU.
    sp_cpu() must apply the same scaling so --spc totals match the
    standalone parse. Regression for a bug where sp_cpu()'s ORCA branch
    returned raw wall time, dropping a factor of nprocs (e.g. 8x for an
    8-core run, which silently turned 3 days of effective CPU into
    9 hours on the --spc path)."""
    from goodvibes.io import sp_cpu, parse_qcdata
    from conftest import orca_path
    parallel = orca_path('16_o2_superoxide_anion.out')

    qcdata = parse_qcdata(parallel)
    assert qcdata.nprocs > 1                       # the fixture must be parallel
    assert sp_cpu(parallel) == qcdata.cpu          # sp_cpu must include nprocs scaling
