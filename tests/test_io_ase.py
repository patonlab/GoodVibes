#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for parsing ASE-driven calculations encoded as extxyz."""

import os
import tempfile

import pytest

from goodvibes.io import (parse_ase_thermo, parse_data, parse_qcdata,
                          read_initial)
from conftest import (ASE_FREQ_FILES, ASE_TS_FILES, ASE_LINEAR_FILES,
                      ASE_SOLVATION_FILES, ase_path)


# ---------------------------------------------------------------------------
# parse_qcdata: program auto-detection from .extxyz
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", ASE_FREQ_FILES + ASE_TS_FILES)
def test_parse_qcdata_detects_ase(filename):
    q = parse_qcdata(ase_path(filename))
    assert q.program == 'ase'


def test_parse_data_ase_program():
    spe, program, *_ = parse_data(ase_path('01_water.extxyz'))
    assert program == 'ase'
    assert spe is not None


# ---------------------------------------------------------------------------
# Energy extraction (Hartree fixtures)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_energy", [
    ('01_water.extxyz',            -76.0105109111),
    ('05_methylene_triplet.extxyz', -39.019780646668),
    ('10_formaldehyde.extxyz',     -114.462988874),
    ('22_hcn_linear.extxyz',        -93.414443348),
    ('44_ts_sn2.extxyz',           -960.457792359),
])
def test_scf_energy_extraction_hartree(filename, expected_energy):
    q = parse_qcdata(ase_path(filename))
    assert abs(q.scf_energy - expected_energy) < 1e-9


def test_scf_energy_unit_conversion_eV_to_hartree():
    """The alanine fixture is encoded in eV; parser should convert to Hartree
    and match the value parsed straight from the matching g16 .log."""
    q_ext = parse_qcdata(ase_path('08_alanine_pcm_water.extxyz'))
    # Source g16 energy: -323.372758893 Hartree
    assert abs(q_ext.scf_energy - (-323.372758893)) < 1e-9


# ---------------------------------------------------------------------------
# Charge and multiplicity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_charge, expected_mult", [
    ('01_water.extxyz', 0, 1),
    ('05_methylene_triplet.extxyz', 0, 3),
    ('10_formaldehyde.extxyz', 0, 1),
    ('22_hcn_linear.extxyz', 0, 1),
    ('44_ts_sn2.extxyz', -1, 1),
    ('08_alanine_pcm_water.extxyz', 0, 1),
])
def test_charge_multiplicity(filename, expected_charge, expected_mult):
    q = parse_qcdata(ase_path(filename))
    assert q.charge == expected_charge
    assert q.multiplicity == expected_mult


# ---------------------------------------------------------------------------
# Frequencies — sign convention splits into real vs imaginary
# ---------------------------------------------------------------------------

def test_frequencies_split_by_sign_ts():
    """SN2 TS has exactly one imaginary mode; rest are positive."""
    q = parse_qcdata(ase_path('44_ts_sn2.extxyz'))
    assert len(q.im_frequency_wn) == 1
    assert q.im_frequency_wn[0] < 0
    assert all(f > 0 for f in q.frequency_wn)


def test_no_imaginary_in_ground_state():
    q = parse_qcdata(ase_path('01_water.extxyz'))
    assert q.im_frequency_wn == []
    assert len(q.frequency_wn) == 3


# ---------------------------------------------------------------------------
# Geometry & molecular mass
# ---------------------------------------------------------------------------

def test_atom_block_parsed():
    q = parse_qcdata(ase_path('01_water.extxyz'))
    assert q.atom_types == ['O', 'H', 'H']
    assert q.atom_nums == [8, 1, 1]
    assert len(q.cartesians) == 3


def test_molecular_mass_computed_when_absent():
    """Generated fixtures don't write molecular_mass; parser should compute
    it from atom_types via ATOMIC_MASSES."""
    q = parse_qcdata(ase_path('01_water.extxyz'))
    # H2O = 1.00782503207 * 2 + 15.99491461957 ≈ 18.0106
    assert abs(q.molecular_mass - 18.01056) < 1e-3


# ---------------------------------------------------------------------------
# Linear molecules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", ASE_LINEAR_FILES)
def test_linear_marker(filename):
    q = parse_qcdata(ase_path(filename))
    assert q.linear_mol is True


# ---------------------------------------------------------------------------
# Solvation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", ASE_SOLVATION_FILES)
def test_solvation_model_propagated(filename):
    q = parse_qcdata(ase_path(filename))
    assert q.solvation_model and q.solvation_model != 'gas phase'


# ---------------------------------------------------------------------------
# Job type detection
# ---------------------------------------------------------------------------

def test_job_type_ts():
    q = parse_qcdata(ase_path('44_ts_sn2.extxyz'))
    assert q.job_type == 'TS'


def test_job_type_freq_default():
    q = parse_qcdata(ase_path('01_water.extxyz'))
    assert q.job_type in ('Freq', 'GSFreq')


# ---------------------------------------------------------------------------
# read_initial: level of theory + solvation extracted from comment line
# ---------------------------------------------------------------------------

def test_read_initial_water_gas_phase():
    lot, solv, prog, _, _ = read_initial(ase_path('01_water.extxyz'))
    assert lot == 'HF/6-31G(d)'
    assert solv == 'gas phase'
    assert prog == 'Normal'


def test_read_initial_alanine_solvated():
    lot, solv, prog, _, _ = read_initial(ase_path('08_alanine_pcm_water.extxyz'))
    assert 'M062X' in lot
    assert solv != 'gas phase'
    assert prog == 'Normal'


# ---------------------------------------------------------------------------
# Optional-keys path: minimal extxyz still parses successfully
# ---------------------------------------------------------------------------

def test_minimal_extxyz_parses():
    """The bare-minimum extxyz (no point_group, symmno, linear_mol, zpe)
    should still parse; mass and rotemp fall back to geometry-derived values."""
    minimal = (
        '3\n'
        'Properties=species:S:1:pos:R:3 program=ase scf_energy=-76.4 charge=0 multiplicity=1 '
        'frequencies="1655 3826 3935"\n'
        'O 0.0 0.0  0.117\n'
        'H 0.0 0.755 -0.471\n'
        'H 0.0 -0.755 -0.471\n'
    )
    with tempfile.NamedTemporaryFile('w', suffix='.extxyz', delete=False) as f:
        f.write(minimal)
        path = f.name
    try:
        q = parse_qcdata(path)
        assert q.program == 'ase'
        assert q.scf_energy == -76.4
        assert len(q.frequency_wn) == 3
        assert q.molecular_mass > 0
        assert any(t > 0 for t in q.rotemp)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# parse_ase_thermo direct call (skip the dispatcher)
# ---------------------------------------------------------------------------

def test_parse_ase_thermo_direct():
    q = parse_ase_thermo(ase_path('01_water.extxyz'))
    assert q.program == 'ase'
    assert q.scf_energy is not None
