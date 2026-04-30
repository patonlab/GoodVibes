#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for parsing Q-Chem 6 output files using goodvibes.io."""

import pytest

from goodvibes.io import (parse_data, parse_qchem_thermo, parse_qcdata,
                          read_initial)
from conftest import (qchem_path, QCHEM_FREQ_FILES, QCHEM_TS_FILES,
                      QCHEM_SP_ONLY_FILES, QCHEM_ERROR_FILES)


# ---------------------------------------------------------------------------
# parse_qcdata: program auto-detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename",
    QCHEM_FREQ_FILES + QCHEM_TS_FILES + QCHEM_SP_ONLY_FILES)
def test_parse_qcdata_detects_qchem(filename):
    q = parse_qcdata(qchem_path(filename))
    assert q.program == 'QChem'


def test_parse_data_qchem_program():
    spe, program, *_ = parse_data(qchem_path('01a_water_hf_freq.out'))
    assert program == 'QChem'
    assert spe is not None


# ---------------------------------------------------------------------------
# Energy extraction — DFT/HF "Total energy" line
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_energy", [
    ('01a_water_hf_freq.out',                        -76.0105109),
    ('05_methylene_triplet_carbene.out',             -39.01978075),  # MP2 final
    ('08_alanine_C1_pcm_water.out',                 -323.37615244),  # CPCM-water
    ('10_formaldehyde_verbose_pop.out',             -114.46116374),
    ('19_acetic_acid_smd_dmso.out',                 -229.18130563),  # SMD-dmso
    ('22_hcn_linear_freq_noraman.out',               -93.41407838),
    ('44_ts_sn2_identity_chloride.out',             -960.4577515),
])
def test_scf_energy_extraction(filename, expected_energy):
    q = parse_qcdata(qchem_path(filename))
    assert abs(q.scf_energy - expected_energy) < 1e-6


# ---------------------------------------------------------------------------
# Energy precedence: CCSD(T), MP2, B2PLYP composite
# ---------------------------------------------------------------------------

def test_ccsdt_total_energy_picked_over_hf_reference():
    """File 11 prints `Total energy = -100.058` (HF ref) AND
    `CCSD(T) total energy = -100.338`. Parser must pick the latter."""
    q = parse_qcdata(qchem_path('11_hf_molecule_ccsdt_gold_standard.out'))
    assert abs(q.scf_energy - (-100.33835614)) < 1e-6


def test_mp2_total_energy_picked_for_mp2_method():
    """UMP2 freq job: parser must pick `MP2 total energy =` (-39.0198)
    instead of the embedded HF reference (-38.9267)."""
    q = parse_qcdata(qchem_path('05_methylene_triplet_carbene.out'))
    assert abs(q.scf_energy - (-39.01978075)) < 1e-6
    assert q.scf_energy < -39.0  # not the HF reference at -38.93


def test_b2plyp_composite_picks_dft_total_not_extra_mp2():
    """File 18 is B3LYP opt+freq then B2PLYP single point. Q-Chem prints
    an extra `MP2 total energy = -119.04` analysis line in the B2PLYP
    step; we must use the B2PLYP `Total energy = -118.87`."""
    q = parse_qcdata(qchem_path('18_propane_linked_composite_dh.out'))
    assert abs(q.scf_energy - (-118.87156552)) < 1e-6


# ---------------------------------------------------------------------------
# Charge & multiplicity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_charge, expected_mult", [
    ('01a_water_hf_freq.out',                  0, 1),
    ('04_benzene_radical_cation.out',          1, 2),
    ('05_methylene_triplet_carbene.out',       0, 3),
    ('06_carbon_atom_single_point.out',        0, 3),
    ('16_o2_superoxide_anion.out',            -1, 2),
    ('31_methylammonium_cpcm_water.out',       1, 1),
    ('44_ts_sn2_identity_chloride.out',       -1, 1),
    ('46_ts_h3_hydrogen_abstraction.out',      0, 2),
    ('47_ts_e2_elimination_ethylchloride.out', -1, 1),
])
def test_charge_multiplicity(filename, expected_charge, expected_mult):
    q = parse_qcdata(qchem_path(filename))
    assert q.charge == expected_charge
    assert q.multiplicity == expected_mult


# ---------------------------------------------------------------------------
# Frequency extraction (real / imaginary count)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, n_real, n_imag", [
    ('01a_water_hf_freq.out',                        3, 0),
    ('22_hcn_linear_freq_noraman.out',               4, 0),  # 3N-5
    ('32_cyclohexane_tpss_meta_gga.out',            48, 0),  # 18 atoms × 3 - 6
    ('44_ts_sn2_identity_chloride.out',             11, 1),  # TS, 1 imag
    ('45_ts_diels_alder_butadiene_ethylene.out',    41, 1),  # Diels-Alder TS
    ('37_planar_cyclohexane_3rd_order_saddle.out',  45, 3),  # 3rd-order saddle
])
def test_frequency_counts(filename, n_real, n_imag):
    q = parse_qcdata(qchem_path(filename))
    assert len(q.frequency_wn) == n_real
    assert len(q.im_frequency_wn) == n_imag


def test_imaginary_frequencies_are_negative():
    q = parse_qcdata(qchem_path('44_ts_sn2_identity_chloride.out'))
    assert all(f < 0 for f in q.im_frequency_wn)
    assert all(f > 0 for f in q.frequency_wn)


# ---------------------------------------------------------------------------
# Linear-molecule detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", ['22_hcn_linear_freq_noraman.out',
                                       '40_n2o_linear_highT_highP.out',
                                       '16_o2_superoxide_anion.out',
                                       '46_ts_h3_hydrogen_abstraction.out'])
def test_linear_detection(filename):
    q = parse_qcdata(qchem_path(filename))
    assert q.linear_mol is True


# ---------------------------------------------------------------------------
# Solvation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected", [
    ('08_alanine_C1_pcm_water.out',           'CPCM'),
    ('19_acetic_acid_smd_dmso.out',           'SMD,dmso'),
    ('28_pyridine_smd_acetonitrile_wb97xd.out', 'SMD,acetonitrile'),
    ('30_phenol_smd_thf_pbe0_d3bj.out',       'SMD,thf'),
])
def test_solvation_model(filename, expected):
    q = parse_qcdata(qchem_path(filename))
    assert q.solvation_model == expected


def test_gas_phase_has_empty_solvation():
    q = parse_qcdata(qchem_path('01a_water_hf_freq.out'))
    assert q.solvation_model == ''


# ---------------------------------------------------------------------------
# Geometry extraction
# ---------------------------------------------------------------------------

def test_atoms_parsed():
    q = parse_qcdata(qchem_path('01a_water_hf_freq.out'))
    assert q.atom_types == ['O', 'H', 'H']
    assert q.atom_nums == [8, 1, 1]
    assert len(q.cartesians) == 3


def test_molecular_mass_parsed():
    q = parse_qcdata(qchem_path('01a_water_hf_freq.out'))
    assert abs(q.molecular_mass - 18.010570) < 1e-3


# ---------------------------------------------------------------------------
# Job type detection
# ---------------------------------------------------------------------------

def test_job_type_ts():
    q = parse_qcdata(qchem_path('44_ts_sn2_identity_chloride.out'))
    assert q.job_type == 'TS'


def test_job_type_sp_only():
    q = parse_qcdata(qchem_path('11_hf_molecule_ccsdt_gold_standard.out'))
    assert q.job_type == 'SP'


# ---------------------------------------------------------------------------
# read_initial: level of theory + solvation + termination status
# ---------------------------------------------------------------------------

def test_read_initial_water_normal():
    lot, solv, prog, _, _ = read_initial(qchem_path('01a_water_hf_freq.out'))
    assert 'hf' in lot.lower()
    assert solv == 'gas phase'
    assert prog == 'Normal'


def test_read_initial_alanine_solvated():
    lot, solv, prog, _, _ = read_initial(qchem_path('08_alanine_C1_pcm_water.out'))
    assert 'm06-2x' in lot.lower()
    assert 'CPCM' in solv
    assert prog == 'Normal'


def test_read_initial_smd():
    lot, solv, prog, _, _ = read_initial(qchem_path('19_acetic_acid_smd_dmso.out'))
    assert solv.startswith('SMD')
    assert 'dmso' in solv
    assert prog == 'Normal'


# ---------------------------------------------------------------------------
# Error fixtures: parser doesn't crash, termination is flagged
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", QCHEM_ERROR_FILES)
def test_error_files_dont_crash(filename):
    """All 8 error fixtures must parse without raising."""
    q = parse_qcdata(qchem_path(filename))
    assert q.program == 'QChem'


@pytest.mark.parametrize("filename", [
    '51_err_scf_convergence_fe_complex.out',
    '53_err_wrong_charge_multiplicity.out',
    '54_err_missing_basis_heavy_atom.out',
    '55_err_insufficient_memory.out',
    '57_err_syntax_route_typo.out',
    '60_err_missing_end_blank_line.out',
])
def test_failed_jobs_have_no_scf_energy(filename):
    """Jobs that died before producing any SCF should report scf_energy=None."""
    q = parse_qcdata(qchem_path(filename))
    assert q.scf_energy is None


# ---------------------------------------------------------------------------
# parse_qchem_thermo direct call (skip the dispatcher)
# ---------------------------------------------------------------------------

def test_parse_qchem_thermo_direct():
    q = parse_qchem_thermo(qchem_path('01a_water_hf_freq.out'))
    assert q.program == 'QChem'
    assert q.scf_energy is not None
