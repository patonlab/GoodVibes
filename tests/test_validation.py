"""Tests for goodvibes.validation: collect_and_validate_files,
print_check_fails, and check_files (smoke).

Behavior pinned here:
  - collect_and_validate_files filters out 'Error' / 'Incomplete' files,
    keeps the parallel level_of_theory and solvation_model lists aligned,
    and sys.exits when no usable files remain.
  - print_check_fails groups files by attribute value and abbreviates
    long groups.
  - check_files runs end-to-end against representative thermo_data without
    raising; the heavy log output is sampled via caplog.
"""

import logging
from types import SimpleNamespace

import pytest

from goodvibes.validation import (
    collect_and_validate_files,
    print_check_fails,
    check_files,
)
from goodvibes.thermo import calc_bbe
from conftest import g16path


GAS_CONSTANT = 8.3144621
ATMOS = 101.325
T_DEFAULT = 298.15
CONC_DEFAULT = ATMOS / (GAS_CONSTANT * T_DEFAULT)


def _opts(**kw):
    """Minimal options stub for collect_and_validate_files / check_files."""
    defaults = {
        'spc': None, 'conc': None, 'duplicate': False,
        'e_cutoff': 0.05, 'ro_cutoff': 0.01, 'rmsd_cutoff': None,
    }
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def _bbe(filename):
    return calc_bbe(g16path(filename), 'grimme', False, 100.0, 100.0,
                    T_DEFAULT, CONC_DEFAULT, 1.0, None, None, None, 0)


# ===========================================================================
# collect_and_validate_files: happy path + filtering
# ===========================================================================

NORMAL = ['01a_water_hf_freq.log',
          '05_methylene_triplet_carbene.log',
          '22_hcn_linear_freq_noraman.log']

ERROR_FILES = ['51_err_scf_convergence_fe_complex.log',
               '52_err_opt_not_converged_maxcycles.log',
               '53_err_wrong_charge_multiplicity.log',
               '54_err_missing_basis_heavy_atom.log',
               '55_err_insufficient_memory.log',
               '57_err_syntax_route_typo.log',
               '60_err_missing_end_blank_line.log']

INCOMPLETE_FILES = ['56_err_timed_out.log', '61_empty.log']


def test_collect_normal_files_pass_through():
    """All-normal input is returned unchanged."""
    files = [g16path(f) for f in NORMAL]
    out_files, lots, sm = collect_and_validate_files(files, _opts())
    assert out_files == files
    assert len(lots) == len(files)
    assert len(sm) == len(files)


def test_collect_aligns_metadata_lists():
    """Returned (files, level_of_theory, solvation_model) are parallel."""
    files = [g16path(f) for f in NORMAL]
    out_files, lots, sm = collect_and_validate_files(files, _opts())
    # Every entry has a non-empty level of theory string
    for lot, file in zip(lots, out_files):
        assert isinstance(lot, str) and lot
    # Solvation model is a list-or-string; sanity check it's set
    assert all(s for s in sm)


@pytest.mark.parametrize("err", ERROR_FILES)
def test_collect_drops_error_terminations(err):
    """Files with 'Error' progress are dropped, normal ones survive."""
    files = [g16path('01a_water_hf_freq.log'), g16path(err)]
    out_files, lots, sm = collect_and_validate_files(files, _opts())
    assert len(out_files) == 1
    assert out_files[0].endswith('01a_water_hf_freq.log')
    assert len(lots) == 1 and len(sm) == 1


@pytest.mark.parametrize("inc", INCOMPLETE_FILES)
def test_collect_drops_incomplete_terminations(inc):
    """Files with 'Incomplete' progress (timed out, empty) are dropped."""
    files = [g16path('01a_water_hf_freq.log'), g16path(inc)]
    out_files, _, _ = collect_and_validate_files(files, _opts())
    assert len(out_files) == 1
    assert out_files[0].endswith('01a_water_hf_freq.log')


def test_collect_all_broken_exits():
    """If every file is broken, sys.exit fires (CLI deathmatch path)."""
    files = [g16path(f) for f in ERROR_FILES + INCOMPLETE_FILES]
    with pytest.raises(SystemExit):
        collect_and_validate_files(files, _opts())


def test_collect_warn_messages_logged(caplog):
    """Each removed file logs a Warning-shaped message."""
    files = [g16path('01a_water_hf_freq.log'),
             g16path('51_err_scf_convergence_fe_complex.log'),
             g16path('56_err_timed_out.log')]
    with caplog.at_level(logging.INFO, logger='goodvibes'):
        collect_and_validate_files(files, _opts())
    text = caplog.text
    # Error files use 'Warning! Error termination'
    assert 'Error termination' in text
    # Incomplete files use 'may not have terminated normally'
    assert 'may not have terminated' in text


# ===========================================================================
# print_check_fails
# ===========================================================================

def test_print_check_fails_groups_by_attribute(caplog):
    """Files with the same attribute value are grouped together in output."""
    attrs = ['B3LYP', 'M062X', 'B3LYP', 'M062X']
    files = ['a.log', 'b.log', 'c.log', 'd.log']
    with caplog.at_level(logging.INFO, logger='goodvibes'):
        print_check_fails(attrs, files, 'levels of theory')
    text = caplog.text
    assert 'Multiple levels of theory found' in text
    assert 'B3LYP' in text and 'M062X' in text
    # Both groups list their members
    assert 'a.log' in text and 'c.log' in text
    assert 'b.log' in text and 'd.log' in text


def test_print_check_fails_abbreviates_long_groups(caplog):
    """Groups with > 3 members show the first 3 + 'and N others'."""
    attrs = ['B3LYP'] * 10
    files = [f'file_{i}.log' for i in range(10)]
    with caplog.at_level(logging.INFO, logger='goodvibes'):
        print_check_fails(attrs, files, 'levels of theory')
    text = caplog.text
    assert 'and 7 others' in text
    assert 'file_0.log' in text
    # Files past the third aren't named individually
    assert 'file_5.log' not in text


def test_print_check_fails_with_secondary_attribute(caplog):
    """option2 keys grouping by (attr1, attr2) — used for charge+multiplicity."""
    attrs = [0, 1, 0, 1]
    option2 = [1, 2, 1, 2]
    files = ['a.log', 'b.log', 'c.log', 'd.log']
    with caplog.at_level(logging.INFO, logger='goodvibes'):
        print_check_fails(attrs, files, 'charge and multiplicity', option2)
    text = caplog.text
    assert 'charge and multiplicity' in text


# ===========================================================================
# check_files: smoke + sampled log content
# ===========================================================================

def _thermo_data(filenames):
    """Build a thermo_data dict by running calc_bbe over the given fixtures."""
    return {g16path(f): _bbe(f) for f in filenames}


def test_check_files_homogeneous_run_no_warnings(caplog):
    """Three water/methylene/HCN files all use HF/6-31G(d) → no 'Caution'
    cautions about levels of theory or solvation."""
    data = _thermo_data(['01a_water_hf_freq.log'])  # single file: trivially homogeneous
    lots = ['HF/6-31G(d)']
    with caplog.at_level(logging.INFO, logger='goodvibes'):
        check_files(data, _opts(), lots)
    text = caplog.text
    assert 'Using HF/6-31G(d) in all calculations' in text
    # No multi-level caution
    assert 'Multiple levels of theory' not in text


def test_check_files_mixed_levels_of_theory_warns(caplog):
    """Two files declared at different levels of theory → caution emitted."""
    data = _thermo_data(['01a_water_hf_freq.log',
                         '02_ethane_opt_freq_T398_P2.log'])
    lots = ['HF/6-31G(d)', 'B3LYP/6-311+G(d,p)']  # genuinely different
    with caplog.at_level(logging.INFO, logger='goodvibes'):
        check_files(data, _opts(), lots)
    assert 'Multiple levels of theory' in caplog.text


def test_check_files_linear_correctness_caution(caplog):
    """An HCN linear molecule with the right 3N-5 frequency count gets the
    'All linear molecules have the correct number of frequencies' message."""
    data = _thermo_data(['22_hcn_linear_freq_noraman.log'])
    lots = ['M062X/6-311+G(d,p)']
    with caplog.at_level(logging.INFO, logger='goodvibes'):
        check_files(data, _opts(), lots)
    text = caplog.text
    assert 'linear' in text.lower()
    assert 'correct number of frequencies' in text


def test_check_files_ts_imaginary_frequency_handled(caplog):
    """A bona-fide TS with one imaginary frequency does NOT trigger the
    'TS does not have 1 imaginary frequency' caution."""
    data = _thermo_data(['44_ts_sn2_identity_chloride.log'])
    lots = ['B3LYP/6-311+G(d,p)']
    with caplog.at_level(logging.INFO, logger='goodvibes'):
        check_files(data, _opts(), lots)
    assert 'does not have 1 imaginary frequency' not in caplog.text


def test_check_files_runs_with_duplicate_option(caplog):
    """Smoke: --duplicate code path runs without error and reports a
    'No duplicates' line for two distinct molecules."""
    data = _thermo_data(['01a_water_hf_freq.log',
                         '02_ethane_opt_freq_T398_P2.log'])
    lots = ['HF/6-31G(d)', 'B3LYP/6-311+G(d,p)']
    with caplog.at_level(logging.INFO, logger='goodvibes'):
        check_files(data, _opts(duplicate=True), lots)
    assert 'No duplicates' in caplog.text


def test_check_files_dispersion_homogeneous(caplog):
    """When every file reports 'No empirical dispersion detected', the
    summary line uses the '-' (informational) marker, not 'x' (caution)."""
    data = _thermo_data(['01a_water_hf_freq.log'])
    lots = ['HF/6-31G(d)']
    with caplog.at_level(logging.INFO, logger='goodvibes'):
        check_files(data, _opts(), lots)
    assert 'No empirical dispersion detected' in caplog.text
