#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for extracted modules: utils, validation."""

from datetime import datetime

import pytest

import logging
import numpy as np
from goodvibes.utils import all_same, add_time, display_name, setup_logging, fatal
from goodvibes.sort import deduplicate


# ===========================================================================
# all_same
# ===========================================================================

def test_all_same_identical():
    assert all_same([1, 1, 1]) is True


def test_all_same_different():
    assert all_same([1, 2, 1]) is False


def test_all_same_single():
    assert all_same([42]) is True


def test_all_same_empty():
    """Empty list: all() returns True vacuously."""
    assert all_same([]) is True


def test_all_same_strings():
    assert all_same(['a', 'a', 'a']) is True
    assert all_same(['a', 'b']) is False


# ===========================================================================
# display_name
# ===========================================================================

def test_display_name_full_path():
    assert display_name('/path/to/ethane.log') == 'ethane'


def test_display_name_out_extension():
    assert display_name('file.out') == 'file'


def test_display_name_no_directory():
    assert display_name('water.log') == 'water'


def test_display_name_nested_path():
    assert display_name('/a/b/c/methylaniline.out') == 'methylaniline'


def test_display_name_no_extension():
    assert display_name('noext') == 'noext'


# ===========================================================================
# add_time
# ===========================================================================

def test_add_time_basic():
    """Adding known CPU time to a datetime."""
    tm = datetime(2020, 1, 5, 10, 30, 0, 0)
    cpu = [0, 1, 30, 45, 500]  # 0 days, 1 hr, 30 min, 45 sec, 500 msec
    result = add_time(tm, cpu)
    assert result.hour == 12
    assert result.minute == 0
    assert result.second == 45


def test_add_time_zero_cpu():
    """Zero CPU time returns same time-of-day."""
    tm = datetime(2020, 1, 3, 8, 0, 0, 0)
    cpu = [0, 0, 0, 0, 0]
    result = add_time(tm, cpu)
    assert result.hour == 8
    assert result.minute == 0
    assert result.second == 0


def test_add_time_with_days():
    """CPU time spanning multiple days."""
    tm = datetime(2020, 1, 1, 0, 0, 0, 0)
    cpu = [2, 5, 30, 0, 0]  # 2 days, 5 hrs, 30 mins
    result = add_time(tm, cpu)
    assert result.day == 3
    assert result.hour == 5
    assert result.minute == 30


# ===========================================================================
# setup_logging / fatal
# ===========================================================================

def test_setup_logging_creates_dat(tmp_path):
    """setup_logging creates a .dat file and messages appear in it."""
    logger = logging.getLogger('goodvibes')
    logger.handlers.clear()
    setup_logging(str(tmp_path / 'test'), 'output')
    logger.info("hello world")
    logging.shutdown()
    logger.handlers.clear()
    dat_file = tmp_path / 'test_output.dat'
    assert dat_file.exists()
    assert dat_file.read_text() == "hello world"


def test_fatal_exits(tmp_path):
    """fatal() logs message and calls sys.exit."""
    logger = logging.getLogger('goodvibes')
    logger.handlers.clear()
    setup_logging(str(tmp_path / 'test'), 'output')
    with pytest.raises(SystemExit):
        fatal("fatal error")
    logger.handlers.clear()


# ===========================================================================
# sort.deduplicate
# ===========================================================================

class MockBBE:
    """Mock object for GoodVibes BBE (energies, frequencies, etc.) for testing."""
    def __init__(self, scf_energy=None, roconst=None, frequency_wn=None, cartesians=None):
        if scf_energy is not None:
            self.scf_energy = scf_energy
        if roconst is not None:
            self.roconst = roconst
        if frequency_wn is not None:
            self.frequency_wn = frequency_wn
        if cartesians is not None:
            self.cartesians = cartesians

def test_deduplicate_identical_structures():
    """Two structures with identical properties should be flagged as duplicates."""
    bbe1 = MockBBE(scf_energy=-100.0, roconst=[10.0, 20.0, 30.0])
    bbe2 = MockBBE(scf_energy=-100.0, roconst=[10.0, 20.0, 30.0])
    thermo_data = {'file1.log': bbe1, 'file2.log': bbe2}

    dups = deduplicate(thermo_data)
    assert len(dups) == 1
    assert dups[0] == ['file2.log', 'file1.log']


def test_deduplicate_different_structures():
    """Two structures with very different energies are not duplicates."""
    bbe1 = MockBBE(scf_energy=-100.0, roconst=[10.0, 20.0, 30.0])
    bbe2 = MockBBE(scf_energy=-99.0, roconst=[10.0, 20.0, 30.0])
    thermo_data = {'file1.log': bbe1, 'file2.log': bbe2}

    dups = deduplicate(thermo_data)
    assert len(dups) == 0


def test_deduplicate_no_files():
    """Empty file list returns no duplicates."""
    assert deduplicate({}) == []


def test_deduplicate_abs_energy_diff():
    """Energy comparison uses absolute value — order should not matter."""
    # file1 has LOWER energy than file2 (1.0 Hartree difference)
    bbe1 = MockBBE(scf_energy=-100.0, roconst=[10.0, 20.0, 30.0])
    bbe2 = MockBBE(scf_energy=-99.0, roconst=[10.0, 20.0, 30.0])

    # Both orderings should reject (1.0 Ha = 627.5 kcal/mol >> 0.05 kcal/mol cutoff)
    assert deduplicate({'file1.log': bbe1, 'file2.log': bbe2}) == []
    assert deduplicate({'file2.log': bbe2, 'file1.log': bbe1}) == []


def test_deduplicate_custom_energy_cutoff():
    """Custom energy cutoff allows larger differences to pass."""
    bbe1 = MockBBE(scf_energy=-100.0, roconst=[10.0, 20.0, 30.0])
    bbe2 = MockBBE(scf_energy=-100.001, roconst=[10.0, 20.0, 30.0])
    thermo_data = {'file1.log': bbe1, 'file2.log': bbe2}

    # Default cutoff (0.05 kcal/mol) should NOT flag (0.001 Ha ~ 0.63 kcal/mol > 0.05)
    assert deduplicate(thermo_data) == []
    # Relaxed cutoff (1.0 kcal/mol) should flag them
    assert len(deduplicate(thermo_data, e_cutoff=1.0)) == 1


def test_deduplicate_custom_roconst_cutoff():
    """Custom rotational constant cutoff."""
    bbe1 = MockBBE(scf_energy=-100.0, roconst=[10.0, 20.0, 30.0])
    bbe2 = MockBBE(scf_energy=-100.0, roconst=[10.2, 20.0, 30.0])
    thermo_data = {'file1.log': bbe1, 'file2.log': bbe2}

    # Relative diff on first constant = 0.2/10.1 ~ 2% > 1% default, so rejected
    assert deduplicate(thermo_data) == []
    # Relaxed cutoff (5%) should flag them
    assert len(deduplicate(thermo_data, ro_cutoff=0.05)) == 1


def test_deduplicate_custom_rmsd_cutoff():
    """RMSD comparison with custom threshold."""
    coords1 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    coords2 = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.0, 0.0]]  # RMSD ~ 0.167
    bbe1 = MockBBE(scf_energy=-100.0, roconst=[10.0, 20.0, 30.0], cartesians=coords1)
    bbe2 = MockBBE(scf_energy=-100.0, roconst=[10.0, 20.0, 30.0], cartesians=coords2)
    thermo_data = {'file1.log': bbe1, 'file2.log': bbe2}

    # Tight cutoff (0.125 Å) should reject (aligned RMSD ~ 0.129 > 0.125)
    assert deduplicate(thermo_data, rmsd_cutoff=0.125) == []
    # Relaxed cutoff should flag them
    assert len(deduplicate(thermo_data, rmsd_cutoff=0.5)) == 1


def test_deduplicate_rmsd_alignment():
    """Kabsch alignment handles translated and rotated identical structures."""
    coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    # Same structure translated by [5, 5, 5]
    coords_translated = [[5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [5.0, 6.0, 5.0]]
    # Same structure rotated 90° around z-axis
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    coords_rotated = (np.array(coords) @ R.T).tolist()

    bbe_orig = MockBBE(scf_energy=-100.0, roconst=[10.0, 20.0, 30.0], cartesians=coords)
    bbe_trans = MockBBE(scf_energy=-100.0, roconst=[10.0, 20.0, 30.0], cartesians=coords_translated)
    bbe_rot = MockBBE(scf_energy=-100.0, roconst=[10.0, 20.0, 30.0], cartesians=coords_rotated)

    # Translated copy should be flagged as duplicate (RMSD ~ 0 after alignment)
    assert len(deduplicate({'a.log': bbe_orig, 'b.log': bbe_trans}, rmsd_cutoff=0.125)) == 1
    # Rotated copy should be flagged as duplicate (RMSD ~ 0 after alignment)
    assert len(deduplicate({'a.log': bbe_orig, 'b.log': bbe_rot}, rmsd_cutoff=0.125)) == 1


def test_deduplicate_tighter_cutoff_rejects():
    """Tighter cutoffs can reject pairs that would pass defaults."""
    bbe1 = MockBBE(scf_energy=-100.0, roconst=[10.0, 20.0, 30.0])
    bbe2 = MockBBE(scf_energy=-100.00005, roconst=[10.0, 20.0, 30.0])
    thermo_data = {'file1.log': bbe1, 'file2.log': bbe2}

    # Default cutoffs: should be flagged
    assert len(deduplicate(thermo_data)) == 1
    # Very tight energy cutoff rejects them
    assert len(deduplicate(thermo_data, e_cutoff=1e-6)) == 0


def test_deduplicate_missing_attributes_not_flagged():
    """Structures missing attributes should not be flagged as duplicates."""
    thermo_data = {'file1.log': MockBBE(), 'file2.log': MockBBE()}
    assert deduplicate(thermo_data) == []


def test_deduplicate_no_cross_pair_leakage():
    """Values from one pair comparison must not leak to the next pair."""
    coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    # First two are near-duplicates (sets e_diff ~ 0)
    bbe1 = MockBBE(scf_energy=-100.0, roconst=[10.0, 20.0, 30.0], frequency_wn=[500.0, 1000.0, 1500.0], cartesians=coords)
    bbe2 = MockBBE(scf_energy=-100.0, roconst=[10.0, 20.0, 30.0], frequency_wn=[500.0, 1000.0, 1500.0], cartesians=coords)
    # Third has no energy — old code would inherit e_diff from pair (1,0)
    bbe3 = MockBBE(roconst=[10.0, 20.0, 30.0], frequency_wn=[500.0, 1000.0, 1500.0], cartesians=coords)

    thermo_data = {'a.log': bbe1, 'b.log': bbe2, 'c.log': bbe3}
    dups = deduplicate(thermo_data)

    # Only (b, a) should be flagged; c should not match anyone
    assert len(dups) == 1
    assert dups[0] == ['b.log', 'a.log']


def test_deduplicate_single_atoms_zero_roconst():
    """Single atoms with all-zero rotational constants should be flagged as duplicates."""
    # Two single atoms: same energy, all-zero rotational constants
    bbe1 = MockBBE(scf_energy=-242.328708, roconst=[0.0, 0.0, 0.0])
    bbe2 = MockBBE(scf_energy=-242.328708, roconst=[0.0, 0.0, 0.0])
    thermo_data = {'Al_298K.log': bbe1, 'Al_400K.log': bbe2}
    assert len(deduplicate(thermo_data)) == 1
