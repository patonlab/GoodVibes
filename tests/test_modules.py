#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for extracted modules: utils, validation."""

from datetime import datetime

import pytest

from goodvibes.utils import all_same, add_time, display_name, Logger


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
# Logger
# ===========================================================================

def test_logger_dat_file(tmp_path):
    """Logger creates a .dat file and writes to it."""
    log = Logger(str(tmp_path / 'test'), 'output', csv=False)
    log.write("hello world")
    log.finalize()
    dat_file = tmp_path / 'test_output.dat'
    assert dat_file.exists()
    assert dat_file.read_text() == "hello world"


def test_logger_csv_file(tmp_path):
    """Logger creates a .csv file when csv=True."""
    log = Logger(str(tmp_path / 'test'), 'output', csv=True)
    log.write("hello world")
    log.finalize()
    csv_file = tmp_path / 'test_output.csv'
    assert csv_file.exists()


def test_logger_csv_thermodata(tmp_path):
    """Logger comma-separates thermodata in CSV mode."""
    log = Logger(str(tmp_path / 'test'), 'output', csv=True)
    log.write("col1 col2 col3", thermodata=True)
    log.finalize()
    content = (tmp_path / 'test_output.csv').read_text()
    assert 'col1,col2,col3,' in content


def test_logger_non_csv_thermodata(tmp_path):
    """Logger does not comma-separate in non-CSV mode even with thermodata=True."""
    log = Logger(str(tmp_path / 'test'), 'output', csv=False)
    log.write("col1 col2 col3", thermodata=True)
    log.finalize()
    content = (tmp_path / 'test_output.dat').read_text()
    assert content == "col1 col2 col3"


def test_logger_fatal_exits(tmp_path):
    """Logger.fatal() calls sys.exit."""
    log = Logger(str(tmp_path / 'test'), 'output', csv=False)
    with pytest.raises(SystemExit):
        log.fatal("fatal error")


# ===========================================================================
# validation.check_dup
# ===========================================================================

def test_check_dup_identical_structures():
    """Two structures with identical properties should be flagged as duplicates."""
    from goodvibes.validation import check_dup

    class MockBBE:
        def __init__(self, scf, roconst, freqs):
            self.scf_energy = scf
            self.roconst = roconst
            self.frequency_wn = freqs

    bbe1 = MockBBE(-100.0, [10.0, 20.0, 30.0], [500.0, 1000.0, 1500.0])
    bbe2 = MockBBE(-100.0, [10.0, 20.0, 30.0], [500.0, 1000.0, 1500.0])
    files = ['file1.log', 'file2.log']
    thermo_data = {'file1.log': bbe1, 'file2.log': bbe2}

    dups = check_dup(files, thermo_data)
    assert len(dups) == 1
    assert dups[0] == ['file2.log', 'file1.log']


def test_check_dup_different_structures():
    """Two structures with very different energies are not duplicates.
    Note: check_dup uses `e_diff < cutoff` (not abs), so bbe_i must have
    higher energy than bbe_j for the energy check to reject the pair."""
    from goodvibes.validation import check_dup

    class MockBBE:
        def __init__(self, scf, roconst, freqs):
            self.scf_energy = scf
            self.roconst = roconst
            self.frequency_wn = freqs

    bbe1 = MockBBE(-100.0, [10.0, 20.0, 30.0], [500.0, 1000.0, 1500.0])
    bbe2 = MockBBE(-99.0, [10.0, 20.0, 30.0], [500.0, 1000.0, 1500.0])
    files = ['file1.log', 'file2.log']
    thermo_data = {'file1.log': bbe1, 'file2.log': bbe2}

    dups = check_dup(files, thermo_data)
    assert len(dups) == 0


def test_check_dup_different_frequencies():
    """Same energy but very different frequencies are not duplicates."""
    from goodvibes.validation import check_dup

    class MockBBE:
        def __init__(self, scf, roconst, freqs):
            self.scf_energy = scf
            self.roconst = roconst
            self.frequency_wn = freqs

    bbe1 = MockBBE(-100.0, [10.0, 20.0, 30.0], [500.0, 1000.0, 1500.0])
    bbe2 = MockBBE(-100.0, [10.0, 20.0, 30.0], [600.0, 1100.0, 1600.0])
    files = ['file1.log', 'file2.log']
    thermo_data = {'file1.log': bbe1, 'file2.log': bbe2}

    dups = check_dup(files, thermo_data)
    assert len(dups) == 0


def test_check_dup_no_files():
    """Empty file list returns no duplicates."""
    from goodvibes.validation import check_dup
    assert check_dup([], {}) == []
