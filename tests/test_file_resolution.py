"""The path a caller passes must be the file that gets parsed.

Regression for the resolution order in io.py: ``x.log`` used to be tried
before the given ``x.out``, so an ORCA ``x.out`` sitting next to a Gaussian
``x.log`` was silently parsed as the Gaussian file.
"""
import os
import shutil

import pytest

from conftest import g16path, orca_path
from goodvibes.io import parse_data, parse_qcdata, resolve_output_file, sp_cpu

GAUSSIAN = g16path('01a_water_hf_freq.log')
ORCA = orca_path('01a_water_hf_freq.out')


@pytest.fixture
def twins(tmp_path):
    """x.log (Gaussian) and x.out (ORCA) side by side."""
    shutil.copy(GAUSSIAN, tmp_path / 'x.log')
    shutil.copy(ORCA, tmp_path / 'x.out')
    return tmp_path


def test_given_path_wins_over_sibling_log(twins):
    out = str(twins / 'x.out')
    assert resolve_output_file(out) == out
    assert parse_qcdata(out).program == 'Orca'
    assert parse_data(out)[1] == 'Orca'
    assert parse_qcdata(str(twins / 'x.log')).program == 'Gaussian'


def test_extensionless_stem_falls_back_to_log_then_out(twins):
    stem = str(twins / 'x')
    assert resolve_output_file(stem) == stem + '.log'
    os.remove(twins / 'x.log')
    assert resolve_output_file(stem) == stem + '.out'
    assert parse_qcdata(stem).program == 'Orca'


def test_missing_file_resolves_to_none(tmp_path):
    assert resolve_output_file(str(tmp_path / 'nope.out')) is None
    assert parse_qcdata(str(tmp_path / 'nope.out')).program == 'unknown'
    with pytest.raises(ValueError):
        sp_cpu(str(tmp_path / 'nope.out'))


def test_sp_cpu_uses_given_file(twins):
    assert sp_cpu(str(twins / 'x.out')) == parse_qcdata(ORCA).cpu
    assert sp_cpu(str(twins / 'x.log')) == parse_qcdata(GAUSSIAN).cpu
