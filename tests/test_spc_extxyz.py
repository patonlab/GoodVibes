"""An ASE extxyz file takes its --spc single point from an extxyz twin.

Regression: find_spc_file tried only ``name_SUFFIX.log`` and
``name_SUFFIX.out``, so ``water.extxyz --spc SP`` never found
``water_SP.extxyz``: the CLI exited with "No SPC calculation file" and the
API warned and kept the frequency-level energy.
"""
import shutil
import warnings

import pytest

from conftest import ase_path
from goodvibes.api import compute_thermo
from goodvibes.io import find_spc_file

from test_cli_errors import gv_logger_cleanup, run_main  # noqa: F401  (fixture re-export)

E_FREQ = -76.0105109111   # scf_energy of 01_water.extxyz
E_SP = -76.0405109111     # the twin's: 0.03 Eh lower


@pytest.fixture
def water_with_extxyz_spc(tmp_path):
    shutil.copy(ase_path('01_water.extxyz'), tmp_path / 'water.extxyz')
    text = (tmp_path / 'water.extxyz').read_text()
    assert f'scf_energy={E_FREQ}' in text
    (tmp_path / 'water_SP.extxyz').write_text(text.replace(f'scf_energy={E_FREQ}', f'scf_energy={E_SP}'))
    return tmp_path


def test_find_spc_file_finds_an_extxyz_twin(water_with_extxyz_spc):
    stem = str(water_with_extxyz_spc / 'water')
    assert find_spc_file(stem, 'SP') == stem + '_SP.extxyz'
    assert find_spc_file(stem, 'TZ') is None


def test_extxyz_twin_supplies_the_single_point_energy(water_with_extxyz_spc):
    path = str(water_with_extxyz_spc / 'water.extxyz')
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        r = compute_thermo(path, spc='SP', strict_spc=True, freq_scale_factor=1.0, zpe_scale_factor=1.0)
    assert r.spc_applied is True
    assert r.sp_energy == pytest.approx(E_SP, abs=1e-10)
    plain = compute_thermo(path, freq_scale_factor=1.0, zpe_scale_factor=1.0)
    assert r.enthalpy - plain.enthalpy == pytest.approx(E_SP - E_FREQ, abs=1e-10)


def test_cli_pairs_extxyz_twins(water_with_extxyz_spc, monkeypatch, gv_logger_cleanup):  # noqa: F811
    # The glob also matches the twin; --spc must skip it as an input and use it as the single point.
    run_main(monkeypatch, water_with_extxyz_spc, ['*.extxyz', '--spc', 'SP', '-v', '1.0'], gv_logger_cleanup)
    dat = (water_with_extxyz_spc / 'GoodVibes_output.dat').read_text()
    rows = [line.split() for line in dat.splitlines() if line.startswith('o ')]
    structures = [r for r in rows if r[1] != 'Requested:']
    assert [r[1] for r in structures] == ['water']
    assert float(structures[0][2]) == pytest.approx(E_SP, abs=1e-6)     # the E_SPC column
