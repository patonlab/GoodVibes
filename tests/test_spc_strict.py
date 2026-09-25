"""A requested single-point correction that cannot be applied is reported.

Regression: `spc_correction = self.sp_energy - self.scf_energy` sat inside
`except TypeError: pass`, so a missing or unparseable --spc partner (or a
--spc link job that is not in the output) silently left H and G at the
frequency-level energy. The CLI pre-checks for missing partner files, but
the library API and unparseable files reached the silent path.
"""
import os
import shutil
import warnings

import pytest

from conftest import datapath, g16path
from goodvibes.api import compute_thermo
from goodvibes.thermo import MissingSinglePointError, ThermoOptions, calc_bbe

from test_cli_errors import gv_logger_cleanup, run_main  # noqa: F401  (fixture re-export)

ETHANE = datapath('ethane.out')          # has ethane_TZ.out next to it
WATER = g16path('01a_water_hf_freq.log')  # no link job


@pytest.fixture
def ethane_with_garbage_spc(tmp_path):
    shutil.copy(ETHANE, tmp_path / 'ethane.out')
    (tmp_path / 'ethane_TZ.out').write_text('not a quantum chemistry output\n')
    return str(tmp_path / 'ethane.out')


@pytest.fixture
def ethane_without_spc(tmp_path):
    shutil.copy(ETHANE, tmp_path / 'ethane.out')
    return str(tmp_path / 'ethane.out')


def _quiet(path, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return compute_thermo(path, freq_scale_factor=1.0, zpe_scale_factor=1.0, **kw)


def test_valid_spc_is_applied_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        r = compute_thermo(ETHANE, spc='TZ', freq_scale_factor=1.0, zpe_scale_factor=1.0)
    assert r.spc_applied is True
    assert r.sp_energy is not None
    plain = _quiet(ETHANE)
    assert plain.spc_applied is False
    assert r.enthalpy - plain.enthalpy == pytest.approx(r.sp_energy - r.scf_energy, abs=1e-12)


@pytest.mark.parametrize('fixture', ['ethane_with_garbage_spc', 'ethane_without_spc'])
def test_unusable_spc_warns_and_keeps_frequency_level_energy(fixture, request):
    path = request.getfixturevalue(fixture)
    with pytest.warns(RuntimeWarning, match="suffix 'TZ'"):
        r = compute_thermo(path, spc='TZ', freq_scale_factor=1.0, zpe_scale_factor=1.0)
    assert r.spc_applied is False and r.sp_energy is None
    assert r.enthalpy == pytest.approx(_quiet(path).enthalpy, abs=1e-12)
    assert 'TZ' in r.bbe.spc_reason


def test_strict_spc_raises(ethane_with_garbage_spc):
    with pytest.raises(MissingSinglePointError, match="suffix 'TZ'"):
        compute_thermo(ethane_with_garbage_spc, spc='TZ', strict_spc=True,
                       freq_scale_factor=1.0, zpe_scale_factor=1.0)
    opts = ThermoOptions(spc='TZ', strict_spc=True, freq_scale_factor=1.0, zpe_scale_factor=1.0)
    with pytest.raises(MissingSinglePointError):
        calc_bbe.from_options(ethane_with_garbage_spc, opts)


def test_link_without_link_job_is_reported():
    with pytest.warns(RuntimeWarning, match="link job"):
        r = compute_thermo(WATER, spc='link', freq_scale_factor=1.0, zpe_scale_factor=1.0)
    assert r.spc_applied is False
    with pytest.raises(MissingSinglePointError, match="link job"):
        compute_thermo(WATER, spc='link', strict_spc=True, freq_scale_factor=1.0, zpe_scale_factor=1.0)


def test_cli_warns_in_dat_and_strict_flag_exits(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    # The CLI already refuses missing or unterminated partner files up front;
    # --spc link on an output without a link job is the case that used to pass silently.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        run_main(monkeypatch, tmp_path, [WATER, '--spc', 'link'])
    text = (tmp_path / 'GoodVibes_output.dat').read_text()
    assert 'Warning: 01a_water_hf_freq:' in text and '--strict-spc' in text
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, tmp_path, [WATER, '--spc', 'link', '--strict-spc', '--output', 'strict'])
    assert exc.value.code == 1
    assert os.path.exists(tmp_path / 'GoodVibes_strict.dat')
    assert 'FATAL ERROR (--strict-spc)' in (tmp_path / 'GoodVibes_strict.dat').read_text()
