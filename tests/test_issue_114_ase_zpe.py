"""Regression tests for issue #114 (ASE extxyz ingest and the ZPE gate).

1. Omitting the optional ``zpe`` key must not silently disable thermochemistry.
2. A supplied ``zpe`` is metadata only; the reported ZPE is recomputed from
   the frequencies (documented behaviour, now true whether or not the key is
   present).
3. ``zpe=0.0`` must not be misread as "monatomic" and drop the rotational
   terms.
4. Frequencies parsed but a prerequisite missing is reported, not silent.
"""
import math

import pytest

from conftest import ase_path, g16path
from goodvibes.api import compute_thermo
from goodvibes.constants import J_TO_AU
from goodvibes.io import parse_qcdata
from goodvibes.thermo import ThermoOptions, calc_bbe, calc_zeropoint_energy

ase = pytest.importorskip('ase')
from ase import Atoms  # noqa: E402
from goodvibes.ase_helper import write_thermo_extxyz  # noqa: E402

FREQS = [1595.0, 3657.0, 3756.0]


def _write(path, **extra):
    h2o = Atoms('OHH', positions=[[0., 0., 0.117], [0., 0.757, -0.469], [0., -0.757, -0.469]])
    write_thermo_extxyz(str(path), h2o, energy=-76.368128, frequencies=FREQS,
                        point_group='C2V', symmno=2, linear_mol=False, **extra)
    return str(path)


def _thermo(path):
    # Explicit scale factors: no level_of_theory in the file, no lookup noise.
    return compute_thermo(path, QH=True, freq_scale_factor=1.0, zpe_scale_factor=1.0)


def test_omitted_zpe_still_computes_thermochemistry(tmp_path):
    r = _thermo(_write(tmp_path / 'h2o.extxyz'))
    assert r.zpe is not None and r.enthalpy is not None and r.qh_gibbs_free_energy is not None
    expected = calc_zeropoint_energy(FREQS) / J_TO_AU
    assert r.zpe == pytest.approx(expected, abs=1e-12)
    assert parse_qcdata(str(tmp_path / 'h2o.extxyz')).zero_point_corr == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize('supplied', [0.0207766, 0.0, 99.0])
def test_supplied_zpe_is_metadata_only_and_never_changes_the_answer(tmp_path, supplied):
    ref = _thermo(_write(tmp_path / 'ref.extxyz'))
    got = _thermo(_write(tmp_path / 'z.extxyz', zpe=supplied))
    assert got.zpe == pytest.approx(ref.zpe, abs=1e-12)
    assert got.enthalpy == pytest.approx(ref.enthalpy, abs=1e-12)
    assert got.gibbs_free_energy == pytest.approx(ref.gibbs_free_energy, abs=1e-12)
    assert got.qh_gibbs_free_energy == pytest.approx(ref.qh_gibbs_free_energy, abs=1e-12)
    assert parse_qcdata(str(tmp_path / 'z.extxyz')).zero_point_corr == supplied


def test_zpe_zero_keeps_rotational_energy(tmp_path):
    # Non-linear triatomic: U_rot = 3/2 RT must be present in H.
    r = _thermo(_write(tmp_path / 'z0.extxyz', zpe=0.0))
    q = parse_qcdata(str(tmp_path / 'z0.extxyz'))
    assert q.rotemp and not q.linear_mol
    # Rotational entropy for a non-linear molecule is > 0; if it had been
    # treated as monatomic entropy would drop by several cal/mol/K.
    r_no_zpe = _thermo(_write(tmp_path / 'ref.extxyz'))
    assert r.entropy == pytest.approx(r_no_zpe.entropy, abs=1e-12)
    assert not math.isclose(r.gibbs_free_energy, r_no_zpe.gibbs_free_energy + 2.23 / 627.509541, abs_tol=1e-4)


def test_gaussian_atom_still_has_no_rotational_terms():
    # Neon: frequencies empty, Gaussian prints ZPE 0.0. Must keep U_rot = S_rot = 0.
    r = compute_thermo(g16path('07_neon_atom_with_freq.log'), freq_scale_factor=1.0, zpe_scale_factor=1.0)
    q = parse_qcdata(g16path('07_neon_atom_with_freq.log'))
    assert q.frequency_wn == [] and r.zpe == 0.0
    # H = E + U_trans + RT = E + 5/2 RT for an atom; no 3/2 RT rotational term.
    R, T = 8.3144621, 298.15
    assert r.enthalpy - r.scf_energy == pytest.approx(2.5 * R * T / J_TO_AU, rel=1e-9)


def test_missing_prerequisite_with_frequencies_warns():
    q = parse_qcdata(ase_path('01_water.extxyz'))
    q.scf_energy = None
    opts = ThermoOptions(freq_scale_factor=1.0, zpe_scale_factor=1.0)
    with pytest.warns(RuntimeWarning, match=r'3 frequencies parsed .* scf_energy'):
        bbe = calc_bbe.from_options(q, opts)
    # calc_bbe leaves the attributes unset (output.py keys "computed?" off
    # hasattr); the public ThermoResult surfaces them as None.
    assert not hasattr(bbe, 'gibbs_free_energy')
    with pytest.warns(RuntimeWarning):
        r = compute_thermo(qcdata=q, freq_scale_factor=1.0, zpe_scale_factor=1.0)
    assert r.enthalpy is None and r.gibbs_free_energy is None


def test_single_point_stays_silent():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        bbe = compute_thermo(g16path('20_benzene_singlepoint.log'), freq_scale_factor=1.0, zpe_scale_factor=1.0)
    assert bbe.enthalpy is None
