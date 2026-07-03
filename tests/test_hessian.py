# -*- coding: utf-8 -*-
"""Tests for io.parse_hessian (Cartesian Hessian + per-atom mass extraction).

The tests are self-validating: the parsed Hessian is mass-weighted and
diagonalized, and the resulting harmonic frequencies are compared against the
frequencies the QC program itself printed (Gaussian .log / ORCA .hess).
Program frequencies have translations/rotations projected out while the raw
Hessian eigenvalues do not, so vibrational modes match only to within a few
cm-1 -- ample to catch any unit, ordering, or symmetry mistake, which would
produce garbage.
"""

import numpy as np
import pytest

from goodvibes.io import parse_hessian, parse_qcdata
from conftest import g16path, orca_path

# Hartree/(Bohr^2 amu) -> (angular frequency)^2 in (cm-1)^2
ENERGY_AU = 4.35974434e-18
BOHR_RADIUS = 5.2917721092e-11
AMU_TO_KG = 1.660538921e-27
SPEED_OF_LIGHT_CM = 2.99792458e10
FORCE_TO_WAVENUMBER2 = (ENERGY_AU / (BOHR_RADIUS ** 2 * AMU_TO_KG)
                        / (SPEED_OF_LIGHT_CM * 2 * np.pi) ** 2)


def hessian_frequencies(hd):
    """Unprojected harmonic frequencies (cm-1) from a HessianData."""
    m = np.repeat(hd.masses, 3)
    mw = hd.hessian / np.sqrt(np.outer(m, m))
    eigs = np.linalg.eigvalsh(mw * FORCE_TO_WAVENUMBER2)
    return np.sign(eigs) * np.sqrt(np.abs(eigs))


@pytest.mark.parametrize("path, natoms", [
    (g16path('01a_water_hf_freq.log'), 3),
    (g16path('01c_water_hf_freq_isotopes.log'), 3),
])
def test_gaussian_hessian_shape_and_symmetry(path, natoms):
    hd = parse_hessian(path)
    assert hd.program == 'Gaussian'
    assert hd.hessian.shape == (3 * natoms, 3 * natoms)
    assert np.allclose(hd.hessian, hd.hessian.T)
    assert len(hd.masses) == natoms


def test_gaussian_hessian_frequencies_match_log():
    path = g16path('01a_water_hf_freq.log')
    freqs = hessian_frequencies(parse_hessian(path))
    printed = sorted(parse_qcdata(path).frequency_wn)
    # highest 3N-6 eigenmodes vs Gaussian's projected frequencies
    assert np.allclose(sorted(freqs)[-len(printed):], printed, atol=5.0)


def test_gaussian_hessian_isotope_masses():
    # 01c is D2O: per-atom masses must reflect the iso= substitution
    hd = parse_hessian(g16path('01c_water_hf_freq_isotopes.log'))
    assert hd.masses[0] == pytest.approx(15.99491, abs=1e-4)
    assert hd.masses[1] == pytest.approx(2.01410, abs=1e-4)
    assert hd.masses[2] == pytest.approx(2.01410, abs=1e-4)


def test_gaussian_isotope_frequencies_differ():
    # D2O stretches must come out far below H2O's from the same force field
    h2o = hessian_frequencies(parse_hessian(g16path('01a_water_hf_freq.log')))
    d2o = hessian_frequencies(parse_hessian(g16path('01c_water_hf_freq_isotopes.log')))
    assert max(d2o) < 0.78 * max(h2o)  # ~1/sqrt(2) shift for O-D stretch


def test_orca_hess_file():
    hd = parse_hessian(orca_path('ts_sn2.hess'))
    assert hd.program == 'Orca'
    assert hd.hessian.shape == (18, 18)
    assert np.allclose(hd.hessian, hd.hessian.T)
    # $atoms: C, Cl, 3H, F with standard masses
    assert hd.masses == pytest.approx([12.011, 35.453, 1.008, 1.008, 1.008, 18.998], abs=1e-3)


def test_orca_hess_frequencies_match_stored():
    path = orca_path('ts_sn2.hess')
    freqs = hessian_frequencies(parse_hessian(path))
    # reference frequencies stored in the same file
    with open(path) as f:
        lines = f.readlines()
    i = next(k for k, line in enumerate(lines)
             if line.strip() == '$vibrational_frequencies')
    n = int(lines[i + 1])
    stored = [float(lines[i + 2 + k].split()[1]) for k in range(n)]
    stored_vib = sorted(v for v in stored if abs(v) > 1e-6)
    computed = sorted(freqs)
    # TS: one imaginary mode, present in both
    assert stored_vib[0] < -300 and computed[0] < -300
    vib = [computed[0]] + computed[-(len(stored_vib) - 1):]
    assert np.allclose(vib, stored_vib, atol=5.0)


def test_orca_out_requires_hess_file(tmp_path):
    # An ORCA .out without its companion .hess must raise FileNotFoundError
    out = tmp_path / "job.out"
    out.write_text(" * O   R   C   A *\n")
    with pytest.raises(FileNotFoundError, match=r"\.hess"):
        parse_hessian(str(out))


def test_unsupported_program_raises(tmp_path):
    out = tmp_path / "job.out"
    out.write_text("Welcome to Q-Chem\n")
    with pytest.raises(ValueError, match="not supported"):
        parse_hessian(str(out))


def test_gaussian_without_archive_raises(tmp_path):
    out = tmp_path / "job.log"
    out.write_text("Gaussian 16\nNAtoms= 3\n")
    with pytest.raises(ValueError, match="NImag"):
        parse_hessian(str(out))


def test_qcdata_per_atom_masses_gaussian():
    # parse_qcdata now exposes the per-atom masses (isotope-aware)
    qc = parse_qcdata(g16path('01c_water_hf_freq_isotopes.log'))
    assert qc.per_atom_masses == pytest.approx([15.99491, 2.01410, 2.01410], abs=1e-4)
