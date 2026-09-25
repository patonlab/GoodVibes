"""Two ASE-path regressions that every MLIP user hits first.

1. invert='auto' keeps the most negative mode of a transition state only when
   the job type says 'TSFreq', but the extxyz parser (and Q-Chem) wrote 'TS',
   so the reaction coordinate of an ASE TS was inverted to a real mode.
2. write_thermo_extxyz did float(f) on ASE's complex frequency array, which
   discards the imaginary part: an imaginary mode became 0.0 cm⁻¹.
"""
import numpy as np
import pytest

from conftest import ase_path, qchem_path
from goodvibes.api import compute_thermo
from goodvibes.io import parse_qcdata
from goodvibes.thermo import _apply_frequency_inversion

SN2 = ase_path('44_ts_sn2.extxyz')


def test_ase_ts_job_type_is_tsfreq():
    q = parse_qcdata(SN2)
    assert q.job_type == 'TSFreq'
    assert len(q.im_frequency_wn) == 1 and q.im_frequency_wn[0] < -50


def test_qchem_ts_job_type_is_tsfreq():
    assert parse_qcdata(qchem_path('44_ts_sn2_identity_chloride.out')).job_type == 'TSFreq'


def test_auto_inversion_keeps_the_reaction_coordinate_for_ase_ts():
    r = compute_thermo(SN2, invert='auto', freq_scale_factor=1.0, zpe_scale_factor=1.0)
    assert len(r.im_frequency_wn or []) == 1
    assert not r.inverted_freqs


@pytest.mark.parametrize('job_type', ['TS', 'TSFreq'])
def test_auto_inversion_rule_accepts_both_spellings(job_type):
    freqs, imags, inverted = _apply_frequency_inversion([100.0, 200.0], [-345.0, -80.0], 'auto', job_type)
    assert imags == [-345.0]          # reaction coordinate kept
    assert inverted == [-80.0]        # the spurious small imaginary mode is inverted to +80
    assert 80.0 in freqs and 345.0 not in freqs


def test_write_thermo_extxyz_keeps_imaginary_modes(tmp_path):
    ase = pytest.importorskip('ase')
    from goodvibes.ase_helper import write_thermo_extxyz
    atoms = ase.Atoms('OHH', positions=[[0., 0., 0.117], [0., 0.757, -0.469], [0., -0.757, -0.469]])
    complex_freqs = np.array([0 + 345.2j, 1595.0 + 0j, 3657.0 + 0j])
    out = tmp_path / 'ts.extxyz'
    write_thermo_extxyz(str(out), atoms, energy=-76.0, frequencies=complex_freqs)
    q = parse_qcdata(str(out))
    assert q.im_frequency_wn == pytest.approx([-345.2])
    assert q.frequency_wn == pytest.approx([1595.0, 3657.0])
    assert q.job_type == 'TSFreq'
    # plain floats and explicitly negative values are untouched
    write_thermo_extxyz(str(out), atoms, energy=-76.0, frequencies=[-345.2, 1595.0, 3657.0])
    assert parse_qcdata(str(out)).im_frequency_wn == pytest.approx([-345.2])
