"""QCData.with_single_point: a single-point energy attached in memory, for
composites such as DFT//MLIP that have no single-point output file."""
import warnings

import pytest

from conftest import g16path
from goodvibes import compute_thermo
from goodvibes.constants import HARTREE_TO_EV
from goodvibes.io import SP_ATTACHED, dict_to_qcdata, parse_qcdata, qcdata_to_dict
from goodvibes.pes_model import ComputedEntry

WATER = g16path("01a_water_hf_freq.log")
SP = -76.330                                  # hartree; the file's HF energy is about -76.0107


@pytest.fixture
def water():
    return parse_qcdata(WATER)


def _shift(result, base):
    return result.qh_gibbs_free_energy - base.qh_gibbs_free_energy


def test_the_attached_energy_replaces_the_electronic_energy(water):
    base = compute_thermo(qcdata=water)
    qc = water.with_single_point(SP, "hartree", "DLPNO-CCSD(T)/def2-TZVP")
    with warnings.catch_warnings():
        warnings.simplefilter("error")        # no missing-single-point warning
        r = compute_thermo(qcdata=qc)
    assert r.spc_applied and r.sp_energy == pytest.approx(SP)
    assert _shift(r, base) == pytest.approx(SP - water.scf_energy, abs=1e-9)
    assert r.enthalpy - base.enthalpy == pytest.approx(SP - water.scf_energy, abs=1e-9)
    assert r.scf_energy == base.scf_energy    # E itself is still the frequency-level energy
    assert r.bbe.sp_level_of_theory == "DLPNO-CCSD(T)/def2-TZVP"
    assert r.scale_factor_source == "truhlar"  # looked up for the frequency level
    assert water.sp_energy is None and water.sp_suffix == ""      # the original is unchanged
    assert (qc.sp_suffix, qc.sp_charge, qc.sp_multiplicity) == (SP_ATTACHED, water.charge, water.multiplicity)


def test_units_are_converted_and_required(water):
    in_ev = water.with_single_point(SP * HARTREE_TO_EV, "eV")
    assert in_ev.sp_energy == pytest.approx(SP)
    with pytest.raises(TypeError):
        water.with_single_point(SP)          # no default units
    with pytest.raises(ValueError):
        water.with_single_point(SP, "furlongs")


def test_the_attached_energy_survives_re_evaluation_and_serialisation(water):
    qc = water.with_single_point(SP, "hartree", "DLPNO")
    r = compute_thermo(qcdata=qc)
    hot = ComputedEntry.from_result(r).bbe(400.0)
    assert hot.spc_applied and hot.sp_energy == pytest.approx(SP)
    back = dict_to_qcdata(qcdata_to_dict(qc))
    assert (back.sp_energy, back.sp_suffix, back.sp_level_of_theory) == (qc.sp_energy, SP_ATTACHED, "DLPNO")
    assert compute_thermo(qcdata=back).qh_gibbs_free_energy == pytest.approx(r.qh_gibbs_free_energy)


def test_an_explicit_spc_still_wins(water):
    qc = water.with_single_point(SP, "hartree")
    with pytest.warns(RuntimeWarning, match="no single-point file with suffix 'nothere'"):
        r = compute_thermo(qcdata=qc, spc="nothere")
    assert not r.spc_applied


def test_dft_on_an_mlip_structure():
    pytest.importorskip("ase")
    from ase.build import molecule
    from goodvibes.io import QCData
    freqs = parse_qcdata(WATER).frequency_wn
    mlip_qc = QCData.from_atoms(molecule("H2O"), -2080.0, frequencies=freqs, method="MACE-OFF23", symm=None)
    composite = mlip_qc.with_single_point(SP, "hartree", "HF/6-31G(d)")
    r = compute_thermo(qcdata=composite)
    base = compute_thermo(qcdata=mlip_qc)
    assert r.scale_factor_source == "mlip-unscaled"
    assert _shift(r, base) == pytest.approx(SP - mlip_qc.scf_energy, abs=1e-9)
    assert r.qh_gibbs_free_energy == pytest.approx(SP + (base.qh_gibbs_free_energy - base.scf_energy), abs=1e-9)
