"""QCData.from_atoms / from_vibrations: file-free thermochemistry for ASE
and MLIP workflows, checked against the extxyz route and against ASE's own
IdealGasThermo."""
import math
import warnings

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase.build import molecule

from goodvibes import compute_batch, compute_thermo
from goodvibes.constants import EV_TO_WAVENUMBER, HARTREE_TO_EV, KCAL_TO_AU
from goodvibes.io import ATOMIC_MASSES, QCData, parse_qcdata

try:
    import pymsym  # noqa: F401
    HAS_PYMSYM = True
except ImportError:
    HAS_PYMSYM = False

WATER_FREQS = [1655.4, 3826.7, 3935.6]


def _water():
    return molecule("H2O")


def test_from_atoms_basic_fields():
    qc = QCData.from_atoms(_water(), -76.4 * HARTREE_TO_EV, frequencies=WATER_FREQS,
                           name="water", method="HF/3-21G", charge=0, multiplicity=1, symm=None)
    assert qc.program == "ase" and qc.file == "water" and qc.level_of_theory == "HF/3-21G"
    assert qc.scf_energy == pytest.approx(-76.4)
    assert qc.frequency_wn == WATER_FREQS and qc.im_frequency_wn == []
    assert qc.job_type == "Freq"
    assert qc.atom_types == ["O", "H", "H"] and qc.atom_nums == [8, 1, 1]
    assert qc.molecular_mass == pytest.approx(ATOMIC_MASSES["O"] + 2 * ATOMIC_MASSES["H"])
    assert qc.per_atom_masses == [ATOMIC_MASSES["O"], ATOMIC_MASSES["H"], ATOMIC_MASSES["H"]]
    assert all(t > 0 for t in qc.rotemp) and not qc.linear_mol
    assert qc.symmno == 1 and qc.point_group == ""
    assert qc.zero_point_corr == pytest.approx(sum(WATER_FREQS) / 2 * 4.556335e-6, rel=1e-4)


def test_from_atoms_reads_defaults_from_atoms_info():
    atoms = _water()
    atoms.info.update(charge=1, multiplicity=2, name="cation", level_of_theory="MACE-OFF23")
    qc = QCData.from_atoms(atoms, 0.0)
    assert (qc.charge, qc.multiplicity, qc.file, qc.level_of_theory) == (1, 2, "cation", "MACE-OFF23")
    assert qc.job_type == "SP" and qc.zero_point_corr is None


def test_from_atoms_units_and_imaginary_modes():
    qc = QCData.from_atoms(_water(), -1.0, energy_units="hartree",
                           frequencies=[0.2, 0.4, -0.05], frequency_units="eV", symm=None)
    assert qc.scf_energy == -1.0
    assert qc.frequency_wn == pytest.approx([0.2 * EV_TO_WAVENUMBER, 0.4 * EV_TO_WAVENUMBER])
    assert qc.im_frequency_wn == pytest.approx([-0.05 * EV_TO_WAVENUMBER])
    assert qc.job_type == "TSFreq"
    qc2 = QCData.from_atoms(_water(), -627.509541, energy_units="kcal/mol",
                            frequencies=[complex(0, 400.0), 1000.0, 2000.0], job_type="TS", symm=None)
    assert qc2.scf_energy == pytest.approx(-1.0)
    assert qc2.im_frequency_wn == [-400.0] and qc2.job_type == "TSFreq"
    with pytest.raises(ValueError, match="frequency units"):
        QCData.from_atoms(_water(), 0.0, frequencies=[1.0], frequency_units="THz")


def test_from_atoms_masses_options():
    atoms = _water()
    iso = QCData.from_atoms(atoms, 0.0, symm=None)
    std = QCData.from_atoms(atoms, 0.0, masses="atoms", symm=None)
    assert std.molecular_mass == pytest.approx(float(atoms.get_masses().sum()))
    assert std.molecular_mass != iso.molecular_mass
    explicit = QCData.from_atoms(atoms, 0.0, masses=[18.0, 2.0, 2.0], symm=None)
    assert explicit.molecular_mass == 22.0 and explicit.rotemp != iso.rotemp
    with pytest.raises(ValueError, match="3 atoms"):
        QCData.from_atoms(atoms, 0.0, masses=[1.0])
    with pytest.raises(ValueError, match="masses must be"):
        QCData.from_atoms(atoms, 0.0, masses="average")


def test_from_atoms_symmetry_options():
    assert QCData.from_atoms(_water(), 0.0, symm=3).symmno == 3
    auto = QCData.from_atoms(_water(), 0.0, symm="auto")
    if HAS_PYMSYM:
        assert auto.symmno == 2 and auto.point_group == "C2v"
    else:
        assert auto.symmno == 1


def test_from_atoms_linear_molecule():
    qc = QCData.from_atoms(molecule("CO2"), 0.0, symm=None)
    assert qc.linear_mol and qc.rotemp[0] == qc.rotemp[1] == qc.rotemp[2] > 0
    forced = QCData.from_atoms(molecule("CO2"), 0.0, symm=None, linear_mol=False)
    assert not forced.linear_mol


def test_from_atoms_matches_the_extxyz_route(tmp_path):
    """The same geometry, energy and frequencies give the same G whether
    they come through QCData.from_atoms or a written-and-parsed extxyz."""
    from goodvibes.ase_helper import write_thermo_extxyz
    atoms = _water()
    path = tmp_path / "water.extxyz"
    write_thermo_extxyz(str(path), atoms, energy=-76.4, frequencies=WATER_FREQS,
                        level_of_theory="B3LYP/6-31G*", job_type="Freq")
    parsed = parse_qcdata(str(path))
    direct = QCData.from_atoms(atoms, -76.4, energy_units="hartree", frequencies=WATER_FREQS,
                               method="B3LYP/6-31G*", symm=None)
    assert parsed.level_of_theory == "B3LYP/6-31G*"
    assert parsed.molecular_mass == pytest.approx(direct.molecular_mass)
    assert parsed.rotemp == pytest.approx(direct.rotemp)
    r_file = compute_thermo(str(path))
    r_direct = compute_thermo(qcdata=direct)
    assert r_direct.qh_gibbs_free_energy == pytest.approx(r_file.qh_gibbs_free_energy, abs=1e-10)
    assert r_direct.entropy == pytest.approx(r_file.entropy, abs=1e-13)
    assert r_direct.level_of_theory == "B3LYP/6-31G*" and r_direct.name == "atoms"
    # the level of theory drives the scale-factor lookup for both
    assert r_direct.bbe.options.freq_scale_factor == r_file.bbe.options.freq_scale_factor != 1.0


def test_from_atoms_unknown_method_is_unscaled():
    r = compute_thermo(qcdata=QCData.from_atoms(_water(), 0.0, frequencies=WATER_FREQS, method="MACE-OFF23"))
    assert r.bbe.options.freq_scale_factor == 1.0 and r.bbe.options.zpe_scale_factor == 1.0


# ---------------------------------------------------------------------------
# from_vibrations
# ---------------------------------------------------------------------------

def test_from_vibrations_drops_translations_and_rotations_and_flags_noise():
    atoms = _water()
    raw = [complex(0, 0.7), 0.1, 0.6, 1.9, 2.4, 3.0, complex(0, 12.0), 194.0, 2406.1]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        qc = QCData.from_vibrations(atoms, raw, -10.0, name="w", symm=None)
    assert qc.frequency_wn == [12.0, 194.0, 2406.1]          # 6 smallest dropped, −12 taken as real
    assert qc.im_frequency_wn == []
    real_ts = QCData.from_vibrations(atoms, raw[:6] + [complex(0, 340.0), 194.0, 2406.1], -10.0, symm=None)
    assert real_ts.im_frequency_wn == [-340.0] and real_ts.job_type == "TSFreq"
    assert any("treated as real" in str(x.message) for x in w)
    assert qc.job_type == "Freq"
    kept = QCData.from_vibrations(atoms, [100.0, 200.0, 300.0], -10.0, symm=None)
    assert kept.frequency_wn == [100.0, 200.0, 300.0]        # already 3N−6: nothing dropped
    nodrop = QCData.from_vibrations(atoms, raw, -10.0, drop_tr_modes=False, symm=None,
                                    imag_threshold_cm1=-100.0)
    assert len(nodrop.frequency_wn) == 9


def test_from_vibrations_warns_on_unexpected_imaginary_counts():
    atoms = _water()
    with pytest.warns(RuntimeWarning, match="expected 1"):
        QCData.from_vibrations(atoms, [100.0, 200.0, 300.0], 0.0, job_type="TS", symm=None)
    with pytest.warns(RuntimeWarning, match="minimum with 1"):
        QCData.from_vibrations(atoms, [-300.0, 200.0, 300.0], 0.0, job_type="Freq", symm=None)
    with pytest.warns(RuntimeWarning, match="2 imaginary modes"):
        QCData.from_vibrations(atoms, [-300.0, -200.0, 300.0], 0.0, symm=None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ts = QCData.from_vibrations(atoms, [-300.0, 200.0, 300.0], 0.0, job_type="TS", symm=None)
    assert ts.job_type == "TSFreq" and ts.im_frequency_wn == [-300.0]


def test_from_vibrations_against_ase_ideal_gas_thermo(tmp_path):
    """EMT water: the RRHO entropy and thermal enthalpy agree with ASE's
    IdealGasThermo on the same 3N−6 modes."""
    from ase.calculators.emt import EMT
    from ase.optimize import BFGS
    from ase.thermochemistry import IdealGasThermo
    from ase.vibrations import Vibrations
    atoms = _water()
    atoms.calc = EMT()
    BFGS(atoms, logfile=None).run(fmax=1e-4)
    vib = Vibrations(atoms, delta=0.01, name=str(tmp_path / "vib"))
    vib.run()
    vd = vib.get_vibrations()
    e = atoms.get_potential_energy()
    qc = QCData.from_vibrations(atoms, vd, e, name="h2o", method="EMT", symm=2)
    assert len(qc.frequency_wn) == 3 and qc.im_frequency_wn == []
    r = compute_thermo(qcdata=qc, temperature=298.15)
    ig = IdealGasThermo(vib_energies=sorted(np.real(vd.get_energies()))[-3:], geometry="nonlinear",
                        atoms=atoms, symmetrynumber=2, spin=0, potentialenergy=e)
    s_ase = ig.get_entropy(298.15, 101325.0, verbose=False)          # eV/K
    h_ase = ig.get_enthalpy(298.15, verbose=False) - e               # eV
    assert r.entropy * HARTREE_TO_EV == pytest.approx(s_ase, rel=1e-4)
    assert (r.enthalpy - r.scf_energy) * HARTREE_TO_EV == pytest.approx(h_ase, rel=1e-6)
    assert r.scf_energy == pytest.approx(e / HARTREE_TO_EV)
    # Vibrations objects are accepted as well as VibrationsData
    assert QCData.from_vibrations(atoms, vib, e, symm=None).frequency_wn == pytest.approx(qc.frequency_wn)
    vib.clean()


# ---------------------------------------------------------------------------
# API integration
# ---------------------------------------------------------------------------

def test_compute_batch_accepts_qcdata_and_paths_mixed():
    qc = QCData.from_atoms(_water(), -76.4, energy_units="hartree", frequencies=WATER_FREQS, symm=None)
    from conftest import g16path
    results = compute_batch([qc, g16path("01a_water_hf_freq.log")], temperature=300.0)
    assert [r.name for r in results] == ["atoms", "01a_water_hf_freq"]
    assert results[0].qh_gibbs_free_energy == pytest.approx(compute_thermo(qcdata=qc, temperature=300.0).qh_gibbs_free_energy)
    parallel = compute_batch([qc, qc], jobs=2, temperature=300.0)
    assert parallel[1].qh_gibbs_free_energy == pytest.approx(results[0].qh_gibbs_free_energy)


def test_from_atoms_results_feed_a_conformer_set_at_other_temperatures():
    from goodvibes import ConformerSet
    qc = QCData.from_atoms(_water(), -76.4, energy_units="hartree", frequencies=WATER_FREQS, symm=None)
    cs = ConformerSet.from_results("w", [compute_thermo(qcdata=qc)])
    assert cs.recomputable
    v = cs.vectors(500.0)[0]
    assert v.qh_gibbs == pytest.approx(compute_thermo(qcdata=qc, temperature=500.0).qh_gibbs_free_energy)
    assert abs(v.qh_gibbs - cs.vectors()[0].qh_gibbs) * KCAL_TO_AU > 1.0
    assert math.isfinite(v.entropy)


def test_mass_table_covers_the_periodic_table_to_uranium():
    from ase.data import atomic_masses, chemical_symbols
    for sym in ("Cs", "Pt", "Au", "Hg", "Pb", "Bi", "U"):
        assert sym in ATOMIC_MASSES
    for sym, m in ATOMIC_MASSES.items():
        assert abs(m - atomic_masses[chemical_symbols.index(sym)]) < 2.5, sym
