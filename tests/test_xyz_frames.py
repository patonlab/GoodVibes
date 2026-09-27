"""read_xyz_frames: multi-frame .xyz / .extxyz files (CREST ensembles, xtb
trajectories, MLIP sweeps) as energy-only QCData."""
import math
import os

import pytest

from goodvibes import ConformerSet, compute_batch, read_xyz_frames
from goodvibes.constants import GAS_CONSTANT, HARTREE_TO_EV, J_TO_AU

WATER = """O   0.000000   0.000000   0.117300
H   0.000000   0.757200  -0.469200
H   0.000000  -0.757200  -0.469200
"""
CREST = f"""3
  -5.07054444
{WATER}   3
       -5.06954444 !CREST
{WATER}3
  -5.06554444
{WATER}"""
XTB = f"""3
 energy: -5.070544440612 gnorm: 0.000123456789 xtb: 6.5.1 (fbbdde1)
{WATER}
3
 energy: -5.069544440612 gnorm: 0.000223456789 xtb: 6.5.1 (fbbdde1)
{WATER}"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_a_crest_ensemble_reads_as_energy_only_frames(tmp_path):
    frames = read_xyz_frames(_write(tmp_path, "crest_conformers.xyz", CREST), method="GFN2-xTB")
    assert [f.scf_energy for f in frames] == pytest.approx([-5.07054444, -5.06954444, -5.06554444])
    assert [os.path.basename(f.file) for f in frames] == ["crest_conformers_001", "crest_conformers_002",
                                                          "crest_conformers_003"]
    f = frames[0]
    assert (f.atom_types, f.job_type, f.program, f.level_of_theory) == (["O", "H", "H"], "SP", "xyz", "GFN2-xTB")
    assert not f.frequency_wn and f.molecular_mass == pytest.approx(18.0106, abs=1e-3)
    assert (f.charge, f.multiplicity) == (0, 1)


def test_the_frames_give_boltzmann_populations_by_electronic_energy(tmp_path):
    frames = read_xyz_frames(_write(tmp_path, "crest_conformers.xyz", CREST))
    results = compute_batch(frames)
    assert all(r.scf_energy is not None and r.qh_gibbs_free_energy is None for r in results)
    assert all(r.scale_factor_source is None for r in results)
    ens = ConformerSet.from_results("water", results, weight_by="electronic")
    rt = GAS_CONSTANT * 298.15 / J_TO_AU
    w = [math.exp(-(e + 5.07054444) / rt) for e in (-5.07054444, -5.06954444, -5.06554444)]
    assert ens.populations(298.15) == pytest.approx([x / sum(w) for x in w])
    assert ens.lowest_index() == 0
    with pytest.raises(ValueError, match="'qh_gibbs' is not available"):
        ens.populations(298.15, "qh_gibbs")
    with pytest.raises(ValueError, match="unless the set is weighted by 'electronic'"):
        ConformerSet.from_results("water", results)
    with pytest.raises(ValueError, match="unless the set is weighted by 'electronic'"):
        ConformerSet.from_results("water", results, weight_by="spc")    # no single point here


def test_energy_only_sets_roll_up_their_electronic_energy(tmp_path):
    frames = read_xyz_frames(_write(tmp_path, "crest_conformers.xyz", CREST))
    ens = ConformerSet.from_results("water", compute_batch(frames), weight_by="electronic")
    p = ens.populations(298.15)
    expected = sum(pi * f.scf_energy for pi, f in zip(p, frames))
    for vec in (ens.boltzmann_weighted(298.15), ens.gconf_corrected(298.15)):
        assert vec.scf_energy == pytest.approx(expected)
        assert vec.qh_gibbs is None and vec.entropy is None and vec.get("e_zpe") is None
    diff = ens.boltzmann_weighted(298.15) - ens.lowest_conformer(298.15)
    assert diff.scf_energy == pytest.approx(expected - frames[0].scf_energy) and diff.gibbs is None


def test_an_xtb_trajectory(tmp_path):
    frames = read_xyz_frames(_write(tmp_path, "xtbopt.log", XTB))
    assert [f.scf_energy for f in frames] == pytest.approx([-5.070544440612, -5.069544440612])


@pytest.mark.parametrize("comment, energy", [
    ("energy: -5 gnorm: 0.1", -5.0),                  # an integer, not the next number
    ("energy: -5.07e0 gnorm: 0.1", -5.07),
    ("energy: -5.0 force=0.01", -5.0),                # a stray key=value is not extxyz
    ("Energy = -4.5 (GFN2)", -4.5),
])
def test_a_labelled_plain_energy(tmp_path, comment, energy):
    frames = read_xyz_frames(_write(tmp_path, "one.xyz", f"3\n{comment}\n{WATER}"))
    assert frames[0].scf_energy == pytest.approx(energy)


def test_an_extxyz_in_ev_with_extra_columns_and_keys(tmp_path):
    lines = []
    for i, e in enumerate((-2080.0, -2079.9)):
        lines += ["3",
                  f'Lattice="0 0 0 0 0 0 0 0 0" Properties=species:S:1:pos:R:3:forces:R:3 energy={e} '
                  f'charge=-1 multiplicity=2 level_of_theory=MACE-OFF23 pbc="F F F"']
        lines += [row + "  0.1 0.2 0.3" for row in WATER.splitlines()]
    frames = read_xyz_frames(_write(tmp_path, "sweep.extxyz", "\n".join(lines) + "\n"))
    assert frames[0].scf_energy == pytest.approx(-2080.0 / HARTREE_TO_EV)
    assert (frames[1].charge, frames[1].multiplicity, frames[1].level_of_theory) == (-1, 2, "MACE-OFF23")
    assert frames[0].cartesians[1] == pytest.approx([0.0, 0.7572, -0.4692])
    kcal = read_xyz_frames(str(tmp_path / "sweep.extxyz"), energy_units="kcal/mol")
    assert kcal[0].scf_energy == pytest.approx(-2080.0 / 627.509, rel=1e-5)


def test_extxyz_energy_key_units_and_names(tmp_path):
    text = ("3\nname=conf_a scf_energy=-76.01 E_dft=-76.2 E_dft_units=hartree\n" + WATER
            + "3\nname=conf_b scf_energy=-76.00 E_dft=-76.1 E_dft_units=hartree\n" + WATER)
    path = _write(tmp_path, "named.extxyz", text)
    frames = read_xyz_frames(path)
    assert [f.scf_energy for f in frames] == pytest.approx([-76.01, -76.00])     # hartree by default
    assert [os.path.basename(f.file) for f in frames] == ["conf_a", "conf_b"]
    assert [f.scf_energy for f in read_xyz_frames(path, energy_key="E_dft")] == pytest.approx([-76.2, -76.1])


def test_ase_writes_what_the_reader_reads(tmp_path):
    ase_io = pytest.importorskip("ase.io")
    from ase.build import molecule
    from ase.calculators.singlepoint import SinglePointCalculator
    images = []
    for e in (-14.2, -14.1, -13.9):
        atoms = molecule("CH4")
        atoms.calc = SinglePointCalculator(atoms, energy=e)
        images.append(atoms)
    path = str(tmp_path / "sweep.extxyz")
    ase_io.write(path, images, format="extxyz")
    frames = read_xyz_frames(path)
    assert [f.scf_energy * HARTREE_TO_EV for f in frames] == pytest.approx([-14.2, -14.1, -13.9])
    assert frames[0].atom_types == ["C", "H", "H", "H", "H"]


@pytest.mark.parametrize("text, message", [
    ("3\nno energy here\n" + WATER, "no energy in the comment line"),
    ("3\nProperties=species:S:1:pos:R:3 foo=bar\n" + WATER, "looked for energy"),
    ("3\nenergy: n/a gnorm: 0.1\n" + WATER, "energy 'n/a' is not a number"),
    ("-2\n-1.0\n" + WATER, "expected a positive atom count"),
    ("0\n-1.0\n", "expected a positive atom count"),
    ("3\n-1.0\nO 0 0 0\n", "truncated"),
    ("three\n-1.0\n" + WATER, "expected a positive atom count"),
    ("3\n-1.0\nO 0 0\nH 0 0 1\nH 0 1 0\n", "cannot read the atom line"),
    ("", "no frames"),
])
def test_malformed_files_are_errors(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        read_xyz_frames(_write(tmp_path, "bad.xyz", text))
