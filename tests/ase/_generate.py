"""Regenerate the .extxyz fixtures in this directory from the matching g16 logs.

Each fixture mirrors a Gaussian calculation in tests/g16/ so the
cross-validation tests in test_thermo_ase.py compare like-for-like:

    parse_qcdata('01_water.extxyz')  →  same H/S/G as
    parse_qcdata('tests/g16/01a_water_hf_freq.log')

Run: ``python tests/ase/_generate.py`` (requires ase). The output should be
deterministic — ``git diff tests/ase/`` is the regression check.
"""
import os

from ase import Atoms

from goodvibes.ase_helper import write_thermo_extxyz
from goodvibes.io import parse_qcdata, read_initial

HERE = os.path.dirname(os.path.abspath(__file__))
G16_DIR = os.path.join(HERE, '..', 'g16')


# (output_name, source_log, level_of_theory_override, energy_units, extras)
# energy_units='eV' triggers an eV-encoded variant for the unit-conversion test.
FIXTURES = [
    ('01_water.extxyz',           '01a_water_hf_freq.log',                 None,            'Hartree', {}),
    ('05_methylene_triplet.extxyz','05_methylene_triplet_carbene.log',     None,            'Hartree', {}),
    ('10_formaldehyde.extxyz',    '10_formaldehyde_verbose_pop.log',       None,            'Hartree', {}),
    ('22_hcn_linear.extxyz',      '22_hcn_linear_freq_noraman.log',        None,            'Hartree', {}),
    ('44_ts_sn2.extxyz',          '44_ts_sn2_identity_chloride.log',       None,            'Hartree', {'job_type': 'TS'}),
    ('08_alanine_pcm_water.extxyz','08_alanine_C1_pcm_water.log',          None,            'eV',      {}),
]


def fixture_from_log(log_path, energy_units, extras):
    """Return (Atoms, kwargs) ready to feed into write_thermo_extxyz."""
    q = parse_qcdata(log_path)
    if not q.atom_types or q.scf_energy is None:
        raise RuntimeError(f"log produced empty QCData: {log_path}")

    lot, solv, _, _, _ = read_initial(log_path)
    if lot.startswith('none/'):
        lot = ''

    atoms = Atoms(symbols=q.atom_types, positions=q.cartesians)

    energy = q.scf_energy
    if energy_units == 'eV':
        energy = energy * 27.211386245988  # Ha → eV

    frequencies = list(q.im_frequency_wn) + list(q.frequency_wn)

    kwargs = dict(
        energy=energy,
        frequencies=frequencies if frequencies else None,
        charge=q.charge if q.charge is not None else 0,
        multiplicity=q.multiplicity,
        level_of_theory=lot or None,
        solvation_model=solv or 'gas phase',
        point_group=q.point_group or None,
        symmno=q.symmno or None,
        linear_mol=q.linear_mol if q.linear_mol else None,
        zpe=q.zero_point_corr,
        energy_units=energy_units,
    )
    kwargs.update(extras)
    return atoms, kwargs


def main():
    for out_name, src_log, _lot_override, energy_units, extras in FIXTURES:
        src = os.path.join(G16_DIR, src_log)
        if not os.path.exists(src):
            print(f"SKIP {out_name}: source log missing ({src})")
            continue
        atoms, kwargs = fixture_from_log(src, energy_units, extras)
        out_path = os.path.join(HERE, out_name)
        write_thermo_extxyz(out_path, atoms, **kwargs)
        print(f"wrote {out_name}")


if __name__ == '__main__':
    main()
