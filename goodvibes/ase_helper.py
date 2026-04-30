"""Optional helper for emitting GoodVibes ASE thermo extxyz files from an
ASE-driven calculation.

ASE is *not* a runtime dependency of GoodVibes. This module imports it lazily
so the rest of the package works without ASE installed; only direct callers
of ``write_thermo_extxyz`` need it. See tests/ase/README.md for the format spec.
"""
from __future__ import annotations


def write_thermo_extxyz(
    path,
    atoms,
    energy,
    frequencies=None,
    charge=0,
    multiplicity=1,
    level_of_theory=None,
    solvation_model='gas phase',
    empirical_dispersion='',
    point_group=None,
    symmno=None,
    linear_mol=None,
    zpe=None,
    job_type=None,
    energy_units='Hartree',
):
    """Write a GoodVibes-compatible ASE thermo extxyz file.

    Parameters
    ----------
    path : str
        Output file path (typically ``*.extxyz``).
    atoms : ase.Atoms
        Geometry. Element symbols + Cartesian positions are written.
    energy : float
        Electronic (SCF) energy. Hartree by default — override with
        ``energy_units='eV'`` (or ``'kcal/mol'`` / ``'kJ/mol'``).
    frequencies : iterable of float, optional
        Vibrational frequencies in cm-1. Negative values are imaginary modes.
    charge, multiplicity : int
        Defaults: 0 / 1 (closed-shell singlet).
    level_of_theory : str, optional
        Free-form string, e.g. ``"B3LYP/6-31G*"``. Used by the GoodVibes scaling
        factor lookup; omit to fall back on ``--freq_scale_factor``.
    zpe : float, optional
        Zero-point energy in Hartree. If omitted, GoodVibes recomputes it from
        ``frequencies`` at thermo time.
    job_type : str, optional
        ``'Freq'`` / ``'GSFreq'`` / ``'TS'`` / ``'SP'``. If omitted the parser
        infers from frequency presence/sign.
    """
    try:
        import ase  # noqa: F401
        from ase.io import write
    except ImportError as e:
        raise ImportError(
            "ASE is required to write extxyz fixtures. Install with: pip install ase"
        ) from e

    info = dict(atoms.info)
    info['program'] = 'ase'
    info['ase_version'] = ase.__version__
    info['scf_energy'] = float(energy)
    info['scf_energy_units'] = energy_units
    info['charge'] = int(charge)
    info['multiplicity'] = int(multiplicity)
    if frequencies is not None:
        info['frequencies'] = ' '.join(f'{float(f):.6f}' for f in frequencies)
        info['frequencies_units'] = 'cm-1'
    if level_of_theory is not None:
        info['level_of_theory'] = str(level_of_theory)
    if solvation_model is not None:
        info['solvation_model'] = str(solvation_model)
    if empirical_dispersion:
        info['empirical_dispersion'] = str(empirical_dispersion)
    if point_group is not None:
        info['point_group'] = str(point_group)
    if symmno is not None:
        info['symmno'] = int(symmno)
    if linear_mol is not None:
        info['linear_mol'] = bool(linear_mol)
    if zpe is not None:
        info['zpe'] = float(zpe)
    if job_type is not None:
        info['job_type'] = str(job_type)

    atoms = atoms.copy()
    atoms.info = info
    write(path, atoms, format='extxyz')
