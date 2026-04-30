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
    """
    Write a GoodVibes-compatible ASE thermo `.extxyz` file for the supplied geometry and metadata.
    
    Parameters:
        path (str): Output file path (typically ending with `.extxyz`).
        atoms (ase.Atoms): Geometry; element symbols and Cartesian positions are written.
        energy (float): Electronic (SCF) energy.
        frequencies (iterable[float], optional): Vibrational frequencies in cm^-1; negative values indicate imaginary modes.
        charge (int, optional): Total molecular charge. Default: 0.
        multiplicity (int, optional): Spin multiplicity. Default: 1.
        level_of_theory (str, optional): Free-form method/basis description (e.g., "B3LYP/6-31G*") used for GoodVibes scaling lookup.
        solvation_model (str or None, optional): Description of solvation model (defaults to 'gas phase'); set to None to omit.
        empirical_dispersion (str, optional): Empirical dispersion tag (included when non-empty).
        point_group (str, optional): Molecular point group symbol.
        symmno (int, optional): Symmetry number.
        linear_mol (bool, optional): Whether the molecule is linear.
        zpe (float, optional): Zero-point energy in the same units as `energy`.
        job_type (str, optional): Job type hint (e.g., 'Freq', 'GSFreq', 'TS', 'SP'); parser may infer if omitted.
        energy_units (str, optional): Units for `energy` (default: 'Hartree').
    
    Raises:
        ImportError: If ASE is not installed (suggests installing via `pip install ase`).
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
