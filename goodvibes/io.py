# -*- coding: utf-8 -*-
import json
import os.path
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np



@dataclass
class QCData:
    """Program-agnostic container for parsed quantum chemistry data.

    Populated by parse_qcdata() in io.py. Consumed by calc_bbe in thermo.py.
    """
    # Provenance
    file: str = ''
    program: str = ''
    version_program: str = ''
    job_type: str = ''

    # Electronic structure
    scf_energy: Optional[float] = None
    charge: Optional[int] = None
    multiplicity: int = 1

    # Model chemistry metadata
    solvation_model: str = ''
    empirical_dispersion: str = ''

    # Molecular properties
    molecular_mass: float = 0.0
    symmno: int = 1
    linear_mol: bool = False
    point_group: str = ''

    # Rotational data
    roconst: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotemp: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    linear_warning: bool = False

    # Frequencies (raw from parser — positive and negative separated)
    frequency_wn: List[float] = field(default_factory=list)
    im_frequency_wn: List[float] = field(default_factory=list)

    # Thermal corrections
    zero_point_corr: Optional[float] = None

    # CPU time [days, hours, mins, secs, msecs]
    cpu: List[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])

    # Number of MPI processes / cores reported by the QC program. Used by
    # the ORCA parser to convert TOTAL RUN TIME (wall) into an effective
    # CPU-time estimate (wall × nprocs); 1 elsewhere.
    nprocs: int = 1

    # ONIOM MM frequency scaling fractions (per-frequency, empty if not ONIOM)
    fract_modelsys: List[float] = field(default_factory=list)
    has_oniom: bool = False

    # Molecular geometry
    atom_nums: List[int] = field(default_factory=list)
    atom_types: List[str] = field(default_factory=list)
    cartesians: List[List[float]] = field(default_factory=list)

    # Per-atom masses in amu, as reported by the QC program for the freq job
    # (respects isotope substitution, e.g. Gaussian's iso= keyword). Empty when
    # the parser does not report them; consumers should fall back to
    # ATOMIC_MASSES lookups by atom_types. Order matches cartesians.
    per_atom_masses: List[float] = field(default_factory=list)

    # Frequency scaling factor applied by QC program (ORCA only; 1.0 means unscaled)
    applied_freq_scale_factor: float = 1.0

    # Single-point-correction (SPC) cache. Populated by calc_bbe when --spc
    # is used so the SPC file isn't re-parsed on subsequent --import runs.
    # `sp_suffix` is the --spc value that produced the cached numbers; on
    # a re-run with the same suffix the cache is reused, on a different
    # suffix it's bypassed and refreshed. `sp_file` is the resolved
    # absolute path of the SPC output (diagnostic only).
    sp_energy: Optional[float] = None
    sp_version_program: str = ''
    sp_solvation_model: str = ''
    sp_charge: Optional[int] = None
    sp_empirical_dispersion: str = ''
    sp_multiplicity: Optional[int] = None
    sp_suffix: str = ''
    sp_file: str = ''


# Cache version for JSON serialization format
CACHE_VERSION = 1


def qcdata_to_dict(qcdata):
    """Convert a QCData instance to a JSON-serializable dictionary."""
    d = asdict(qcdata)
    d['_cache_version'] = CACHE_VERSION
    return d


def dict_to_qcdata(d):
    """Reconstruct a QCData instance from a dictionary."""
    clean = {k: v for k, v in d.items() if not k.startswith('_')}
    return QCData(**clean)


def save_cache(cache_dict, path):
    """Write QCData cache to a JSON file.

    Parameters
    ----------
    cache_dict : dict
        Mapping of {basename_key: qcdata_dict} where each value is from qcdata_to_dict().
    path : str
        Output JSON file path.
    """
    envelope = {
        '_cache_version': CACHE_VERSION,
        '_created': datetime.now().isoformat(),
        'entries': cache_dict,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(envelope, f, indent=2)


def load_cache(path):
    """Load a QCData cache from a JSON file.

    Auto-detects the format. v1.0+ payloads (the unified schema written
    by `--export` / `--json`) are read by extracting each result's
    `qcdata` block; the legacy cache envelope (`{_cache_version, entries}`,
    written by pre-v5.0 `--cache-save`) is also accepted for back-compat.

    Parameters
    ----------
    path : str
        Path to JSON file (either v1.0 unified schema or legacy cache).

    Returns
    -------
    dict
        Mapping of {basename_key: QCData}.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'schema_version' in data:
        # Unified v1.0+ payload — use each result's qcdata block.
        from .schema import validate
        validate(data)
        cache = {}
        for entry in data.get('results', []):
            qc = entry.get('qcdata')
            if qc is None:
                continue
            name = entry.get('name')
            if name is None:
                # Fall back to deriving from the file path.
                fp = entry.get('file', '')
                name = os.path.splitext(os.path.basename(fp))[0]
            cache[name] = dict_to_qcdata(qc)
        return cache
    # Legacy envelope: {_cache_version, _created, entries}
    version = data.get('_cache_version', 0)
    if version != CACHE_VERSION:
        raise ValueError(
            "Cache version mismatch: expected {} (legacy) or a v1.0+ "
            "schema_version, got _cache_version={}".format(CACHE_VERSION, version)
        )
    return {key: dict_to_qcdata(val) for key, val in data['entries'].items()}


# PHYSICAL CONSTANTS                                      UNITS
KCAL_TO_AU = 627.509541  # UNIT CONVERSION

# Radii used to determine connectivity in symmetry corrections
# Covalent radii taken from Cambridge Structural Database
RADII = {'H': 0.32, 'He': 0.93, 'Li': 1.23, 'Be': 0.90, 'B': 0.82, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.72,
         'Ne': 0.71, 'Na': 1.54, 'Mg': 1.36, 'Al': 1.18, 'Si': 1.11, 'P': 1.06, 'S': 1.02, 'Cl': 0.99, 'Ar': 0.98,
         'K': 2.03, 'Ca': 1.74, 'Sc': 1.44, 'Ti': 1.32, 'V': 1.22, 'Cr': 1.18, 'Mn': 1.17, 'Fe': 1.17, 'Co': 1.16,
         'Ni': 1.15, 'Cu': 1.17, 'Zn': 1.25, 'Ga': 1.26, 'Ge': 1.22, 'As': 1.20, 'Se': 1.16, 'Br': 1.14, 'Kr': 1.12,
         'Rb': 2.16, 'Sr': 1.91, 'Y': 1.62, 'Zr': 1.45, 'Nb': 1.34, 'Mo': 1.30, 'Tc': 1.27, 'Ru': 1.25, 'Rh': 1.25,
         'Pd': 1.28, 'Ag': 1.34, 'Cd': 1.48, 'In': 1.44, 'Sn': 1.41, 'Sb': 1.40, 'Te': 1.36, 'I': 1.33, 'Xe': 1.31,
         'Cs': 2.35, 'Ba': 1.98, 'La': 1.69, 'Lu': 1.60, 'Hf': 1.44, 'Ta': 1.34, 'W': 1.30, 'Re': 1.28, 'Os': 1.26,
         'Ir': 1.27, 'Pt': 1.30, 'Au': 1.34, 'Hg': 1.49, 'Tl': 1.48, 'Pb': 1.47, 'Bi': 1.46, 'X': 0}
# Bondi van der Waals radii for all atoms from: Bondi, A. J. Phys. Chem. 1964, 68, 441-452,
# except hydrogen, which is taken from Rowland, R. S.; Taylor, R. J. Phys. Chem. 1996, 100, 7384-7391.
# Radii unavailable in either of these publications are set to 2 Angstrom
# (Unfinished)
BONDI = {'H': 1.09, 'He': 1.40, 'Li': 1.82, 'Be': 2.00, 'B': 2.00, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
         'Ne': 1.54}

# Some useful arrays
periodictable = ["", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si",
                 "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                 "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
                 "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",
                 "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
                 "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
                 "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
                 "Rg", "Uub", "Uut", "Uuq", "Uup", "Uuh", "Uus", "Uuo"]

def element_id(massno, num=False):
    """
    Get element symbol from mass number.

    Used in parsing output files to determine elements present in file.

    Parameter:
    massno (int): mass of element.

    Returns:
    str: element symbol, or 'XX' if not found in periodic table.
    """
    try:
        if num:
            return periodictable.index(massno)
        return periodictable[massno]
    except IndexError:
        return "XX"


# Standard atomic weights (IUPAC 2021), most common isotope where applicable.
# Used by parse_ase_thermo when an extxyz fixture omits molecular_mass / rotational
# constants and they must be computed from the geometry.
ATOMIC_MASSES = {
    'H': 1.00782503207, 'He': 4.002602,
    'Li': 7.0160034366, 'Be': 9.012183065, 'B': 11.00930536, 'C': 12.0,
    'N': 14.0030740048, 'O': 15.99491461957, 'F': 18.998403163, 'Ne': 19.9924401762,
    'Na': 22.9897692820, 'Mg': 23.985041697, 'Al': 26.98153853, 'Si': 27.97692653465,
    'P': 30.97376199842, 'S': 31.9720711744, 'Cl': 34.968852682, 'Ar': 39.9623831237,
    'K': 38.96370648, 'Ca': 39.96259098, 'Sc': 44.95590828, 'Ti': 47.94794198,
    'V': 50.94395704, 'Cr': 51.94050623, 'Mn': 54.93804451, 'Fe': 55.93493633,
    'Co': 58.93319429, 'Ni': 57.93534241, 'Cu': 62.92959772, 'Zn': 63.92914201,
    'Ga': 68.9255735, 'Ge': 73.921177761, 'As': 74.92159457, 'Se': 79.9165218,
    'Br': 78.9183376, 'Kr': 83.9114977282,
    'Rb': 84.91178974, 'Sr': 87.9056125, 'Y': 88.9058403, 'Zr': 89.9046977,
    'Nb': 92.906373, 'Mo': 97.90540482, 'Tc': 98.9062508, 'Ru': 101.9043441,
    'Rh': 102.905498, 'Pd': 105.9034804, 'Ag': 106.9050916, 'Cd': 113.90336509,
    'In': 114.903878776, 'Sn': 119.90220163, 'Sb': 120.903812, 'Te': 129.906222748,
    'I': 126.9044719, 'Xe': 131.9041550856,
}


def _principal_moments_of_inertia(atom_types, cartesians):
    """Return (total_mass [amu], [I_a, I_b, I_c] in amu*Angstrom^2).

    Used as a fallback when an extxyz fixture omits molecular_mass / roconst /
    rotemp; computes them from the geometry using ATOMIC_MASSES.
    """
    masses = np.array([ATOMIC_MASSES.get(a, 0.0) for a in atom_types])
    coords = np.array(cartesians, dtype=float)
    total = float(masses.sum())
    if total <= 0.0 or len(coords) == 0:
        return total, np.array([0.0, 0.0, 0.0])
    com = (masses[:, None] * coords).sum(axis=0) / total
    r = coords - com
    Ixx = float((masses * (r[:, 1] ** 2 + r[:, 2] ** 2)).sum())
    Iyy = float((masses * (r[:, 0] ** 2 + r[:, 2] ** 2)).sum())
    Izz = float((masses * (r[:, 0] ** 2 + r[:, 1] ** 2)).sum())
    Ixy = float(-(masses * r[:, 0] * r[:, 1]).sum())
    Ixz = float(-(masses * r[:, 0] * r[:, 2]).sum())
    Iyz = float(-(masses * r[:, 1] * r[:, 2]).sum())
    inertia = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    eigvals = np.linalg.eigvalsh(inertia)
    eigvals = np.sort(np.maximum(eigvals, 0.0))
    return total, eigvals

def _fill_mass_and_rotemp_from_geometry(qcdata):
    """Populate molecular_mass and rotational constants from geometry.

    Fallback used when a QC program omits the THERMOCHEMISTRY block
    (e.g. Gaussian freq-without-opt, ORCA --hess-only) — recomputes from
    atom_types + cartesians via ATOMIC_MASSES so that downstream
    translational/rotational entropy doesn't see mass=0 or rotemp=[0,0,0].
    No-op when the parser already populated the fields.
    """
    if not (qcdata.atom_types and qcdata.cartesians):
        return
    needs_mass = qcdata.molecular_mass == 0.0
    needs_rotemp = qcdata.rotemp == [0.0, 0.0, 0.0]
    if not (needs_mass or needs_rotemp):
        return
    total_mass, eigvals = _principal_moments_of_inertia(
        qcdata.atom_types, qcdata.cartesians)
    if needs_mass:
        qcdata.molecular_mass = total_mass
    if needs_rotemp:
        HC_OVER_KB = 1.4387768775  # K per cm⁻¹
        roconst_cm = [16.857629 / I if I > 1e-3 else 0.0 for I in eigvals]
        qcdata.roconst = [b * 29.9792458 for b in roconst_cm]
        qcdata.rotemp = [HC_OVER_KB * b for b in roconst_cm]


def compute_connectivity(atom_types, cartesians, tolerance=0.2):
    """Compute molecular connectivity based on covalent radii."""
    connectivity = []
    for i, ai in enumerate(atom_types):
        row = []
        for j, aj in enumerate(atom_types):
            if i == j:
                continue
            cutoff = RADII[ai] + RADII[aj] + tolerance
            distance = np.linalg.norm(np.array(cartesians[i]) - np.array(cartesians[j]))
            if distance < cutoff:
                row.append(j)
        connectivity.append(row)
    return connectivity

def write_xyz(filepath, files, thermo_data):
    """Write optimized Cartesian coordinates to a multi-structure .xyz file.

    Each structure block contains the atom count, a comment line with the
    filename and SCF energy, and the Cartesian coordinates.

    Parameters:
        filepath (str): output .xyz file path.
        files (list): output file paths whose coordinates to write.
        thermo_data (dict): file path → calc_bbe mapping.
    """
    from .utils import display_name
    with open(filepath, 'w', encoding='utf-8') as f:
        for file in files:
            bbe = thermo_data[file]
            f.write(f'{len(bbe.atom_types)}\n')
            if bbe.scf_energy is not None:
                f.write(f'{display_name(file):<39} {"Eopt":>13} {bbe.scf_energy:.6f}\n')
            else:
                f.write(f'{display_name(file):<39}\n')
            if bbe.atom_types and bbe.cartesians:
                for atom, carts in zip(bbe.atom_types, bbe.cartesians):
                    f.write(f'{atom:>1}{carts[0]:13.6f}{carts[1]:13.6f}{carts[2]:13.6f}\n')

def find_spc_file(name, spc):
    """Locate the single-point-correction output paired with ``name``.

    Pattern matched: ``{name}_{spc}.{log,out}`` — exact match. ``--spc TZVP``
    finds ``filename_TZVP.log`` only; it will NOT match ``filename_def2_TZVP.log``.
    To pair with a file like ``filename_def2_TZVP.log``, pass the full suffix
    (``--spc def2_TZVP``).

    Parameters
    ----------
    name : str
        Path stem of the parent output file (no extension).
    spc : str
        Suffix passed via ``--spc``.

    Returns
    -------
    str or None
        Path to the matching file, or None if nothing matches.
    """
    for ext in ('.log', '.out'):
        candidate = f'{name}_{spc}{ext}'
        if os.path.isfile(candidate):
            return candidate
    return None


def parse_data(file):
    """
    Read computational chemistry output file.

    Attempt to obtain single point energy, program type, program version, solvation_model,
    charge, empirical_dispersion, and multiplicity from file.

    Parameter:
    file (str): name of file to be parsed.

    Returns:
    float: single point energy.
    str: program used to run calculation.
    str: version of program used to run calculation.
    str: solvation model used in chemical calculation (if any).
    str: original filename parsed.
    int: overall charge of molecule or chemical system.
    str: empirical dispersion used in chemical calculation (if any).
    int: multiplicity of molecule or chemical system.
    """
    spe, program, data, version_program, solvation_model, keyword_line, a, charge, multiplicity = 'none', 'none', [], '', '', '', 0, None, None
    # Default when no route section is present (e.g. a truncated Gaussian
    # output); otherwise referenced-before-assignment at the return below.
    empirical_dispersion = ''

    data = None
    stub = os.path.splitext(file)[0]
    possible_filenames = (stub + ".log", stub + ".out", stub + ".extxyz", file)
    actual_file = file
    for possible_filename in possible_filenames:
        if os.path.exists(possible_filename):
            actual_file = possible_filename
            with open(possible_filename, encoding='utf-8', errors='replace') as f:
                data = f.readlines()
            break

    if data is None:
        raise ValueError("File {} does not exist".format(file))

    # ASE extxyz: delegate to parse_ase_thermo and translate to parse_data's
    # tuple shape; ASE files have no separate single-point semantics.
    if len(data) >= 2 and 'program=ase' in data[1]:
        q = parse_ase_thermo(actual_file)
        return (q.scf_energy if q.scf_energy is not None else 'none',
                'ase',
                q.version_program,
                q.solvation_model or 'gas phase',
                file,
                q.charge if q.charge is not None else 0,
                q.empirical_dispersion or 'No empirical dispersion detected',
                q.multiplicity)

    # Q-Chem: delegate to parse_qchem_thermo (Q-Chem energy lines, MP2/CCSD
    # precedence, and $molecule echo are all already handled there).
    if any('You are running Q-Chem' in ln or 'Welcome to Q-Chem' in ln for ln in data[:80]):
        q = parse_qchem_thermo(actual_file)
        return (q.scf_energy if q.scf_energy is not None else 'none',
                'QChem',
                q.version_program,
                q.solvation_model or 'gas phase',
                file,
                q.charge if q.charge is not None else 0,
                q.empirical_dispersion or 'No empirical dispersion detected',
                q.multiplicity)

    for line in data:
        if "Gaussian" in line:
            program = "Gaussian"
            break
        if "* O   R   C   A *" in line:
            program = "Orca"
            break
        if "NWChem" in line:
            program = "NWChem"
            break
        if "x T B" in line or "xtb version" in line:
            program = "xtb"
            break
    repeated_link1 = 0
    freq_started = False  # Guard against VPT2 displaced geometry energies
    zero_point_corr_G4 = 0.0

    for line in data:
        if program == "Gaussian":
            # Reset freq_started at each new link (linked jobs have separate links)
            if 'Normal termination' in line:
                freq_started = False
            if line.strip().startswith('Frequencies -- '):
                freq_started = True
            if not freq_started and line.strip().startswith('SCF Done:'):
                spe = float(line.strip().split()[4])
            elif not freq_started and line.strip().startswith('E2('):
                spe_value = line.strip().split()[-1]
                spe = float(spe_value.replace('D','E'))
            elif not freq_started and 'EUMP2 =' in line.strip():
                spe = float((line.strip().split()[5]).replace('D', 'E'))
            elif 'CCSD(T)=' in line.strip():
                raw = line.strip().split('CCSD(T)=')[1].split('\\')[0].split()[0]
                spe = float(raw.replace('D', 'E'))
            elif 'CCSD=' in line.strip() and 'CCSD(T)' not in line.strip():
                raw = line.strip().split('CCSD=')[1].split('\\')[0].split()[0]
                spe = float(raw.replace('D', 'E'))
            elif line.strip().startswith('Counterpoise corrected energy'):
                spe = float(line.strip().split()[4])
            # For ONIOM calculations use the extrapolated value rather than SCF value
            elif "ONIOM: extrapolated energy" in line.strip():
                spe = (float(line.strip().split()[4]))
            # For G4 calculations look for G4 energies (Gaussian16a bug prints G4(0 K) as DE(HF)) --Brian modified to work for G16c-where bug is fixed.
            elif line.strip().startswith('G4(0 K)'):
                spe = float(line.strip().split()[2])
                spe -= zero_point_corr_G4 #Remove G4 ZPE
            elif line.strip().startswith('E(ZPE)='): #Get G4 ZPE
                zero_point_corr_G4 = float(line.strip().split()[1])
            # For TD calculations look for SCF energies of the first excited state
            elif 'E(TD-HF/TD-DFT)' in line.strip():
                spe = float(line.strip().split()[4])
            # For Semi-empirical or Molecular Mechanics calculations
            elif "Energy= " in line.strip() and "Predicted" not in line.strip() and "Thermal" not in line.strip() and "G4" not in line.strip():
                spe = (float(line.strip().split()[1]))
            elif "Gaussian" in line and "Revision" in line and repeated_link1 == 0:
                for i in range(len(line.strip(",").split(",")) - 1):
                    line.strip(",").split(",")[i]
                    version_program += line.strip(",").split(",")[i]
                    repeated_link1 = 1
                version_program = version_program[1:]
            # Charge and multiplicity
            elif 'Charge' in line and 'Multiplicity' in line:
                try:
                    parts = line.split()
                    charge = int(parts[parts.index('Charge') + 2])
                    multiplicity = int(parts[parts.index('Multiplicity') + 2])
                except (ValueError, IndexError):
                    pass
        elif program == "Orca":
            if 'Program Version' in line.strip():
                version_program = "ORCA version " + line.split()[2]
            if line.strip().startswith('FINAL SINGLE POINT ENERGY'):
                spe = float(line.strip().split()[-1])
            if "Total Charge" in line.strip() and "...." in line.strip():
                charge = int(line.strip("=").split()[-1])
            if "Multiplicity" in line.strip() and "...." in line.strip():
                multiplicity = int(line.strip("=").split()[-1])
        elif program == "NWChem":
            if 'nwchem branch' in line.strip():
                version_program = "NWChem version " + line.split()[3]
            if line.strip().startswith('Total DFT energy'):
                spe = float(line.strip().split()[4])
            if "charge" in line.strip():
                charge = int(line.strip().split()[-1])
            if "mult " in line.strip():
                multiplicity = int(line.strip().split()[-1])
            elif "Spin multiplicity:" in line:
                try:
                    multiplicity = int(line.split(":", 1)[1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
        elif program == "xtb":
            stripped = line.strip()
            if 'xtb version' in stripped and not version_program:
                parts = stripped.split()
                try:
                    version_program = "xtb version " + parts[parts.index('version') + 1]
                except (ValueError, IndexError):
                    pass
            if stripped.startswith(':: total energy'):
                parts = stripped.split()
                try:
                    spe = float(parts[3])
                except (ValueError, IndexError):
                    pass
            if 'net charge' in stripped and stripped.startswith(':'):
                parts = stripped.split()
                try:
                    charge = int(parts[parts.index('charge') + 1])
                except (ValueError, IndexError):
                    pass
            if 'unpaired electrons' in stripped and stripped.startswith(':'):
                parts = stripped.split()
                try:
                    multiplicity = int(parts[parts.index('electrons') + 1]) + 1
                except (ValueError, IndexError):
                    pass

    # Solvation model and empirical dispersion detection
    if 'Gaussian' in version_program.strip():
        for i, line in enumerate(data):
            if '#' in line.strip() and a == 0:
                for j, line in enumerate(data[i:i + 10]):
                    if '--' in line.strip():
                        a = a + 1
                        break
                    if a != 0:
                        break
                    else:
                        for k in range(len(line.strip().split("\n"))):
                            line.strip().split("\n")[k]
                            keyword_line += line.strip().split("\n")[k]
        keyword_line = keyword_line.lower()
        if 'scrf' not in keyword_line.strip():
            solvation_model = "gas phase"
        else:
            start_scrf = keyword_line.strip().find('scrf') + 4
            if '(' in keyword_line[start_scrf:start_scrf + 4]:
                start_scrf += keyword_line[start_scrf:start_scrf + 4].find('(') + 1
                end_scrf = keyword_line.find(")", start_scrf)
                display_solvation_model = "scrf=(" + ','.join(
                    keyword_line[start_scrf:end_scrf].lower().split(',')) + ')'
                sorted_solvation_model = "scrf=(" + ','.join(
                    sorted(keyword_line[start_scrf:end_scrf].lower().split(','))) + ')'
            else:
                if ' = ' in keyword_line[start_scrf:start_scrf + 4]:
                    start_scrf += keyword_line[start_scrf:start_scrf + 4].find(' = ') + 3
                elif ' =' in keyword_line[start_scrf:start_scrf + 4]:
                    start_scrf += keyword_line[start_scrf:start_scrf + 4].find(' =') + 2
                elif '=' in keyword_line[start_scrf:start_scrf + 4]:
                    start_scrf += keyword_line[start_scrf:start_scrf + 4].find('=') + 1
                end_scrf = keyword_line.find(" ", start_scrf)
                if end_scrf == -1:
                    display_solvation_model = "scrf=(" + ','.join(keyword_line[start_scrf:].lower().split(',')) + ')'
                    sorted_solvation_model = "scrf=(" + ','.join(
                        sorted(keyword_line[start_scrf:].lower().split(','))) + ')'
                else:
                    display_solvation_model = "scrf=(" + ','.join(
                        keyword_line[start_scrf:end_scrf].lower().split(',')) + ')'
                    sorted_solvation_model = "scrf=(" + ','.join(
                        sorted(keyword_line[start_scrf:end_scrf].lower().split(','))) + ')'
        if solvation_model != "gas phase":
            solvation_model = [sorted_solvation_model, display_solvation_model]
        empirical_dispersion = ''
        if keyword_line.strip().find('empiricaldispersion') == -1 and keyword_line.strip().find(
                'emp=') == -1 and keyword_line.strip().find('emp =') == -1 and keyword_line.strip().find('emp(') == -1:
            empirical_dispersion = "No empirical dispersion detected"
        elif keyword_line.strip().find('empiricaldispersion') > -1:
            start_emp_disp = keyword_line.strip().find('empiricaldispersion') + 19
            if '(' in keyword_line[start_emp_disp:start_emp_disp + 4]:
                start_emp_disp += keyword_line[start_emp_disp:start_emp_disp + 4].find('(') + 1
                end_emp_disp = keyword_line.find(")", start_emp_disp)
                empirical_dispersion = 'empiricaldispersion=(' + ','.join(
                    sorted(keyword_line[start_emp_disp:end_emp_disp].lower().split(','))) + ')'
            else:
                if ' = ' in keyword_line[start_emp_disp:start_emp_disp + 4]:
                    start_emp_disp += keyword_line[start_emp_disp:start_emp_disp + 4].find(' = ') + 3
                elif ' =' in keyword_line[start_emp_disp:start_emp_disp + 4]:
                    start_emp_disp += keyword_line[start_emp_disp:start_emp_disp + 4].find(' =') + 2
                elif '=' in keyword_line[start_emp_disp:start_emp_disp + 4]:
                    start_emp_disp += keyword_line[start_emp_disp:start_emp_disp + 4].find('=') + 1
                end_emp_disp = keyword_line.find(" ", start_emp_disp)
                if end_emp_disp == -1:
                    empirical_dispersion = "empiricaldispersion=(" + ','.join(
                        sorted(keyword_line[start_emp_disp:].lower().split(','))) + ')'
                else:
                    empirical_dispersion = "empiricaldispersion=(" + ','.join(
                        sorted(keyword_line[start_emp_disp:end_emp_disp].lower().split(','))) + ')'
        elif keyword_line.strip().find('emp=') > -1 or keyword_line.strip().find(
                'emp =') > -1 or keyword_line.strip().find('emp(') > -1:
            # Check for temp keyword
            temp, emp_e, emp_p = False, False, False
            check_temp = keyword_line.strip().find('emp=')
            start_emp_disp = keyword_line.strip().find('emp=')
            if check_temp == -1:
                check_temp = keyword_line.strip().find('emp =')
                start_emp_disp = keyword_line.strip().find('emp =')
            if check_temp == -1:
                check_temp = keyword_line.strip().find('emp=(')
                start_emp_disp = keyword_line.strip().find('emp(')
            check_temp += -1
            if keyword_line[check_temp].lower() == 't':
                temp = True  # Look for a new one
                if keyword_line.strip().find('emp=', check_temp + 5) > -1:
                    emp_e = True
                    start_emp_disp = keyword_line.strip().find('emp=', check_temp + 5) + 3
                elif keyword_line.strip().find('emp =', check_temp + 5) > -1:
                    emp_e = True
                    start_emp_disp = keyword_line.strip().find('emp =', check_temp + 5) + 3
                elif keyword_line.strip().find('emp(', check_temp + 5) > -1:
                    emp_p = True
                    start_emp_disp = keyword_line.strip().find('emp(', check_temp + 5) + 3
                else:
                    empirical_dispersion = "No empirical dispersion detected"
            else:
                start_emp_disp += 3
            if (temp and emp_e) or (not temp and keyword_line.strip().find('emp=') > -1) or (
                    not temp and keyword_line.strip().find('emp =')):
                if '(' in keyword_line[start_emp_disp:start_emp_disp + 4]:
                    start_emp_disp += keyword_line[start_emp_disp:start_emp_disp + 4].find('(') + 1
                    end_emp_disp = keyword_line.find(")", start_emp_disp)
                    empirical_dispersion = 'empiricaldispersion=(' + ','.join(
                        sorted(keyword_line[start_emp_disp:end_emp_disp].lower().split(','))) + ')'
                else:
                    if ' = ' in keyword_line[start_emp_disp:start_emp_disp + 4]:
                        start_emp_disp += keyword_line[start_emp_disp:start_emp_disp + 4].find(' = ') + 3
                    elif ' =' in keyword_line[start_emp_disp:start_emp_disp + 4]:
                        start_emp_disp += keyword_line[start_emp_disp:start_emp_disp + 4].find(' =') + 2
                    elif '=' in keyword_line[start_emp_disp:start_emp_disp + 4]:
                        start_emp_disp += keyword_line[start_emp_disp:start_emp_disp + 4].find('=') + 1
                    end_emp_disp = keyword_line.find(" ", start_emp_disp)
                    if end_emp_disp == -1:
                        empirical_dispersion = "empiricaldispersion=(" + ','.join(
                            sorted(keyword_line[start_emp_disp:].lower().split(','))) + ')'
                    else:
                        empirical_dispersion = "empiricaldispersion=(" + ','.join(
                            sorted(keyword_line[start_emp_disp:end_emp_disp].lower().split(','))) + ')'
            elif (temp and emp_p) or (not temp and keyword_line.strip().find('emp(') > -1):
                start_emp_disp += keyword_line[start_emp_disp:start_emp_disp + 4].find('(') + 1
                end_emp_disp = keyword_line.find(")", start_emp_disp)
                empirical_dispersion = 'empiricaldispersion=(' + ','.join(
                    sorted(keyword_line[start_emp_disp:end_emp_disp].lower().split(','))) + ')'
    if 'ORCA' in version_program.strip():
        keyword_line_1 = "gas phase"
        keyword_line_2 = ''
        keyword_line_3 = ''
        for i, line in enumerate(data):
            if 'CPCM SOLVATION MODEL' in line.strip():
                keyword_line_1 = "CPCM,"
            if 'SMD CDS free energy correction energy' in line.strip():
                keyword_line_2 = "SMD,"
            if "Solvent:              " in line.strip():
                keyword_line_3 = line.strip().split()[-1]
        solvation_model = keyword_line_1 + keyword_line_2 + keyword_line_3
        empirical_dispersion1 = 'No empirical dispersion detected'
        empirical_dispersion2 = ''
        empirical_dispersion3 = ''
        for i, line in enumerate(data):
            if keyword_line.strip().find('DFT DISPERSION CORRECTION') > -1:
                empirical_dispersion1 = ''
            if keyword_line.strip().find('DFTD3') > -1:
                empirical_dispersion2 = "D3"
            if keyword_line.strip().find('USING zero damping') > -1:
                empirical_dispersion3 = ' with zero damping'
        empirical_dispersion = empirical_dispersion1 + empirical_dispersion2 + empirical_dispersion3
    if 'NWChem' in version_program.strip():
        empirical_dispersion1 = 'No empirical dispersion detected'
        empirical_dispersion2 = ''
        empirical_dispersion3 = ''
        for i, line in enumerate(data):
            if keyword_line.strip().find('Dispersion correction') > -1:
                empirical_dispersion1 = ''
            if keyword_line.strip().find('disp vdw 3') > -1:
                empirical_dispersion2 = "D3"
            if keyword_line.strip().find('disp vdw 4') > -1:
                empirical_dispersion2 = "D3BJ"
        empirical_dispersion = empirical_dispersion1 + empirical_dispersion2 + empirical_dispersion3
    if 'xtb' in version_program.strip():
        solvation_model = 'gas phase'
        for line in data:
            stripped = line.strip()
            if 'program call' in stripped and ':' in stripped:
                solvation_model = _xtb_solvation_from_call(stripped.split(':', 1)[1].strip())
                break
        empirical_dispersion = 'No empirical dispersion detected'

    return spe, program, version_program, solvation_model, file, charge, empirical_dispersion, multiplicity

def sp_cpu(file):
    """Read single-point output for CPU time.

    Delegates to ``parse_qcdata(file).cpu`` so the SPC CPU is identical
    to what a standalone parse of the same file would report. This
    guarantees that ``goodvibes <parents> --spc <suffix>`` totals
    correctly reflect the SPC files' CPU — including program-specific
    accumulation (Gaussian sums multi-step ``Job cpu time`` lines) and
    nprocs scaling (ORCA reports wall time only; the parser scales by
    parallel-MPI process count to estimate effective CPU).
    """
    candidates = [
        os.path.splitext(file)[0] + '.log',
        os.path.splitext(file)[0] + '.out',
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return parse_qcdata(cand).cpu
    raise ValueError("File {} does not exist".format(file))



# Tokens on the ORCA ``!`` keyword line that are NOT method or basis set.
_ORCA_SKIP_TOKENS = {
    # Job types
    'OPT', 'OPTTS', 'FREQ', 'NUMFREQ', 'NUMHESS', 'ANFREQ', 'SP', 'NMR',
    'VPT2', 'SCANTS', 'NEB-TS', 'FAST-NEB-TS', 'NEB-CI',
    # Solvation
    'CPCM', 'SMD',
    # Dispersion (standalone keywords, not embedded in functional name)
    'D3BJ', 'D3', 'D4', 'D3ZERO',
    # SCF convergence
    'TIGHTSCF', 'NORMALSCF', 'LOOSESCF', 'VERYTIGHTSCF', 'EXTREMESCF',
    'SLOWCONV', 'NOCONV',
    # RI approximations
    'RIJCOSX', 'RI', 'RIJK', 'RIJONX', 'AUTOAUX', 'RIJDX',
    # Grid
    'GRID4', 'GRID5', 'GRID6', 'GRID7',
    'FINALGRID4', 'FINALGRID5', 'FINALGRID6', 'FINALGRID7',
    # Output verbosity
    'PRINT', 'MINIPRINT', 'LARGEPRINT', 'PRINTBASIS', 'PRINTMOS', 'NOPRINT',
    # Reference (DFT)
    'UKS', 'RKS',
    # Other
    'NOFROZENCORE', 'FROZENCORE', 'MOREAD', 'XYZFILE',
    'SCALFREQ', 'QUASIRRHO',
}

# Reference keywords that map to HF when no other method is present
_ORCA_HF_REFS = {'UHF', 'RHF', 'ROHF'}

# Regex for recognizing basis set tokens (case-insensitive matching)
_BASIS_PATTERNS = [
    re.compile(r'^\d+-\d+\+*G', re.IGNORECASE),          # Pople: 6-31G, 6-311+G, etc.
    re.compile(r'^(AUG-)?CC-PV[DTQR56]Z$', re.IGNORECASE),  # Dunning
    re.compile(r'^DEF2-', re.IGNORECASE),                  # Karlsruhe def2-
    re.compile(r'^(SV|TZV|QZV)P?P?$', re.IGNORECASE),     # Ahlrichs without def2
    re.compile(r'^SV\(P\)$', re.IGNORECASE),               # SV(P)
]


def _is_basis_set(token):
    """Return True if *token* looks like a basis set name."""
    return any(pat.match(token) for pat in _BASIS_PATTERNS)


def _is_auxiliary_basis(token):
    """Return True if *token* is an auxiliary basis set (e.g. cc-pVTZ/C, def2/J)."""
    upper = token.upper()
    if '/' in upper:
        suffix = upper.rsplit('/', 1)[1]
        if suffix in ('C', 'J', 'JK', 'JKFIT', 'CFIT'):
            return True
        # def2/J, def2/JK style
        if upper.startswith('DEF2/'):
            return True
    return False


def _parse_orca_lot(data):
    """Extract (method, basis_set) from ORCA output file lines.

    Parses the ``!`` keyword line(s) in the ORCA input section and classifies
    tokens into method, basis set, or known keywords to skip.  Falls back to
    the ``Your calculation utilizes the basis:`` line when no basis set is
    found on the ``!`` line.
    """
    kw_tokens = []
    fallback_basis = None

    for line in data:
        stripped = line.strip()

        # Collect tokens from ORCA input keyword lines: |  N> ! ...
        # The ``!`` must be the first non-whitespace after ``>`` to avoid
        # matching ``!`` inside comments on other input lines.
        if '|' in line and '>' in stripped:
            after_angle = stripped.split('>', 1)[1].lstrip()
            if after_angle.startswith('!'):
                after_bang = after_angle.split('!', 1)[1]
                kw_tokens.extend(after_bang.split())

        # Fallback basis from ORCA output section
        if 'Your calculation utilizes the basis:' in stripped and fallback_basis is None:
            fallback_basis = stripped.split(':')[-1].strip()

    # Classify tokens
    method = None
    basis = None
    hf_ref = None  # Track UHF/RHF/ROHF in case it's the only method

    for token in kw_tokens:
        upper = token.upper()

        # Skip parallelization (pal4, pal8, pal16, ...)
        if re.match(r'^PAL\d+$', upper):
            continue

        # Skip known ORCA keywords
        if upper in _ORCA_SKIP_TOKENS:
            continue

        # Skip auxiliary basis sets
        if _is_auxiliary_basis(token):
            continue

        # Skip GCP(...) tokens
        if upper.startswith('GCP(') or upper == 'GCP':
            continue

        # Skip QM/QM2 compound job keyword
        if upper == 'QM/QM2':
            continue

        # Track HF reference keywords
        if upper in _ORCA_HF_REFS:
            hf_ref = 'HF'
            continue

        # Classify as basis set or method
        if _is_basis_set(token) and basis is None:
            basis = token
        elif method is None:
            method = token

    # Fall back: if only HF reference found and no other method
    if method is None and hf_ref is not None:
        method = hf_ref

    # Fall back to "utilizes the basis" line
    if basis is None and fallback_basis is not None:
        basis = fallback_basis

    return method or 'none', basis or 'none'


def level_of_theory(file):
    """Read output for the level of theory and basis set used."""
    repeated_theory = 0
    with open(file, encoding='utf-8', errors='replace') as f:
        data = f.readlines()
    level, bs = 'none', 'none'

    # Detect ORCA and use dedicated parser
    for line in data:
        if '* O   R   C   A *' in line:
            level, bs = _parse_orca_lot(data)
            return '/'.join([level, bs])
        if 'Gaussian' in line:
            break
        if 'NWChem' in line:
            break
        if 'x T B' in line or 'xtb version' in line:
            for ln in data:
                if 'Hamiltonian' in ln and 'xTB' in ln:
                    parts = ln.split()
                    for tok in parts:
                        if 'xTB' in tok:
                            return tok + '/none'
            return 'GFN2-xTB/none'

    for line in data:
        if line.strip().find('External calculation') > -1:
            level, bs = 'ext', 'ext'
            break
        if '\\Freq\\' in line.strip() and repeated_theory == 0:
            try:
                level, bs = (line.strip().split("\\")[4:6])
                repeated_theory = 1
            except IndexError:
                pass
        elif '|Freq|' in line.strip() and repeated_theory == 0:
            try:
                level, bs = (line.strip().split("|")[4:6])
                repeated_theory = 1
            except IndexError:
                pass
        if '\\SP\\' in line.strip() and repeated_theory == 0:
            try:
                level, bs = (line.strip().split("\\")[4:6])
                repeated_theory = 1
            except IndexError:
                pass
        elif '|SP|' in line.strip() and repeated_theory == 0:
            try:
                level, bs = (line.strip().split("|")[4:6])
                repeated_theory = 1
            except IndexError:
                pass
        if 'DLPNO BASED TRIPLES CORRECTION' in line.strip():
            level = 'DLPNO-CCSD(T)'
        if 'Estimated CBS total energy' in line.strip():
            try:
                bs = ("Extrapol." + line.strip().split()[4])
            except IndexError:
                pass
        # Remove the restricted R or unrestricted U label
        if level[0] in ('R', 'U'):
            level = level[1:]
    level_of_theory = '/'.join([level, bs])
    return level_of_theory

def read_initial(file):
    """At beginning of procedure, read level of theory, solvation model, and check for normal termination"""
    with open(file, encoding='utf-8', errors='replace') as f:
        data = f.readlines()
    level, bs, program, keyword_line = 'none', 'none', 'none', 'none'
    solvation_model = "gas phase"
    progress, orientation = 'Incomplete', 'Input'
    a, repeated_theory = 0, 0
    no_grid = True
    dft_used = 'F'
    grid_lookup = {1: 'sg1', 2: 'coarse', 4: 'fine', 5: 'ultrafine', 7: 'superfine'}

    for line in data:
        # Determine program
        if "Gaussian" in line:
            program = "Gaussian"
            break
        if "* O   R   C   A *" in line:
            program = "Orca"
            break
        if "NWChem" in line:
            program = "NWChem"
            break
        if "x T B" in line or "xtb version" in line:
            program = "xtb"
            break
        if "You are running Q-Chem" in line or "Welcome to Q-Chem" in line:
            program = "QChem"
            break
    # ASE extxyz: program=ase appears in line 2 (the comment line)
    if program == 'none' and len(data) >= 2 and 'program=ase' in data[1]:
        program = 'ase'
    for line in data:
        # Grab pertinent information from file
        if line.strip().find('External calculation') > -1:
            level, bs = 'ext', 'ext'
        if line.strip().find('Standard orientation:') > -1:
            orientation = 'Standard'
        if line.strip().find('IExCor=') > -1 and no_grid:
            try:
                dft_used = line.split('=')[2].split()[0]
                _ = grid_lookup[int(dft_used)]
                no_grid = False
            except (KeyError, ValueError, IndexError):
                pass
        if '\\Freq\\' in line.strip() and repeated_theory == 0:
            try:
                level, bs = (line.strip().split("\\")[4:6])
                repeated_theory = 1
            except IndexError:
                pass
        elif '|Freq|' in line.strip() and repeated_theory == 0:
            try:
                level, bs = (line.strip().split("|")[4:6])
                repeated_theory = 1
            except IndexError:
                pass
        if '\\SP\\' in line.strip() and repeated_theory == 0:
            try:
                level, bs = (line.strip().split("\\")[4:6])
                repeated_theory = 1
            except IndexError:
                pass
        elif '|SP|' in line.strip() and repeated_theory == 0:
            try:
                level, bs = (line.strip().split("|")[4:6])
                repeated_theory = 1
            except IndexError:
                pass
        if 'DLPNO BASED TRIPLES CORRECTION' in line.strip():
            level = 'DLPNO-CCSD(T)'
        if 'Estimated CBS total energy' in line.strip():
            try:
                bs = ("Extrapol." + line.strip().split()[4])
            except IndexError:
                pass
        # Remove the restricted R or unrestricted U label
        if level[0] in ('R', 'U'):
            level = level[1:]

    #NWChem specific parsing
    if program == 'NWChem':
        keyword_line_1 = "gas phase"
        keyword_line_2 = ''
        keyword_line_3 = ''
        for i, line in enumerate(data):
            if line.strip().startswith("xc "):
                level=line.strip().split()[1]
            if line.strip().startswith("* library "):
                bs = line.strip().replace("* library ",'')
            #need to update these tags for NWChem solvation later
            if 'CPCM SOLVATION MODEL' in line.strip():
                keyword_line_1 = "CPCM,"
            if 'SMD CDS free energy correction energy' in line.strip():
                keyword_line_2 = "SMD,"
            if "Solvent:              " in line.strip():
                keyword_line_3 = line.strip().split()[-1]
            #need to update NWChem keyword for error calculation
            if 'Total times' in line:
                progress = 'Normal'
            elif 'error termination' in line:
                progress = 'Error'
        solvation_model = keyword_line_1 + keyword_line_2 + keyword_line_3

    # Grab solvation models - Gaussian files
    if program == 'Gaussian':
        for i, line in enumerate(data):
            if '#' in line.strip() and a == 0:
                for j, line in enumerate(data[i:i + 10]):
                    if '--' in line.strip():
                        a = a + 1
                        break
                    if a != 0:
                        break
                    else:
                        for k in range(len(line.strip().split("\n"))):
                            keyword_line += line.strip().split("\n")[k]
            if 'Normal termination' in line:
                progress = 'Normal'
            elif 'Error termination' in line:
                progress = 'Error'
        keyword_line = keyword_line.lower()
        if 'scrf' not in keyword_line.strip():
            solvation_model = "gas phase"
        else:
            start_scrf = keyword_line.strip().find('scrf') + 5
            if start_scrf >= len(keyword_line):
                # Bare `scrf` with no body — Gaussian defaults to PCM/water.
                solvation_model = "scrf"
            elif keyword_line[start_scrf] == "(":
                end_scrf = keyword_line.find(")", start_scrf)
                solvation_model = "scrf=" + keyword_line[start_scrf:end_scrf]
                if solvation_model[-1] != ")":
                    solvation_model = solvation_model + ")"
            else:
                start_scrf2 = keyword_line.strip().find('scrf') + 4
                if keyword_line.find(" ", start_scrf) > -1:
                    end_scrf = keyword_line.find(" ", start_scrf)
                else:
                    end_scrf = len(keyword_line)
                if keyword_line[start_scrf2] == "(":
                    solvation_model = "scrf=(" + keyword_line[start_scrf:end_scrf]
                    if solvation_model[-1] != ")":
                        solvation_model = solvation_model + ")"
                else:
                    if keyword_line.find(" ", start_scrf) > -1:
                        end_scrf = keyword_line.find(" ", start_scrf)
                    else:
                        end_scrf = len(keyword_line)
                    solvation_model = "scrf=" + keyword_line[start_scrf:end_scrf]
    # xtb parsing for level of theory and solvation model
    elif program == 'xtb':
        # Hamiltonian (GFN0/1/2-xTB) → "level"; xtb has no basis set in the QC sense
        for ln in data:
            if 'Hamiltonian' in ln and 'xTB' in ln:
                for tok in ln.split():
                    if 'xTB' in tok:
                        level = tok
                        break
                if level != 'none':
                    break
        bs = 'none'
        # Solvation from program-call line
        solvation_model = 'gas phase'
        for ln in data:
            stripped = ln.strip()
            if 'program call' in stripped and ':' in stripped:
                solvation_model = _xtb_solvation_from_call(stripped.split(':', 1)[1].strip())
                break
        # Termination status
        for ln in data:
            if '[ERROR]' in ln or 'fatal error' in ln.lower():
                progress = 'Error'
                break
            if 'finished run on' in ln:
                progress = 'Normal'

    # Q-Chem: parse $rem METHOD/BASIS from the input echo, solvation from
    # the run banner, and termination from "Thank you very much" footer.
    elif program == 'QChem':
        in_rem = False
        method, basis = 'none', 'none'
        cpcm = smd = solvent_name = ''
        for line in data:
            s = line.strip()
            if s.lower() == '$rem':
                in_rem = True
                continue
            if in_rem:
                if s == '$end' or s.startswith('$'):
                    in_rem = False
                else:
                    parts = s.split()
                    if len(parts) >= 2:
                        key = parts[0].lower()
                        if key == 'method' and method == 'none':
                            method = parts[1]
                        elif key == 'basis' and basis == 'none':
                            basis = parts[1]
            if 'Using C-PCM' in s or 'C-PCM dielectric factor' in s:
                cpcm = 'CPCM'
            elif 'Using the SMD solvation model' in s:
                smd = 'SMD'
            elif s.startswith('Solvent:'):
                try:
                    solvent_name = s.split(':', 1)[1].strip()
                except IndexError:
                    pass
            if 'Thank you very much for using Q-Chem' in line:
                progress = 'Normal'
            elif 'Q-Chem fatal error' in line or 'fatal error occurred' in line:
                progress = 'Error'
        level, bs = method, basis
        if smd:
            solvation_model = 'SMD,' + solvent_name if solvent_name else 'SMD'
        elif cpcm:
            solvation_model = 'CPCM,' + solvent_name if solvent_name else 'CPCM'

    # ASE extxyz: pull metadata directly from the comment-line key=values
    elif program == 'ase':
        info = _parse_extxyz_comment(data[1]) if len(data) >= 2 else {}
        lot = info.get('level_of_theory', '')
        if '/' in lot:
            level, bs = lot.split('/', 1)
        elif lot:
            level, bs = lot, 'none'
        else:
            level, bs = 'none', 'none'
        solvation_model = info.get('solvation_model', 'gas phase') or 'gas phase'
        try:
            float(info.get('scf_energy', ''))
            progress = 'Normal'
        except ValueError:
            progress = 'Incomplete'

    # ORCA parsing for solvation model and level of theory
    elif program == 'Orca':
        level, bs = _parse_orca_lot(data)
        keyword_line_1 = "gas phase"
        keyword_line_2 = ''
        keyword_line_3 = ''
        orca_abort_line = -1
        last_freq_line = -1
        for i, line in enumerate(data):
            if 'CPCM SOLVATION MODEL' in line.strip():
                keyword_line_1 = "CPCM,"
            if 'SMD CDS free energy correction energy' in line.strip():
                keyword_line_2 = "SMD,"
            if "Solvent:              " in line.strip():
                keyword_line_3 = line.strip().split()[-1]
            if 'ORCA TERMINATED NORMALLY' in line:
                progress = 'Normal'
            elif 'error termination' in line:
                progress = 'Error'
            if 'ORCA will abort' in line:
                orca_abort_line = i
            if 'VIBRATIONAL FREQUENCIES' in line and '---' not in line:
                last_freq_line = i
        # Abort after the last freq section means no valid thermo was produced
        if orca_abort_line > last_freq_line:
            progress = 'Error'
        solvation_model = keyword_line_1 + keyword_line_2 + keyword_line_3
    level_of_theory = '/'.join([level, bs])

    return level_of_theory, solvation_model, progress, orientation, dft_used

def gaussian_jobtype(filename):
    """Read the jobtype from a Gaussian archive string."""
    job = ''
    with open(filename, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.strip().find('\\SP\\') > -1:
                job += 'SP'
            if line.strip().find('\\FOpt\\') > -1:
                job += 'GS'
            if line.strip().find('\\FTS\\') > -1:
                job += 'TS'
            if line.strip().find('\\Freq\\') > -1:
                job += 'Freq'
    return job


def parse_gaussian_thermo(file):
    """Parse Gaussian output for all thermochemistry-relevant data.

    Returns QCData with raw frequencies (negative = imaginary, no inversion
    applied). Frequency inversion is a user policy decision handled in
    thermo.py.

    Parameters
    ----------
    file : str
        Path to Gaussian output file.
    """
    qcdata = QCData(file=file, program='Gaussian')

    # Delegate solvation, dispersion, version, charge to parse_data
    (_, _, version_program, solvation_model, _, charge,
     empirical_dispersion, multiplicity) = parse_data(file)
    qcdata.version_program = version_program
    qcdata.solvation_model = solvation_model
    qcdata.empirical_dispersion = empirical_dispersion
    if charge is not None:
        qcdata.charge = charge
    if multiplicity is not None:
        qcdata.multiplicity = multiplicity

    # Job type from archive string
    qcdata.job_type = gaussian_jobtype(file)

    # Read file
    with open(file, encoding='utf-8', errors='replace') as f:
        g_output = f.readlines()

    # Auto-detect G4 composite method
    g4 = any('G4(0 K)' in line for line in g_output)

    # Detect ONIOM
    is_oniom = any('ONIOM: extrapolated energy' in line for line in g_output)
    qcdata.has_oniom = is_oniom

    # --- First pass: find link structure ---
    linkmax = 0
    freqloc = 0
    for line in g_output:
        if 'Normal termination' in line:
            linkmax += 1
        if 'Frequencies --' in line:
            freqloc = linkmax

    # --- Second pass: extract data ---
    link = 0
    frequency_wn = []
    im_frequency_wn = []
    fract_modelsys = []
    per_atom_masses = []  # Gaussian's per-atom masses (respects iso= keyword)
    freq_started = False  # True once we encounter the first "Frequencies --" in this link
    freq_done = False     # True once VPT2 "Recovering" marker is seen (guards against duplicates)

    if freqloc == 0:
        freqloc = len(g_output)

    for i, line in enumerate(g_output):
        # Link counter
        if 'Normal termination' in line:
            link += 1
            if link == freqloc:
                frequency_wn = []
                im_frequency_wn = []
                fract_modelsys = []
                freq_started = False
                freq_done = False

        # Stop after freq link unless G4/composite
        if not g4 and link > freqloc:
            break

        # VPT2/anharmonic: "Recovering previously computed normal modes" means
        # the frequencies about to appear are a duplicate of the harmonic set
        if 'Recovering previously computed normal modes' in line:
            freq_done = True

        # Frequencies
        if not freq_done and line.strip().startswith('Frequencies -- '):
            freq_started = True
            if is_oniom:
                fract_line = g_output[i + 3]
            for j in range(2, 5):
                try:
                    x = float(line.strip().split()[j])
                    if x > 0.0:
                        frequency_wn.append(x)
                        if is_oniom:
                            try:
                                y = float(fract_line.strip().split()[j]) / 100.0
                                y = float('{:.6f}'.format(y))
                                fract_modelsys.append(y)
                            except (IndexError, ValueError):
                                fract_modelsys.append(1.0)
                    elif x < 0.0:
                        im_frequency_wn.append(x)
                except IndexError:
                    pass

        # --- SCF energy (all variants, last one wins) ---
        # Guard against VPT2 displaced geometry energies: once frequencies are
        # read, ignore further SCF Done / E2 / EUMP2 lines (but still allow
        # archive-line energies like CCSD= which appear after frequencies).
        elif not freq_started and line.strip().startswith('SCF Done:'):
            qcdata.scf_energy = float(line.strip().split()[4])
        elif not freq_started and line.strip().startswith('E2('):
            spe_value = line.strip().split()[-1]
            qcdata.scf_energy = float(spe_value.replace('D', 'E'))
        elif line.strip().startswith('Counterpoise corrected energy'):
            qcdata.scf_energy = float(line.strip().split()[4])
        elif not freq_started and 'EUMP2 =' in line.strip():
            qcdata.scf_energy = float((line.strip().split()[5]).replace('D', 'E'))
        elif 'CCSD(T)=' in line.strip():
            raw = line.strip().split('CCSD(T)=')[1].split('\\')[0].split()[0]
            qcdata.scf_energy = float(raw.replace('D', 'E'))
        elif 'CCSD=' in line.strip() and 'CCSD(T)' not in line.strip():
            raw = line.strip().split('CCSD=')[1].split('\\')[0].split()[0]
            qcdata.scf_energy = float(raw.replace('D', 'E'))
        elif 'ONIOM: extrapolated energy' in line.strip():
            qcdata.scf_energy = float(line.strip().split()[4])
        elif line.strip().startswith('G4(0 K)'):
            qcdata.scf_energy = float(line.strip().split()[2])
            if qcdata.zero_point_corr is not None:
                qcdata.scf_energy -= qcdata.zero_point_corr
        elif line.strip().startswith('E(ZPE)='):
            qcdata.zero_point_corr = float(line.strip().split()[1])
        elif 'E(TD-HF/TD-DFT)' in line.strip():
            qcdata.scf_energy = float(line.strip().split()[4])
        elif ('Energy= ' in line.strip()
              and 'Predicted' not in line.strip()
              and 'Thermal' not in line.strip()
              and 'G4' not in line.strip()):
            qcdata.scf_energy = float(line.strip().split()[1])

        # Coordinates: take the LAST "Standard orientation" or "Input orientation" block
        elif 'Standard orientation:' in line or 'Input orientation:' in line:
            atom_nums_tmp = []
            cartesians_tmp = []
            for k in range(i + 5, len(g_output)):
                if '-----' in g_output[k]:
                    break
                parts = g_output[k].split()
                atom_nums_tmp.append(int(parts[1]))
                cartesians_tmp.append([float(parts[3]), float(parts[4]), float(parts[5])])
            qcdata.atom_nums = atom_nums_tmp
            qcdata.cartesians = cartesians_tmp

        # Zero-point correction
        elif line.strip().startswith('Zero-point correction='):
            qcdata.zero_point_corr = float(line.strip().split()[2])

        # Multiplicity
        elif 'Multiplicity' in line.strip():
            try:
                qcdata.multiplicity = int(line.split('=')[-1].strip().split()[0])
            except (ValueError, IndexError):
                qcdata.multiplicity = int(line.split()[-1])

        # Per-atom isotope masses Gaussian printed for this freq job. Captured
        # so we can recompute high-precision rotational constants from the
        # geometry — the "Rotational temperatures" block prints only 3 sig figs,
        # which loses ~3 µHartree in G for heavy or floppy systems.
        elif (line.strip().startswith('Atom')
              and 'has atomic number' in line
              and 'and mass' in line):
            try:
                per_atom_masses.append(float(line.split('and mass')[1].split()[0]))
            except (ValueError, IndexError):
                pass

        # Molecular mass — also marks the end of one per-atom mass block, so
        # subsequent thermo sections (rare; G4 composites) start fresh.
        elif line.strip().startswith('Molecular mass:'):
            qcdata.molecular_mass = float(line.strip().split()[2])

        # Rotational symmetry number
        elif line.strip().startswith('Rotational symmetry number'):
            qcdata.symmno = int((line.strip().split()[3]).split('.')[0])

        # Point group / linearity
        elif line.strip().startswith('Full point group'):
            pg = line.strip().split()[3]
            qcdata.point_group = pg
            if pg in ('D*H', 'C*V'):
                qcdata.linear_mol = True

        # Rotational constants (GHz)
        elif line.strip().startswith('Rotational constants (GHZ):'):
            try:
                parts = line.strip().replace(':', ' ').split()
                qcdata.roconst = [float(parts[3]), float(parts[4]), float(parts[5])]
            except ValueError:
                if '********' in line.strip():
                    qcdata.linear_warning = True
                    parts = line.strip().replace(':', ' ').split()
                    qcdata.roconst = [float(parts[4]), float(parts[5])]

        # Rotational temperatures
        elif line.strip().startswith('Rotational temperature '):
            qcdata.rotemp = [float(line.strip().split()[3])]
        elif line.strip().startswith('Rotational temperatures'):
            try:
                qcdata.rotemp = [float(line.strip().split()[3]),
                                 float(line.strip().split()[4]),
                                 float(line.strip().split()[5])]
            except ValueError:
                if '********' in line.strip():
                    qcdata.linear_warning = True
                    qcdata.rotemp = [float(line.strip().split()[4]),
                                     float(line.strip().split()[5])]

        # CPU time (not elif — checked independently for every line)
        if 'Job cpu time' in line.strip():
            qcdata.cpu = [
                int(line.split()[3]) + qcdata.cpu[0],
                int(line.split()[5]) + qcdata.cpu[1],
                int(line.split()[7]) + qcdata.cpu[2],
                0 + qcdata.cpu[3],
                int(float(line.split()[9]) * 1000.0) + qcdata.cpu[4],
            ]

    qcdata.frequency_wn = frequency_wn
    qcdata.im_frequency_wn = im_frequency_wn
    if is_oniom:
        qcdata.fract_modelsys = fract_modelsys
    if qcdata.atom_nums:
        qcdata.atom_types = [periodictable[n] for n in qcdata.atom_nums]

    # High-precision rotational constants: Gaussian's "Rotational temperatures"
    # printout is only 3 sig figs, which propagates ~3 µHartree of error into G
    # via S_rot for systems with small θ_rot. Recompute MOI from the geometry
    # using the per-atom masses Gaussian itself reported (this respects the
    # iso= keyword), then derive roconst / rotemp at full double precision.
    # G4 composites can print masses for several thermo sections; the final
    # len(cartesians) entries always belong to the freq job we care about.
    if qcdata.cartesians and len(per_atom_masses) >= len(qcdata.cartesians):
        qcdata.per_atom_masses = per_atom_masses[-len(qcdata.cartesians):]
    if (qcdata.cartesians
            and len(per_atom_masses) >= len(qcdata.cartesians)
            and not qcdata.linear_warning
            and len(qcdata.rotemp) == 3):
        masses = np.asarray(per_atom_masses[-len(qcdata.cartesians):], float)
        coords = np.asarray(qcdata.cartesians, float)
        total_mass = float(masses.sum())
        com = (masses[:, None] * coords).sum(axis=0) / total_mass
        r = coords - com
        Ixx = float((masses * (r[:, 1] ** 2 + r[:, 2] ** 2)).sum())
        Iyy = float((masses * (r[:, 0] ** 2 + r[:, 2] ** 2)).sum())
        Izz = float((masses * (r[:, 0] ** 2 + r[:, 1] ** 2)).sum())
        Ixy = float(-(masses * r[:, 0] * r[:, 1]).sum())
        Ixz = float(-(masses * r[:, 0] * r[:, 2]).sum())
        Iyz = float(-(masses * r[:, 1] * r[:, 2]).sum())
        inertia = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
        eigvals = np.sort(np.maximum(np.linalg.eigvalsh(inertia), 0.0))
        if all(I > 1e-3 for I in eigvals):
            HC_OVER_KB = 1.4387768775  # K per cm⁻¹
            roconst_cm = [16.857629 / I for I in eigvals]
            qcdata.roconst = [b * 29.9792458 for b in roconst_cm]
            qcdata.rotemp = [HC_OVER_KB * b for b in roconst_cm]
            qcdata.molecular_mass = total_mass

    _fill_mass_and_rotemp_from_geometry(qcdata)

    return qcdata


def parse_nwchem_thermo(file):
    """Parse NWChem output for all thermochemistry-relevant data.

    Returns QCData with raw frequencies (negative = imaginary, no inversion
    applied).

    Parameters
    ----------
    file : str
        Path to NWChem output file.
    """
    qcdata = QCData(file=file, program='NWChem')

    # Delegate version, charge, multiplicity to parse_data
    try:
        (_, _, version_program, solvation_model, _, charge,
         empirical_dispersion, multiplicity) = parse_data(file)
        qcdata.version_program = version_program
        qcdata.solvation_model = solvation_model
        qcdata.empirical_dispersion = empirical_dispersion
        if charge is not None:
            qcdata.charge = charge
        if multiplicity is not None:
            qcdata.multiplicity = multiplicity
    except (ValueError, IndexError):
        pass

    # NWChem has no archive string for job type; detect from content
    with open(file, encoding='utf-8', errors='replace') as f:
        g_output = f.readlines()

    frequency_wn = []
    im_frequency_wn = []

    for i, line in enumerate(g_output):
        # Frequencies (up to 6 per line)
        if line.strip().startswith('P.Frequency'):
            for j in range(1, 7):
                try:
                    x = float(line.strip().split()[j])
                    if x > 0.0:
                        frequency_wn.append(x)
                    elif x < 0.0:
                        im_frequency_wn.append(x)
                except IndexError:
                    pass

        # SCF energy
        elif line.strip().startswith('Total DFT energy ='):
            qcdata.scf_energy = float(line.strip().split()[4])

        # Zero-point correction
        elif line.strip().startswith('Zero-Point'):
            qcdata.zero_point_corr = float(line.strip().split()[8])

        # Multiplicity — match input-deck echo "mult <N>" or runtime
        # "Spin multiplicity: <N>" (NWChem 7.x).
        elif 'mult ' in line.strip():
            try:
                qcdata.multiplicity = int(line.split()[1])
            except (ValueError, IndexError):
                qcdata.multiplicity = 1
        elif 'Spin multiplicity:' in line:
            try:
                qcdata.multiplicity = int(line.split(":", 1)[1].strip().split()[0])
            except (ValueError, IndexError):
                pass

        # Coordinates: take the LAST "Output coordinates in angstroms" block
        elif 'Output coordinates in angstroms' in line:
            atom_nums_tmp = []
            cartesians_tmp = []
            for k in range(i + 4, len(g_output)):
                cline = g_output[k].strip()
                if cline == '':
                    break
                parts = cline.split()
                atom_nums_tmp.append(int(float(parts[2])))
                cartesians_tmp.append([float(parts[3]), float(parts[4]), float(parts[5])])
            qcdata.atom_nums = atom_nums_tmp
            qcdata.cartesians = cartesians_tmp

        # Molecular mass
        elif line.strip().find('mol. weight') != -1:
            qcdata.molecular_mass = float(line.strip().split()[-1][0:-1])

        # Rotational symmetry number
        elif line.strip().find('symmetry #') != -1:
            qcdata.symmno = int(line.strip().split()[-1][0:-1])

        # Point group / linearity
        elif line.strip().find('symmetry detected') != -1:
            pg = line.strip().split()[0]
            qcdata.point_group = pg
            if pg in ('D*H', 'C*V'):
                qcdata.linear_mol = True

        # Version (fallback if parse_data failed)
        elif 'nwchem branch' in line.strip() and not qcdata.version_program:
            qcdata.version_program = 'NWChem version ' + line.split()[3]

        # Rotational constants (convert cm⁻¹ to GHz) and temperatures
        elif line.strip().startswith('A=') or line.strip().startswith('B=') or line.strip().startswith('C='):
            letter = line.strip()[0]
            h = {'A': 0, 'B': 1, 'C': 2}[letter]
            qcdata.roconst[h] = float(line.strip().split()[1]) * 29.9792458
            qcdata.rotemp[h] = float(line.strip().split()[4])

        # CPU time
        if 'Total times' in line.strip():
            secs = float(line.strip().split()[3][0:-1])
            qcdata.cpu = [0, 0, 0, secs, 0.0]

    # Determine job type from content
    if len(frequency_wn) > 0 or len(im_frequency_wn) > 0:
        qcdata.job_type = 'Freq'
    else:
        qcdata.job_type = 'SP'

    qcdata.frequency_wn = frequency_wn
    qcdata.im_frequency_wn = im_frequency_wn
    if qcdata.atom_nums:
        qcdata.atom_types = [periodictable[n] for n in qcdata.atom_nums]

    _fill_mass_and_rotemp_from_geometry(qcdata)

    return qcdata


def _scale_cpu(cpu, factor):
    """Multiply a [days, hours, mins, secs, msecs] tuple by an integer factor.

    Used by the ORCA parser to convert wall time × nprocs → effective CPU.
    """
    total_ms = (
        cpu[0] * 86_400_000
        + cpu[1] * 3_600_000
        + cpu[2] * 60_000
        + cpu[3] * 1_000
        + cpu[4]
    ) * factor
    days, total_ms = divmod(total_ms, 86_400_000)
    hours, total_ms = divmod(total_ms, 3_600_000)
    mins, total_ms = divmod(total_ms, 60_000)
    secs, msecs = divmod(total_ms, 1_000)
    return [days, hours, mins, secs, msecs]


def parse_orca_thermo(file):
    """Parse ORCA output for all thermochemistry-relevant data.

    Uses native line-by-line parsing.

    Parameters
    ----------
    file : str
        Path to ORCA output file.
    """
    qcdata = QCData(file=file, program='Orca')

    with open(file, encoding='utf-8', errors='replace') as f:
        output = f.readlines()

    frequency_wn = []
    im_frequency_wn = []
    in_freq_section = False
    solvation_type = ''
    solvent_name = ''
    applied_freq_scale_factor = 1.0
    _has_opt = False
    _has_ts = False
    _has_freq_kw = False
    _orca_abort_line = -1
    _last_freq_line = -1
    # Every "FINAL SINGLE POINT ENERGY" in the file (full precision), and
    # the electronic energy ORCA itself reports inside its thermochemistry
    # block. The latter is the energy H/G are built on; we use it to pick
    # the right SCF energy so a linked single point printed *after* the
    # frequencies doesn't silently replace the opt/freq-level energy
    # (issue #101). Resolved after the loop.
    _all_fspe = []
    _thermo_elec_energy = None

    for i, line in enumerate(output):
        stripped = line.strip()

        # --- Frequency section state machine ---
        if 'VIBRATIONAL FREQUENCIES' in stripped and '---' not in stripped:
            in_freq_section = True
            frequency_wn = []
            im_frequency_wn = []
            _last_freq_line = i
        elif in_freq_section and 'cm**-1' in stripped:
            parts = stripped.split()
            try:
                freq_val = float(parts[1])
                if freq_val > 0.0:
                    frequency_wn.append(freq_val)
                elif freq_val < 0.0:
                    im_frequency_wn.append(freq_val)
                # freq_val == 0.0 → skip (translational/rotational modes)
            except (IndexError, ValueError):
                pass
        elif in_freq_section:
            if stripped == '' and (frequency_wn or im_frequency_wn):
                in_freq_section = False

        # --- Main field parsing ---
        # Coordinates: take the LAST "CARTESIAN COORDINATES (ANGSTROEM)" block
        if 'CARTESIAN COORDINATES (ANGSTROEM)' in stripped:
            atom_nums_tmp = []
            atom_types_tmp = []
            cartesians_tmp = []
            for k in range(i + 2, len(output)):
                cline = output[k].strip()
                if cline == '' or '---' in cline:
                    break
                parts = cline.split()
                elem = parts[0]
                atom_types_tmp.append(elem)
                atom_nums_tmp.append(element_id(elem, num=True))
                cartesians_tmp.append([float(parts[1]), float(parts[2]), float(parts[3])])
            qcdata.atom_nums = atom_nums_tmp
            qcdata.atom_types = atom_types_tmp
            qcdata.cartesians = cartesians_tmp

        # SCF energy: collect every occurrence; the right one is chosen
        # after the loop (see _resolve below). Don't commit to "last wins"
        # here — a linked SPC after the freqs would clobber the freq-level
        # energy that the thermochemistry is built on (issue #101).
        elif stripped.startswith('FINAL SINGLE POINT ENERGY'):
            try:
                _all_fspe.append(float(stripped.split()[-1]))
            except (ValueError, IndexError):
                pass

        # Electronic energy as reported in ORCA's own thermochemistry
        # block ("Electronic energy   ...   -123.45678901 Eh"). This is
        # unambiguously the energy paired with the printed H and G, i.e.
        # the opt/freq level, never a later single point.
        elif stripped.startswith('Electronic energy') and '...' in stripped and stripped.endswith('Eh'):
            try:
                _thermo_elec_energy = float(stripped.split()[-2])
            except (ValueError, IndexError):
                pass

        # Collect input keywords for job type detection (handled after loop)
        elif '|' in line and '>' in stripped and '!' in stripped:
            kw = stripped.split('!', 1)[1].upper()
            if 'OPTTS' in kw:
                _has_ts = True
            elif 'OPT' in kw:
                _has_opt = True
            if 'FREQ' in kw:
                _has_freq_kw = True

        # Charge
        elif 'Total Charge' in stripped and '....' in stripped:
            qcdata.charge = int(stripped.split()[-1])

        # Multiplicity
        elif 'Multiplicity' in stripped and 'Mult' in stripped and '....' in stripped:
            qcdata.multiplicity = int(stripped.split()[-1])

        # Program version
        elif 'Program Version' in stripped:
            qcdata.version_program = 'ORCA version ' + line.split()[2]

        # Zero-point energy (Eh)
        elif stripped.startswith('Zero point energy') and '...' in stripped:
            parts = stripped.split()
            try:
                dot_idx = parts.index('...')
                qcdata.zero_point_corr = float(parts[dot_idx + 1])
            except (ValueError, IndexError):
                pass

        # Molecular mass (AMU)
        elif 'Total Mass' in stripped and '...' in stripped:
            parts = stripped.split()
            try:
                dot_idx = parts.index('...')
                qcdata.molecular_mass = float(parts[dot_idx + 1])
            except (ValueError, IndexError):
                pass

        # Rotational constants in cm⁻¹ → convert to GHz and rotational temps
        elif stripped.startswith('Rotational constants in cm-1:'):
            rparts = stripped.split(':')[1].split()
            try:
                roconst_cm = [float(rparts[0]), float(rparts[1]), float(rparts[2])]
                qcdata.roconst = [b * 29.9792458 for b in roconst_cm]
                # Rotational temperature: T_rot = h*c*B/k_B where B in cm⁻¹
                HC_OVER_KB = 1.4387768775  # K per cm⁻¹
                qcdata.rotemp = [HC_OVER_KB * b for b in roconst_cm]
            except (IndexError, ValueError):
                pass

        # Point group and symmetry number (same line)
        elif 'Point Group:' in stripped and 'Symmetry Number:' in stripped:
            pg_part = stripped.split('Point Group:')[1].split(',')[0].strip()
            qcdata.point_group = pg_part
            # Linear point groups: ORCA 6 prints "C(inf)v"/"D(inf)h",
            # ORCA 5 prints "Cinfv"/"Dinfh" — match both via the "inf" stem.
            if 'inf' in pg_part.lower():
                qcdata.linear_mol = True
            try:
                symm_part = stripped.split('Symmetry Number:')[1].strip()
                qcdata.symmno = int(symm_part)
            except (ValueError, IndexError):
                pass

        # Solvation model detection
        elif 'CPCM SOLVATION MODEL' in stripped:
            solvation_type = 'CPCM'
        elif 'SMD CDS free energy correction energy' in stripped:
            solvation_type = 'SMD'
        elif stripped.startswith('Solvent:') and '...' in stripped:
            solvent_name = stripped.split()[-1]

        # Frequency scaling factor (ORCA pre-applies this to reported frequencies)
        elif stripped.startswith('Scaling factor for frequencies'):
            try:
                applied_freq_scale_factor = float(stripped.split('=')[1].split('(')[0].strip())
            except (IndexError, ValueError):
                pass

        # CPU time. ORCA prints wall time only ("TOTAL RUN TIME:"); we scale
        # by the number of MPI processes (parsed from "Program running with
        # N parallel MPI-processes") to get an effective CPU time, matching
        # how Gaussian/NWChem/xTB report theirs.
        elif stripped.startswith('TOTAL RUN TIME:'):
            parts = stripped.split()
            try:
                cpu_list = [
                    int(parts[3]),   # days
                    int(parts[5]),   # hours
                    int(parts[7]),   # minutes
                    int(parts[9]),   # seconds
                    int(parts[11]),  # msec
                ]
                if qcdata.nprocs > 1:
                    cpu_list = _scale_cpu(cpu_list, qcdata.nprocs)
                qcdata.cpu = cpu_list
            except (IndexError, ValueError):
                pass

        # MPI process count (printed once per task; we keep the first hit).
        elif 'parallel MPI-processes' in stripped and qcdata.nprocs == 1:
            try:
                qcdata.nprocs = int(stripped.split('with')[1].split('parallel')[0].strip())
            except (IndexError, ValueError):
                pass

        # ORCA abort: optimization failed before freq calculation could run.
        if 'ORCA will abort' in stripped:
            _orca_abort_line = i

    # Resolve the SCF energy (issue #101). When ORCA printed a
    # thermochemistry block, its "Electronic energy" is the energy H/G are
    # built on; choose the full-precision FINAL SINGLE POINT ENERGY that
    # matches it. This keeps the opt/freq-level energy even when a linked
    # single point is printed afterwards in the same file (the linked SPC
    # is then applied only on explicit --spc link, which surfaces both the
    # opt energy and the SPC energy). With no thermochemistry block (a
    # single-point-only file), the last energy wins, as before.
    if _thermo_elec_energy is not None and _all_fspe:
        best = min(_all_fspe, key=lambda e: abs(e - _thermo_elec_energy))
        qcdata.scf_energy = best if abs(best - _thermo_elec_energy) < 1e-3 else _thermo_elec_energy
    elif _thermo_elec_energy is not None:
        qcdata.scf_energy = _thermo_elec_energy
    elif _all_fspe:
        qcdata.scf_energy = _all_fspe[-1]

    # If the last abort comes after the last freq section (or there are no freqs),
    # any frequencies/ZPE present are from calc_hess, not the actual freq job — discard.
    if _orca_abort_line > _last_freq_line:
        frequency_wn = []
        im_frequency_wn = []
        qcdata.zero_point_corr = None

    # Assemble solvation model
    if solvation_type:
        if solvent_name:
            qcdata.solvation_model = solvation_type + ',' + solvent_name
        else:
            qcdata.solvation_model = solvation_type

    # Un-scale frequencies if ORCA applied a scaling factor, so that
    # frequency_wn always contains unscaled values (matching Gaussian convention)
    qcdata.applied_freq_scale_factor = applied_freq_scale_factor
    if applied_freq_scale_factor != 1.0 and applied_freq_scale_factor > 0.0:
        frequency_wn = [f / applied_freq_scale_factor for f in frequency_wn]
        im_frequency_wn = [f / applied_freq_scale_factor for f in im_frequency_wn]
    qcdata.frequency_wn = frequency_wn
    qcdata.im_frequency_wn = im_frequency_wn

    # Fallback: parse coordinates from "* xyz charge mult" input block
    # when CARTESIAN COORDINATES (ANGSTROEM) block is absent (e.g., thermo-only re-analysis)
    if not qcdata.atom_nums:
        for i, line in enumerate(output):
            stripped = line.strip()
            if stripped.startswith('|') and '* xyz' in stripped:
                atom_nums_tmp = []
                atom_types_tmp = []
                cartesians_tmp = []
                for k in range(i + 1, len(output)):
                    cline = output[k].strip()
                    # Input lines start with "| N>" prefix
                    if '|' in cline:
                        content = cline.split('>', 1)[-1].strip() if '>' in cline else cline.strip('| ').strip()
                    else:
                        break
                    if content == '*' or content == '' or 'END OF INPUT' in content:
                        break
                    parts = content.split()
                    if len(parts) >= 4:
                        elem = parts[0]
                        atom_types_tmp.append(elem)
                        atom_nums_tmp.append(element_id(elem, num=True))
                        cartesians_tmp.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if atom_nums_tmp:
                    qcdata.atom_nums = atom_nums_tmp
                    qcdata.atom_types = atom_types_tmp
                    qcdata.cartesians = cartesians_tmp
                break

    # For linear molecules, filter rotemp to non-zero values only
    # (linear molecules have one near-zero rotational constant along the axis)
    if qcdata.linear_mol:
        nonzero_rotemp = [t for t in qcdata.rotemp if t > 1e-10]
        if nonzero_rotemp:
            qcdata.rotemp = nonzero_rotemp

    # Determine job type from combined input keywords and output content
    has_freq = _has_freq_kw or len(frequency_wn) > 0 or len(im_frequency_wn) > 0
    if _has_ts:
        qcdata.job_type = 'TSFreq' if has_freq else 'TS'
    elif _has_opt:
        qcdata.job_type = 'GSFreq' if has_freq else 'GS'
    elif has_freq:
        qcdata.job_type = 'Freq'
    else:
        qcdata.job_type = 'SP'

    _fill_mass_and_rotemp_from_geometry(qcdata)

    return qcdata


def _xtb_solvation_from_call(call_line):
    """Parse the xtb ``program call`` line for an implicit-solvation flag.

    Returns a string like 'GBSA,water', 'CPCM-X,dimethylsulfoxide',
    'ALPB,toluene', or 'gas phase'.
    """
    tokens = call_line.split()
    for i, tok in enumerate(tokens):
        if tok in ('-g', '--gbsa') and i + 1 < len(tokens):
            return 'GBSA,' + tokens[i + 1]
        if tok == '--cpcmx' and i + 1 < len(tokens):
            return 'CPCM-X,' + tokens[i + 1]
        if tok == '--alpb' and i + 1 < len(tokens):
            return 'ALPB,' + tokens[i + 1]
    return 'gas phase'


def _xtb_read_xyz_fallback(file):
    """Read the paired .xyz file when the .out has no final-geometry block.

    Returns (atom_nums, atom_types, cartesians) — empty lists if not readable.
    """
    xyz = os.path.splitext(file)[0] + '.xyz'
    if not os.path.exists(xyz):
        return [], [], []
    try:
        with open(xyz, encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
    except OSError:
        return [], [], []
    if len(lines) < 2:
        return [], [], []
    try:
        natoms = int(lines[0].strip())
    except (ValueError, IndexError):
        return [], [], []
    atom_nums, atom_types, cartesians = [], [], []
    for line in lines[2:2 + natoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        elem = parts[0]
        try:
            atom_types.append(elem)
            atom_nums.append(element_id(elem, num=True))
            cartesians.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue
    return atom_nums, atom_types, cartesians


def parse_xtb_thermo(file):
    """Parse xtb output for all thermochemistry-relevant data.

    xtb takes a ``.xyz`` coordinate file plus CLI flags rather than an input
    deck. The parser handles single-point (``xtb file.xyz``), Hessian-only
    (``--hess``), and optimization+Hessian (``--ohess``) jobs, plus implicit
    solvation (GBSA via ``-g``, ALPB via ``--alpb``, CPCM-X via ``--cpcmx``).

    For runs without ``--opt``/``--ohess`` (no optimized-geometry block in the
    output), Cartesians fall back to the paired ``.xyz`` input file.

    Parameters
    ----------
    file : str
        Path to xtb output file.
    """
    qcdata = QCData(file=file, program='xtb')

    with open(file, encoding='utf-8', errors='replace') as f:
        output = f.readlines()

    # Empty / aborted runs: bail out early but record the program.
    if not output or any('[ERROR]' in line for line in output):
        if any('xtb version' in line for line in output):
            for line in output:
                if 'xtb version' in line:
                    parts = line.split()
                    try:
                        qcdata.version_program = 'xtb version ' + parts[parts.index('version') + 1]
                    except (ValueError, IndexError):
                        pass
                    break
        return qcdata

    HC_OVER_KB = 1.4387768775  # K per cm⁻¹

    frequency_wn = []
    im_frequency_wn = []
    last_freq_idx = -1
    program_call = ''

    # First pass: locate the LAST vibrational-frequencies header.  Linear
    # molecules and atoms use the ``vibrational frequencies (cm⁻¹)`` form;
    # everything else uses ``projected vibrational frequencies (cm⁻¹)``.
    for i, line in enumerate(output):
        if 'vibrational frequencies' in line and '(cm' in line:
            last_freq_idx = i

    # Second pass: extract scalar fields and geometry; freq parsing happens once
    for i, line in enumerate(output):
        stripped = line.strip()

        # --- Version
        if 'xtb version' in stripped and not qcdata.version_program:
            parts = stripped.split()
            try:
                qcdata.version_program = 'xtb version ' + parts[parts.index('version') + 1]
            except (ValueError, IndexError):
                pass

        # --- Program call line (for solvation flag parsing)
        elif 'program call' in stripped and ':' in stripped and not program_call:
            program_call = stripped.split(':', 1)[1].strip()

        # --- Charge: ":  net charge   <n>   :"
        elif 'net charge' in stripped and stripped.startswith(':'):
            parts = stripped.split()
            try:
                qcdata.charge = int(parts[parts.index('charge') + 1])
            except (ValueError, IndexError):
                pass

        # --- Multiplicity from "unpaired electrons" (mult = unpaired + 1)
        elif 'unpaired electrons' in stripped and stripped.startswith(':'):
            parts = stripped.split()
            try:
                unpaired = int(parts[parts.index('electrons') + 1])
                qcdata.multiplicity = unpaired + 1
            except (ValueError, IndexError):
                pass

        # --- Electronic SCF energy ":: total energy   <E> Eh ::"
        # Last occurrence wins (final SCC after opt, or sole SCC for SP)
        elif stripped.startswith(':: total energy'):
            parts = stripped.split()
            try:
                qcdata.scf_energy = float(parts[3])
            except (ValueError, IndexError):
                pass

        # --- Zero-point energy
        elif stripped.startswith(':: zero point energy'):
            parts = stripped.split()
            try:
                qcdata.zero_point_corr = float(parts[4])
            except (ValueError, IndexError):
                pass

        # --- Molecular mass: "molecular mass/u    :   <m>"
        elif stripped.startswith('molecular mass/u'):
            parts = stripped.split()
            try:
                qcdata.molecular_mass = float(parts[-1])
            except (ValueError, IndexError):
                pass

        # --- Rotational constants in cm⁻¹: "rotational constants/cm⁻¹ : B1 B2 B3"
        # xtb prints "Infinity" for atoms (no rotation); skip in that case.
        elif stripped.startswith('rotational constants/cm'):
            parts = stripped.split(':', 1)[1].split() if ':' in stripped else stripped.split()
            if 'Infinity' in parts[:3]:
                pass  # atom: leave roconst/rotemp at defaults
            else:
                try:
                    roconst_cm = [float(parts[0]), float(parts[1]), float(parts[2])]
                    qcdata.roconst = [b * 29.9792458 for b in roconst_cm]
                    qcdata.rotemp = [HC_OVER_KB * b for b in roconst_cm]
                except (IndexError, ValueError):
                    pass

        # --- Linear marker (thermo SETUP block)
        elif 'linear?' in stripped and stripped.startswith(':'):
            if 'true' in stripped:
                qcdata.linear_mol = True

        # --- Point group from "symmetry  <pg>"
        elif stripped.startswith(':') and 'symmetry' in stripped and 'rotational' not in stripped:
            parts = stripped.split()
            try:
                idx = parts.index('symmetry')
                if idx + 1 < len(parts) and parts[idx + 1] != ':':
                    pg = parts[idx + 1]
                    if pg not in ('GBSA', 'found', 'elements:'):
                        qcdata.point_group = pg
            except ValueError:
                pass

        # --- Symmetry number ":  rotational number  <n>  :"
        elif 'rotational number' in stripped and stripped.startswith(':'):
            parts = stripped.split()
            try:
                qcdata.symmno = int(parts[parts.index('number') + 1])
            except (ValueError, IndexError):
                pass

        # --- Final optimized geometry block (only present for --opt/--ohess)
        elif stripped == 'final structure:':
            # Skip the "===" line; line i+1 is "===", i+2 is atom count, i+3 is comment
            try:
                natoms = int(output[i + 2].strip())
            except (ValueError, IndexError):
                continue
            atom_nums_tmp = []
            atom_types_tmp = []
            cartesians_tmp = []
            for k in range(i + 4, i + 4 + natoms):
                if k >= len(output):
                    break
                parts = output[k].split()
                if len(parts) < 4:
                    break
                try:
                    atom_types_tmp.append(parts[0])
                    atom_nums_tmp.append(element_id(parts[0], num=True))
                    cartesians_tmp.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    break
            qcdata.atom_nums = atom_nums_tmp
            qcdata.atom_types = atom_types_tmp
            qcdata.cartesians = cartesians_tmp

        # --- CPU time from "total:" block: "* cpu-time: <D> d, <H> h, <M> min, <S> sec"
        # xtb prints two spaces ("*  cpu-time:") so match the substring.
        elif 'cpu-time:' in stripped and stripped.startswith('*') and qcdata.cpu == [0, 0, 0, 0, 0]:
            # First (= "total:") cpu-time line; subsequent are SCF/ANC/hess subtimers
            parts = stripped.replace(',', ' ').split()
            try:
                d_idx = parts.index('d')
                h_idx = parts.index('h')
                m_idx = parts.index('min')
                s_idx = parts.index('sec')
                days = int(parts[d_idx - 1])
                hours = int(parts[h_idx - 1])
                mins = int(parts[m_idx - 1])
                secs_float = float(parts[s_idx - 1])
                secs = int(secs_float)
                msecs = int((secs_float - secs) * 1000)
                qcdata.cpu = [days, hours, mins, secs, msecs]
            except (ValueError, IndexError):
                pass

    # Frequencies from the LAST projected-vibrational-frequencies block.
    # xtb prints all 3N modes; the first 5 (linear) or 6 (non-linear) are the
    # zero translational/rotational eigenvalues from projection — skip them by
    # filtering |v| < 1.0 cm⁻¹. Negatives below that are imaginary modes.
    if last_freq_idx >= 0:
        for k in range(last_freq_idx + 1, len(output)):
            parts = output[k].split()
            if not parts or parts[0] != 'eigval':
                break
            for tok in parts[2:]:
                try:
                    v = float(tok)
                except ValueError:
                    continue
                if abs(v) < 1.0:
                    continue
                if v > 0:
                    frequency_wn.append(v)
                else:
                    im_frequency_wn.append(v)
    qcdata.frequency_wn = frequency_wn
    qcdata.im_frequency_wn = im_frequency_wn

    # Multiplicity: if `unpaired electrons` was missing from output, leave at default 1.

    # Solvation: parse from program call line
    if program_call:
        qcdata.solvation_model = _xtb_solvation_from_call(program_call)
    else:
        qcdata.solvation_model = 'gas phase'

    # xtb has no DFT dispersion; GFN includes its own D3/D4 implicitly.
    qcdata.empirical_dispersion = 'No empirical dispersion detected'

    # Geometry fallback: SP-only and --hess runs lack a 'final structure:' block.
    if not qcdata.atom_nums:
        atom_nums, atom_types, cartesians = _xtb_read_xyz_fallback(file)
        if atom_nums:
            qcdata.atom_nums = atom_nums
            qcdata.atom_types = atom_types
            qcdata.cartesians = cartesians

    # Mass / rotational fallback: --hess-only runs print frequencies but skip
    # the THERMODYNAMIC block, so 'molecular mass / u' and 'rotational
    # constants / cm⁻¹' are absent.  Recompute from the geometry we just loaded.
    _fill_mass_and_rotemp_from_geometry(qcdata)

    # Linear molecules: keep only physical rotational temperatures.  xtb
    # inverts a near-zero moment of inertia for the linear axis, producing
    # either a huge positive number (e.g. 1e27 K) or a huge negative one;
    # restrict to 0 < T_rot < 1e10 K to drop both pathological cases.
    if qcdata.linear_mol and qcdata.rotemp:
        physical = [t for t in qcdata.rotemp if 0.0 < t < 1.0e10]
        if physical:
            qcdata.rotemp = physical
            qcdata.roconst = [t / HC_OVER_KB * 29.9792458 for t in physical]

    # Job type from observed content
    has_opt_geom = any('final structure:' in line for line in output)
    has_freq = bool(frequency_wn) or bool(im_frequency_wn)
    if has_opt_geom and has_freq:
        qcdata.job_type = 'GSFreq'
    elif has_opt_geom:
        qcdata.job_type = 'GS'
    elif has_freq:
        qcdata.job_type = 'Freq'
    else:
        qcdata.job_type = 'SP'

    return qcdata


# Bohr → Ångström conversion (squared, for amu*Bohr² → amu*Å²)
_BOHR2_TO_ANGSTROM2 = 0.5291772109 ** 2


def parse_qchem_thermo(file):
    """Parse a Q-Chem 6 output file for all thermochemistry-relevant data.

    Handles single-job and multi-job (`@@@`-separated opt + freq + sp) inputs:
    the LAST occurrence wins for SCF energy, geometry, and frequencies, so
    the converged opt geometry and post-correlation energies are picked up
    correctly. Recognises native HF/DFT total energy, MP2 total energy, CCSD
    total energy, and CCSD(T) total energy via the precedence
    CCSD(T) > CCSD > MP2 > Total energy.

    Parameters
    ----------
    file : str
        Path to a Q-Chem 6 output file.
    """
    qcdata = QCData(file=file, program='QChem')

    with open(file, encoding='utf-8', errors='replace') as f:
        output = f.readlines()

    if not output:
        return qcdata

    HC_OVER_KB = 1.4387768775  # K per cm⁻¹

    # First pass: locate the last freq block, last geometry block, last
    # thermo header, and the input $molecule + $rem echoes (for charge,
    # multiplicity, and the requested correlation method).
    last_geom_header = -1
    last_thermo_header = -1
    last_freq_header = -1
    molecule_block_lines = []
    rem_methods = []
    in_molecule_block = False
    in_rem_block = False
    user_input_seen = False

    for i, line in enumerate(output):
        stripped = line.strip()
        if 'You are running Q-Chem version' in stripped:
            try:
                qcdata.version_program = 'Q-Chem version ' + stripped.split('version:')[1].strip()
            except IndexError:
                pass
        elif 'User input:' in stripped:
            user_input_seen = True
        elif user_input_seen and stripped == '$molecule' and not molecule_block_lines:
            in_molecule_block = True
            continue
        elif in_molecule_block:
            if stripped == '$end' or stripped.startswith('$'):
                in_molecule_block = False
            elif stripped:
                molecule_block_lines.append(stripped)
        elif user_input_seen and stripped.lower() == '$rem':
            in_rem_block = True
            continue
        elif in_rem_block:
            if stripped == '$end' or stripped.startswith('$'):
                in_rem_block = False
            elif stripped and not stripped.startswith('!'):
                parts = stripped.split()
                if len(parts) >= 2 and parts[0].lower() == 'method':
                    rem_methods.append(parts[1].lower())
        if 'Standard Nuclear Orientation (Angstroms)' in line:
            last_geom_header = i
        # Anchor on the harmonic block ("AT 298.15 K AND ..."); skip the
        # anharmonic VPT2/TOSH summaries that print only ZPE.
        if 'STANDARD THERMODYNAMIC QUANTITIES AT' in line:
            last_thermo_header = i
        if stripped.startswith('Frequency:'):
            last_freq_header = i

    # Method of the FINAL job step decides which energy line to trust.
    final_method = rem_methods[-1] if rem_methods else ''

    # Charge / multiplicity from the input $molecule block.
    # First non-`READ` line is "<charge> <mult>".
    for line in molecule_block_lines:
        s = line.strip()
        if s.upper() == 'READ':
            continue
        parts = s.split()
        if len(parts) >= 2:
            try:
                qcdata.charge = int(parts[0])
                qcdata.multiplicity = int(parts[1])
            except ValueError:
                pass
            break

    # Multiplicity fallback from "There are N alpha and M beta electrons".
    if qcdata.multiplicity == 1:
        for line in output:
            if 'alpha and' in line and 'beta electrons' in line:
                parts = line.split()
                try:
                    nalpha = int(parts[parts.index('alpha') - 1])
                    nbeta = int(parts[parts.index('beta') - 1])
                    qcdata.multiplicity = abs(nalpha - nbeta) + 1
                except (ValueError, IndexError):
                    pass

    # Energies — collect candidates; pick by FINAL method, not by precedence.
    # Q-Chem prints "MP2 total energy" as an extra analysis even when the
    # method is e.g. B2PLYP (a double hybrid that's properly captured by the
    # SCF "Total energy" line), so we can't blindly prefer correlated lines.
    scf_total = None
    mp2_total = None
    ccsd_total = None
    ccsdt_total = None
    for line in output:
        stripped = line.strip()
        if stripped.startswith('Total energy ='):
            try:
                scf_total = float(stripped.split('=')[1].split()[0])
            except (IndexError, ValueError):
                pass
        elif 'MP2' in stripped and 'total energy =' in stripped:
            try:
                mp2_total = float(stripped.split('=')[1].split()[0])
            except (IndexError, ValueError):
                pass
        elif stripped.startswith('CCSD total energy') or stripped.startswith('CCSD   total energy'):
            try:
                ccsd_total = float(stripped.split('=')[1].split()[0])
            except (IndexError, ValueError):
                pass
        elif stripped.startswith('CCSD(T) total energy'):
            try:
                ccsdt_total = float(stripped.split('=')[1].split()[0])
            except (IndexError, ValueError):
                pass
    if 'ccsd(t)' in final_method and ccsdt_total is not None:
        qcdata.scf_energy = ccsdt_total
    elif final_method == 'ccsd' and ccsd_total is not None:
        qcdata.scf_energy = ccsd_total
    elif final_method in ('mp2', 'rimp2', 'ump2') and mp2_total is not None:
        qcdata.scf_energy = mp2_total
    else:
        qcdata.scf_energy = scf_total

    # Solvation
    solvation_type = ''
    solvent_name = ''
    for line in output:
        stripped = line.strip()
        if 'Using C-PCM' in stripped or 'C-PCM dielectric factor' in stripped:
            solvation_type = solvation_type or 'CPCM'
        elif 'Using the SMD solvation model' in stripped:
            solvation_type = 'SMD'
        elif stripped.startswith('Solvent:'):
            try:
                solvent_name = stripped.split(':', 1)[1].strip()
            except IndexError:
                pass
    if solvation_type:
        qcdata.solvation_model = (solvation_type + ',' + solvent_name) if solvent_name else solvation_type

    # Final geometry — read from the LAST Standard Nuclear Orientation block.
    if last_geom_header >= 0:
        atom_nums, atom_types, cartesians = [], [], []
        # Skip 2 header lines + the dashes line before atoms.
        i = last_geom_header + 3
        while i < len(output):
            parts = output[i].split()
            if len(parts) >= 5:
                try:
                    int(parts[0])  # row index
                    elem = parts[1]
                    xyz = [float(parts[2]), float(parts[3]), float(parts[4])]
                    atom_types.append(elem)
                    atom_nums.append(element_id(elem, num=True))
                    cartesians.append(xyz)
                    i += 1
                    continue
                except ValueError:
                    pass
            break
        qcdata.atom_types = atom_types
        qcdata.atom_nums = atom_nums
        qcdata.cartesians = cartesians

    # Frequencies — collect from contiguous Frequency: lines starting at the last header.
    if last_freq_header >= 0:
        # Walk forward from the last_freq_header but we want all preceding
        # contiguous Frequency: lines that belong to the same final block.
        # Strategy: scan the file once more and collect freqs from the LAST
        # contiguous run of Frequency: lines (separated only by columns of
        # mode data, never by another major section header).
        freq_blocks = []
        current = []
        for line in output:
            stripped = line.strip()
            if stripped.startswith('Frequency:'):
                try:
                    nums = [float(x) for x in stripped.split(':', 1)[1].split()]
                    current.extend(nums)
                except ValueError:
                    pass
            elif current and ('STANDARD THERMODYNAMIC' in line or
                              'Standard Nuclear Orientation' in line):
                # End of a freq block; new section ahead.
                freq_blocks.append(current)
                current = []
        if current:
            freq_blocks.append(current)
        if freq_blocks:
            freqs = freq_blocks[-1]
            qcdata.frequency_wn = [f for f in freqs if f >= 0]
            qcdata.im_frequency_wn = [f for f in freqs if f < 0]

    # Point group: printed once per SCF (before the thermo block); take LAST.
    for line in output:
        if 'Molecular Point Group' in line:
            try:
                after = line.split('Molecular Point Group', 1)[1].strip()
                qcdata.point_group = after.split()[0]
            except (IndexError, ValueError):
                pass

    # Thermo block: ZPE, mass, symmno, principal moments (last block wins).
    if last_thermo_header >= 0:
        for line in output[last_thermo_header:]:
            stripped = line.strip()
            if stripped.startswith('Zero point vibrational energy:'):
                # "Zero point vibrational energy: 14.050 kcal/mol"
                try:
                    zpe_kcal = float(stripped.split(':')[1].split()[0])
                    qcdata.zero_point_corr = zpe_kcal / 627.509541
                except (IndexError, ValueError):
                    pass
            elif stripped.startswith('Molecular Mass:'):
                try:
                    qcdata.molecular_mass = float(stripped.split(':')[1].split()[0])
                except (IndexError, ValueError):
                    pass
            elif stripped.startswith('Eigenvalues --'):
                try:
                    eigs_bohr2 = [float(x) for x in stripped.split('--', 1)[1].split()][:3]
                    eigs_a2 = [e * _BOHR2_TO_ANGSTROM2 for e in eigs_bohr2]
                    nonzero = [e for e in eigs_a2 if e > 1e-6]
                    if len(nonzero) == 2:  # linear
                        qcdata.linear_mol = True
                        B_cm = 16.857629 / nonzero[-1]
                        qcdata.roconst = [B_cm * 29.9792458] * 3
                        qcdata.rotemp = [HC_OVER_KB * B_cm] * 3
                    elif len(nonzero) == 3:
                        roconst_cm = [16.857629 / I for I in nonzero]  # noqa: E741
                        qcdata.roconst = [b * 29.9792458 for b in roconst_cm]
                        qcdata.rotemp = [HC_OVER_KB * b for b in roconst_cm]
                except (IndexError, ValueError):
                    pass
            elif 'Rotational Symmetry Number is' in stripped:
                try:
                    qcdata.symmno = int(stripped.split('is')[1].split()[0])
                except (IndexError, ValueError):
                    pass

    # CPU time: "Total job time:  1.70s(wall), 21.27s(cpu)" — take the (cpu)
    # value (the wall × ncores estimate Q-Chem prints) so the sum matches
    # the Gaussian/NWChem/xTB convention of true summed CPU.
    for line in reversed(output):
        if 'Total job time:' in line:
            try:
                cpu_s = float(line.split(',')[1].split('s(cpu)')[0].strip())
                days = int(cpu_s // 86400)
                rem = cpu_s - days * 86400
                hours = int(rem // 3600)
                rem -= hours * 3600
                mins = int(rem // 60)
                rem -= mins * 60
                secs = int(rem)
                ms = int(round((rem - secs) * 1000))
                qcdata.cpu = [days, hours, mins, secs, ms]
            except (IndexError, ValueError):
                pass
            break

    # Job type from observed content.
    has_freq = bool(qcdata.frequency_wn) or bool(qcdata.im_frequency_wn)
    has_geom_block = last_geom_header >= 0
    if qcdata.im_frequency_wn:
        qcdata.job_type = 'TS'
    elif has_geom_block and has_freq:
        qcdata.job_type = 'GSFreq'
    elif has_freq:
        qcdata.job_type = 'Freq'
    else:
        qcdata.job_type = 'SP'

    _fill_mass_and_rotemp_from_geometry(qcdata)

    return qcdata


# Regex matching extxyz comment-line tokens: bare key=value or key="value with spaces"
_EXTXYZ_TOKEN = re.compile(r'(\w+)=("(?:[^"\\]|\\.)*"|\S+)')


def _parse_extxyz_comment(line):
    """Parse an extxyz comment line into a {key: str} dict.

    Strips surrounding double quotes from quoted values; leaves unquoted values
    untouched. Caller is responsible for type coercion.
    """
    out = {}
    for key, val in _EXTXYZ_TOKEN.findall(line):
        if len(val) >= 2 and val[0] == '"' and val[-1] == '"':
            val = val[1:-1]
        out[key] = val
    return out


def _extxyz_bool(s):
    """Decode an extxyz boolean (T/F/True/False) to a Python bool."""
    return str(s).strip().lower() in ('t', 'true', '1', 'yes')


def parse_ase_thermo(file):
    """Parse a GoodVibes ASE thermo extxyz file.

    The format is standard extended XYZ with thermochemistry metadata in the
    second line (key=value, ASE Atoms.info convention). Required keys:
    ``program=ase``, ``scf_energy``. Frequencies are space-separated in cm-1
    (negatives are imaginary). See tests/ase/README.md for the full spec.
    """
    qcdata = QCData(file=file, program='ase')

    with open(file, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    if len(lines) < 2:
        return qcdata
    try:
        natoms = int(lines[0].strip())
    except ValueError:
        return qcdata

    info = _parse_extxyz_comment(lines[1])

    # Provenance
    qcdata.version_program = info.get('ase_version', 'ase')

    # Charge / multiplicity (defaults: 0 / 1)
    try:
        qcdata.charge = int(float(info.get('charge', '0')))
    except ValueError:
        qcdata.charge = 0
    try:
        qcdata.multiplicity = int(float(info.get('multiplicity', '1')))
    except ValueError:
        qcdata.multiplicity = 1

    # Electronic energy (Hartree by default; eV converted)
    if 'scf_energy' in info:
        try:
            e = float(info['scf_energy'])
            units = info.get('scf_energy_units', 'Hartree').lower()
            if units in ('ev',):
                e = e / 27.211386245988
            elif units in ('kcal/mol', 'kcalmol'):
                e = e / 627.509541
            elif units in ('kj/mol', 'kjmol'):
                e = e / 2625.4996394799
            qcdata.scf_energy = e
        except ValueError:
            pass

    # Zero-point energy (Hartree)
    if 'zpe' in info:
        try:
            qcdata.zero_point_corr = float(info['zpe'])
        except ValueError:
            pass

    # Frequencies (cm-1; negatives are imaginary)
    if 'frequencies' in info:
        for tok in info['frequencies'].split():
            try:
                v = float(tok)
            except ValueError:
                continue
            if v < 0:
                qcdata.im_frequency_wn.append(v)
            else:
                qcdata.frequency_wn.append(v)

    # Optional model-chemistry metadata
    qcdata.solvation_model = info.get('solvation_model', '')
    qcdata.empirical_dispersion = info.get('empirical_dispersion', '')
    qcdata.point_group = info.get('point_group', '')
    if 'symmno' in info:
        try:
            qcdata.symmno = int(float(info['symmno']))
        except ValueError:
            pass
    if 'linear_mol' in info:
        qcdata.linear_mol = _extxyz_bool(info['linear_mol'])
    if 'applied_freq_scale_factor' in info:
        try:
            qcdata.applied_freq_scale_factor = float(info['applied_freq_scale_factor'])
        except ValueError:
            pass

    # Atom block: species, x, y, z (additional columns ignored)
    atom_nums, atom_types, cartesians = [], [], []
    for line in lines[2:2 + natoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        elem = parts[0]
        try:
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        except ValueError:
            continue
        atom_types.append(elem)
        atom_nums.append(element_id(elem, num=True))
        cartesians.append(xyz)
    qcdata.atom_types = atom_types
    qcdata.atom_nums = atom_nums
    qcdata.cartesians = cartesians

    # Molecular mass: explicit override, else compute from atomic masses
    if 'molecular_mass' in info:
        try:
            qcdata.molecular_mass = float(info['molecular_mass'])
        except ValueError:
            pass
    if not qcdata.molecular_mass and atom_types:
        qcdata.molecular_mass = sum(ATOMIC_MASSES.get(a, 0.0) for a in atom_types)

    # Rotational constants: explicit override (cm-1), else derived from inertia
    HC_OVER_KB = 1.4387768775  # K per cm⁻¹
    if 'roconst_cm' in info:
        try:
            roconst_cm = [float(t) for t in info['roconst_cm'].split()][:3]
            qcdata.roconst = [b * 29.9792458 for b in roconst_cm]
            qcdata.rotemp = [HC_OVER_KB * b for b in roconst_cm]
        except ValueError:
            pass
    elif 'rotemp' in info:
        try:
            qcdata.rotemp = [float(t) for t in info['rotemp'].split()][:3]
            qcdata.roconst = [t / HC_OVER_KB * 29.9792458 for t in qcdata.rotemp]
        except ValueError:
            pass
    if not any(qcdata.rotemp) and len(atom_types) >= 2:
        _mass, eigvals = _principal_moments_of_inertia(atom_types, cartesians)
        # Linear molecules: smallest principal moment is ~0.
        nonzero = [e for e in eigvals if e > 1e-3]
        if not qcdata.linear_mol and len(nonzero) == 2:
            qcdata.linear_mol = True
        if qcdata.linear_mol:
            # Use the largest two equal moments; physics needs a single B.
            B_cm = 16.857629 / nonzero[-1]  # cm⁻¹
            qcdata.roconst = [B_cm * 29.9792458] * 3
            qcdata.rotemp = [HC_OVER_KB * B_cm] * 3
        elif len(nonzero) == 3:
            roconst_cm = [16.857629 / I for I in nonzero]  # noqa: E741
            qcdata.roconst = [b * 29.9792458 for b in roconst_cm]
            qcdata.rotemp = [HC_OVER_KB * b for b in roconst_cm]

    # Job type: explicit override, else infer from frequency presence/sign
    if 'job_type' in info:
        qcdata.job_type = info['job_type']
    else:
        if qcdata.im_frequency_wn:
            qcdata.job_type = 'TS'
        elif qcdata.frequency_wn:
            qcdata.job_type = 'Freq'
        else:
            qcdata.job_type = 'SP'

    return qcdata


def _detect_program(data):
    """Detect the QC program from output file lines.

    Scans the first ~80 lines (Q-Chem banner is past the SLURM preamble);
    ASE extxyz is recognized by ``program=ase`` in the comment (2nd) line.

    Parameters
    ----------
    data : list of str
        Output file contents as lines.

    Returns
    -------
    str
        One of 'Gaussian', 'Orca', 'NWChem', 'xtb', 'QChem', 'ase', 'unknown'.
    """
    for line in data[:80]:
        if 'Gaussian' in line:
            return 'Gaussian'
        if '* O   R   C   A *' in line:
            return 'Orca'
        if 'NWChem' in line:
            return 'NWChem'
        if 'x T B' in line or 'xtb version' in line:
            return 'xtb'
        if 'You are running Q-Chem' in line or 'Welcome to Q-Chem' in line:
            return 'QChem'
    if len(data) >= 2 and 'program=ase' in data[1]:
        return 'ase'
    return 'unknown'


def parse_qcdata(file):
    """Parse any supported output file into a QCData object.

    Detects program from file content and delegates to the correct parser.

    Parameters
    ----------
    file : str
        Path to quantum chemistry output file.
    """
    stub = os.path.splitext(file)[0]
    possible_filenames = (stub + '.log', stub + '.out', stub + '.extxyz', file)
    data = None
    actual_file = file
    for possible_filename in possible_filenames:
        if os.path.exists(possible_filename):
            actual_file = possible_filename
            with open(possible_filename, encoding='utf-8', errors='replace') as f:
                data = f.readlines()
            break

    if data is None:
        return QCData(file=file, program='unknown')

    program = _detect_program(data)

    if program == 'Gaussian':
        return parse_gaussian_thermo(actual_file)
    elif program == 'Orca':
        return parse_orca_thermo(actual_file)
    elif program == 'NWChem':
        return parse_nwchem_thermo(actual_file)
    elif program == 'xtb':
        return parse_xtb_thermo(actual_file)
    elif program == 'QChem':
        return parse_qchem_thermo(actual_file)
    elif program == 'ase':
        return parse_ase_thermo(actual_file)
    else:
        return QCData(file=file, program='unknown')


@dataclass(frozen=True)
class HessianData:
    """Cartesian Hessian and the masses needed to mass-weight it.

    Produced by parse_hessian(). Units follow the QC programs' native
    conventions: the Hessian is in Hartree/Bohr^2 and masses are in amu.
    Row/column order is 3N Cartesian components (x1, y1, z1, x2, ...)
    matching the atom order of the parsed geometry.
    """
    hessian: np.ndarray       # (3N, 3N) symmetric, Hartree/Bohr^2
    masses: List[float]       # length N, amu (respects isotope substitution)
    program: str              # 'Gaussian' or 'Orca'
    source: str               # file the Hessian was actually read from


def _parse_gaussian_hessian(data, file):
    """Extract the Cartesian Hessian from a Gaussian frequency job.

    Reads the lower-triangular force-constant matrix from the archive
    (final "1\\1\\..." block, NImag section) written on normal termination
    of a freq job, and the per-atom masses from the "has atomic number ...
    and mass" lines (these respect the iso= keyword).

    Parameters
    ----------
    data : list of str
        Output file contents as lines.
    file : str
        Path (recorded as HessianData.source).

    Returns
    -------
    HessianData
    """
    natoms = None
    per_atom_masses = []
    start = None
    for i, line in enumerate(data):
        s = line.strip()
        if s.startswith('NAtoms='):
            try:
                natoms = int(s.split()[1])
            except (ValueError, IndexError):
                pass
        elif (s.startswith('Atom') and 'has atomic number' in s
                and 'and mass' in s):
            try:
                per_atom_masses.append(float(s.split('and mass')[1].split()[0]))
            except (ValueError, IndexError):
                pass
        elif 'NImag' in line:
            start = i
    if start is None:
        raise ValueError(
            "No archive force-constant block (NImag) found in %s -- "
            "the file must be a normally-terminated Gaussian freq job" % file)

    # The archive is wrapped at 70 characters mid-token; concatenate the
    # stripped lines up to the terminating '@'. Windows builds of Gaussian
    # use '|' instead of '\' as the archive separator.
    longline = ''
    for line in data[start:]:
        longline += line.strip()
        if '@' in line:
            break
    if '\\' not in longline:
        longline = longline.replace('|', '\\')
    try:
        forces = longline.split('NImag')[1].split('\\')[2].split(',')
        tri = [float(x) for x in forces]
    except (IndexError, ValueError):
        raise ValueError("Could not parse archive force constants in %s" % file)

    # Lower-triangle length L = dof*(dof+1)/2
    dof = int((-1.0 + (1.0 + 8.0 * len(tri)) ** 0.5) / 2.0 + 0.5)
    if dof * (dof + 1) // 2 != len(tri):
        raise ValueError(
            "Archive force-constant block in %s has %d elements, which is "
            "not a triangular number" % (file, len(tri)))
    if natoms is not None and dof != 3 * natoms:
        raise ValueError(
            "Archive Hessian dimension %d does not match NAtoms=%d in %s"
            % (dof, natoms, file))

    hessian = np.zeros((dof, dof))
    idx = 0
    for m in range(dof):
        for n in range(m + 1):
            hessian[m, n] = hessian[n, m] = tri[idx]
            idx += 1

    # G4-style composites print masses for several sections; the final
    # N entries belong to the freq job (same convention as parse_gaussian_thermo)
    n_atoms = dof // 3
    masses = per_atom_masses[-n_atoms:] if len(per_atom_masses) >= n_atoms else []
    return HessianData(hessian=hessian, masses=masses, program='Gaussian', source=file)


def _parse_orca_hess(hess_path):
    """Read an ORCA .hess file ($hessian and $atoms sections).

    The $hessian section stores the full (3N, 3N) matrix in column blocks:
    a header line of column indices followed by 3N rows of
    "row_index value value ...". The $atoms section provides per-atom
    masses (respecting mass overrides in the ORCA input).

    Parameters
    ----------
    hess_path : str
        Path to the .hess file.

    Returns
    -------
    HessianData
    """
    with open(hess_path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    hessian = None
    masses = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s == '$hessian':
            dim = int(lines[i + 1].split()[0])
            hessian = np.zeros((dim, dim))
            j = i + 2
            cols_done = 0
            while cols_done < dim:
                col_idx = [int(t) for t in lines[j].split()]
                j += 1
                for _ in range(dim):
                    toks = lines[j].split()
                    j += 1
                    row = int(toks[0])
                    for c, v in zip(col_idx, toks[1:]):
                        hessian[row, c] = float(v)
                cols_done += len(col_idx)
            i = j
        elif s == '$atoms':
            nat = int(lines[i + 1].split()[0])
            for k in range(nat):
                masses.append(float(lines[i + 2 + k].split()[1]))
            i = i + 2 + nat
        else:
            i += 1

    if hessian is None:
        raise ValueError("No $hessian section found in %s" % hess_path)
    if masses and hessian.shape[0] != 3 * len(masses):
        raise ValueError(
            "$hessian dimension %d does not match %d atoms in %s"
            % (hessian.shape[0], len(masses), hess_path))
    # Guard against small numerical asymmetry
    hessian = 0.5 * (hessian + hessian.T)
    return HessianData(hessian=hessian, masses=masses, program='Orca', source=hess_path)


def parse_hessian(file):
    """Parse the Cartesian Hessian for a frequency job into HessianData.

    Unlike parse_qcdata(), which extracts the program's printed frequencies,
    this returns the raw second-derivative matrix -- needed by consumers that
    re-mass-weight it, e.g. isotopologue analysis in Kinisot.

    Supported sources:
      * Gaussian: the archive force-constant block of a freq .log/.out.
      * ORCA: the companion .hess file (same stub as the .out), or a .hess
        path given directly.

    Parameters
    ----------
    file : str
        QC output file (.log/.out) or an ORCA .hess file.

    Returns
    -------
    HessianData
        hessian (Hartree/Bohr^2), masses (amu), program, source.

    Raises
    ------
    FileNotFoundError, ValueError
        If no file, no Hessian data, or an unsupported program.
    """
    stub, ext = os.path.splitext(file)
    if ext == '.hess':
        return _parse_orca_hess(file)

    actual_file = None
    for candidate in (stub + '.log', stub + '.out', file):
        if os.path.exists(candidate):
            actual_file = candidate
            break
    if actual_file is None:
        raise FileNotFoundError("No output file found for %s" % file)

    with open(actual_file, encoding='utf-8', errors='replace') as f:
        data = f.readlines()
    program = _detect_program(data)

    if program == 'Gaussian':
        return _parse_gaussian_hessian(data, actual_file)
    if program == 'Orca':
        hess_path = stub + '.hess'
        if not os.path.exists(hess_path):
            raise FileNotFoundError(
                "ORCA stores the Hessian in a separate file: expected %s "
                "alongside %s" % (hess_path, actual_file))
        return _parse_orca_hess(hess_path)
    raise ValueError(
        "Hessian extraction is not supported for program %r (%s); "
        "supported: Gaussian freq outputs and ORCA .hess files"
        % (program, actual_file))
