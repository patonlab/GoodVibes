# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import os.path
from dataclasses import dataclass, field
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

    # ONIOM MM frequency scaling fractions (per-frequency, empty if not ONIOM)
    fract_modelsys: List[float] = field(default_factory=list)
    has_oniom: bool = False

    # Molecular geometry
    atom_nums: List[int] = field(default_factory=list)
    atom_types: List[str] = field(default_factory=list)
    cartesians: List[List[float]] = field(default_factory=list)


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

class xyz_out:
    """
    Enables output of optimized coordinates to a single xyz-formatted file.

    Writes Cartesian coordinates of parsed chemical input.

    Attributes:
        xyz (file object): path in current working directory to write Cartesian coordinates.
    """
    def __init__(self, filein, suffix, append):
        self.xyz = open('{}_{}.{}'.format(filein, append, suffix), 'w')

    def write_text(self, message):
        self.xyz.write(message + "\n")

    def write_coords(self, atoms, coords):
        for n, carts in enumerate(coords):
            self.xyz.write('{:>1}'.format(atoms[n]))
            for cart in carts:
                self.xyz.write('{:13.6f}'.format(cart))
            self.xyz.write('\n')

    def finalize(self):
        self.xyz.close()

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

    data = None
    stub = os.path.splitext(file)[0]
    possible_filenames = (stub + ".log", stub + ".out")
    for possible_filename in possible_filenames:
        if os.path.exists(possible_filename):
            with open(possible_filename) as f:
                data = f.readlines()

    if data is None:
        raise ValueError("File {} does not exist".format(file))

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

    return spe, program, version_program, solvation_model, file, charge, empirical_dispersion, multiplicity

def sp_cpu(file):
    """Read single-point output for cpu time."""
    program, data, cpu = None, [], None

    if os.path.exists(os.path.splitext(file)[0] + '.log'):
        with open(os.path.splitext(file)[0] + '.log') as f:
            data = f.readlines()
    elif os.path.exists(os.path.splitext(file)[0] + '.out'):
        with open(os.path.splitext(file)[0] + '.out') as f:
            data = f.readlines()
    else:
        raise ValueError("File {} does not exist".format(file))

    for line in data:
        if line.find("Gaussian") > -1:
            program = "Gaussian"
            break
        if line.find("* O   R   C   A *") > -1:
            program = "Orca"
            break
        if line.find("NWChem") > -1:
            program = "NWChem"
            break

    for line in data:
        if program == "Gaussian":
            if line.strip().find("Job cpu time") > -1:
                days = int(line.split()[3])
                hours = int(line.split()[5])
                mins = int(line.split()[7])
                secs = 0
                msecs = int(float(line.split()[9]) * 1000.0)
                cpu = [days, hours, mins, secs, msecs]
        if program == "Orca":
            if line.strip().find("TOTAL RUN TIME") > -1:
                days = int(line.split()[3])
                hours = int(line.split()[5])
                mins = int(line.split()[7])
                secs = int(line.split()[9])
                msecs = float(line.split()[11])
                cpu = [days, hours, mins, secs, msecs]
        if program == "NWChem":
            if line.strip().find("Total times") > -1:
                days = 0
                hours = 0
                mins = 0
                secs = float(line.split()[3][0:-1])
                msecs = 0.0
                cpu = [days, hours, mins, secs, msecs]

    return cpu


import re

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
    with open(file) as f:
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
    with open(file) as f:
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
            if keyword_line[start_scrf] == "(":
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
    # ORCA parsing for solvation model and level of theory
    elif program == 'Orca':
        level, bs = _parse_orca_lot(data)
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
            if 'ORCA TERMINATED NORMALLY' in line:
                progress = 'Normal'
            elif 'error termination' in line:
                progress = 'Error'
        solvation_model = keyword_line_1 + keyword_line_2 + keyword_line_3
    level_of_theory = '/'.join([level, bs])

    return level_of_theory, solvation_model, progress, orientation, dft_used

def gaussian_jobtype(filename):
    """Read the jobtype from a Gaussian archive string."""
    job = ''
    with open(filename) as f:
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


def parse_gaussian_thermo(file, ssymm=False):
    """Parse Gaussian output for all thermochemistry-relevant data.

    Returns QCData with raw frequencies (negative = imaginary, no inversion
    applied). Frequency inversion is a user policy decision handled in
    thermo.py.

    Parameters
    ----------
    file : str
        Path to Gaussian output file.
    ssymm : bool
        If True, skip rotational symmetry number from file (use default 1).
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
    with open(file) as f:
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

        # Molecular mass
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

    return qcdata


def parse_nwchem_thermo(file, ssymm=False):
    """Parse NWChem output for all thermochemistry-relevant data.

    Returns QCData with raw frequencies (negative = imaginary, no inversion
    applied).

    Parameters
    ----------
    file : str
        Path to NWChem output file.
    ssymm : bool
        If True, skip rotational symmetry number from file (use default 1).
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
    with open(file) as f:
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

        # Multiplicity
        elif 'mult ' in line.strip():
            try:
                qcdata.multiplicity = int(line.split()[1])
            except (ValueError, IndexError):
                qcdata.multiplicity = 1

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

    return qcdata


def parse_orca_thermo(file, ssymm=False):
    """Parse ORCA output for all thermochemistry-relevant data.

    Uses native line-by-line parsing.

    Parameters
    ----------
    file : str
        Path to ORCA output file.
    ssymm : bool
        If True, skip symmetry number from file (use default 1).
    """
    qcdata = QCData(file=file, program='Orca')

    with open(file) as f:
        output = f.readlines()

    frequency_wn = []
    im_frequency_wn = []
    in_freq_section = False
    solvation_type = ''
    solvent_name = ''
    _has_opt = False
    _has_ts = False
    _has_freq_kw = False

    for i, line in enumerate(output):
        stripped = line.strip()

        # --- Frequency section state machine ---
        if 'VIBRATIONAL FREQUENCIES' in stripped and '---' not in stripped:
            in_freq_section = True
            frequency_wn = []
            im_frequency_wn = []
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

        # SCF energy (last occurrence wins for linked jobs)
        elif stripped.startswith('FINAL SINGLE POINT ENERGY'):
            qcdata.scf_energy = float(stripped.split()[-1])

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
            if '(inf)' in pg_part:
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

        # CPU time
        elif stripped.startswith('TOTAL RUN TIME:'):
            parts = stripped.split()
            try:
                qcdata.cpu = [
                    int(parts[3]),   # days
                    int(parts[5]),   # hours
                    int(parts[7]),   # minutes
                    int(parts[9]),   # seconds
                    int(parts[11]),  # msec
                ]
            except (IndexError, ValueError):
                pass

    # Assemble solvation model
    if solvation_type:
        if solvent_name:
            qcdata.solvation_model = solvation_type + ',' + solvent_name
        else:
            qcdata.solvation_model = solvation_type

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

    return qcdata


def parse_qcdata(file, ssymm=False):
    """Parse any supported output file into a QCData object.

    Detects program from file content and delegates to the correct parser.

    Parameters
    ----------
    file : str
        Path to quantum chemistry output file.
    ssymm : bool
        If True, skip rotational symmetry number from file (use default 1).
    """
    stub = os.path.splitext(file)[0]
    possible_filenames = (stub + '.log', stub + '.out')
    data = None
    actual_file = file
    for possible_filename in possible_filenames:
        if os.path.exists(possible_filename):
            actual_file = possible_filename
            with open(possible_filename) as f:
                data = f.readlines()
            break

    if data is None:
        return QCData(file=file, program='unknown')

    # Detect program from first ~50 lines
    program = 'unknown'
    for line in data[:50]:
        if 'Gaussian' in line:
            program = 'Gaussian'
            break
        if '* O   R   C   A *' in line:
            program = 'Orca'
            break
        if 'NWChem' in line:
            program = 'NWChem'
            break

    if program == 'Gaussian':
        return parse_gaussian_thermo(actual_file, ssymm=ssymm)
    elif program == 'Orca':
        return parse_orca_thermo(actual_file, ssymm=ssymm)
    elif program == 'NWChem':
        return parse_nwchem_thermo(actual_file, ssymm=ssymm)
    else:
        return QCData(file=file, program='unknown')
