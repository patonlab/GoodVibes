# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import os.path, sys
import numpy as np

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

class getoutData:
    """
    Read molecule data from a computational chemistry output file.

    Currently supports Gaussian and ORCA output types.

    Attributes:
        FREQS (list): list of frequencies parsed from Gaussian file.
        REDMASS (list): list of reduced masses parsed from Gaussian file.
        FORCECONST (list): list of force constants parsed from Gaussian file.
        NORMALMODE (list): list of normal modes parsed from Gaussian file.
        atom_nums (list): list of atom number IDs.
        atom_types (list): list of atom element symbols.
        cartesians (list): list of cartesian coordinates for each atom.
        atomictypes (list): list of atomic types output in Gaussian files.
        connectivity (list): list of atomic connectivity in a molecule, based on covalent radii
    """
    def __init__(self, file):
        with open(file) as f:
            data = f.readlines()
        program = 'none'

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

        def get_freqs(self, outlines, natoms, format):
            self.FREQS = []
            self.REDMASS = []
            self.FORCECONST = []
            self.NORMALMODE = []
            freqs_so_far = 0
            if format == "Gaussian":
                for i in range(0, len(outlines)):
                    if outlines[i].find(" Frequencies -- ") > -1:
                        nfreqs = len(outlines[i].split())
                        for j in range(2, nfreqs):
                            self.FREQS.append(float(outlines[i].split()[j]))
                            self.NORMALMODE.append([])
                        for j in range(3, nfreqs + 1): self.REDMASS.append(float(outlines[i + 1].split()[j]))
                        for j in range(3, nfreqs + 1): self.FORCECONST.append(float(outlines[i + 2].split()[j]))

                        for j in range(0, natoms):
                            for k in range(0, nfreqs - 2):
                                self.NORMALMODE[(freqs_so_far + k)].append(
                                    [float(outlines[i + 5 + j].split()[3 * k + 2]),
                                     float(outlines[i + 5 + j].split()[3 * k + 3]),
                                     float(outlines[i + 5 + j].split()[3 * k + 4])])
                        freqs_so_far = freqs_so_far + nfreqs - 2

        def getatom_types(self, outlines, program):
            if program == "Gaussian":
                for i, oline in enumerate(outlines):
                    if "Input orientation" in oline or "Standard orientation" in oline:
                        self.atom_nums, self.atom_types, self.cartesians, self.atomictypes, carts = [], [], [], [], \
                                                                                                    outlines[i + 5:]
                        for j, line in enumerate(carts):
                            if "-------" in line:
                                break
                            self.atom_nums.append(int(line.split()[1]))
                            self.atom_types.append(element_id(int(line.split()[1])))
                            self.atomictypes.append(int(line.split()[2]))
                            if len(line.split()) > 5:
                                self.cartesians.append(
                                    [float(line.split()[3]), float(line.split()[4]), float(line.split()[5])])
                            else:
                                self.cartesians.append(
                                    [float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])
            if program == "Orca":
                for i, oline in enumerate(outlines):
                    if "*" in oline and ">" in oline and "xyz" in oline:
                        self.atom_nums, self.atom_types, self.cartesians, carts = [], [], [], outlines[i + 1:]
                        for j, line in enumerate(carts):
                            if ">" in line and "*" in line:
                                break
                            if len(line.split()) > 5:
                                self.cartesians.append(
                                    [float(line.split()[3]), float(line.split()[4]), float(line.split()[5])])
                                self.atom_types.append(line.split()[2])
                                self.atom_nums.append(element_id(line.split()[2], num=True))
                            else:
                                self.cartesians.append(
                                    [float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])
                                self.atom_types.append(line.split()[1])
                                self.atom_nums.append(element_id(line.split()[1], num=True))
            if program == "NWChem":
                for i, oline in enumerate(outlines):
                    if "Output coordinates" in oline:
                        self.atom_nums, self.atom_types, self.cartesians, self.atomictypes, carts = [], [], [], [], outlines[i+4:]
                        for j, line in enumerate(carts):
                            if line.strip()=='' :
                                break
                            self.atom_nums.append(int(float(line.split()[2])))
                            self.atom_types.append(element_id(int(float(line.split()[2]))))
                            self.atomictypes.append(int(float(line.split()[2])))
                            self.cartesians.append([float(line.split()[3]),float(line.split()[4]),float(line.split()[5])])

        getatom_types(self, data, program)
        natoms = len(self.atom_types)
        try:
            get_freqs(self, data, natoms, program)
        except:
            pass

    # Convert coordinates to string that can be used by the symmetry.c program
    def coords_string(self):
        xyzstring = str(len(self.atom_nums)) + '\n'
        for atom, xyz in zip(self.atom_nums, self.cartesians):
            xyzstring += "{0} {1:.6f} {2:.6f} {3:.6f}\n".format(atom, *xyz)
        return xyzstring

    # Obtain molecule connectivity to be used for internal symmetry determination
    def get_connectivity(self):
        connectivity = []
        tolerance = 0.2

        for i, ai in enumerate(self.atom_types):
            row = []
            for j, aj in enumerate(self.atom_types):
                if i == j:
                    continue
                cutoff = RADII[ai] + RADII[aj] + tolerance
                distance = np.linalg.norm(np.array(self.cartesians[i]) - np.array(self.cartesians[j]))
                if distance < cutoff:
                    row.append(j)
            connectivity.append(row)
            self.connectivity = connectivity

def cosmo_rs_out(datfile, names, interval=False):
    """
    Read solvation free energies from a COSMO-RS data file

    Parameters:
    datfile (str): name of COSMO-RS output file.
    names (list): list of species in COSMO-RS file that correspond to names of other computational output files.
    interval (bool): flag for parser to read COSMO-RS temperature interval calculation.
    """
    gsolv = {}
    if os.path.exists(datfile):
        with open(datfile) as f:
            data = f.readlines()
    else:
        raise ValueError("File {} does not exist".format(datfile))

    temp = 0
    t_interval = []
    gsolv_dicts = []
    found = False
    oldtemp = 0
    gsolv_temp = {}
    if interval:
        for i, line in enumerate(data):
            for name in names:
                if line.find('(' + name.split('.')[0] + ')') > -1 and line.find('Compound') > -1:
                    if data[i - 5].find('Temperature') > -1:
                        temp = data[i - 5].split()[2]
                    if float(temp) > float(interval[0]) and float(temp) < float(interval[1]):
                        if float(temp) not in t_interval:
                            t_interval.append(float(temp))
                        if data[i + 10].find('Gibbs') > -1:
                            gsolv = float(data[i + 10].split()[6].strip()) / KCAL_TO_AU
                            gsolv_temp[name] = gsolv

                            found = True
            if found:
                if oldtemp is 0:
                    oldtemp = temp
                if temp is not oldtemp:
                    gsolv_dicts.append(gsolv)  # Store dict at one temp
                    gsolv = {}  # Clear gsolv
                    gsolv.update(gsolv_temp)  # Grab the first one for the new temp
                    oldtemp = temp
                gsolv.update(gsolv_temp)
                gsolv_temp = {}
                found = False
        gsolv_dicts.append(gsolv)  # Grab last dict
    else:
        for i, line in enumerate(data):
            for name in names:
                if line.find('(' + name.split('.')[0] + ')') > -1 and line.find('Compound') > -1:
                    if data[i + 11].find('Gibbs') > -1:
                        gsolv = float(data[i + 11].split()[6].strip()) / KCAL_TO_AU
                        gsolv[name] = gsolv

    if interval:
        return t_interval, gsolv_dicts
    else:
        return gsolv

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

    if os.path.exists(os.path.splitext(file)[0] + '.log'):
        with open(os.path.splitext(file)[0] + '.log') as f:
            data = f.readlines()
    elif os.path.exists(os.path.splitext(file)[0] + '.out'):
        with open(os.path.splitext(file)[0] + '.out') as f:
            data = f.readlines()
    else:
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
    for line in data:
        if program == "Gaussian":
            if line.strip().startswith('SCF Done:'):
                spe = float(line.strip().split()[4])
            elif line.strip().startswith('Counterpoise corrected energy'):
                spe = float(line.strip().split()[4])
            # For MP2 calculations replace with EUMP2
            elif 'EUMP2 =' in line.strip():
                spe = float((line.strip().split()[5]).replace('D', 'E'))
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
            elif "Charge" in line.strip() and "Multiplicity" in line.strip():
                charge = int(line.split('Multiplicity')[0].split('=')[-1].strip())
                multiplicity = line.split('=')[-1].strip()
        if program == "Orca":
            if line.strip().startswith('FINAL SINGLE POINT ENERGY'):
                spe = float(line.strip().split()[4])
            if 'Program Version' in line.strip():
                version_program = "ORCA version " + line.split()[2]
            if "Total Charge" in line.strip() and "...." in line.strip():
                charge = int(line.strip("=").split()[-1])
            if "Multiplicity" in line.strip() and "...." in line.strip():
                multiplicity = int(line.strip("=").split()[-1])
        if program == "NWChem":
            if line.strip().startswith('Total DFT energy'):
                spe = float(line.strip().split()[4])
            if 'nwchem branch' in line.strip():
                version_program = "NWChem version " + line.split()[3]
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
        # keyword_line_1 = "gas phase"
        # keyword_line_2 = ''
        # keyword_line_3 = ''
        # for i, line in enumerate(data):
        #     if 'CPCM SOLVATION MODEL' in line.strip():
        #         keyword_line_1 = "CPCM,"
        #     if 'SMD CDS free energy correction energy' in line.strip():
        #         keyword_line_2 = "SMD,"
        #     if "Solvent:              " in line.strip():
        #         keyword_line_3 = line.strip().split()[-1]
        # solvation_model = keyword_line_1 + keyword_line_2 + keyword_line_3
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
    spe, program, data, cpu = None, None, [], None

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
            if line.strip().startswith('SCF Done:'):
                spe = float(line.strip().split()[4])
            if line.strip().find("Job cpu time") > -1:
                days = int(line.split()[3])
                hours = int(line.split()[5])
                mins = int(line.split()[7])
                secs = 0
                msecs = int(float(line.split()[9]) * 1000.0)
                cpu = [days, hours, mins, secs, msecs]
        if program == "Orca":
            if line.strip().startswith('FINAL SINGLE POINT ENERGY'):
                spe = float(line.strip().split()[4])
            if line.strip().find("TOTAL RUN TIME") > -1:
                days = int(line.split()[3])
                hours = int(line.split()[5])
                mins = int(line.split()[7])
                secs = int(line.split()[9])
                msecs = float(line.split()[11])
                cpu = [days, hours, mins, secs, msecs]
        if program == "NWChem":
            if line.strip().startswith('Total DFT energy ='):
                spe = float(line.strip().split()[4])
            if line.strip().find("Total times") > -1:
                days = 0
                hours = 0
                mins = 0
                secs = float(line.split()[3][0:-1])
                msecs = 0
                cpu = [days,hours,mins,secs,msecs]

    return cpu

def level_of_theory(file):
    """Read output for the level of theory and basis set used."""
    repeated_theory = 0
    with open(file) as f:
        data = f.readlines()
    level, bs = 'none', 'none'

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
    progress, orientation = 'Incomplete', 'Input'
    a, repeated_theory = 0, 0
    no_grid = True
    DFT, dft_used, level, bs, scf_iradan, cphf_iradan = False, 'F', 'none', 'none', False, False
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
                grid = grid_lookup[int(dft_used)]
                no_grid = False
            except:
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
    if program is 'NWChem':
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
    if program is 'Gaussian':
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
    # ORCA parsing for solvation model
    elif program is 'Orca':
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

def jobtype(file):
    """Read output for the level of theory and basis set used."""
    with open(file) as f:
        data = f.readlines()
    job = ''
    for line in data:
        if line.strip().find('\\SP\\') > -1:
            job += 'SP'
        if line.strip().find('\\FOpt\\') > -1:
            job += 'GS'
        if line.strip().find('\\FTS\\') > -1:
            job += 'TS'
        if line.strip().find('\\Freq\\') > -1:
            job += 'Freq'
    return job
