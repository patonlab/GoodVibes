# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import ctypes, math, os.path, sys
import numpy as np

try:
    import goodvibes.thermo as thermo
except:
    import thermo as thermo

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

# Some useful arrays
periodictable = ["", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si",
                 "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                 "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
                 "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",
                 "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
                 "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
                 "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
                 "Rg", "Uub", "Uut", "Uuq", "Uup", "Uuh", "Uus", "Uuo"]

# Symmetry numbers for different point groups
pg_sm = {"C1": 1, "Cs": 1, "Ci": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6, "C7": 7, "C8": 8, "D2": 4, "D3": 6,
         "D4": 8, "D5": 10, "D6": 12, "D7": 14, "D8": 16, "C2v": 2, "C3v": 3, "C4v": 4, "C5v": 5, "C6v": 6, "C7v": 7,
         "C8v": 8, "C2h": 2, "C3h": 3, "C4h": 4, "C5h": 5, "C6h": 6, "C7h": 7, "C8h": 8, "D2h": 4, "D3h": 6, "D4h": 8,
         "D5h": 10, "D6h": 12, "D7h": 14, "D8h": 16, "D2d": 4, "D3d": 6, "D4d": 8, "D5d": 10, "D6d": 12, "D7d": 14,
         "D8d": 16, "S4": 4, "S6": 6, "S8": 8, "T": 6, "Th": 12, "Td": 12, "O": 12, "Oh": 24, "Cinfv": 1, "Dinfh": 2,
         "I": 30, "Ih": 60, "Kh": 1}

def sharepath(filename):
    """
    Get absolute pathway to GoodVibes project.

    Used in finding location of compiled C files used in symmetry corrections.

    Parameter:
    filename (str): name of compiled C file, OS specific.

    Returns:
    str: absolute path on machine to compiled C file.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, 'share', filename)

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

class Logger:
    """
    Enables output to terminal and to text file.

    Writes GV output to .dat or .csv files.

    Attributes:
        csv (bool): decides if comma separated value file is written.
        log (file object): file to write GV output to.
        thermodata (bool): decides if string passed to logger is thermochemical data, needing to be separated by commas
    """
    def __init__(self, filein, append, csv):
        self.csv = csv
        if not self.csv:
            suffix = 'dat'
        else:
            suffix = 'csv'
        self.log = open('{0}_{1}.{2}'.format(filein, append, suffix), 'w')

    def write(self, message, thermodata=False):
        self.thermodata = thermodata
        print(message, end='')
        if self.csv and self.thermodata:
            items = message.split()
            message = ",".join(items)
            message = message + ","
        self.log.write(message)

    def fatal(self, message):
        print(message + "\n")
        self.log.write(message + "\n")
        self.finalize()
        sys.exit(1)

    def finalize(self):
        self.log.close()

class SDFWriter:
    """
    A class that acts like a file. If num_individual_files is positive, it also
    creates an individual file for each conformation, incrementing a counter and opening
    a new file until num_individual_files is reached.  Call .next_conformation() to
    move to a new file.
    """
    def __init__(self, main_file_path, num_individual_files=0):
        self.num_individual_files = num_individual_files
        self.make_individual_files = num_individual_files > 0
        self.individual_file = None
        self.main_file_name = main_file_path
        self.main_file = open(main_file_path, 'w')
        _, self.extension = os.path.splitext(self.main_file_name)
        if len(self.extension) == 0:
            raise Exception('SDFWriter assumes filenames have an extension at the end.')
        self.counter = 0
        self.next_conformation()

    def _get_individual_file(self):
        if self.individual_file:
            self.individual_file.close()

        # insert _<num> just before the extension
        new_file_name = self.main_file_name.replace(self.extension, '_%d%s' % (self.counter, self.extension))
        return open(new_file_name, 'w')


    def write(self, data):
        self.main_file.write(data)
        if self.make_individual_files:
            self.individual_file.write(data)

    def next_conformation(self):
        self.counter += 1

        if self.counter > self.num_individual_files:
            self.make_individual_files = False

        if self.make_individual_files:
            self.individual_file = self._get_individual_file()

    def close(self):
        self.main_file.close()
        if self.individual_file:
            self.individual_file.close()

class sdf_out:
    #Write a SDF file for viewing that contains the low energy conformations in ascending order of energy.
    # Provide an integer to make_individual_files to additionally make that number of individual files, one per conformation.
    def __init__(self, sdffile_name, file_data, num_individual_files=0):
        sdffile = SDFWriter(sdffile_name, num_individual_files)

        for file in file_data:
            sdffile.write(os.path.splitext(os.path.basename(file.name))[0]+"\n")
            if hasattr(file, "scfenergies"):
                sdffile.write("     E="+str(file.scfenergies[-1])+"\n\n")
            else:
                sdffile.write("     E=None\n\n")
            nbonds = 0
            if hasattr(file, 'connectivity'):
                for atomi in range(0,file.natom):
                    for atomj in range(atomi,file.natom):
                        if atomj in file.connectivity[atomi]:
                            nbonds += 1
            if hasattr(file, 'connectivity'):
                sdffile.write(str(file.natom).rjust(3)+str(nbonds).rjust(3)+"  0  0  0  0  0  0  0  0  0999 V2000")
            if hasattr(file, 'atomcoords') and hasattr(file, 'atomnos'):
                for j,atom in enumerate(file.atomnos):
                    x = "%.4f" % file.atomcoords[j][0]
                    y = "%.4f" % file.atomcoords[j][1]
                    z = "%.4f" % file.atomcoords[j][2]
                    sdffile.write("\n"+x.rjust(10)+y.rjust(10)+z.rjust(10)+periodictable[int(atom)].rjust(2)+"   0  0  0  0  0  0  0  0  0  0  0  0")
            if hasattr(file, 'connectivity'):
                #print(len(file.connectivity), file.connectivity)
                for atomi in range(0,file.natom):
                    for atomj in range(atomi,file.natom):
                        if atomj in file.connectivity[atomi]:
                            sdffile.write("\n"+str(atomi+1).rjust(3)+str(atomj+1).rjust(3)+str(1).rjust(2)+" 0")
            sdffile.write("\nM  END\n$$$$    \n")
            sdffile.next_conformation()
        sdffile.close()

class xyz_out:
    """
    Enables output of optimized coordinates to a single xyz-formatted file.

    Writes Cartesian coordinates of parsed chemical input.

    Attributes:
        xyz (file object): path in current working directory to write Cartesian coordinates.
    """
    def __init__(self, xyz_file, file_data):

        self.xyz = open(xyz_file, 'w')

        for file in file_data:
            if hasattr(file, 'natom'): self.xyz.write(str(file.natom)+"\n")
            if hasattr(file, "scfenergies"):
                self.xyz.write(
                    '{:<39} {:>13} {:13.6f}\n'.format(os.path.splitext(os.path.basename(file.name))[0], 'Eopt',
                                                    file.scfenergies[-1]))
            else:
                self.xyz.write('{:<39}\n'.format(os.path.splitext(os.path.basename(file.name))[0]))
            if hasattr(file, 'atomcoords') and hasattr(file, 'atomnos'):
                for n, atom in enumerate(file.atomnos):
                    self.xyz.write('{:>1}'.format(periodictable[int(atom)]))
                    for cart in file.atomcoords[n]:
                        self.xyz.write('{:13.6f}'.format(cart))
                    self.xyz.write('\n')

        self.xyz.close()

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

class getoutData:
    """
    Read molecule data from a computational chemistry output file.

    Currently supports Gaussian and ORCA output types.

    Attributes:
        vibfreqs (list): list of frequencies parsed from Gaussian file.
        REDMASS (list): list of reduced masses parsed from Gaussian file.
        FORCECONST (list): list of force constants parsed from Gaussian file.
        NORMALMODE (list): list of normal modes parsed from Gaussian file.
        atom_nums (list): list of atom number IDs.
        atom_types (list): list of atom element symbols.
        cartesians (list): list of cartesian coordinates for each atom.
        atomictypes (list): list of atomic types output in Gaussian files.
        connectivity (list): list of atomic connectivity in a molecule, based on covalent radii
    """
    def __init__(self, file, options=None):
        with open(file) as f:
            data = f.readlines()
        self.program = 'none'

        self.name = file #os.path.splitext(file)[0]

        for line in data:
            if "Gaussian" in line:
                self.program = "Gaussian"
                break
            if "* O   R   C   A *" in line:
                self.program = "Orca"
                break
            if "NWChem" in line:
                self.program = "NWChem"
                break

        def get_cpu(self, outlines, format):
            """Read output for cpu time."""
            days, hours, mins, secs, msecs = 0, 0, 0, 0, 0

            for line in outlines:
                if format == "Gaussian":
                    if line.strip().find("Job cpu time") > -1:
                        days += int(line.split()[3])
                        hours += int(line.split()[5])
                        mins += int(line.split()[7])
                        msecs += int(float(line.split()[9]) * 1000.0)
                if format == "Orca":
                    if line.strip().find("TOTAL RUN TIME") > -1:
                        days += int(line.split()[3])
                        hours += int(line.split()[5])
                        mins += int(line.split()[7])
                        secs += int(line.split()[9])
                        msecs += float(line.split()[11])
                if format == "NWChem":
                    if line.strip().find("Total times") > -1:
                        secs += float(line.split()[3][0:-1])

            return [days,hours,mins,secs,msecs]

        def get_energy(self, outlines, format):
            scfenergies = []
            for line in outlines:
                if format == "Gaussian":
                    if line.strip().startswith('SCF Done:'):
                        scfenergies.append(float(line.strip().split()[4]))
                    # For Counterpoise calculations the corrected energy value will be taken
                    elif line.strip().startswith('Counterpoise corrected energy'):
                        scfenergies.append(float(line.strip().split()[4]))
                    # For MP2 calculations replace with EUMP2
                    elif 'EUMP2 =' in line.strip():
                        scfenergies.append(float((line.strip().split()[5]).replace('D', 'E')))
                    # For ONIOM calculations use the extrapolated value rather than SCF value
                    elif "ONIOM: extrapolated energy" in line.strip():
                        scfenergies.append((float(line.strip().split()[4])))
                    # For Semi-empirical or Molecular Mechanics calculations
                    elif "Energy= " in line.strip() and "Predicted" not in line.strip() and "Thermal" not in line.strip():
                        scfenergies.append((float(line.strip().split()[1])))
                if format == "Orca":
                    if line.strip().startswith('FINAL SINGLE POINT ENERGY'):
                        scfenergies.append(float(line.strip().split()[4]))
                if format == "NWChem":
                    if line.strip().startswith('Total DFT energy ='):
                        scfenergies.append(float(line.strip().split()[4]))
            return scfenergies

        def get_freqs(self, outlines, natoms, format):
            self.vibfreqs = []
            self.REDMASS = []
            self.FORCECONST = []
            self.NORMALMODE = []
            self.linear_mol = 0
            freqs_so_far = 0
            if format == "Gaussian":
                for i in range(0, len(outlines)):
                    if outlines[i].find(" Frequencies -- ") > -1:
                        nfreqs = len(outlines[i].split())
                        for j in range(2, nfreqs):
                            self.vibfreqs.append(float(outlines[i].split()[j]))
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
                for line in outlines:
                    # Look for thermal corrections, paying attention to point group symmetry
                    if line.strip().startswith('Zero-point correction='):
                        self.zero_point_corr = float(line.strip().split()[2])
                    # Grab Multiplicity
                    elif 'Multiplicity' in line.strip():
                        try:
                            self.mult = int(line.split('=')[-1].strip().split()[0])
                        except:
                            self.mult = int(line.split()[-1])
                    # Grab molecular mass
                    elif line.strip().startswith('Molecular mass:'):
                        self.molecular_mass = float(line.strip().split()[2])
                    # Grab rational symmetry number
                    elif line.strip().startswith('Rotational symmetry number'):
                        self.symmno = int((line.strip().split()[3]).split(".")[0])
                    # Grab point group
                    elif line.strip().startswith('Full point group'):
                        if line.strip().split()[3] == 'D*H' or line.strip().split()[3] == 'C*V':
                            self.linear_mol = 1
                        self.point_group = line.strip().split()[3]
                    # Grab rotational constants
                    elif line.strip().startswith('Rotational constants (GHZ):'):
                        try:
                            self.roconst = [float(line.strip().replace(':', ' ').split()[3]),
                                            float(line.strip().replace(':', ' ').split()[4]),
                                            float(line.strip().replace(':', ' ').split()[5])]
                        except ValueError:
                            if line.strip().find('********'):
                                self.linear_warning = True
                                self.roconst = [float(line.strip().replace(':', ' ').split()[4]),
                                                float(line.strip().replace(':', ' ').split()[5])]
                    # Grab rotational temperatures
                    elif line.strip().startswith('Rotational temperature '):
                        self.rotemp = [float(line.strip().split()[3])]
                    elif line.strip().startswith('Rotational temperatures'):
                        try:
                            self.rotemp = [float(line.strip().split()[3]), float(line.strip().split()[4]),
                                      float(line.strip().split()[5])]
                        except ValueError:
                            self.rotemp = None
                            if line.strip().find('********'):
                                self.linear_warning = True
                                self.rotemp = [float(line.strip().split()[4]), float(line.strip().split()[5])]

        def getatom_types(self, outlines, program):
            if program == "Gaussian":
                for i, oline in enumerate(outlines):
                    if "Input orientation" in oline or "Standard orientation" in oline:
                        self.atomnos, self.atom_types, self.atomcoords, self.atomictypes, carts = [], [], [], [], \
                                                                                                    outlines[i + 5:]
                        for j, line in enumerate(carts):
                            if "-------" in line:
                                break
                            self.atomnos.append(int(line.split()[1]))
                            self.atom_types.append(element_id(int(line.split()[1])))
                            self.atomictypes.append(int(line.split()[2]))
                            if len(line.split()) > 5:
                                self.atomcoords.append(
                                    [float(line.split()[3]), float(line.split()[4]), float(line.split()[5])])
                            else:
                                self.atomcoords.append(
                                    [float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])
            if program == "Orca":
                for i, oline in enumerate(outlines):
                    if "*" in oline and ">" in oline and "xyz" in oline:
                        self.atomnos, self.atom_types, self.atomcoords, carts = [], [], [], outlines[i + 1:]
                        for j, line in enumerate(carts):
                            if ">" in line and "*" in line:
                                break
                            if len(line.split()) > 5:
                                self.atomcoords.append(
                                    [float(line.split()[3]), float(line.split()[4]), float(line.split()[5])])
                                self.atom_types.append(line.split()[2])
                                self.atomnos.append(element_id(line.split()[2], num=True))
                            else:
                                self.atomcoords.append(
                                    [float(line.split()[2]), float(line.split()[3]), float(line.split()[4])])
                                self.atom_types.append(line.split()[1])
                                self.atomnos.append(element_id(line.split()[1], num=True))
            if program == "NWChem":
                for i, oline in enumerate(outlines):
                    if "Output coordinates" in oline:
                        self.atomnos, self.atom_types, self.atomcoords, self.atomictypes, carts = [], [], [], [], outlines[i+4:]
                        for j, line in enumerate(carts):
                            if line.strip()=='' :
                                break
                            self.atomnos.append(int(float(line.split()[2])))
                            self.atom_types.append(element_id(int(float(line.split()[2]))))
                            self.atomictypes.append(int(float(line.split()[2])))
                            self.atomcoords.append([float(line.split()[3]),float(line.split()[4]),float(line.split()[5])])

        def get_level_of_theory(self, outlines, program):
            """Read output for the level of theory and basis set used."""
            repeated_theory = 0
            level, bs = 'none', 'none'

            for line in outlines:
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
            self.level_of_theory = level_of_theory
            self.functional = level
            self.basis_set = bs

        def get_jobtype(self, outlines, program):
            """Read output for the level of theory and basis set used."""
            job = ''
            for line in outlines:
                if line.strip().find('\\SP\\') > -1:
                    job += 'SP'
                if line.strip().find('\\FOpt\\') > -1:
                    job += 'GS'
                if line.strip().find('\\FTS\\') > -1:
                    job += 'TS'
                if line.strip().find('\\Freq\\') > -1:
                    job += 'Freq'
            self.job = job

        def read_initial(self, data, program):
            """At beginning of procedure, read level of theory, solvation model, and check for normal termination"""
            level, bs, program, keyword_line = 'none', 'none', 'none', 'none'
            progress, orientation = 'Incomplete', 'Input'
            a, repeated_theory = 0, 0
            no_grid = True
            DFT, dft_used, level, bs, scf_iradan, cphf_iradan = False, 'F', 'none', 'none', False, False
            grid_lookup = {1: 'sg1', 2: 'coarse', 4: 'fine', 5: 'ultrafine', 7: 'superfine'}

            for line in data:
                # Determine program to find solvation model used
                if "Gaussian" in line:
                    program = "Gaussian"
                if "* O   R   C   A *" in line:
                    program = "Orca"
                if "NWChem" in line:
                    program = "NWChem"
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

            self.solvation_model = solvation_model
            self.progress = progress
            self.orientation = orientation
            self.dft_used = dft_used

        # extract various things from file
        read_initial(self, data, self.program)
        get_level_of_theory(self, data, self.program)
        get_jobtype(self, data, self.program)
        getatom_types(self, data, self.program)
        self.scfenergies = get_energy(self, data, self.program)
        if options.spc == 'link': del self.scfenergies[-1]
        self.cpu = get_cpu(self, data, self.program)
        self.natom = len(self.atom_types)
        try:
            get_freqs(self, data, self.natom, self.program)
        except:
            pass
        try:
            self.ex_sym, self.ex_pgroup = self.ex_sym(file.split('.')[0].replace('/', '_'))
        except:
            pass

        if hasattr(options, 'spc'):
            self.sp_energy = 0.0
            name = os.path.splitext(file)[0]
            spc_file = None

            if options.spc == 'link':
                spc_file = file

            if options.spc != 'link' and options.spc != False:
                for filename in [name+'_'+options.spc+'.out', name+'_'+options.spc+'.log', name+'-'+options.spc+'.out', name+'-'+options.spc+'.log']:
                    if os.path.exists(filename):
                        spc_file = filename

            if spc_file != None:
                with open(spc_file) as f:
                    spcdata = f.readlines()
                self.spc_program = 'none'

                for line in spcdata:
                    if "Gaussian" in line:
                        self.spc_program = "Gaussian"
                        break
                    if "* O   R   C   A *" in line:
                        self.spc_program = "Orca"
                        break
                    if "NWChem" in line:
                        self.spc_program = "NWChem"
                        break

                self.sp_energy = get_energy(self, spcdata, self.spc_program)[-1]
                if options.spc != 'link': self.sp_cpu = get_cpu(self, spcdata, self.spc_program) # the CPU has already been included for link job

    # Get external symmetry number
    def ex_sym(self, file):
        coords_string = self.coords_string()
        coords = coords_string.encode('utf-8')
        c_coords = ctypes.c_char_p(coords)

        # Determine OS with sys.platform to see what compiled symmetry file to use
        platform = sys.platform
        if platform.startswith('linux'):  # linux - .so file
            path1 = sharepath('symmetry_linux.so')
            newlib = 'lib_' + file + '.so'
            path2 = sharepath(newlib)
            copy = 'cp ' + path1 + ' ' + path2
            os.popen(copy).close()
            symmetry = ctypes.CDLL(path2)
        elif platform.startswith('darwin'):  # macOS - .dylib file
            path1 = sharepath('symmetry_mac.dylib')
            newlib = 'lib_' + file + '.dylib'
            path2 = sharepath(newlib)
            copy = 'cp ' + path1 + ' ' + path2
            os.popen(copy).close()
            symmetry = ctypes.CDLL(path2)
        elif platform.startswith('win'):  # windows - .dll file
            path1 = sharepath('symmetry_windows.dll')
            newlib = 'lib_' + file + '.dll'
            path2 = sharepath(newlib)
            copy = 'copy ' + path1 + ' ' + path2
            os.popen(copy).close()
            symmetry = ctypes.cdll.LoadLibrary(path2)

        symmetry.symmetry.restype = ctypes.c_char_p
        pgroup = symmetry.symmetry(c_coords).decode('utf-8')
        ex_sym = pg_sm.get(pgroup)

        # Remove file
        if platform.startswith('linux'):  # linux - .so file
            remove = 'rm ' + path2
            os.popen(remove).close()
        elif platform.startswith('darwin'):  # macOS - .dylib file
            remove = 'rm ' + path2
            os.popen(remove).close()
        elif platform.startswith('win'):  # windows - .dll file
            handle = symmetry._handle
            del symmetry
            ctypes.windll.kernel32.FreeLibrary(ctypes.c_void_p(handle))
            remove = 'Del /F "' + path2 + '"'
            os.popen(remove).close()

        return ex_sym, pgroup

    def int_sym(self):
        self.get_connectivity()
        cap = [1, 9, 17]
        neighbor = [5, 6, 7, 8, 14, 15, 16]
        int_sym = 1

        for i, row in enumerate(self.connectivity):
            if self.atomnos[i] != 6: continue
            As = np.array(self.atomnos)[row]
            if len(As == 4):
                neighbors = [x for x in As if x in neighbor]
                caps = [x for x in As if x in cap]
                if (len(neighbors) == 1) and (len(set(caps)) == 1):
                    int_sym *= 3
        return int_sym

    # Convert coordinates to string that can be used by the symmetry.c program
    def coords_string(self):
        xyzstring = str(len(self.atomnos)) + '\n'
        for atom, xyz in zip(self.atomnos, self.atomcoords):
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
                distance = np.linalg.norm(np.array(self.atomcoords[i]) - np.array(self.atomcoords[j]))
                if distance < cutoff:
                    row.append(j)
            connectivity.append(row)
            self.connectivity = connectivity
