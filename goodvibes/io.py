''' Parsing and Writing Functions for the Goodvibes package.'''

# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import os.path
import sys
import time
from glob import glob
import pandas as pd
from rdkit import Chem
import cclib

import goodvibes.xyz2mol as xyz2mol
from goodvibes.utils import KCAL_TO_AU
from goodvibes.version import __version__

# compchem packages supported by GoodVibes
SUPPORTED_PACKAGES = set(('Gaussian', 'ORCA', 'QChem'))

# most compchem outputs look like this:
SUPPORTED_EXTENSIONS = set(('.out', '.log'))

# Some literature references
GRIMME_REF = "Grimme, S. Chem. Eur. J. 2012, 18, 9955-9964"
TRUHLAR_REF = "Ribeiro, R. F.; Marenich, A. V.; Cramer, C. J.; Truhlar, D. G. J. Phys. Chem. B 2011, 115, 14556-14562"
HEAD_GORDON_REF = "Li, Y.; Gomes, J.; Sharada, S. M.; Bell, A. T.; Head-Gordon, M. J. Phys. Chem. C 2015, 119, 1840-1850"
GOODVIBES_REF = ("Luchini, G.; Alegre-Requena, J. V.; Funes-Ardoiz, I.; Paton, R. S. F1000Research, 2020, 9, 291."
                 "\n   DOI: 10.12688/f1000research.22758.1")
SIMON_REF = "Simon, L.; Paton, R. S. J. Am. Chem. Soc. 2018, 140, 5412-5420"


''' Functions used to load and parse compchem output files '''
def load_filelist(arglist, spc = False):
    '''returns file list with acceptable extensions and user-requested arguments'''
    files = []
    sp_files = []
    user_args = '   Requested: '
    for elem in arglist:
        try:
            if os.path.splitext(elem)[1].lower() in SUPPORTED_EXTENSIONS:  # Look for file names
                for file in glob(elem):
                    # if we don't expect single point calculations then grab everything
                    if spc is False or spc == 'link':
                        files.append(file)
                    # need to check for both opt and single point calculations separately
                    elif file.find('_' + spc + ".") == -1:
                        files.append(file)
                        name, ext = os.path.splitext(file)
                    else:
                        sp_files.append(file)
            else:
                user_args += elem + ' '
        except IndexError:
            pass

    # check that we have corresponding singlepoint outputs
    if spc is not False and spc != 'link':
        filenames = [os.path.basename(file).split('.')[0] for file in files]
        spnames = [os.path.basename(file).split('.')[0] for file in sp_files]
        sp_starts = [name.split('_'+spc)[0] for name in spnames]

        for name in filenames:
            if not name+'_'+spc in spnames:
                sys.exit(f"\n   Error! SPC output {name}_{spc} not found!\n")

        # remove any superfluous sp files
        sp_files = [file for file,start in zip(sp_files,sp_starts) if start in filenames]

    return files, sp_files, user_args

def get_cc_packages(log, files):
    '''Use cclib to detect which package(s) were used to generate the output files'''
    package_list = []
    for file in files:
        try:
            parser = cclib.io.ccopen(file)
            data = parser.parse()
            package = data.metadata['package']
            package_list.append(package)
        except KeyError:
            package_list.append('Unknown')
            log.write(f'\n   ! Unable to parse {file} !\n')

    for package in list(set(package_list)):
        if package not in SUPPORTED_PACKAGES:
            log.write('\nx  Warning: Unsupported package detected: ' + package + ' !\n')
        else:
            log.write('\no  Supported compchem packages detected: ' + package+'\n')

    return package_list

def get_cc_species(log, files, package_list):
    '''Use cclib to parse the output files and return a list of cclib data objects'''
    species_list = []
    for file, package in zip(files, package_list):
        if package in SUPPORTED_PACKAGES:
            parser = cclib.io.ccopen(file)
            data = parser.parse()
            data.name = os.path.basename(file.split('.')[0])
            species_list.append(data)
        else:
            log.write(f'\n   ! Unable to parse {file} !\n')
    return species_list

def get_levels_of_theory(log, species_list):
    '''Determine the level of theory used for each species in the list of cclib data objects'''
    level_of_theory = []
    for species in species_list:
        try:
            level = species.metadata['functional'] + '/' + species.metadata['basis_set']
        except KeyError:
            level = 'Unknown'
        level_of_theory.append(level)

    # remove duplicates
    model_chemistry =  (list(set(level_of_theory)))

    if len(model_chemistry) == 1:
        if model_chemistry[0] != 'Unknown':
            log.write('\no  A model chemistry detected: ' + model_chemistry[0] + '!')
        model = model_chemistry[0]

    else:
        for model in model_chemistry:
            log.write('\n   Multiple levels of theory detected: ' + model)
        model = 'mixed'

    return model

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
        raise ValueError(f"File {datfile} does not exist")

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
                if oldtemp == 0:
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


''' Functions used to write structure files '''
class Xyz_Out:
    """
    Enables output of optimized coordinates to a single xyz-formatted file.
    Writes Cartesian coordinates of parsed chemical input.
    Attributes:
        xyz (file object): path in current working directory to write Cartesian coordinates.
    """
    def __init__(self, filename):
        self.xyz = open(filename, 'w')

    def write_text(self, message):
        '''Writes a string to the xyz file.'''
        self.xyz.write(message + "\n")

    def write_coords(self, atoms, coords):
        '''Writes the atoms and coordinates to the xyz file.'''
        for n, carts in enumerate(coords):
            self.xyz.write(f'{atoms[n]:>1}')
            for cart in carts:
                self.xyz.write(f'{cart:13.6f}')
            self.xyz.write('\n')

    def finalize(self):
        '''Closes the xyz file.'''
        self.xyz.close()

def write_to_xyz(log, thermo_data, filename):
    '''Writes the optimized coordinates of the species to a single xyz-formatted file.'''
    xyz = Xyz_Out(filename)
    for bbe in thermo_data:
        if hasattr(bbe, "atomtypes") and hasattr(bbe, "cartesians") and  hasattr(bbe, "qh_gibbs_free_energy"):
            xyz.write_text(str(len(bbe.atomtypes)))
            xyz.write_text(
                    f'{bbe.name:<39} qh-g: {bbe.qh_gibbs_free_energy:13.6f}')
            xyz.write_coords(bbe.atomtypes, bbe.cartesians)
        else:
            log.write(f"\nx  Error writing {bbe.name} to XYZ ...")
    xyz.finalize()

def write_to_sdf(log, thermo_data, filename):
    '''Writes the optimized coordinates of the species to a single sdf-formatted file.'''
    with Chem.SDWriter(filename) as w:
        # create mol objects using xyz2mol based on atoms, coordinates and total charge
        for bbe in thermo_data:
            try:
                if hasattr(bbe, "atomtypes") and hasattr(bbe, "cartesians") and  hasattr(bbe, "qh_gibbs_free_energy"):
                    mol = xyz2mol.xyz2mol(bbe.atomnos.tolist(), bbe.cartesians.tolist(), charge=bbe.charge)[0]
                    mol.SetProp('_Name', bbe.name)
                    mol.SetProp('E', str(bbe.scf_energy))
                    mol.SetProp('ZPE', str(bbe.zpe))
                    mol.SetProp('H', str(bbe.enthalpy))
                    mol.SetProp('S', str(bbe.entropy))
                    mol.SetProp('qh-S', str(bbe.qh_entropy))
                    mol.SetProp('G(T)', str(bbe.gibbs_free_energy))
                    mol.SetProp('qh-G(T)', str(bbe.qh_gibbs_free_energy))
                    mol.SetProp('im freq', str(bbe.im_freq))
                    w.write(mol)
            except ValueError:
                log.write(f"\nx  Error writing {bbe.name} to SDF ...")


''' Logger class and functions used for writing to terminal and file'''
class Logger:
    """
    Enables output to terminal and to text file.

    Writes GV output to .dat or .csv files.

    Attributes:
        csv (bool): decides if comma separated value file is written.
        log (file object): file to write GV output to.
        thermodata (bool): decides if string passed to logger is thermochemical data, needing to be separated by commas
    """
    def __init__(self, filein, append):
        log_file = filein+'_'+append+'.dat'
        self.log = open(log_file, 'w')

    def write(self, message):
        '''Writes a string to the log file.'''
        print(message, end='')
        self.log.write(message)

    def write_df(self, df, dp=5):
        '''Writes a dataframe to the log file using {dp} decimal places.'''

        if dp == 3:
            pd.options.display.float_format = '{:.3f}'.format
        if dp == 4:
            pd.options.display.float_format = '{:.4f}'.format
        elif dp == 6:
            pd.options.display.float_format = '{:.6f}'.format
        else:
            pd.options.display.float_format = '{:.5f}'.format

        print(df)
        self.log.write(df.to_string())

    def fatal(self, message):
        '''Writes a fatal error message to the log file and exits the program.'''
        print(message + "\n")
        self.log.write(message + "\n")
        self.finalize()
        sys.exit(1)

    def finalize(self):
        '''Closes the log file.'''
        self.log.close()

def gv_header(log, files, temp=298.15, temperature_interval=False, conc=False, cosmo_int=False, cosmo = False, s_freq_cutoff=100.0, h_freq_cutoff=100.0, qs='grimme', qh=False, spc=False, media=False):
    '''Prints the header of the GoodVibes output file.'''

    start = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    log.write("   GoodVibes v" + __version__ + " " + start + "\n   REF: " + GOODVIBES_REF + "\n\n")

    # Check if user has specified any files, if not quit now
    if len(files) == 0:
        sys.exit("\n   Please provide GoodVibes with calculation output files on the command line.\n"
                "   For help, use option '-h'\n")

    if temperature_interval is False:
        log.write("   Temperature = " + str(temp) + " Kelvin")

    # If not at standard temp, need to correct the molarity of 1 atmosphere (assuming pressure is still 1 atm)
    if conc:
        log.write("   Concentration = " + str(conc) + " mol/L")
    else:
        log.write("   Pressure = 1 atm")
    log.write('\n   All energetic values below shown in Hartree unless otherwise specified.')

    # COSMO-RS temperature interval
    if cosmo_int:
        args = cosmo_int.split(',')
        cfile = args[0]
        cinterval = args[1:]
        log.write('\n\n   Reading COSMO-RS file: ' + cfile + ' over a T range of ' + cinterval[0] + '-' +
                cinterval[1] + ' K.')

        t_interval, gsolv_dicts = cosmo_rs_out(cfile, files, interval=cinterval)
        temperature_interval = True

    elif cosmo is not False:  # Read from COSMO-RS output
        try:
            cosmo_solv = cosmo_rs_out(cosmo, files)
            log.write('\n\n   Reading COSMO-RS file: ' + cosmo)
        except ValueError:
            cosmo_solv = None
            log.write('\n\n   Warning! COSMO-RS file ' + cosmo + ' requested but not found')

    # Summary of the quasi-harmonic treatment; print out the relevant reference
    log.write("\n\n   Entropic quasi-harmonic treatment: frequency cut-off value of " + str(
        s_freq_cutoff) + " wavenumbers will be applied.")
    if qs == "grimme":
        log.write("\n   QS = Grimme: Using a mixture of RRHO and Free-rotor vibrational entropies.")
        qs_ref = GRIMME_REF
    elif qs == "truhlar":
        log.write("\n   QS = Truhlar: Using an RRHO treatment where low frequencies are adjusted to the cut-off value.")
        qs_ref = TRUHLAR_REF
    else:
        log.fatal("\n   FATAL ERROR: Unknown quasi-harmonic model " + qs + " specified (QS must = grimme or truhlar).")
    log.write("\n   REF: " + qs_ref + '\n')

    # Check if qh-H correction should be applied
    if qh is not False:
        log.write("\n\n   Enthalpy quasi-harmonic treatment: frequency cut-off value of " + str(
            h_freq_cutoff) + " wavenumbers will be applied.")
        log.write("\n   QH = Head-Gordon: Using an RRHO treatement with an approximation term for vibrational energy.")
        qh_ref = HEAD_GORDON_REF
        log.write("\n   REF: " + qh_ref + '\n')

    # Whether single-point energies are to be used
    if spc:
        log.write("\n   Combining final single point energy with thermal corrections.")
    # Solvent correction message
    if media:
        log.write("\n   Applying standard concentration correction (based on density at 20C) to solvent media.")

def gv_tabulate(thermo_data):
    '''Returns a pandas DataFrame of the thermochemical data.'''
    thermo_df = pd.DataFrame([vars(dat) for dat in thermo_data])
    return thermo_df

def gv_summary(gv_df, spc=False, symm=False, imag=False, boltz=False):
    '''Print a summary of the thermochemistry for the species in the log files'''    

    if spc is not False:
        columns = ['name', 'charge', 'mult', 'sp_energy', 'scf_energy', 'zpe', 'sp_enthalpy', 'ts', 'qhts',
            'sp_gibbs_free_energy', 'sp_qh_gibbs_free_energy']

    else:
        columns = ['name', 'charge', 'mult', 'scf_energy', 'zpe', 'enthalpy', 'ts', 'qhts',
            'gibbs_free_energy', 'qh_gibbs_free_energy']

    if symm is not False:
        columns += ['point_group']
    if imag is not False:
        columns += ['im_freq']
    if boltz is not False:
        columns += ['boltz_fac']

    nice_df = gv_df[columns].copy()

    nice_df.rename(columns={'name': 'Species', 'charge': 'chg', 'sp_energy': 'E spc', 'scf_energy': 'E', 'zpe': 'ZPE', 'sp_gibbs_free_energy': 'G(T) spc', 'sp_qh_gibbs_free_energy': 'qh-G(T) spc', 'gibbs_free_energy': 'G(T)', 'qh_gibbs_free_energy': 'qh-G(T)', 'sp_enthalpy': 'H spc','enthalpy': 'H', 'ts': 'T.S', 'qhts': 'T.qh-S'}, inplace=True)
    nice_df.rename(columns={'point_group': 'PG'}, inplace=True)

    if imag is not False:
        nice_df['v_im'] = [str(round(val[0],2)) if len(val) > 0 else '' for val in nice_df["im_freq"]]
        nice_df.drop(columns=['im_freq'], inplace=True)

    if boltz is not False:
        nice_df['Boltz'] = [str(round(val,3)) for val in nice_df["boltz_fac"]]
        nice_df.drop(columns=['boltz_fac'], inplace=True)

    return nice_df
