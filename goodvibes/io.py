''' Parsing and Writing Functions for the Goodvibes package.'''

# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import os.path
import sys
import time
from glob import glob
import xyz2mol
from rdkit import Chem
import cclib

from goodvibes.utils import ATMOS, GAS_CONSTANT, KCAL_TO_AU

# compchem packages supported by GoodVibes
SUPPORTED_PACKAGES = set(('Gaussian', 'Orca'))

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
                        if not (os.path.exists(name + '_' + spc + '.log') or os.path.exists(
                                name + '_' + spc + '.out')) and spc != 'link':
                            sys.exit(f"\n   Error! SPC output file '{name}+'_'+{spc}' not found! "
                                    "files should be named 'filename_spc' or specify link job.'\n")
            else:
                user_args += elem + ' '
        except IndexError:
            pass
    return files, user_args

def get_cc_packages(log, files):
    '''Use cclib to detect which package(s) were used to generate the output files'''
    package_list = []
    for file in files:
        try:
            parser = cclib.io.ccopen(file)
            package = parser.metadata['package']
            package_list.append(package)
        except:
            package_list.append('Unknown')
            log.write('\n   ! Unable to parse {} !'.format(file) + '\n')

    for package in list(set(package_list)):
        if package not in SUPPORTED_PACKAGES:
            log.write('\nx  Warning: Unsupported package detected: ' + package + ' !\n')
        else:
            log.write('\no  Supported compchem packages detected: ' + package + '\n')

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
            log.write('\n   ! Unable to parse {} !'.format(file) + '\n')
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
        log.write('   A model chemistry detected: ' + model_chemistry[0] + '\n')
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
class xyz_out:
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
            self.xyz.write('{:>1}'.format(atoms[n]))
            for cart in carts:
                self.xyz.write('{:13.6f}'.format(cart))
            self.xyz.write('\n')

    def finalize(self):
        '''Closes the xyz file.'''
        self.xyz.close()

def write_to_xyz(log, thermo_data, filename):
    '''Writes the optimized coordinates of the species to a single xyz-formatted file.'''
    xyz = xyz_out(filename)
    for bbe in thermo_data:
        if hasattr(bbe, "atomtypes") and hasattr(bbe, "cartesians") and  hasattr(bbe, "qh_gibbs_free_energy"):
            xyz.write_text(str(len(bbe.atomtypes)))
            xyz.write_text(
                    '{:<39} {:>13} {:13.6f}'.format(os.path.splitext(os.path.basename(bbe.name))[0], 'qh-G(',
                                                    bbe.qh_gibbs_free_energy))
            xyz.write_coords(bbe.atomtypes, bbe.cartesians)
        else:
            log.write("\nx  Error writing {} to XYZ ...".format(bbe.name))
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
                    mol.SetProp('im freq', str(bbe.im_frequency_wn))
                    w.write(mol)
            except:
                log.write("\nx  Error writing {} to SDF ...".format(bbe.name))


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
    def __init__(self, filein, append, csv):
        self.csv = csv
        if not self.csv:
            suffix = 'dat'
        else:
            suffix = 'csv'
        self.log = open('{0}_{1}.{2}'.format(filein, append, suffix), 'w')

    def write(self, message, thermodata=False):
        '''Writes a string to the log file.'''
        self.thermodata = thermodata
        print(message, end='')
        if self.csv and self.thermodata:
            items = message.split()
            message = ",".join(items)
            message = message + ","
        self.log.write(message)

    def fatal(self, message):
        '''Writes a fatal error message to the log file and exits the program.'''
        print(message + "\n")
        self.log.write(message + "\n")
        self.finalize()
        sys.exit(1)

    def finalize(self):
        '''Closes the log file.'''
        self.log.close()

def gv_header(log, files, options, __version__):
    '''Prints the header of the GoodVibes output file.'''

    start = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    log.write("   GoodVibes v" + __version__ + " " + start + "\n   REF: " + GOODVIBES_REF + "\n\n")

    # Check if user has specified any files, if not quit now
    if len(files) == 0:
        sys.exit("\n   Please provide GoodVibes with calculation output files on the command line.\n"
                "   For help, use option '-h'\n")

    if options.temperature_interval is False:
        log.write("   Temperature = " + str(options.temperature) + " Kelvin")

    # If not at standard temp, need to correct the molarity of 1 atmosphere (assuming pressure is still 1 atm)
    if options.conc:
        log.write("   Concentration = " + str(options.conc) + " mol/L")
    else:
        options.conc = ATMOS / (GAS_CONSTANT * options.temperature)
        log.write("   Pressure = 1 atm")
    log.write('\n   All energetic values below shown in Hartree unless otherwise specified.')

    # COSMO-RS temperature interval
    if options.cosmo_int:
        args = options.cosmo_int.split(',')
        cfile = args[0]
        cinterval = args[1:]
        log.write('\n\n   Reading COSMO-RS file: ' + cfile + ' over a T range of ' + cinterval[0] + '-' +
                cinterval[1] + ' K.')

        t_interval, gsolv_dicts = cosmo_rs_out(cfile, files, interval=cinterval)
        options.temperature_interval = True

    elif options.cosmo is not False:  # Read from COSMO-RS output
        try:
            cosmo_solv = cosmo_rs_out(options.cosmo, files)
            log.write('\n\n   Reading COSMO-RS file: ' + options.cosmo)
        except ValueError:
            cosmo_solv = None
            log.write('\n\n   Warning! COSMO-RS file ' + options.cosmo + ' requested but not found')

    if options.freq_cutoff != 100.0:
        options.S_freq_cutoff = options.freq_cutoff
        options.H_freq_cutoff = options.freq_cutoff

    # Summary of the quasi-harmonic treatment; print out the relevant reference
    log.write("\n\n   Entropic quasi-harmonic treatment: frequency cut-off value of " + str(
        options.S_freq_cutoff) + " wavenumbers will be applied.")
    if options.QS == "grimme":
        log.write("\n   QS = Grimme: Using a mixture of RRHO and Free-rotor vibrational entropies.")
        qs_ref = GRIMME_REF
    elif options.QS == "truhlar":
        log.write("\n   QS = Truhlar: Using an RRHO treatment where low frequencies are adjusted to the cut-off value.")
        qs_ref = TRUHLAR_REF
    else:
        log.fatal("\n   FATAL ERROR: Unknown quasi-harmonic model " + options.QS + " specified (QS must = grimme or truhlar).")
    log.write("\n   REF: " + qs_ref + '\n')

    # Check if qh-H correction should be applied
    if options.QH:
        log.write("\n\n   Enthalpy quasi-harmonic treatment: frequency cut-off value of " + str(
            options.H_freq_cutoff) + " wavenumbers will be applied.")
        log.write("\n   QH = Head-Gordon: Using an RRHO treatement with an approximation term for vibrational energy.")
        qh_ref = HEAD_GORDON_REF
        log.write("\n   REF: " + qh_ref + '\n')

    # Whether single-point energies are to be used
    if options.spc:
        log.write("\n   Combining final single point energy with thermal corrections.")
    # Solvent correction message
    if options.media:
        log.write("\n   Applying standard concentration correction (based on density at 20C) to solvent media.")

def gv_summary(log, thermo_data, options):
    '''Prints the main summary to the GoodVibes output file.'''
    if options.spc is False:
        log.write("\n\n   ")
        if options.QH:
            log.write('{:<39} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} '
                        '{:>13}'.format("Structure", "E", "ZPE", "H", "qh-H", "T.S", "T.qh-S", "G(T)", "qh-G(T)"),
                        thermodata=True)
        else:
            log.write('{:<39} {:>13} {:>10} {:>13} {:>10} {:>10} {:>13} {:>13}'.format("Structure", "E", "ZPE", "H",
                                                                                        "T.S", "T.qh-S", "G(T)",
                                                                                        "qh-G(T)"), thermodata=True)
    else:
        log.write("\n\n   ")
        if options.QH:
            log.write('{:<39} {:>13} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} '
                        '{:>13}'.format("Structure", "E_SPC", "E", "ZPE", "H_SPC", "qh-H_SPC", "T.S", "T.qh-S",
                                        "G(T)_SPC", "qh-G(T)_SPC"), thermodata=True)
        else:
            log.write('{:<39} {:>13} {:>13} {:>10} {:>13} {:>10} {:>10} {:>13} '
                        '{:>13}'.format("Structure", "E_SPC", "E", "ZPE", "H_SPC", "T.S", "T.qh-S", "G(T)_SPC",
                                        "qh-G(T)_SPC"), thermodata=True)
    if options.cosmo is not False:
        log.write('{:>13} {:>16}'.format("COSMO-RS", "COSMO-qh-G(T)"), thermodata=True)
    if options.boltz is True:
        log.write('{:>7}'.format("Boltz"), thermodata=True)
    if options.imag_freq is True:
        log.write('{:>9}'.format("im freq"), thermodata=True)
    if options.ssymm:
        log.write('{:>13}'.format("Point Group"), thermodata=True)
    #log.write("\n" + stars + "")

    # Look for duplicates or enantiomers
    #if options.duplicate:
    #    dup_list = check_dup(files, thermo_data)
    #else:
    #    dup_list = []

    # Boltzmann factors and averaging over clusters
    if options.boltz is not False:
        boltz_facs, weighted_free_energy, boltz_sum = get_boltz(files, thermo_data, clustering, clusters,
                                                                options.temperature, dup_list)

    for bbe in thermo_data:  # Loop over the output files and compute thermochemistry
        #print(dir(bbe))
        #duplicate = False
        #if len(dup_list) != 0:
        #    for dup in dup_list:
        #        if dup[0] == file:
        #            duplicate = True
        #            log.write('\nx  {} is a duplicate or enantiomer of {}'.format(dup[0].rsplit('.', 1)[0],
        #                                                                            dup[1].rsplit('.', 1)[0]))
        #            break
        #if not duplicate:
        #    bbe = thermo_data[file]

            #if options.cputime != False:  # Add up CPU times
            #    if hasattr(bbe, "cpu"):
            #        if bbe.cpu != None:
            #            total_cpu_time = add_time(total_cpu_time, bbe.cpu)
            #    if hasattr(bbe, "sp_cpu"):
            #        if bbe.sp_cpu != None:
            #            total_cpu_time = add_time(total_cpu_time, bbe.sp_cpu)
            #if total_cpu_time.month > 1:
            #    add_days += 31

        # Check for possible error in Gaussian calculation of linear molecules which can return 2 rotational constants instead of 3
        if bbe.linear_warning:
            log.write("\nx  " + '{:<39}'.format(os.path.splitext(os.path.basename(bbe.name))[0]))
            log.write('          ----   Caution! Potential invalid calculation of linear molecule from Gaussian')
        else:
            if hasattr(bbe, "gibbs_free_energy"):
                if options.spc is not False:
                    if bbe.sp_energy != '!':
                        log.write("\no  ")
                        log.write('{:<39}'.format(os.path.splitext(os.path.basename(bbe.name))[0]), thermodata=True)
                        log.write(' {:13.6f}'.format(bbe.sp_energy), thermodata=True)
                    if bbe.sp_energy == '!':
                        log.write("\nx  ")
                        log.write('{:<39}'.format(os.path.splitext(os.path.basename(bbe.name))[0]), thermodata=True)
                        log.write(' {:>13}'.format('----'), thermodata=True)
                else:
                    log.write("\no  ")
                    log.write('{:<39}'.format(os.path.splitext(os.path.basename(bbe.name))[0]), thermodata=True)
            # Gaussian SPC file handling
            if hasattr(bbe, "scf_energy") and not hasattr(bbe, "gibbs_free_energy"):
                log.write("\nx  " + '{:<39}'.format(os.path.splitext(os.path.basename(bbe.name))[0]))
            # ORCA spc files
            elif not hasattr(bbe, "scf_energy") and not hasattr(bbe, "gibbs_free_energy"):
                log.write("\nx  " + '{:<39}'.format(os.path.splitext(os.path.basename(bbe.name))[0]))
            if hasattr(bbe, "scf_energy"):
                log.write(' {:13.6f}'.format(bbe.scf_energy), thermodata=True)
            # No freqs found
            if not hasattr(bbe, "gibbs_free_energy"):
                log.write("   Warning! Couldn't find frequency information ...")
            else:
                if all(getattr(bbe, attrib) for attrib in
                        ["enthalpy", "entropy", "qh_entropy", "gibbs_free_energy", "qh_gibbs_free_energy"]):
                    if options.QH:
                        log.write(' {:10.6f} {:13.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(
                            bbe.zpe, bbe.enthalpy, bbe.qh_enthalpy, (options.temperature * bbe.entropy),
                            (options.temperature * bbe.qh_entropy), bbe.gibbs_free_energy,
                            bbe.qh_gibbs_free_energy), thermodata=True)
                    else:
                        log.write(' {:10.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} '
                                    '{:13.6f}'.format(bbe.zpe, bbe.enthalpy,
                                                    (options.temperature * bbe.entropy),
                                                    (options.temperature * bbe.qh_entropy),
                                                    bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy),
                                    thermodata=True)

                if options.media is not False and options.media.lower() in solvents and options.media.lower() == \
                        os.path.splitext(os.path.basename(file))[0].lower():
                    log.write("  Solvent: {:4.2f}M ".format(media_conc))

        # Append requested options to end of output
        if options.cosmo and cosmo_solv is not None:
            log.write('{:13.6f} {:16.6f}'.format(cosmo_solv[file], bbe.qh_gibbs_free_energy + cosmo_solv[file]))
        if options.boltz is True:
            log.write('{:7.3f}'.format(boltz_facs[file] / boltz_sum), thermodata=True)
        if options.imag_freq is True and hasattr(bbe, "im_frequency_wn"):
            for freq in bbe.im_frequency_wn:
                log.write('{:9.2f}'.format(freq), thermodata=True)
        if options.ssymm:
            if hasattr(bbe, "qh_gibbs_free_energy"):
                log.write('{:>13}'.format(bbe.point_group))
            else:
                log.write('{:>37}'.format('---'))
#log.write("\n" + stars + "\n")

#log.write('   {:<13} {:>2} {:>4} {:>2} {:>3} {:>2} {:>4} {:>2} '
#            '{:>4}\n'.format('TOTAL CPU', total_cpu_time.day + add_days - 1, 'days', total_cpu_time.hour, 'hrs',
#                            total_cpu_time.minute, 'mins', total_cpu_time.second, 'secs'))
