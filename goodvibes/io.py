# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import os.path, time
import numpy as np

from .utils import *

from cclib.io import ccread
from cclib.parser.utils import convertor

# Some literature references
grimme_ref = "Grimme, S. Chem. Eur. J. 2012, 18, 9955-9964"
truhlar_ref = "Ribeiro, R. F.; Marenich, A. V.; Cramer, C. J.; Truhlar, D. G. J. Phys. Chem. B 2011, 115, 14556-14562"
head_gordon_ref = "Li, Y.; Gomes, J.; Sharada, S. M.; Bell, A. T.; Head-Gordon, M. J. Phys. Chem. C 2015, 119, 1840-1850"
goodvibes_ref = ("Luchini, G.; Alegre-Requena, J. V.; Funes-Ardoiz, I.; Paton, R. S. F1000Research, 2020, 9, 291."
                 "\n   DOI: 10.12688/f1000research.22758.1")
csd_ref = ("C. R. Groom, I. J. Bruno, M. P. Lightfoot and S. C. Ward, Acta Cryst. 2016, B72, 171-179"
           "\n   Cordero, B.; Gomez V.; Platero-Prats, A. E.; Reves, M.; Echeverria, J.; Cremades, E.; Barragan, F.; Alvarez, S. Dalton Trans. 2008, 2832-2838")
oniom_scale_ref = "Simon, L.; Paton, R. S. J. Am. Chem. Soc. 2018, 140, 5412-5420"
d3_ref = "Grimme, S.; Atony, J.; Ehrlich S.; Krieg, H. J. Chem. Phys. 2010, 132, 154104"
d3bj_ref = "Grimme S.; Ehrlich, S.; Goerigk, L. J. Comput. Chem. 2011, 32, 1456-1465"
atm_ref = "Axilrod, B. M.; Teller, E. J. Chem. Phys. 1943, 11, 299 \n Muto, Y. Proc. Phys. Math. Soc. Jpn. 1944, 17, 629"

alphabet = 'abcdefghijklmnopqrstuvwxyz'

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
        

def gv_summary(log, files, thermo_data, options):    
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
    if options.duplicate:
        dup_list = check_dup(files, thermo_data)
    else:
        dup_list = []

    # Boltzmann factors and averaging over clusters
    if options.boltz != False:
        boltz_facs, weighted_free_energy, boltz_sum = get_boltz(files, thermo_data, clustering, clusters,
                                                                options.temperature, dup_list)

    for file in files:  # Loop over the output files and compute thermochemistry
        duplicate = False
        if len(dup_list) != 0:
            for dup in dup_list:
                if dup[0] == file:
                    duplicate = True
                    log.write('\nx  {} is a duplicate or enantiomer of {}'.format(dup[0].rsplit('.', 1)[0],
                                                                                    dup[1].rsplit('.', 1)[0]))
                    break
        if not duplicate:
            bbe = thermo_data[file]
            
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
                log.write("\nx  " + '{:<39}'.format(os.path.splitext(os.path.basename(file))[0]))
                log.write('          ----   Caution! Potential invalid calculation of linear molecule from Gaussian')
            else:
                if hasattr(bbe, "gibbs_free_energy"):
                    if options.spc is not False:
                        if bbe.sp_energy != '!':
                            log.write("\no  ")
                            log.write('{:<39}'.format(os.path.splitext(os.path.basename(file))[0]), thermodata=True)
                            log.write(' {:13.6f}'.format(bbe.sp_energy), thermodata=True)
                        if bbe.sp_energy == '!':
                            log.write("\nx  ")
                            log.write('{:<39}'.format(os.path.splitext(os.path.basename(file))[0]), thermodata=True)
                            log.write(' {:>13}'.format('----'), thermodata=True)
                    else:
                        log.write("\no  ")
                        log.write('{:<39}'.format(os.path.splitext(os.path.basename(file))[0]), thermodata=True)
                # Gaussian SPC file handling
                if hasattr(bbe, "scf_energy") and not hasattr(bbe, "gibbs_free_energy"):
                    log.write("\nx  " + '{:<39}'.format(os.path.splitext(os.path.basename(file))[0]))
                # ORCA spc files
                elif not hasattr(bbe, "scf_energy") and not hasattr(bbe, "gibbs_free_energy"):
                    log.write("\nx  " + '{:<39}'.format(os.path.splitext(os.path.basename(file))[0]))
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


def write_to_xyz(log, files, thermo_data):
    xyz = xyz_out("Goodvibes", "xyz", "output")
    for file in files:
        bbe = thermo_data[file]        
        xyzdata = getoutData(file)
        xyz.write_text(str(len(xyzdata.atom_types)))
        if hasattr(bbe, "scf_energy"):
            xyz.write_text(
                '{:<39} {:>13} {:13.6f}'.format(os.path.splitext(os.path.basename(file))[0], 'Eopt',
                                                bbe.scf_energy))
        else:
            xyz.write_text('{:<39}'.format(os.path.splitext(os.path.basename(file))[0]))
        if hasattr(xyzdata, 'cartesians') and hasattr(xyzdata, 'atom_types'):
            xyz.write_coords(xyzdata.atom_types, xyzdata.cartesians)
    xyz.finalize()

def gv_header(log, files, options, __version__):
    # Start printing results
    
    start = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    log.write("   GoodVibes v" + __version__ + " " + start + "\n   Citation: " + goodvibes_ref + "\n")
    # Check if user has specified any files, if not quit now
    if len(files) == 0:
        sys.exit("\nPlease provide GoodVibes with calculation output files on the command line.\n"
                "For help, use option '-h'\n")

    
    if options.temperature_interval == False:
        log.write("   Temperature = " + str(options.temperature) + " Kelvin")
    # If not at standard temp, need to correct the molarity of 1 atmosphere (assuming pressure is still 1 atm)
    if options.conc:
        gas_phase = False
        log.write("   Concentration = " + str(options.conc) + " mol/L")
    else:
        gas_phase = True
        options.conc = ATMOS / (GAS_CONSTANT * options.temperature)
        log.write("   Pressure = 1 atm")
    log.write('\n   All energetic values below shown in Hartree unless otherwise specified.')
    # Initial read of files,
    # Grab level of theory, solvation model, check for Normal Termination
    l_o_t, s_m, progress, spc_progress, orientation, grid = [], [], {}, {}, {}, {}
    for file in files:
        lot_sm_prog = read_initial(file)
        l_o_t.append(lot_sm_prog[0])
        s_m.append(lot_sm_prog[1])
        progress[file] = lot_sm_prog[2]
        orientation[file] = lot_sm_prog[3]
        grid[file] = lot_sm_prog[4]
        #check spc files for normal termination
        if options.spc is not False and options.spc != 'link':
            name, ext = os.path.splitext(file)
            if os.path.exists(name + '_' + options.spc + '.log'):
                spc_file = name + '_' + options.spc + '.log'
            elif os.path.exists(name + '_' + options.spc + '.out'):
                spc_file = name + '_' + options.spc + '.out'
            lot_sm_prog = read_initial(spc_file)
            spc_progress[spc_file] = lot_sm_prog[2]

    remove_key = []
    # Remove problem files and print errors
    for i, key in enumerate(files):
        if progress[key] == 'Error':
            log.write("\n\nx  Warning! Error termination found in file {}. This file will be omitted from further "
                    "calculations.".format(key))
            remove_key.append([i, key])
        elif progress[key] == 'Incomplete':
            log.write("\n\nx  Warning! File {} may not have terminated normally or the calculation may still be "
                    "running. This file will be omitted from further calculations.".format(key))
            remove_key.append([i, key])
    #check spc files for normal termination
    if spc_progress:
        for key in spc_progress:
            if spc_progress[key] == 'Error':
                sys.exit("\n\nx  ERROR! Error termination found in file {} calculations.".format(key))
            elif spc_progress[key] == 'Incomplete':
                sys.exit("\n\nx  ERROR! File {} may not have terminated normally or the "
                    "calculation may still be running.".format(key))

    for [i, key] in list(reversed(remove_key)):
        files.remove(key)
        del l_o_t[i]
        del s_m[i]
        del orientation[key]
        del grid[key]
    if len(files) == 0:
        sys.exit("\n\nPlease try again with normally terminated output files.\nFor help, use option '-h'\n")
    # Attempt to automatically obtain frequency scale factor,
    # Application of freq scale factors requires all outputs to be same level of theory
    if options.freq_scale_factor is not False:
        if 'ONIOM' not in l_o_t[0]:
            log.write("\n\n   User-defined vibrational scale factor " + str(options.freq_scale_factor) + " for " +
                    l_o_t[0] + " level of theory")
        else:
            log.write("\n\n   User-defined vibrational scale factor " + str(options.freq_scale_factor) +
                    " for QM region of " + l_o_t[0])
    else:
        # Look for vibrational scaling factor automatically
        if all_same(l_o_t):
            level = l_o_t[0].upper()
            for data in (scaling_data_dict, scaling_data_dict_mod):
                if level in data:
                    options.freq_scale_factor = data[level].zpe_fac
                    ref = scaling_refs[data[level].zpe_ref]
                    log.write("\n\no  Found vibrational scaling factor of {:.3f} for {} level of theory\n"
                            "   {}".format(options.freq_scale_factor, l_o_t[0], ref))
                    break
        else:  # Print files and different levels of theory found
            files_l_o_t, levels_l_o_t, filtered_calcs_l_o_t = [], [], []
            for file in files:
                files_l_o_t.append(file)
            for i in l_o_t:
                levels_l_o_t.append(i)
            filtered_calcs_l_o_t.append(files_l_o_t)
            filtered_calcs_l_o_t.append(levels_l_o_t)
            print_check_fails(log, filtered_calcs_l_o_t[1], filtered_calcs_l_o_t[0], "levels of theory")

    # Exit program if a comparison of Boltzmann factors is requested and level of theory is not uniform across all files
    if not all_same(l_o_t) and (options.boltz is not False or options.ee is not False):
        sys.exit("\n\nERROR: When comparing files using Boltzmann factors (boltz or ee input options), the level of "
                "theory used should be the same for all files.\n ")
    # Exit program if molecular mechanics scaling factor is given and all files are not ONIOM calculations
    if options.mm_freq_scale_factor is not False:
        if all_same(l_o_t) and 'ONIOM' in l_o_t[0]:
            log.write("\n\n   User-defined vibrational scale factor " +
                    str(options.mm_freq_scale_factor) + " for MM region of " + l_o_t[0])
            log.write("\n   REF: {}".format(oniom_scale_ref))
        else:
            sys.exit("\n   Option --vmm is only for use in ONIOM calculation output files.\n   "
                    " help use option '-h'\n")

    if options.freq_scale_factor is False:
        options.freq_scale_factor = 1.0  # If no scaling factor is found use 1.0
        if all_same(l_o_t):
            log.write("\n\n   Using vibrational scale factor {} for {} level of "
                    "theory".format(options.freq_scale_factor, l_o_t[0]))
        else:
            log.write("\n\n   Using vibrational scale factor {}: differing levels of theory "
                    "detected.".format(options.freq_scale_factor))
    
    # Checks to see whether the available free space of a requested solvent is defined
    #freespace = get_free_space(options.freespace)
    #if freespace != 1000.0:
    #    log.write("\n   Specified solvent " + options.freespace + ": free volume " + str(
    #        "%.3f" % (freespace / 10.0)) + " (mol/l) corrects the translational entropy")

    # Check for implicit solvation
    printed_solv_warn = False
    for i in s_m:
        if ('smd' in i.lower() or 'cpcm' in i.lower()) and not printed_solv_warn:
            log.write("\n\n   Caution! Implicit solvation (SMD/CPCM) detected. Enthalpic and entropic terms cannot be "
                    "safely separated. Use them at your own risk!")
            printed_solv_warn = True

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
        qs_ref = grimme_ref
    elif options.QS == "truhlar":
        log.write("\n   QS = Truhlar: Using an RRHO treatment where low frequencies are adjusted to the cut-off value.")
        qs_ref = truhlar_ref
    else:
        log.fatal("\n   FATAL ERROR: Unknown quasi-harmonic model " + options.QS + " specified (QS must = grimme or truhlar).")
    log.write("\n   REF: " + qs_ref + '\n')

    # Check if qh-H correction should be applied
    if options.QH:
        log.write("\n\n   Enthalpy quasi-harmonic treatment: frequency cut-off value of " + str(
            options.H_freq_cutoff) + " wavenumbers will be applied.")
        log.write("\n   QH = Head-Gordon: Using an RRHO treatement with an approximation term for vibrational energy.")
        qh_ref = head_gordon_ref
        log.write("\n   REF: " + qh_ref + '\n')

    # Check if entropy symmetry correction should be applied
    if options.ssymm:
        log.write('\n\n   Ssymm requested. Symmetry contribution to entropy to be calculated using S. Patchkovskii\'s \n   open source software "Brute Force Symmetry Analyzer" available under GNU General Public License.')
        log.write('\n   REF: (C) 1996, 2003 S. Patchkovskii, Serguei.Patchkovskii@sympatico.ca')
        log.write('\n\n   Atomic radii used to calculate internal symmetry based on Cambridge Structural Database covalent radii.')
        log.write("\n   REF: " + csd_ref + '\n')

    # Whether single-point energies are to be used
    if options.spc:
        log.write("\n   Combining final single point energy with thermal corrections.")
    # Solvent correction message
    if options.media:
        log.write("\n   Applying standard concentration correction (based on density at 20C) to solvent media.")

    # Check for special options
    inverted_freqs, inverted_files = [], []
    if options.ssymm:
        ssymm_option = options.ssymm
    else:
        ssymm_option = False
    if options.mm_freq_scale_factor is not False:
        vmm_option = options.mm_freq_scale_factor
    else:
        vmm_option = False

  
class getoutData:
    """
    Read molecule data from a computational chemistry output file.

    Attributes:
        FREQS (list): list of frequencies parsed from Gaussian file.
        REDMASS (list): list of reduced masses parsed from Gaussian file.
        FORCECONST (list): list of force constants parsed from Gaussian file.
        NORMALMODE (list): list of normal modes parsed from Gaussian file.
        atom_nums (list): list of atom number IDs.
        atom_types (list): list of atom element symbols.
        cartesians (list): list of cartesian coordinates for each atom.
        connectivity (list): list of atomic connectivity in a molecule, based on covalent radii
    """
    def __init__(self, filename):
        data = ccread(filename)
        try:
            self.FREQS = data.vibfreqs.tolist()
            self.REDMASS = data.vibrmasses.tolist()
            self.FORCECONST = data.vibfconsts.tolist()
            self.NORMALMODE = data.vibdisps.tolist()
        except:
            pass

        self.atom_nums = data.atomnos.tolist()
        self.atom_types = [periodictable[atomnum] for atomnum in self.atom_nums]
        # Assuming that the output file doesn't contain a geometry
        # optimization at the beginning, we take the first set of atomic
        # coordinates rather than the last, in the even that a finite
        # difference frequency calculation was performed and the displaced
        # geometries are printed.
        self.cartesians = data.atomcoords[0].tolist()

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
    
    if program != "Orca":
        try:
            possible_filenames = (stub + ".log", stub + ".out")
            for possible_filename in possible_filenames:
                if os.path.exists(possible_filename):
                    ccdata = ccread(possible_filename)
        except IndexError:
            ccdata = None
        if ccdata:
            try:
                spe = ccdata.scfenergies[-1]
                if hasattr(ccdata, "mpenergies"):
                    spe = ccdata.mpenergies[-1]
                if hasattr(ccdata, "ccenergies"):
                    spe = ccdata.ccenergies[-1]
                spe = convertor(spe, "eV", "hartree")
                charge = ccdata.charge
                multiplicity = ccdata.mult
            except AttributeError:
                pass

    for line in data:
        if program == "Gaussian":
            if line.strip().startswith('E2('):
                spe_value = line.strip().split()[-1]
                spe = float(spe_value.replace('D','E'))
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
        elif program == "Orca":
            if 'Program Version' in line.strip():
                version_program = "ORCA version " + line.split()[2]
            if line.strip().startswith('FINAL SINGLE POINT ENERGY'):
                spe = float(line.strip().split()[4])
            if "Total Charge" in line.strip() and "...." in line.strip():
                charge = int(line.strip("=").split()[-1])
            if "Multiplicity" in line.strip() and "...." in line.strip():
                multiplicity = int(line.strip("=").split()[-1])
        elif program == "NWChem":
            if 'nwchem branch' in line.strip():
                version_program = "NWChem version " + line.split()[3]
            if ccdata == None:
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
    # ORCA parsing for solvation model
    elif program == 'Orca':
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
