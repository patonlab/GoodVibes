#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

"""####################################################################
#                           GoodVibes.py                              #
#  Evaluation of quasi-harmonic thermochemistry from Gaussian.        #
#  Partion functions are evaluated from vibrational frequencies       #
#  and rotational temperatures from the standard output.              #
#######################################################################
#  The rigid-rotor harmonic oscillator approximation is used as       #
#  standard for all frequencies above a cut-off value. Below this,    #
#  two treatments can be applied to entropic values:                  #
#    (a) low frequencies are shifted to the cut-off value (as per     #
#    Cramer-Truhlar)                                                  #
#    (b) a free-rotor approximation is applied below the cut-off (as  #
#    per Grimme). In this approach, a damping function interpolates   #
#    between the RRHO and free-rotor entropy treatment of Svib to     #
#    avoid a discontinuity.                                           #
#  Both approaches avoid infinitely large values of Svib as wave-     #
#  numbers tend to zero. With a cut-off set to 0, the results will be #
#  identical to standard values output by the Gaussian program.       #
#######################################################################
#  Enthalpy values below the cutoff value are treated similarly to    #
#  Grimme's method (as per Head-Gordon) where below the cutoff value, #
#  a damping function is applied as the value approaches a value of   #
#  0.5RT, approprate for zeolitic systems                             #
#######################################################################
#  The free energy can be evaluated for variable temperature,         #
#  concentration, vibrational scaling factor, and with a haptic       #
#  correction of the translational entropy in different solvents,     #
#  according to the amount of free space available.                   #
#######################################################################
#  A potential energy surface may be evaluated for a given set of     #
#  structures or conformers, in which case a correction to the free-  #
#  energy due to multiple conformers is applied.                      #
#  Enantiomeric excess, diastereomeric ratios and ddG can also be     #
#  calculated to show preference of stereoisomers.                    #
#######################################################################
#  Careful checks may be applied to compare variables between         #
#  multiple files such as Gaussian version, solvation models, levels  #
#  of theory, charge and multiplicity, potential duplicate structures #
#  errors in potentail linear molecules, correct or incorrect         #
#  transition states, and empirical dispersion models.                #
#######################################################################


#######################################################################
###########  Authors:     Rob Paton, Ignacio Funes-Ardoiz  ############
###########               Guilian Luchini, Juan V. Alegre- ############
###########               Requena, Yanfei Guan, Sibo Lin   ############
###########  Last modified:  May 27, 2020                 ############
####################################################################"""

import math, os.path, sys, time
from argparse import ArgumentParser
from datetime import datetime, timedelta
from glob import glob
import numpy as np

# Importing regardless of relative import
#try:
from goodvibes.vib_scale_factors import scaling_data_dict, scaling_data_dict_mod, scaling_refs
import goodvibes.pes as pes
import goodvibes.io as io
import goodvibes.thermo as thermo
#except:
#    from vib_scale_factors import scaling_data_dict, scaling_data_dict_mod, scaling_refs
#    import pes as pes
#    import io as io
#    import thermo as thermo


# VERSION NUMBER
__version__ = "3.0.2"

SUPPORTED_EXTENSIONS = set(('.out', '.log'))

# Some literature references
grimme_ref = "Grimme, S. Chem. Eur. J. 2012, 18, 9955-9964"
truhlar_ref = "Ribeiro, R. F.; Marenich, A. V.; Cramer, C. J.; Truhlar, D. G. J. Phys. Chem. B 2011, 115, 14556-14562"
head_gordon_ref = "Li, Y.; Gomes, J.; Sharada, S. M.; Bell, A. T.; Head-Gordon, M. J. Phys. Chem. C 2015, 119, 1840-1850"
goodvibes_ref = ("Luchini, G.; Alegre-Requena, J. V.; Funes-Ardoiz, I.; Paton, R. S. F1000Research, 2020, 9, 291."
                 "\n   GoodVibes version " + __version__ + " DOI: 10.12688/f1000research.22758.1")
csd_ref = ("C. R. Groom, I. J. Bruno, M. P. Lightfoot and S. C. Ward, Acta Cryst. 2016, B72, 171-179"
           "\n   Cordero, B.; Gomez V.; Platero-Prats, A. E.; Reves, M.; Echeverria, J.; Cremades, E.; Barragan, F.; Alvarez, S. Dalton Trans. 2008, 2832-2838")
oniom_scale_ref = "Simon, L.; Paton, R. S. J. Am. Chem. Soc. 2018, 140, 5412-5420"
d3_ref = "Grimme, S.; Atony, J.; Ehrlich S.; Krieg, H. J. Chem. Phys. 2010, 132, 154104"
d3bj_ref = "Grimme S.; Ehrlich, S.; Goerigk, L. J. Comput. Chem. 2011, 32, 1456-1465"
atm_ref = "Axilrod, B. M.; Teller, E. J. Chem. Phys. 1943, 11, 299 \n Muto, Y. Proc. Phys. Math. Soc. Jpn. 1944, 17, 629"

alphabet = 'abcdefghijklmnopqrstuvwxyz'


def all_same(items):
    """Returns bool for checking if all items in a list are the same."""
    return all(x == items[0] for x in items)


# Calculate elapsed time
def add_time(tm, cpu):
    [days, hrs, mins, secs, msecs] = cpu
    fulldate = datetime(100, 1, tm.day, tm.hour, tm.minute, tm.second, tm.microsecond)
    fulldate = fulldate + timedelta(days=days, hours=hrs, minutes=mins, seconds=secs, microseconds=msecs * 1000)
    return fulldate


def calc_cpu(thermo_data, options, log):
    # Initialize the total CPU time
    add_days = 0
    cpu = datetime(100, 1, 1, 00, 00, 00, 00)
    for key in thermo_data:
        bbe = thermo_data[key]
        if hasattr(bbe, "cpu"):
            if bbe.cpu != None:
                cpu = add_time(cpu, bbe.cpu)

    if cpu.month > 1: add_days += 31 * (cpu.month -1)
    else: add_days = 0
    log.write('   {:<13} {:>2} {:>4} {:>2} {:>3} {:>2} {:>4} {:>2} '
              '{:>4}\n'.format('TOTAL CPU', cpu.day + add_days - 1, 'days', cpu.hour, 'hrs',
                               cpu.minute, 'mins', cpu.second, 'secs'))


def get_selectivity(files, options, boltz_facs, boltz_sum, log, dup_list=[]):
    """
    Calculate selectivity as enantioselectivity/diastereomeric ratio.

    Parameters:
    pattern (str): pattern to recognize for selectivity calculation, i.e. "R":"S".
    files (str): files to use for selectivity calculation.
    boltz_facs (dict): dictionary of Boltzmann factors for each file used in the calculation.
    boltz_sum (float)
    temperature (float)

    Returns:
    float: enantiomeric/diasteriomeric ratio.
    str: pattern used to identify ratio.
    float: Gibbs free energy barrier.
    bool: flag for failed selectivity calculation.
    str: preferred enantiomer/diastereomer configuration.
    """
    # Grab files for selectivity calcs
    # list the directories to look in
    dirs = []
    for file in files:
        dirs.append(os.path.dirname(file))
    dirs = list(set(dirs))

    a_files, b_files, a_sum, b_sum, failed, pref = [], [], 0.0, 0.0, False, ''

    pattern = options.ee
    try:
        [a_regex,b_regex] = pattern.split(':')
        [a_regex,b_regex] = [a_regex.strip(), b_regex.strip()]

        A = ''.join(a for a in a_regex if a.isalnum())
        B = ''.join(b for b in b_regex if b.isalnum())

        for dir in dirs:
            a_files.extend(glob(dir+'/'+a_regex))
            b_files.extend(glob(dir+'/'+b_regex))
    except:
        pass

    if len(a_files) is 0 or len(b_files) is 0:
        log.write("\n   Warning! Filenames have not been formatted correctly for determining selectivity\n")
        log.write("   Make sure the filename contains either " + A + " or " + B + "\n")
        sys.exit("   Please edit either your filenames or selectivity pattern argument and try again\n")

    # Grab Boltzmann sums
    for file in files:
        if file not in [dup[1] for dup in dup_list]:
            for a_file in a_files:
                if file in a_file:
                    a_sum += boltz_facs[file] / boltz_sum
            for b_file in b_files:
                if file in b_file:
                    b_sum += boltz_facs[file] / boltz_sum

    # Get ratios
    A_round = round(a_sum * 100)
    B_round = round(b_sum * 100)
    r = str(A_round) + ':' + str(B_round)
    if a_sum > b_sum:
        pref = A
        try:
            ratio = a_sum / b_sum
            if ratio < 3:
                ratio = str(round(ratio, 1)) + ':1'
            else:
                ratio = str(round(ratio)) + ':1'
        except ZeroDivisionError:
            ratio = '1:0'
    else:
        pref = B
        try:
            ratio = b_sum / a_sum
            if ratio < 3:
                ratio = '1:' + str(round(ratio, 1))
            else:
                ratio = '1:' + str(round(ratio))
        except ZeroDivisionError:
            ratio = '0:1'
    ee = (a_sum - b_sum) * 100.
    if ee == 0:
        log.write("\n   Warning! No files found for selectivity analysis, adjust the names and try again.\n")
        failed = True
    ee = abs(ee)
    if ee > 99.99:
        ee = 99.99
    try:
        dd_free_energy = thermo.GAS_CONSTANT / thermo.J_TO_AU * options.temperature * math.log((50 + abs(ee) / 2.0) / (50 - abs(ee) / 2.0)) * thermo.KCAL_TO_AU
    except ZeroDivisionError:
        dd_free_energy = 0.0

    if not failed:
        selec_stars = "   " + '*' * 109
        log.write("\n   " + '{:<39} {:>13} {:>13} {:>13} {:>13} {:>13}'.format("Selectivity", "Excess (%)", "Ratio (%)", "Ratio", "Major", "DDG kcal/mol"), thermodata=True)
        log.write("\n" + selec_stars)
        log.write('\no {:<40} {:13.2f} {:>13} {:>13} {:>13} {:13.2f}'.format('', ee, r, ratio, pref,
                                                                             dd_free_energy), thermodata=True)
        log.write("\n" + selec_stars + "\n")

    return ee, r, ratio, dd_free_energy, failed, pref


def get_boltz(thermo_data, options, clusters=[], dup_list=[]):
    """
    Obtain Boltzmann factors, Boltzmann sums, and weighted free energy values.

    Used for selectivity and boltzmann requested options.

    Parameters:
    files (list): list of files to find Boltzmann factors for.
    thermo_data (dict): dict of calc_bbe objects with thermodynamic data to use for Boltzmann averaging.
    clustering (bool): flag for file clustering
    clusters (list): definitions for the requested clusters
    temperature (float): temperature to compute Boltzmann populations at
    dup_list (list): list of potential duplicates

    Returns:boltz_facs, weighted_free_energy, boltz_sum
    dict: dictionary of files with corresponding Boltzmann factors.
    dict: dictionary of files with corresponding weighted Gibbs free energy.
    float: Boltzmann sum computed from Boltzmann factors and Gibbs free energy.
    """
    boltz_facs, weighted_free_energy, e_rel, e_min, boltz_sum = {}, {}, {}, sys.float_info.max, 0.0

    for file in thermo_data:  # Need the most stable structure
        bbe = thermo_data[file]
        if hasattr(bbe, "qh_gibbs_free_energy"):
            if bbe.qh_gibbs_free_energy != None:
                if bbe.qh_gibbs_free_energy < e_min:
                    e_min = bbe.qh_gibbs_free_energy

    if options.clustering:
        for n, cluster in enumerate(clusters):
            boltz_facs['cluster-' + alphabet[n].upper()] = 0.0
            weighted_free_energy['cluster-' + alphabet[n].upper()] = 0.0

    # Calculate E_rel and Boltzmann factors
    for file in thermo_data:
        if file not in [dup[1] for dup in dup_list]:
            bbe = thermo_data[file]
            if hasattr(bbe, "qh_gibbs_free_energy"):
                if bbe.qh_gibbs_free_energy != None:
                    e_rel[file] = bbe.qh_gibbs_free_energy - e_min
                    boltz_facs[file] = math.exp(-e_rel[file] * thermo.J_TO_AU / thermo.GAS_CONSTANT / options.temperature)
                    if options.clustering:
                        for n, cluster in enumerate(clusters):
                            for structure in cluster:
                                if structure == file:
                                    boltz_facs['cluster-' + alphabet[n].upper()] += math.exp(
                                        -e_rel[file] * thermo.J_TO_AU / thermo.GAS_CONSTANT / options.temperature)
                                    weighted_free_energy['cluster-' + alphabet[n].upper()] += math.exp(
                                        -e_rel[file] * thermo.J_TO_AU / thermo.GAS_CONSTANT / options.temperature) * bbe.qh_gibbs_free_energy
                    boltz_sum += math.exp(-e_rel[file] * thermo.J_TO_AU / thermo.GAS_CONSTANT / options.temperature)

    return boltz_facs, weighted_free_energy, boltz_sum


def check_dup(files, thermo_data):
    """
    Check for duplicate species from among all files based on energy, rotational constants and frequencies

    Energy cutoff = 1 microHartree
    RMS Rotational Constant cutoff = 1kHz
    RMS Freq cutoff = 10 wavenumbers
    """
    e_cutoff = 1e-4
    ro_cutoff = 1e-4
    freq_cutoff = 100
    mae_freq_cutoff = 10
    max_freq_cutoff = 10
    dup_list = []
    freq_diff, mae_freq_diff, max_freq_diff, e_diff, ro_diff = 100, 3, 10, 1, 1
    for i, file in enumerate(files):
        for j in range(0, i):
            bbe_i, bbe_j = thermo_data[files[i]], thermo_data[files[j]]
            if hasattr(bbe_i, "scf_energy") and hasattr(bbe_j, "scf_energy"):
                e_diff = bbe_i.scf_energy - bbe_j.scf_energy
            if hasattr(bbe_i, "roconst") and hasattr(bbe_j, "roconst"):
                if len(bbe_i.roconst) == len(bbe_j.roconst):
                    ro_diff = np.linalg.norm(np.array(bbe_i.roconst) - np.array(bbe_j.roconst))
            if hasattr(bbe_i, "frequency_wn") and hasattr(bbe_j, "frequency_wn"):
                if len(bbe_i.frequency_wn) == len(bbe_j.frequency_wn) and len(bbe_i.frequency_wn) > 0:
                    freq_diff = [np.linalg.norm(freqi - freqj) for freqi, freqj in
                                 zip(bbe_i.frequency_wn, bbe_j.frequency_wn)]
                    mae_freq_diff, max_freq_diff = np.mean(freq_diff), np.max(freq_diff)
                elif len(bbe_i.frequency_wn) == len(bbe_j.frequency_wn) and len(bbe_i.frequency_wn) == 0:
                    mae_freq_diff, max_freq_diff = 0., 0.
            if e_diff < e_cutoff and ro_diff < ro_cutoff and mae_freq_diff < mae_freq_cutoff and max_freq_diff < max_freq_cutoff:
                dup_list.append([files[i], files[j]])
    return dup_list


def print_check_fails(log, check_attribute, file, attribute, option2=False):
    """Function for printing checks to the terminal"""
    unique_attr = {}
    for i, attr in enumerate(check_attribute):
        if option2 is not False: attr = (attr, option2[i])
        if attr not in unique_attr:
            unique_attr[attr] = [file[i]]
        else:
            unique_attr[attr].append(file[i])
    log.write("\nx  Caution! Different {} found: ".format(attribute))
    for attr in unique_attr:
        if option2 is not False:
            if float(attr[0]) < 0:
                log.write('\n       {} {}: '.format(attr[0], attr[1]))
            else:
                log.write('\n        {} {}: '.format(attr[0], attr[1]))
        else:
            log.write('\n        -{}: '.format(attr))
        for filename in unique_attr[attr]:
            if filename is unique_attr[attr][-1]:
                log.write('{}'.format(filename))
            else:
                log.write('{}, '.format(filename))


# Perform careful checks on calculation output files
# Check for Gaussian version, solvation state/gas phase consistency, level of theory/basis set consistency,
# charge and multiplicity consistency, standard concentration used, potential linear molecule error,
# transition state verification, empirical dispersion models.
def check_files(file_data, thermo_data, options, log):
    STARS = '*' * 50
    l_o_t = ['']
    log.write("\n   Checks for thermochemistry calculations (frequency calculations):")
    log.write("\n" + STARS)
    # Check program used and version
    version_check = [file.program for file in file_data]
    file_check = [file.name for file in file_data]
    if all_same(version_check) != False:
        log.write("\no  Using {} in all calculations.".format(version_check[0]))
    else:
        print_check_fails(log, version_check, file_check, "programs or versions")

    # Check level of theory
    if all_same(l_o_t) is not False:
        log.write("\no  Using {} in all calculations.".format(l_o_t[0]))
    elif all_same(l_o_t) is False:
        print_check_fails(log, l_o_t, file_check, "levels of theory")

    # Check for solvent models
    solvent_check = [thermo_data[key].solvation_model[0] for key in thermo_data]
    if all_same(solvent_check):
        solvent_check = [thermo_data[key].solvation_model[1] for key in thermo_data]
        log.write("\no  Using {} in all calculations.".format(solvent_check[0]))
    else:
        solvent_check = [thermo_data[key].solvation_model[1] for key in thermo_data]
        print_check_fails(log, solvent_check, file_check, "solvation models")

    # Check for -c 1 when solvent is added
    if all_same(solvent_check):
        if solvent_check[0] == "gas phase" and str(round(options.conc, 4)) == str(round(0.0408740470708, 4)):
            log.write("\no  Using a standard concentration of 1 atm for gas phase.")
        elif solvent_check[0] == "gas phase" and str(round(options.conc, 4)) != str(round(0.0408740470708, 4)):
            log.write("\nx  Caution! Standard concentration is not 1 atm for gas phase (using {} M).".format(options.conc))
        elif solvent_check[0] != "gas phase" and str(round(options.conc, 4)) == str(round(0.0408740470708, 4)):
            log.write("\nx  Using a standard concentration of 1 atm for solvent phase (option -c 1 should be included for 1 M).")
        elif solvent_check[0] != "gas phase" and str(options.conc) == str(1.0):
            log.write("\no  Using a standard concentration of 1 M for solvent phase.")
        elif solvent_check[0] != "gas phase" and str(round(options.conc, 4)) != str(round(0.0408740470708, 4)) and str(
                options.conc) != str(1.0):
            log.write("\nx  Caution! Standard concentration is not 1 M for solvent phase (using {} M).".format(options.conc))
    if all_same(solvent_check) == False and "gas phase" in solvent_check:
        log.write("\nx  Caution! The right standard concentration cannot be determined because the calculations use a combination of gas and solvent phases.")
    if all_same(solvent_check) == False and "gas phase" not in solvent_check:
        log.write("\nx  Caution! Different solvents used, fix this issue and use option -c 1 for a standard concentration of 1 M.")

    # Check charge and multiplicity
    charge_check = [thermo_data[key].charge for key in thermo_data]
    multiplicity_check = [thermo_data[key].multiplicity for key in thermo_data]
    if all_same(charge_check) != False and all_same(multiplicity_check) != False:
        log.write("\no  Using charge {} and multiplicity {} in all calculations.".format(charge_check[0],
                                                                                         multiplicity_check[0]))
    else:
        print_check_fails(log, charge_check, file_check, "charge and multiplicity", multiplicity_check)

    # Check for duplicate structures
    dup_list = check_dup(files, thermo_data)
    if len(dup_list) == 0:
        log.write("\no  No duplicates or enantiomers found")
    else:
        log.write("\nx  Caution! Possible duplicates or enantiomers found:")
        for dup in dup_list:
            log.write('\n        {} and {}'.format(dup[0], dup[1]))

    # Check for linear molecules with incorrect number of vibrational modes
    linear_fails, linear_fails_atom, linear_fails_cart, linear_fails_files, linear_fails_list = [], [], [], [], []
    frequency_list = []

    for file in files:
        linear_fails = getoutData(file)
        linear_fails_cart.append(linear_fails.cartesians)
        linear_fails_atom.append(linear_fails.atom_types)
        linear_fails_files.append(file)
        frequency_list.append(thermo_data[file].frequency_wn)

    linear_fails_list.append(linear_fails_atom)
    linear_fails_list.append(linear_fails_cart)
    linear_fails_list.append(frequency_list)
    linear_fails_list.append(linear_fails_files)

    linear_mol_correct, linear_mol_wrong = [], []
    for i in range(len(linear_fails_list[0])):
        count_linear = 0
        if len(linear_fails_list[0][i]) == 2:
            if len(linear_fails_list[2][i]) == 1:
                linear_mol_correct.append(linear_fails_list[3][i])
            else:
                linear_mol_wrong.append(linear_fails_list[3][i])
        if len(linear_fails_list[0][i]) == 3:
            if linear_fails_list[0][i] == ['I', 'I', 'I'] or linear_fails_list[0][i] == ['O', 'O', 'O'] or \
                    linear_fails_list[0][i] == ['N', 'N', 'N'] or linear_fails_list[0][i] == ['H', 'C', 'N'] or \
                    linear_fails_list[0][i] == ['H', 'N', 'C'] or linear_fails_list[0][i] == ['C', 'H', 'N'] or \
                    linear_fails_list[0][i] == ['C', 'N', 'H'] or linear_fails_list[0][i] == ['N', 'H', 'C'] or \
                    linear_fails_list[0][i] == ['N', 'C', 'H']:
                if len(linear_fails_list[2][i]) == 4:
                    linear_mol_correct.append(linear_fails_list[3][i])
                else:
                    linear_mol_wrong.append(linear_fails_list[3][i])
            else:
                for j in range(len(linear_fails_list[0][i])):
                    for k in range(len(linear_fails_list[0][i])):
                        if k > j:
                            for l in range(len(linear_fails_list[1][i][j])):
                                if linear_fails_list[0][i][j] == linear_fails_list[0][i][k]:
                                    if linear_fails_list[1][i][j][l] > (-linear_fails_list[1][i][k][l] - 0.1) and \
                                            linear_fails_list[1][i][j][l] < (-linear_fails_list[1][i][k][l] + 0.1):
                                        count_linear = count_linear + 1
                                        if count_linear == 3:
                                            if len(linear_fails_list[2][i]) == 4:
                                                linear_mol_correct.append(linear_fails_list[3][i])
                                            else:
                                                linear_mol_wrong.append(linear_fails_list[3][i])
        if len(linear_fails_list[0][i]) == 4:
            if linear_fails_list[0][i] == ['C', 'C', 'H', 'H'] or linear_fails_list[0][i] == ['C', 'H', 'C', 'H'] or \
                    linear_fails_list[0][i] == ['C', 'H', 'H', 'C'] or linear_fails_list[0][i] == ['H', 'C', 'C', 'H'] or \
                    linear_fails_list[0][i] == ['H', 'C', 'H', 'C'] or linear_fails_list[0][i] == ['H', 'H', 'C', 'C']:
                if len(linear_fails_list[2][i]) == 7:
                    linear_mol_correct.append(linear_fails_list[3][i])
                else:
                    linear_mol_wrong.append(linear_fails_list[3][i])
    linear_correct_print, linear_wrong_print = "", ""
    for i in range(len(linear_mol_correct)):
        linear_correct_print += ', ' + linear_mol_correct[i]
    for i in range(len(linear_mol_wrong)):
        linear_wrong_print += ', ' + linear_mol_wrong[i]
    linear_correct_print = linear_correct_print[1:]
    linear_wrong_print = linear_wrong_print[1:]
    if len(linear_mol_correct) == 0:
        if len(linear_mol_wrong) == 0:
            log.write("\n-  No linear molecules found.")
        if len(linear_mol_wrong) >= 1:
            log.write("\nx  Caution! Potential linear molecules with wrong number of frequencies found "
                      "(correct number = 3N-5) -{}.".format(linear_wrong_print))
    elif len(linear_mol_correct) >= 1:
        if len(linear_mol_wrong) == 0:
            log.write("\no  All the linear molecules have the correct number of frequencies -{}.".format(linear_correct_print))
        if len(linear_mol_wrong) >= 1:
            log.write("\nx  Caution! Potential linear molecules with wrong number of frequencies found -{}. Correct "
                      "number of frequencies (3N-5) found in other calculations -{}.".format(linear_wrong_print,
                                                                                             linear_correct_print))

    # Checks whether any TS have > 1 imaginary frequency and any GS have any imaginary frequencies
    for file in files:
        bbe = thermo_data[file]
        if bbe.job_type.find('TS') > -1 and len(bbe.im_frequency_wn) != 1:
            log.write("\nx  Caution! TS {} does not have 1 imaginary frequency greater than -50 wavenumbers.".format(file))
        if bbe.job_type.find('GS') > -1 and bbe.job_type.find('TS') == -1 and len(bbe.im_frequency_wn) != 0:
            log.write("\nx  Caution: GS {} has 1 or more imaginary frequencies greater than -50 wavenumbers.".format(file))

    # Check for empirical dispersion
    dispersion_check = [thermo_data[key].empirical_dispersion for key in thermo_data]
    if all_same(dispersion_check):
        if dispersion_check[0] == 'No empirical dispersion detected':
            log.write("\n-  No empirical dispersion detected in any of the calculations.")
        else:
            log.write("\no  Using " + dispersion_check[0] + " in all calculations.")
    else:
        print_check_fails(log, dispersion_check, file_check, "dispersion models")
    log.write("\n" + STARS + "\n")

    # Check for single-point corrections
    if options.spc is not False:
        log.write("\n   Checks for single-point corrections:")
        log.write("\n" + STARS)
        names_spc, version_check_spc = [], []
        for file in files:
            name, ext = os.path.splitext(file)
            if os.path.exists(name + '_' + options.spc + '.log'):
                names_spc.append(name + '_' + options.spc + '.log')
            elif os.path.exists(name + '_' + options.spc + '.out'):
                names_spc.append(name + '_' + options.spc + '.out')

        # Check SPC program versions
        version_check_spc = [thermo_data[key].sp_version_program for key in thermo_data]
        if all_same(version_check_spc):
            log.write("\no  Using {} in all the single-point corrections.".format(version_check_spc[0]))
        else:
            print_check_fails(log, version_check_spc, file_check, "programs or versions")

        # Check SPC solvation
        solvent_check_spc = [thermo_data[key].sp_solvation_model for key in thermo_data]
        if all_same(solvent_check_spc):
            log.write("\no  Using " + solvent_check_spc[0] + " in all the single-point corrections.")
        else:
            print_check_fails(log, solvent_check_spc, file_check, "solvation models")

        # Check SPC level of theory
        l_o_t_spc = [level_of_theory(name) for name in names_spc]
        if all_same(l_o_t_spc):
            log.write("\no  Using {} in all the single-point corrections.".format(l_o_t_spc[0]))
        else:
            print_check_fails(log, l_o_t_spc, file_check, "levels of theory")

        # Check SPC charge and multiplicity
        charge_spc_check = [thermo_data[key].sp_charge for key in thermo_data]
        multiplicity_spc_check = [thermo_data[key].sp_multiplicity for key in thermo_data]
        if all_same(charge_spc_check) != False and all_same(multiplicity_spc_check) != False:
            log.write("\no  Using charge and multiplicity {} {} in all the single-point corrections.".format(
                charge_spc_check[0], multiplicity_spc_check[0]))
        else:
            print_check_fails(log, charge_spc_check, file_check, "charge and multiplicity", multiplicity_spc_check)

        # Check if the geometries of freq calculations match their corresponding structures in single-point calculations
        geom_duplic_list, geom_duplic_list_spc, geom_duplic_cart, geom_duplic_files, geom_duplic_cart_spc, geom_duplic_files_spc = [], [], [], [], [], []
        for file in files:
            geom_duplic = getoutData(file)
            geom_duplic_cart.append(geom_duplic.cartesians)
            geom_duplic_files.append(file)
        geom_duplic_list.append(geom_duplic_cart)
        geom_duplic_list.append(geom_duplic_files)

        for name in names_spc:
            geom_duplic_spc = getoutData(name)
            geom_duplic_cart_spc.append(geom_duplic_spc.cartesians)
            geom_duplic_files_spc.append(name)
        geom_duplic_list_spc.append(geom_duplic_cart_spc)
        geom_duplic_list_spc.append(geom_duplic_files_spc)
        spc_mismatching = "Caution! Potential differences found between frequency and single-point geometries -"
        if len(geom_duplic_list[0]) == len(geom_duplic_list_spc[0]):
            for i in range(len(files)):
                count = 1
                for j in range(len(geom_duplic_list[0][i])):
                    if count == 1:
                        if geom_duplic_list[0][i][j] == geom_duplic_list_spc[0][i][j]:
                            count = count
                        elif '{0:.3f}'.format(geom_duplic_list[0][i][j][0]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][0] * (-1)) or '{0:.3f}'.format(geom_duplic_list[0][i][j][0]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][0]):
                            if '{0:.3f}'.format(geom_duplic_list[0][i][j][1]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][1] * (-1)) or '{0:.3f}'.format(geom_duplic_list[0][i][j][1]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][1] * (-1)):
                                count = count
                            if '{0:.3f}'.format(geom_duplic_list[0][i][j][2]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][2] * (-1)) or '{0:.3f}'.format(
                                geom_duplic_list[0][i][j][2]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][2] * (-1)):
                                count = count
                        else:
                            spc_mismatching += ", " + geom_duplic_list[1][i]
                            count = count + 1
            if spc_mismatching == "Caution! Potential differences found between frequency and single-point geometries -":
                log.write("\no  No potential differences found between frequency and single-point geometries (based on input coordinates).")
            else:
                spc_mismatching_1 = spc_mismatching[:84]
                spc_mismatching_2 = spc_mismatching[85:]
                log.write("\nx  " + spc_mismatching_1 + spc_mismatching_2 + '.')
        else:
            log.write("\nx  One or more geometries from single-point corrections are missing.")

        # Check for SPC dispersion models
        dispersion_check_spc = [thermo_data[key].sp_empirical_dispersion for key in thermo_data]
        if all_same(dispersion_check_spc):
            if dispersion_check_spc[0] == 'No empirical dispersion detected':
                log.write("\n-  No empirical dispersion detected in any of the calculations.")
            else:
                log.write("\no  Using " + dispersion_check_spc[0] + " in all the singe-point calculations.")
        else:
            print_check_fails(log, dispersion_check_spc, file_check, "dispersion models")
        log.write("\n" + STARS + "\n")


def intro(options, log):
    log.write("\n\n   GoodVibes v" + __version__ + " " + options.start + "\n   " + goodvibes_ref + "\n")

    # Summary of the quasi-harmonic treatment; print out the relevant reference
    if options.temperature_interval is False:
        log.write("   Temperature = " + str(options.temperature) + " Kelvin")
    # If not at standard temp, need to correct the molarity of 1 atmosphere (assuming pressure is still 1 atm)
    if options.gas_phase:
        log.write("   Pressure = 1 atm")
    else:
        log.write("   Concentration = " + str(options.conc) + " mol/L")

    log.write('\n   All energetic values below shown in Hartree unless otherwise specified.')

    log.write("\n\no  Entropic quasi-harmonic treatment: frequency cut-off value of " + str(
        options.S_freq_cutoff) + " wavenumbers will be applied.")
    if options.QS == "grimme":
        log.write("\n   QS = Grimme: Using a mixture of RRHO and Free-rotor vibrational entropies.")
        qs_ref = grimme_ref
    elif options.QS == "truhlar":
        log.write("\n   QS = Truhlar: Using an RRHO treatment where low frequencies are adjusted to the cut-off value.")
        qs_ref = truhlar_ref
    else:
        log.fatal("\n   FATAL ERROR: Unknown quasi-harmonic model " + options.QS + " specified (QS must = grimme or truhlar).")
    log.write("\n   " + qs_ref + '\n')

    # Check if qh-H correction should be applied
    if options.QH:
        log.write("\n\n   Enthalpy quasi-harmonic treatment: frequency cut-off value of " + str(
            options.H_freq_cutoff) + " wavenumbers will be applied.")
        log.write("\n   QH = Head-Gordon: Using an RRHO treatement with an approximation term for vibrational energy.")
        qh_ref = head_gordon_ref
        log.write("\n   REF: " + qh_ref + '\n')

    # Check if D3 corrections should be applied
    if options.D3:
        log.write("\no  D3-Dispersion energy with zero-damping will be calculated and included in the energy and enthalpy terms.")
        log.write("\n   " + d3_ref + '\n')
    if options.D3BJ:
        log.write("\no  D3-Dispersion energy with Becke-Johnson damping will be calculated and added to the energy terms.")
        log.write("\n   " + d3bj_ref + '\n')
    if options.ATM:
        log.write("\n   The repulsive Axilrod-Teller-Muto 3-body term will be included in the dispersion correction.")
        log.write("\n   " + atm_ref + '\n')

    # Check if entropy symmetry correction should be applied
    if options.ssymm:
        log.write('\n   Ssymm requested. Symmetry contribution to entropy to be calculated using S. Patchkovskii\'s \n   open source software "Brute Force Symmetry Analyzer" available under GNU General Public License.')
        log.write('\n   REF: (C) 1996, 2003 S. Patchkovskii, Serguei.Patchkovskii@sympatico.ca')
        log.write('\n\n   Atomic radii used to calculate internal symmetry based on Cambridge Structural Database covalent radii.')
        log.write("\n   REF: " + csd_ref + '\n')

    # Whether linked single-point energies are to be used
    if options.spc:
        log.write("\no  Combining final single point energy with thermal corrections.")

    log.write('\n'+options.command)


def get_vib_scale_factor(level_of_theory, options, log):
    ''' Attempt to automatically obtain frequency scale factor
    Application of freq scale factors requires all outputs to be same level of theory'''

    if options.freq_scale_factor is not False:
        if 'ONIOM' not in level_of_theory[0]:
            log.write("\n\n   User-defined vibrational scale factor " + str(options.freq_scale_factor) + " for " +
                      level_of_theory[0] + " level of theory")
        else:
            log.write("\n\n   User-defined vibrational scale factor " + str(options.freq_scale_factor) +
                      " for QM region")

    else:
        # Look for vibrational scaling factor automatically
        if all_same(level_of_theory):
            level = level_of_theory[0].upper()

            for data in (scaling_data_dict, scaling_data_dict_mod):
                if level in data:

                    options.freq_scale_factor = data[level].zpe_fac
                    ref = scaling_refs[data[level].zpe_ref]
                    log.write("\n\no  Found vibrational scaling factor of {:.3f} for {} level of theory\n"
                              "   {}".format(options.freq_scale_factor, level_of_theory[0], ref))
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

    # Exit program if molecular mechanics scaling factor is given and all files are not ONIOM calculations
    if options.mm_freq_scale_factor is not False:
        if all_same(l_o_t) and 'ONIOM' in l_o_t[0]:
            log.write("\n\no  User-defined vibrational scale factor " +
                      str(options.mm_freq_scale_factor) + " for MM region of " + l_o_t[0])
            log.write("\n   REF: {}".format(oniom_scale_ref))
        else:
            sys.exit("\n   Option --vmm is only for use in ONIOM calculation output files.\n   "
                     " help use option '-h'\n")

    if options.freq_scale_factor is False:
        options.freq_scale_factor = 1.0  # If no scaling factor is found use 1.0
        if all_same(level_of_theory):
            log.write("\n\n   Using vibrational scale factor {} for {} level of "
                      "theory".format(options.freq_scale_factor, level_of_theory[0]))
        else:
            log.write("\n\n   Using vibrational scale factor {}: differing levels of theory "
                      "detected.".format(options.freq_scale_factor))

    return options.freq_scale_factor, options.mm_freq_scale_factor


def summary(thermo_data, options, log, dup_list=[], clusters=[]):
    ''' print table of absolute values'''
    if options.QH:
        stars = "   " + "*" * 142
    else:
        stars = "   " + "*" * 128
    if options.spc is not False: stars += '*' * 14
    if options.cosmo is not False: stars += '*' * 30
    if options.imag_freq is True: stars += '*' * 9
    if options.boltz is True: stars += '*' * 7
    if options.ssymm is True: stars += '*' * 13

    # Boltzmann factors and averaging over clusters
    if options.boltz != False or options.ee != False:
        boltz_facs, weighted_free_energy, boltz_sum = get_boltz(thermo_data, options, clusters, dup_list)

    # Standard mode: tabulate thermochemistry ouput from file(s) at a single temperature and concentration
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
        log.write('{:>13} {:>16}'.format("COSMO-RS", "Solv-qh-G(T)"), thermodata=True)
    if options.boltz is True:
        log.write('{:>7}'.format("Boltz"), thermodata=True)
    if options.imag_freq is True:
        log.write('{:>9}'.format("im freq"), thermodata=True)
    if options.ssymm:
        log.write('{:>13}'.format("Point Group"), thermodata=True)
    log.write("\n" + stars + "")

    for file in thermo_data:  # Loop over the output files and compute thermochemistry
        if file not in [dup[1] for dup in dup_list]:
            try:
                bbe = thermo_data[file]

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
                        if not options.media:
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
                        else:
                            try:
                                from .media import solvents
                            except:
                                from media import solvents
                                # Media correction based on standard concentration of solvent
                            if options.media.lower() in solvents and options.media.lower() == \
                                    os.path.splitext(os.path.basename(file))[0].lower():
                                mw_solvent = solvents[options.media.lower()][0]
                                density_solvent = solvents[options.media.lower()][1]
                                concentration_solvent = (density_solvent * 1000) / mw_solvent
                                media_correction = -(GAS_CONSTANT / J_TO_AU) * math.log(concentration_solvent)
                                if all(getattr(bbe, attrib) for attrib in ["enthalpy", "entropy", "qh_entropy",
                                                                           "gibbs_free_energy", "qh_gibbs_free_energy"]):
                                    if options.QH:
                                        log.write(' {:10.6f} {:13.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} '
                                                  '{:13.6f}'.format(bbe.zpe, bbe.enthalpy, bbe.qh_enthalpy,
                                                                    (options.temperature * (bbe.entropy + media_correction)),
                                                                    (options.temperature * (bbe.qh_entropy + media_correction)),
                                                                    bbe.gibbs_free_energy + (options.temperature * (-media_correction)),
                                                                    bbe.qh_gibbs_free_energy + (options.temperature * (-media_correction))),
                                                  thermodata=True)
                                        log.write("  Solvent")
                                    else:
                                        log.write(' {:10.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} '
                                                  '{:13.6f}'.format(bbe.zpe, bbe.enthalpy,
                                                                    (options.temperature * (bbe.entropy + media_correction)),
                                                                    (options.temperature * (bbe.qh_entropy + media_correction)),
                                                                    bbe.gibbs_free_energy + (options.temperature * (-media_correction)),
                                                                    bbe.qh_gibbs_free_energy + (options.temperature * (-media_correction))), thermodata=True)
                                        log.write("  Solvent")
                            else:
                                if all(getattr(bbe, attrib) for attrib in ["enthalpy", "entropy", "qh_entropy",
                                                                           "gibbs_free_energy", "qh_gibbs_free_energy"]):
                                    if options.QH:
                                        log.write(' {:10.6f} {:13.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} '
                                                  '{:13.6f}'.format(bbe.zpe, bbe.enthalpy, bbe.qh_enthalpy, (options.temperature * bbe.entropy), (options.temperature * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy), thermodata=True)
                                    else:
                                        log.write(' {:10.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} '
                                                  '{:13.6f}'.format(bbe.zpe, bbe.enthalpy,
                                                                    (options.temperature * bbe.entropy),
                                                                    (options.temperature * bbe.qh_entropy),
                                                                    bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy),
                                                  thermodata=True)
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

            except:
                pass

        # Cluster files if requested
        if options.clustering:
            dashes = "-" * (len(stars) - 3)
            for n, cluster in enumerate(clusters):
                for id, structure in enumerate(cluster):
                    if structure == file:
                        if id == len(cluster) - 1:
                            log.write("\n   " + dashes)
                            log.write("\n   " + '{name:<{var_width}} {gval:13.6f} {weight:6.2f}'.format(
                                name='Boltzmann-weighted Cluster ' + alphabet[n].upper(), var_width=len(stars) - 24,
                                gval=weighted_free_energy['cluster-' + alphabet[n].upper()] / boltz_facs[
                                    'cluster-' + alphabet[n].upper()],
                                weight=100 * boltz_facs['cluster-' + alphabet[n].upper()] / boltz_sum),
                                      thermodata=True)
                            log.write("\n   " + dashes)

    log.write("\n" + stars + "\n")


class GVOptions:
    '''This allows you to call GV externally and inherit all default options without an arg parser'''
    def __init__(self):
        self.command = ''
        self.clustering = False
        self.Q = False
        self.QH = False
        self.QS = 'grimme'
        self.freq_cutoff = 100.0
        self.S_freq_cutoff = 100.0
        self.H_freq_cutoff = 100.0
        self.temperature = 298.15
        self.conc = thermo.ATMOS / (thermo.GAS_CONSTANT * self.temperature)
        self.gas_phase = True
        self.temperature_interval = False
        self.freq_scale_factor = 1
        self.mm_freq_scale_factor = False
        self.vmm = False
        self.spc = False
        self.boltz = False
        self.cputime = False
        self.D3 = False
        self.D3BJ = False
        self.ATM = False
        self.xyz = False
        self.sdf = False
        self.csv = False
        self.imag_freq = False
        self.invert = False
        self.freespace = 'none'
        self.duplicate = False
        self.cosmo = False
        self.cosmo_int = False
        self.output = 'output'
        self.pes = False
        self.gconf = True
        self.ee = False
        self.check = False
        self.media = False
        self.graph = False
        self.ssymm = False
        self.inertia = 'global'
        self.start = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())


def main():
    # Get command line inputs. Use -h to list all possible arguments and default values
    parser = ArgumentParser()
    parser.add_argument("-q", dest="Q", action="store_true", default=False,
                        help="Quasi-harmonic entropy correction and enthalpy correction applied (default S=Grimme, "
                             "H=Head-Gordon)")
    parser.add_argument("--qs", dest="QS", default="grimme", type=str.lower, metavar="QS",
                        choices=('grimme', 'truhlar'),
                        help="Type of quasi-harmonic entropy correction (Grimme or Truhlar) (default Grimme)", )
    parser.add_argument("--qh", dest="QH", action="store_true", default=False,
                        help="Type of quasi-harmonic enthalpy correction (Head-Gordon)")
    parser.add_argument("-f", dest="freq_cutoff", default=100, type=float, metavar="FREQ_CUTOFF",
                        help="Cut-off frequency for both entropy and enthalpy (wavenumbers) (default = 100)", )
    parser.add_argument("--fs", dest="S_freq_cutoff", default=100.0, type=float, metavar="S_FREQ_CUTOFF",
                        help="Cut-off frequency for entropy (wavenumbers) (default = 100)")
    parser.add_argument("--fh", dest="H_freq_cutoff", default=100.0, type=float, metavar="H_FREQ_CUTOFF",
                        help="Cut-off frequency for enthalpy (wavenumbers) (default = 100)")
    parser.add_argument("-t", dest="temperature", default=298.15, type=float, metavar="TEMP",
                        help="Temperature (K) (default 298.15)")
    parser.add_argument("-c", dest="conc", default=False, type=float, metavar="CONC",
                        help="Concentration (mol/l) (default 1 atm)")
    parser.add_argument("--ti", dest="temperature_interval", default=False, metavar="TI",
                        help="Initial temp, final temp, step size (K)")
    parser.add_argument("-v", dest="freq_scale_factor", default=False, type=float, metavar="SCALE_FACTOR",
                        help="Frequency scaling factor. If not set, try to find a suitable value in database. "
                             "If not found, use 1.0")
    parser.add_argument("--vmm", dest="mm_freq_scale_factor", default=False, type=float, metavar="MM_SCALE_FACTOR",
                        help="Additional frequency scaling factor used in ONIOM calculations")
    parser.add_argument("--spc", dest="spc", type=str, default=False, metavar="SPC",
                        help="Indicates single point corrections (default False)")
    parser.add_argument("--boltz", dest="boltz", action="store_true", default=False,
                        help="Show Boltzmann factors")
    parser.add_argument("--cpu", dest="cputime", action="store_true", default=False,
                        help="Total CPU time")
    parser.add_argument("--d3", dest="D3", action="store_true", default=False,
                        help="Zero-damped DFTD3 correction will be computed")
    parser.add_argument("--d3bj", dest="D3BJ", action="store_true", default=False,
                        help="Becke-Johnson damped DFTD3 correction will be computed")
    parser.add_argument("--atm", dest="ATM", action="store_true", default=False,
                        help="Axilrod-Teller-Muto 3-body dispersion correction will be computed")
    parser.add_argument("--xyz", dest="xyz", action="store_true", default=False,
                        help="Write Cartesians to a .xyz file (default False)")
    parser.add_argument("--sdf", dest="sdf", action="store_true", default=False,
                        help="Write Cartesians to a .sdf file (default False)")
    parser.add_argument("--csv", dest="csv", action="store_true", default=False,
                        help="Write .csv output file format")
    parser.add_argument("--imag", dest="imag_freq", action="store_true", default=False,
                        help="Print imaginary frequencies (default False)")
    parser.add_argument("--invertifreq", dest="invert", nargs='?', const=True, default=False,
                        help="Make low lying imaginary frequencies positive (cutoff > -50.0 wavenumbers)")
    parser.add_argument("--freespace", dest="freespace", default="none", type=str, metavar="FREESPACE",
                        help="Solvent (H2O, toluene, DMF, AcOH, chloroform) (default none)")
    parser.add_argument("--dup", dest="duplicate", action="store_true", default=False,
                        help="Remove possible duplicates from thermochemical analysis")
    parser.add_argument("--cosmo", dest="cosmo", default=False, metavar="COSMO-RS",
                        help="Filename of a COSMO-RS .tab output file")
    parser.add_argument("--cosmo_int", dest="cosmo_int", default=False, metavar="COSMO-RS",
                        help="Filename of a COSMO-RS .tab output file along with a temperature range (K): "
                             "file.tab,'Initial_T, Final_T'")
    parser.add_argument("--output", dest="output", default="output", metavar="OUTPUT",
                        help="Change the default name of the output file to GoodVibes_\"output\".dat")
    parser.add_argument("--pes", dest="pes", default=False, metavar="PES",
                        help="Tabulate relative values")
    parser.add_argument("--nogconf", dest="gconf", action="store_false", default=True,
                        help="Calculate a free-energy correction related to multi-configurational space (default "
                             "calculate Gconf)")
    parser.add_argument("--ee", dest="ee", default=False, type=str,
                        help="Tabulate selectivity values (excess, ratio) from a mixture, provide pattern for two "
                             "types such as *_R*,*_S*")
    parser.add_argument("--check", dest="check", action="store_true", default=False,
                        help="Checks if calculations were done with the same program, level of theory and solvent, "
                             "as well as detects potential duplicates")
    parser.add_argument("--media", dest="media", default=False, metavar="MEDIA",
                        help="Entropy correction for standard concentration of solvents")
    parser.add_argument("--custom_ext", type=str, default='',
                        help="List of additional file extensions to support, beyond .log or .out, use separated by "
                             "commas (ie, '.qfi, .gaussian'). It can also be specified with environment variable "
                             "GOODVIBES_CUSTOM_EXT")
    parser.add_argument("--graph", dest='graph', default=False, metavar="GRAPH",
                        help="Graph a reaction profile based on free energies calculated. ")
    parser.add_argument("--ssymm", dest='ssymm', action="store_true", default=False,
                        help="Turn on the symmetry correction.")
    parser.add_argument("--bav", dest='inertia', default="global",type=str,choices=['global','conf'],
                        help="Choice of how the moment of inertia is computed. Options = 'global' or 'conf'."
                            "'global' will use the same moment of inertia for all input molecules of 10*10-44,"
                            "'conf' will compute moment of inertia from parsed rotational constants from each Gaussian output file.")
    # Parse Arguments
    (options, args) = parser.parse_known_args()
    options.start = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    options.command = 'o  Requested: '
    files, bbe_vals = [], []

    # If requested, turn on head-gordon enthalpy correction
    if options.Q: options.QH = True

    # If user has specified different file extensions
    if options.custom_ext or os.environ.get('GOODVIBES_CUSTOM_EXT', ''):
        custom_extensions = options.custom_ext.split(',') + os.environ.get('GOODVIBES_CUSTOM_EXT', '').split(',')
        for ext in custom_extensions:
            SUPPORTED_EXTENSIONS.add(ext.strip())

    # Default value for inverting imaginary frequencies
    if options.invert: options.invert == -50.0
    elif options.invert > 0: options.invert *= -1

    # Start a log for the results
    log = io.Logger("Goodvibes", options.output, options.csv)

    # figure out whether conformer clustering is required
    options.clustering = False; clusters = []
    if len(args) > 1:
        for elem in args:
            if elem == 'clust:':
                options.clustering, options.boltz, nclust = True, True, -1

    # Get the filenames from the command line prompt
    args = sys.argv[1:]
    for elem in args:
        if options.clustering:
            if elem == 'clust:':
                clusters.append([])
                nclust += 0
        try:
            if os.path.splitext(elem)[1].lower() in SUPPORTED_EXTENSIONS:  # Look for file names
                for file in glob(elem):
                    if options.spc is False or options.spc is 'link':
                        if file is not options.cosmo:
                            files.append(file)
                        if options.clustering:
                            clusters[nclust].append(file)
                    else:
                        # expects an underscore before spc text in filename ...
                        if file.find('_' + options.spc + ".") == -1:
                            files.append(file)
                            if options.clustering:
                                clusters[nclust].append(file)
                            name, ext = os.path.splitext(file)
                            if not (os.path.exists(name + '_' + options.spc + '.log') or os.path.exists(
                                    name + '_' + options.spc + '.out')) and options.spc != 'link':
                                sys.exit("\nError! SPC calculation file '{}' not found! Make sure files are named with "
                                         "the convention: 'filename_spc' or specify link job.\nFor help, use option '-h'\n"
                                         "".format(name + '_' + options.spc))
            elif elem != 'clust:':  # Look for requested options
                options.command += elem + ' '
        except IndexError:
            pass

    # Start printing results
    start = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    # Check if user has specified any files, if not quit now
    if len(files) == 0:
        sys.exit("\nPlease provide GoodVibes with calculation output files on the command line.\n"
                 "For help, use option '-h'\n")
    if options.clustering:
        options.command += '(clustering active)'

    # If not at standard temp, need to correct the molarity of 1 atmosphere (assuming pressure is still 1 atm)
    if options.conc:
        options.gas_phase = False
    else:
        options.gas_phase = True
        options.conc = thermo.ATMOS / (thermo.GAS_CONSTANT * options.temperature)

    # Initial read of files,
    # Grab level of theory, solvation model, check for Normal Termination
    file_data = []
    for i, file in enumerate(files):
        cc_data = io.getoutData(file, options)

        if hasattr(cc_data, 'program'):
            if cc_data.program != 'Gaussian':
                log.write('\nx  {} format {} not yet supported!'.format(file, cc_data.program))
            elif cc_data.progress != 'Normal':
                log.write('\nx  Warning! Error termination found in {}. This file will be omitted from analysis'.format(file))
            else:
                try: cc_data.functional
                except KeyError: cc_data.functional = 'unknown'
                try: cc_data.basis_set
                except KeyError: cc_data.basis_set = 'unknown'

                log.write('\n   ' + file + ': ' + cc_data.program +' '+cc_data.functional+'/'+cc_data.basis_set+' job terminated normally')
                file_data.append(cc_data)

    files = [file.name for file in file_data] # remove files for which no thermochemical data could be obtained

    # Check if user has specified any files, if not quit now
    if len(files) == 0 or len(file_data) == 0:
        sys.exit("\n\nPlease try again with normally terminated output files.\nFor help, use option '-h'\n")

    # Check the level of theory is consistent across all files
    try:
        level_of_theory = [file.functional + '/' + file.basis_set for file in file_data]
        implicit_solvation = [file.solvation_model for file in file_data]
        if all_same(level_of_theory):
            log.write('\no  All jobs performed at the ' + level_of_theory[-1] + ' level of theory')

        # Exit program if a comparison of Boltzmann factors is requested and level of theory is not uniform across all files
        if not all_same(level_of_theory) and (options.boltz is not False or options.ee is not False):
            sys.exit("\n\nERROR: When comparing files with Boltzmann factors (with bolts, ee, dr options), the level of "
             "theory used should be the same for all files.\n ")
    except ValueError: pass

    # Check for implicit solvation
    printed_solv_warn = False
    try:
        for solv in implicit_solvation:
            if ('smd' in solv[0].lower() or 'pcm' in solv[0].lower()) and not printed_solv_warn:
                log.write("\n   Implicit solvation (SMD/CPCM) detected. Enthalpic and entropic terms are not separable "
                      "safely separated. Use them at your own risk!")
                printed_solv_warn = True
    except ValueError: pass

    # Checks to see whether the available free space of a requested solvent is defined
    freespace = thermo.get_free_space(options.freespace)
    if freespace != 1000.0:
        log.write("\n   Specified solvent " + options.freespace + ": free volume " + str(
            "%.3f" % (freespace / 10.0)) + " (mol/l) corrects the translational entropy")

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

    # Look up vibration scaling factor if not already supplied
    if all_same(level_of_theory):
        options.freq_scale_factor, options.mm_freq_scale_factor = get_vib_scale_factor(level_of_theory, options, log)
    else:
        options.freq_scale_factor = 1.0

    if options.freq_cutoff != 100.0:
        options.S_freq_cutoff = options.freq_cutoff
        options.H_freq_cutoff = options.freq_cutoff

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

    # Loop over all specified output files and compute thermochemistry
    for file in files:
        if options.cosmo:
            cosmo_option = cosmo_solv[file]
        else:
            cosmo_option = None

    # this is the actual thermochemistry calculation!
    bbe_vals = [thermo.calc_bbe(file, options, cosmo=cosmo_option, ssymm=ssymm_option, mm_freq_scale_factor=vmm_option) for file in file_data]

    # Creates a new dictionary object thermo_data, which attaches the bbe data to each file-name
    thermo_data = dict(zip(files, bbe_vals))  # The collected thermochemical data for all files
    interval_bbe_data, interval_thermo_data = [], []

    inverted_freqs, inverted_files = [], []
    for file in files:
        if len(thermo_data[file].inverted_freqs) > 0:
            inverted_freqs.append(thermo_data[file].inverted_freqs)
            inverted_files.append(file)

    # Standard goodvibes analysis (single temperature) requested
    gv_intro = intro(options, log)

    # Standard mode: tabulate thermochemistry ouput from file(s) at a single temperature and concentration
    if options.temperature_interval is False:
        # Look for duplicates or enantiomers if requested
        if options.duplicate: dup_list = check_dup(species, thermo_data, log)
        else: dup_list = []

        # Printing results
        gv_summary = summary(thermo_data, options, log, dup_list, clusters)

    # If necessary, create a file with Cartesians
    if options.xyz:
        xyz = io.xyz_out("Goodvibes_output.xyz", file_data)
    elif options.sdf:
        sdf = io.sdf_out("Goodvibes_output.sdf", file_data)

    # Running a variable temperature analysis of the enthalpy, entropy and the free energy
    elif options.temperature_interval:
        log.write("\n\n   Variable-Temperature analysis of the enthalpy, entropy and the entropy at a constant pressure between")
        if options.cosmo_int is False:
            temperature_interval = [float(temp) for temp in options.temperature_interval.split(',')]
            # If no temperature step was defined, divide the region into 10
            if len(temperature_interval) == 2:
                temperature_interval.append((temperature_interval[1] - temperature_interval[0]) / 10.0)
            interval = range(int(temperature_interval[0]), int(temperature_interval[1] + 1),
                             int(temperature_interval[2]))
            log.write("\n   T init:  %.1f,  T final:  %.1f,  T interval: %.1f" % (
                temperature_interval[0], temperature_interval[1], temperature_interval[2]))
        else:
            interval = t_interval
            log.write("\n   T init:  %.1f,   T final: %.1f" % (interval[0], interval[-1]))

        if options.QH:
            qh_print_format = "\n\n   {:<39} {:>13} {:>24} {:>13} {:>10} {:>10} {:>13} {:>13}"
            if options.spc and options.cosmo_int:
                log.write(qh_print_format.format("Structure", "Temp/K", "H_SPC", "qh-H_SPC", "T.S", "T.qh-S",
                                                 "G(T)_SPC", "COSMO-RS-qh-G(T)_SPC"), thermodata=True)
            elif options.cosmo_int:
                log.write(qh_print_format.format("Structure", "Temp/K", "H", "qh-H", "T.S", "T.qh-S", "G(T)",
                                                 "qh-G(T)", "COSMO-RS-qh-G(T)"), thermodata=True)
            elif options.spc:
                log.write(qh_print_format.format("Structure", "Temp/K", "H_SPC", "qh-H_SPC", "T.S", "T.qh-S",
                                                 "G(T)_SPC", "qh-G(T)_SPC"), thermodata=True)
            else:
                log.write(qh_print_format.format("Structure", "Temp/K", "H", "qh-H", "T.S", "T.qh-S", "G(T)",
                                                 "qh-G(T)"), thermodata=True)
        else:
            print_format_3 = '\n\n   {:<39} {:>13} {:>24} {:>10} {:>10} {:>13} {:>13}'
            if options.spc and options.cosmo_int:
                log.write(print_format_3.format("Structure", "Temp/K", "H_SPC", "T.S", "T.qh-S", "G(T)_SPC",
                                                "COSMO-RS-qh-G(T)_SPC"), thermodata=True)
            elif options.cosmo_int:
                log.write(print_format_3.format("Structure", "Temp/K", "H", "T.S", "T.qh-S", "G(T)", "qh-G(T)",
                                                "COSMO-RS-qh-G(T)"), thermodata=True)
            elif options.spc:
                log.write(print_format_3.format("Structure", "Temp/K", "H_SPC", "T.S", "T.qh-S", "G(T)_SPC",
                                                "qh-G(T)_SPC"), thermodata=True)
            else:
                log.write(print_format_3.format("Structure", "Temp/K", "H", "T.S", "T.qh-S", "G(T)", "qh-G(T)"),
                          thermodata=True)

        for h, file in enumerate(files):  # Temperature interval
            log.write("\n" + stars)
            interval_bbe_data.append([])
            for i in range(len(interval)):  # Iterate through the temperature range
                temp = interval[i]
                if gas_phase:
                    conc = ATMOS / GAS_CONSTANT / temp
                else:
                    conc = options.conc
                linear_warning = []
                if options.cosmo_int is False:
                    cosmo_option = False
                else:
                    cosmo_option = gsolv_dicts[i][file]
                if options.cosmo_int is False:
                    # haven't implemented D3 for this option
                    bbe = calc_bbe(file, options.QS, options.QH, options.S_freq_cutoff, options.H_freq_cutoff, temp,
                                   conc, options.freq_scale_factor, options.freespace, options.spc, options.invert,
                                   0.0, cosmo=cosmo_option, inertia=options.inertia)
                interval_bbe_data[h].append(bbe)
                linear_warning.append(bbe.linear_warning)
                if linear_warning == [['Warning! Potential invalid calculation of linear molecule from Gaussian.']]:
                    log.write("\nx  ")
                    log.write('{:<39}'.format(os.path.splitext(os.path.basename(file))[0]), thermodata=True)
                    log.write('             Warning! Potential invalid calculation of linear molecule from Gaussian ...')
                else:
                    # Gaussian spc files
                    if hasattr(bbe, "scf_energy") and not hasattr(bbe, "gibbs_free_energy"):
                        log.write("\nx  " + '{:<39}'.format(os.path.splitext(os.path.basename(file))[0]))
                    # ORCA spc files
                    elif not hasattr(bbe, "scf_energy") and not hasattr(bbe, "gibbs_free_energy"):
                        log.write("\nx  " + '{:<39}'.format(os.path.splitext(os.path.basename(file))[0]))
                    if not hasattr(bbe, "gibbs_free_energy"):
                        log.write("Warning! Couldn't find frequency information ...")
                    else:
                        log.write("\no  ")
                        log.write('{:<39} {:13.1f}'.format(os.path.splitext(os.path.basename(file))[0], temp),
                                  thermodata=True)
                        # if not options.media:
                        if all(getattr(bbe, attrib) for attrib in
                               ["enthalpy", "entropy", "qh_entropy", "gibbs_free_energy", "qh_gibbs_free_energy"]):
                            if options.QH:
                                if options.cosmo_int:
                                    log.write(' {:24.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(
                                        bbe.enthalpy, bbe.qh_enthalpy, (temp * bbe.entropy),
                                        (temp * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.cosmo_qhg),
                                        thermodata=True)
                                else:
                                    log.write(' {:24.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(
                                        bbe.enthalpy, bbe.qh_enthalpy, (temp * bbe.entropy),
                                        (temp * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy),
                                        thermodata=True)
                            else:
                                if options.cosmo_int:
                                    log.write(' {:24.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(bbe.enthalpy, (
                                            temp * bbe.entropy), (temp * bbe.qh_entropy), bbe.gibbs_free_energy,
                                                                                                     bbe.cosmo_qhg),
                                              thermodata=True)
                                else:
                                    log.write(' {:24.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(bbe.enthalpy, (
                                            temp * bbe.entropy), (temp * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy),
                                              thermodata=True)
                        if options.media is not False and options.media.lower() in solvents and options.media.lower() == \
                                os.path.splitext(os.path.basename(file))[0].lower():
                            log.write("  Solvent: {:4.2f}M ".format(media_conc))

            log.write("\n" + stars + "\n")

    # Perform checks for consistent options provided in calculation files (level of theory)
    if options.check:
        check_files(file_data, thermo_data, options, log)

    # Print CPU usage if requested
    if options.cputime:
        cpu = calc_cpu(thermo_data, options, log)

    # Tabulate relative values
    if options.pes:
        species, table = pes.tabulate(thermo_data, options, log, show=True)

    # Compute enantiomeric excess
    if options.ee is not False:
        boltz_facs, weighted_free_energy, boltz_sum = get_boltz(thermo_data, options, clusters, dup_list)
        ee, er, ratio, dd_free_energy, failed, preference = get_selectivity(files, options, boltz_facs, boltz_sum, log, dup_list)

    # Graph reaction profiles
    if options.graph is not False:
        graph_data = pes.get_pes(thermo_data, options, log)
        pes.graph_reaction_profile(graph_data, options, log)

    # Close the log
    log.finalize()

if __name__ == "__main__":
    main()
