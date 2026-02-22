#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Quasi-harmonic thermochemical corrections from electronic structure calculations."""
from __future__ import print_function, absolute_import

import math
import os.path
import sys
import time
from datetime import datetime, timedelta
from glob import glob
from argparse import ArgumentParser
import numpy as np

# Importing regardless of relative import
try:
    from .vib_scale_factors import scaling_data_dict, scaling_refs, canonicalize_level
    from .pes import get_pes, graph_reaction_profile
    from .io import level_of_theory, read_initial, xyz_out, parse_qcdata
    from .thermo import calc_bbe, get_free_space
except ImportError:
    from vib_scale_factors import scaling_data_dict, scaling_refs, canonicalize_level
    from pes import get_pes, graph_reaction_profile
    from io import level_of_theory, read_initial, xyz_out, parse_qcdata
    from thermo import calc_bbe, get_free_space

# VERSION NUMBER
__version__ = "4.0"

SUPPORTED_EXTENSIONS = set(('.out', '.log'))

# PHYSICAL CONSTANTS                                      UNITS
GAS_CONSTANT = 8.3144621  # J / K / mol
ATMOS = 101.325  # UNIT CONVERSION
J_TO_AU = 4.184 * 627.509541 * 1000.0  # UNIT CONVERSION
KCAL_TO_AU = 627.509541  # UNIT CONVERSION

# Some literature references
grimme_ref = "Grimme, S. Chem. Eur. J. 2012, 18, 9955-9964"
truhlar_ref = "Ribeiro, R. F.; Marenich, A. V.; Cramer, C. J.; Truhlar, D. G. J. Phys. Chem. B 2011, 115, 14556-14562"
head_gordon_ref = "Li, Y.; Gomes, J.; Sharada, S. M.; Bell, A. T.; Head-Gordon, M. J. Phys. Chem. C 2015, 119, 1840-1850"
goodvibes_ref = ("Luchini, G.; Alegre-Requena, J. V.; Funes-Ardoiz, I.; Paton, R. S. F1000Research, 2020, 9, 291."
                 "\n   DOI: 10.12688/f1000research.22758.1")
csd_ref = ("C. R. Groom, I. J. Bruno, M. P. Lightfoot and S. C. Ward, Acta Cryst. 2016, B72, 171-179"
           "\n   Cordero, B.; Gomez V.; Platero-Prats, A. E.; Reves, M.; Echeverria, J.; Cremades, E.; Barragan, F.; Alvarez, S. Dalton Trans. 2008, 2832-2838")
oniom_scale_ref = "Simon, L.; Paton, R. S. J. Am. Chem. Soc. 2018, 140, 5412-5420"
alphabet = 'abcdefghijklmnopqrstuvwxyz'


def all_same(items):
    """Returns bool for checking if all items in a list are the same."""
    return all(x == items[0] for x in items)


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


def add_time(tm, cpu):
    """Calculate elapsed time."""
    [days, hrs, mins, secs, msecs] = cpu
    fulldate = datetime(100, 1, tm.day, tm.hour, tm.minute, tm.second, tm.microsecond)
    fulldate = fulldate + timedelta(days=days, hours=hrs, minutes=mins, seconds=secs, microseconds=msecs * 1000)
    return fulldate


def get_selectivity(pattern, files, boltz_facs, boltz_sum, temperature, log, dup_list):
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
    dirs = []
    for file in files:
        dirs.append(os.path.dirname(file))
    dirs = list(set(dirs))
    a_files, b_files, a_sum, b_sum, failed, pref = [], [], 0.0, 0.0, False, ''

    [a_regex,b_regex] = pattern.split(':')
    [a_regex,b_regex] = [a_regex.strip(), b_regex.strip()]

    A = ''.join(a for a in a_regex if a.isalnum())
    B = ''.join(b for b in b_regex if b.isalnum())

    if len(dirs) > 1 or dirs[0] != '':
        for dir in dirs:
            a_files.extend(glob(dir+'/'+a_regex))
            b_files.extend(glob(dir+'/'+b_regex))
    else:
        a_files.extend(glob(a_regex))
        b_files.extend(glob(b_regex))


    if len(a_files) == 0 or len(b_files) == 0:
        log.write("\n   Warning! Filenames have not been formatted correctly for determining selectivity\n")
        log.write("   Make sure the filename contains either " + A + " or " + B + "\n")
        sys.exit("   Please edit either your filenames or selectivity pattern argument and try again\n")
    # Grab Boltzmann sums
    for file in files:
        duplicate = False
        if len(dup_list) != 0:
            for dup in dup_list:
                if dup[0] == file:
                    duplicate = True
        if not duplicate:
            if file in a_files:
                a_sum += boltz_facs[file] / boltz_sum
            elif file in b_files:
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
        log.write("\n   Warning! No files found for an enantioselectivity analysis, adjust the stereodetermining step name and try again.\n")
        failed = True
    ee = abs(ee)
    try:
        dd_free_energy = GAS_CONSTANT / J_TO_AU * temperature * math.log((50 + abs(ee) / 2.0) / (50 - abs(ee) / 2.0)) * KCAL_TO_AU
    except ZeroDivisionError:
        dd_free_energy = 0.0
    return ee, r, ratio, dd_free_energy, failed, pref


def get_boltz(files, thermo_data, clustering, clusters, temperature, dup_list):
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

    for file in files:  # Need the most stable structure
        bbe = thermo_data[file]
        if hasattr(bbe, "qh_gibbs_free_energy"):
            if bbe.qh_gibbs_free_energy is not None:
                if bbe.qh_gibbs_free_energy < e_min:
                    e_min = bbe.qh_gibbs_free_energy

    if clustering:
        for n, cluster in enumerate(clusters):
            boltz_facs['cluster-' + alphabet[n].upper()] = 0.0
            weighted_free_energy['cluster-' + alphabet[n].upper()] = 0.0
    # Calculate E_rel and Boltzmann factors
    for file in files:
        duplicate = False
        if len(dup_list) != 0:
            for dup in dup_list:
                if dup[0] == file:
                    duplicate = True
        if not duplicate:

            bbe = thermo_data[file]
            if hasattr(bbe, "qh_gibbs_free_energy"):
                if bbe.qh_gibbs_free_energy is not None:
                    e_rel[file] = bbe.qh_gibbs_free_energy - e_min
                    boltz_facs[file] = math.exp(-e_rel[file] * J_TO_AU / GAS_CONSTANT / temperature)
                    if clustering:
                        for n, cluster in enumerate(clusters):
                            for structure in cluster:
                                if structure == file:
                                    boltz_facs['cluster-' + alphabet[n].upper()] += math.exp(
                                        -e_rel[file] * J_TO_AU / GAS_CONSTANT / temperature)
                                    weighted_free_energy['cluster-' + alphabet[n].upper()] += math.exp(
                                        -e_rel[file] * J_TO_AU / GAS_CONSTANT / temperature) * bbe.qh_gibbs_free_energy
                    boltz_sum += math.exp(-e_rel[file] * J_TO_AU / GAS_CONSTANT / temperature)

    return boltz_facs, weighted_free_energy, boltz_sum


def check_dup(files, thermo_data):
    """
    Check for duplicate species from among all files based on energy, rotational constants and frequencies

    Energy cutoff = 1 microHartree
    RMS Rotational Constant cutoff = 1kHz
    RMS Freq cutoff = 10 wavenumbers
    """
    e_cutoff = 1e-4
    ro_cutoff = 0.1
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
        if option2 is not False:
            attr = (attr, option2[i])
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


def check_files(log, files, thermo_data, options, STARS, l_o_t, solvation_model, orientation, grid):
    """
    Perform checks for consistency in calculation output files for computational projects

    Check for consistency in: Gaussian version, solvation state/gas phase,
    level of theory/basis set, charge and multiplicity, standard concentration,
    potential linear molecule errors, transition state verification, empirical dispersion models
    """
    log.write("\n   Checks for thermochemistry calculations (frequency calculations):")
    log.write("\n" + STARS)
    # Check program used and version
    version_check = [thermo_data[key].version_program for key in thermo_data]
    file_check = [thermo_data[key].file for key in thermo_data]
    if all_same(version_check):
        log.write("\no  Using {} in all calculations.".format(version_check[0]))
    else:
        print_check_fails(log, version_check, file_check, "programs or versions")

    # Check level of theory
    if all_same(l_o_t):
        log.write("\no  Using {} in all calculations.".format(l_o_t[0]))
    else:
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
    if not all_same(solvent_check) and "gas phase" in solvent_check:
        log.write("\nx  Caution! The right standard concentration cannot be determined because the calculations use a combination of gas and solvent phases.")
    if not all_same(solvent_check) and "gas phase" not in solvent_check:
        log.write("\nx  Caution! Different solvents used, fix this issue and use option -c 1 for a standard concentration of 1 M.")

    # Check charge and multiplicity
    charge_check = [thermo_data[key].charge for key in thermo_data]
    multiplicity_check = [thermo_data[key].multiplicity for key in thermo_data]
    if all_same(charge_check) and all_same(multiplicity_check):
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
    linear_fails_atom, linear_fails_cart, linear_fails_files, linear_fails_list = [], [], [], []
    frequency_list = []
    for file in files:
        bbe = thermo_data[file]
        linear_fails_cart.append(bbe.cartesians)
        linear_fails_atom.append(bbe.atom_types)
        linear_fails_files.append(file)
        frequency_list.append(bbe.frequency_wn)

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
                            for ci in range(len(linear_fails_list[1][i][j])):
                                if linear_fails_list[0][i][j] == linear_fails_list[0][i][k]:
                                    if linear_fails_list[1][i][j][ci] > (-linear_fails_list[1][i][k][ci] - 0.1) and \
                                            linear_fails_list[1][i][j][ci] < (-linear_fails_list[1][i][k][ci] + 0.1):
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
            if isinstance(solvent_check_spc[0],list):
                log.write("\no  Using " + solvent_check_spc[0][0] + " in all single-point corrections.")
            else:
                log.write("\no  Using " + solvent_check_spc[0] + " in all single-point corrections.")
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
        if all_same(charge_spc_check) and all_same(multiplicity_spc_check):
            log.write("\no  Using charge and multiplicity {} {} in all the single-point corrections.".format(
                charge_spc_check[0], multiplicity_spc_check[0]))
        else:
            print_check_fails(log, charge_spc_check, file_check, "charge and multiplicity", multiplicity_spc_check)

        # Check if the geometries of freq calculations match their corresponding structures in single-point calculations
        geom_duplic_list, geom_duplic_list_spc, geom_duplic_cart, geom_duplic_files, geom_duplic_cart_spc, geom_duplic_files_spc = [], [], [], [], [], []
        for file in files:
            geom_duplic_cart.append(thermo_data[file].cartesians)
            geom_duplic_files.append(file)
        geom_duplic_list.append(geom_duplic_cart)
        geom_duplic_list.append(geom_duplic_files)

        for name in names_spc:
            spc_qcdata = parse_qcdata(name)
            geom_duplic_cart_spc.append(spc_qcdata.cartesians)
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
                            pass
                        elif '{0:.3f}'.format(geom_duplic_list[0][i][j][0]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][0] * (-1)) or '{0:.3f}'.format(geom_duplic_list[0][i][j][0]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][0]):
                            if '{0:.3f}'.format(geom_duplic_list[0][i][j][1]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][1] * (-1)) or '{0:.3f}'.format(geom_duplic_list[0][i][j][1]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][1] * (-1)):
                                pass
                            if '{0:.3f}'.format(geom_duplic_list[0][i][j][2]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][2] * (-1)) or '{0:.3f}'.format(
                                geom_duplic_list[0][i][j][2]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][2] * (-1)):
                                pass
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


def display_name(file):
    """Return the basename without extension, used for output display."""
    return os.path.splitext(os.path.basename(file))[0]


def parse_arguments():
    """Parse command-line arguments and return (options, args, command, clustering, clusters)."""
    files = []
    clusters = []
    command = 'o  Requested: '
    clustering = False
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
    parser.add_argument("--xyz", dest="xyz", action="store_true", default=False,
                        help="Write Cartesians to a .xyz file (default False)")
    parser.add_argument("--csv", dest="csv", action="store_true", default=False,
                        help="Write .csv output file format")
    parser.add_argument("--imag", dest="imag_freq", action="store_true", default=False,
                        help="Print imaginary frequencies (default False)")
    parser.add_argument("--invertifreq", dest="invert", nargs='?', const=True, default=False,
                        help="Make low lying imaginary frequencies positive (cutoff > -50.0 wavenumbers)")
    parser.add_argument("--freespace", dest="freespace", default=None, type=str, metavar="FREESPACE",
                        help="Solvent (H2O, toluene, DMF, AcOH, chloroform) (default none)")
    parser.add_argument("--dup", dest="duplicate", action="store_true", default=False,
                        help="Remove possible duplicates from thermochemical analysis")
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
    parser.add_argument("--g4", dest="g4", action="store_true", default=False,
                        help="Use this option when using G4 calculations in Gaussian")
    # Parse Arguments
    (options, args) = parser.parse_known_args()
    # If requested, turn on head-gordon enthalpy correction
    if options.Q:
        options.QH = True
    # If user has specified different file extensions
    if options.custom_ext or os.environ.get('GOODVIBES_CUSTOM_EXT', ''):
        custom_extensions = options.custom_ext.split(',') + os.environ.get('GOODVIBES_CUSTOM_EXT', '').split(',')
        for ext in custom_extensions:
            SUPPORTED_EXTENSIONS.add(ext.strip())

    # Default value for inverting imaginary frequencies
    if options.invert is True:
        options.invert = -50.0
    elif options.invert is not False and options.invert > 0:
        options.invert = -1 * options.invert

    if len(args) > 1:
        for elem in args:
            if elem == 'clust:':
                clustering = True
                options.boltz = True
                nclust = -1
    # Get the filenames from the command line prompt
    args = sys.argv[1:]
    for elem in args:
        if clustering:
            if elem == 'clust:':
                clusters.append([])
                nclust += 1
        try:
            if os.path.splitext(elem)[1].lower() in SUPPORTED_EXTENSIONS:  # Look for file names
                for file in glob(elem):
                    if options.spc is False or options.spc == 'link':
                        files.append(file)
                        if clustering:
                            clusters[nclust].append(file)
                    else:
                        if file.find('_' + options.spc + ".") == -1:
                            files.append(file)
                            if clustering:
                                clusters[nclust].append(file)
                            name, ext = os.path.splitext(file)
                            if not (os.path.exists(name + '_' + options.spc + '.log') or os.path.exists(
                                    name + '_' + options.spc + '.out')) and options.spc != 'link':
                                sys.exit("\nError! SPC calculation file '{}' not found! Make sure files are named with "
                                         "the convention: 'filename_spc' or specify link job.\nFor help, use option '-h'\n"
                                         "".format(name + '_' + options.spc))
            elif elem != 'clust:':  # Look for requested options
                command += elem + ' '
        except IndexError:
            pass

    if clustering:
        command += '(clustering active)'

    return options, files, command, clustering, clusters


def collect_and_validate_files(files, options, log):
    """Read initial data from files, remove error-terminated ones. Returns (files, l_o_t, s_m, orientation, grid)."""
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

    return files, l_o_t, s_m, orientation, grid


def resolve_scaling_factor(files, options, l_o_t, log):
    """Attempt to automatically obtain frequency scale factor and validate level of theory."""
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
            level = canonicalize_level(l_o_t[0])
            if level in scaling_data_dict:
                options.freq_scale_factor = scaling_data_dict[level].zpe_fac
                ref = scaling_refs[scaling_data_dict[level].zpe_ref]
                log.write("\n\no  Found vibrational scaling factor of {:.3f} for {} level of theory\n"
                          "   {}".format(options.freq_scale_factor, l_o_t[0], ref))
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


def validate_and_configure(options, s_m, log):
    """Validate solvent, print QH/QS configuration, and return (ssymm_option, vmm_option)."""
    # Checks to see whether the available free space of a requested solvent is defined
    if options.freespace is not None:
        freespace = get_free_space(options.freespace)
        if freespace != 1000.0:
            log.write("\n   Specified solvent " + options.freespace + ": free volume " + str(
                "%.3f" % (freespace / 10.0)) + " (mol/l) corrects the translational entropy")

    # Check for implicit solvation
    printed_solv_warn = False
    for i in s_m:
        if ('smd' in i.lower() or 'cpcm' in i.lower()) and not printed_solv_warn:
            log.write("\n\n   Caution! Implicit solvation (SMD/CPCM) detected. Enthalpic and entropic terms cannot be "
                      "safely separated. Use them at your own risk!")
            printed_solv_warn = True

    if options.freq_cutoff != 100.0:
        options.S_freq_cutoff = options.freq_cutoff
        options.H_freq_cutoff = options.freq_cutoff

    # Summary of the quasi-harmonic treatment; print out the relevant reference
    log.write("\n\n   Entropic mRRHO treatment: frequency cut-off value (tau) of " + str(
        options.S_freq_cutoff) + " cm-1 will be applied.")
    if options.QS == "grimme":
        log.write("\n   QS = Grimme: Using a mixture of RRHO and free-rotor vibrational entropies.")
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
    ssymm_option = options.ssymm if options.ssymm else False
    vmm_option = options.mm_freq_scale_factor if options.mm_freq_scale_factor is not False else False

    return ssymm_option, vmm_option


def compute_thermochemistry(files, options, ssymm_option, vmm_option, log):
    """Run calc_bbe for each file. Returns (thermo_data, bbe_vals, media_conc)."""
    bbe_vals = []
    media_conc = None
    for file in files:
        conc = options.conc
        #check if media correction should be applied
        if options.media is not False:
            try:
                from .media import solvents
            except ImportError:
                from media import solvents
            if options.media.lower() in solvents and options.media.lower() == display_name(file).lower():
                mweight = solvents[options.media.lower()][0]
                density = solvents[options.media.lower()][1]
                conc = (density * 1000) / mweight
                media_conc = conc
        bbe = calc_bbe(file, options.QS, options.QH, options.S_freq_cutoff, options.H_freq_cutoff, options.temperature,
                       conc, options.freq_scale_factor, options.freespace, options.spc, options.invert,
                       ssymm=ssymm_option, mm_freq_scale_factor=vmm_option,
                       inertia=options.inertia, g4=options.g4)

        # Populate bbe_vals with indivual bbe entries for each file
        bbe_vals.append(bbe)

    # Creates a new dictionary object thermo_data, which attaches the bbe data to each file-name
    file_list = list(files)
    thermo_data = dict(zip(file_list, bbe_vals))  # The collected thermochemical data for all files
    return thermo_data, bbe_vals, media_conc


def print_results(files, thermo_data, options, log, stars, clustering, clusters, xyz=None, media_conc=None):
    """Print single-temperature thermochemistry output table."""
    # Check if user has chosen to make any low lying imaginary frequencies positive
    inverted_freqs, inverted_files = [], []
    for file in files:
        if len(thermo_data[file].inverted_freqs) > 0:
            inverted_freqs.append(thermo_data[file].inverted_freqs)
            inverted_files.append(file)
    if options.invert is not False:
        for i, file in enumerate(inverted_files):
            if len(inverted_freqs[i]) == 1:
                log.write("\n\n   The following frequency was made positive and used in calculations: " +
                          str(inverted_freqs[i][0]) + " from " + file)
            elif len(inverted_freqs[i]) > 1:
                log.write("\n\n   The following frequencies were made positive and used in calculations: " +
                          str(inverted_freqs[i]) + " from " + file)

    # Adjust printing according to options requested
    if options.spc is not False:
        stars += '*' * 14
    if options.imag_freq is True:
        stars += '*' * 9
    if options.boltz is True:
        stars += '*' * 7
    if options.ssymm is True:
        stars += '*' * 13

    total_cpu_time, add_days = datetime(100, 1, 1, 00, 00, 00, 00), 0

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
    if options.boltz is True:
        log.write('{:>7}'.format("Boltz"), thermodata=True)
    if options.imag_freq is True:
        log.write('{:>9}'.format("im freq"), thermodata=True)
    if options.ssymm:
        log.write('{:>13}'.format("Point Group"), thermodata=True)
    log.write("\n" + stars + "")

    # Look for duplicates or enantiomers
    if options.duplicate:
        dup_list = check_dup(files, thermo_data)
    else:
        dup_list = []

    # Boltzmann factors and averaging over clusters
    boltz_facs, boltz_sum = None, None
    if options.boltz:
        boltz_facs, weighted_free_energy, boltz_sum = get_boltz(files, thermo_data, clustering, clusters,
                                                                options.temperature, dup_list)

    try:
        from .media import solvents
    except ImportError:
        try:
            from media import solvents
        except ImportError:
            solvents = {}

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
            if options.cputime:  # Add up CPU times
                if hasattr(bbe, "cpu"):
                    if bbe.cpu is not None:
                        total_cpu_time = add_time(total_cpu_time, bbe.cpu)
                if hasattr(bbe, "sp_cpu"):
                    if bbe.sp_cpu is not None:
                        total_cpu_time = add_time(total_cpu_time, bbe.sp_cpu)
            if total_cpu_time.month > 1:
                add_days += 31

            if xyz:  # Write Cartesians
                xyz.write_text(str(len(bbe.atom_types)))
                if hasattr(bbe, "scf_energy"):
                    xyz.write_text(
                        '{:<39} {:>13} {:13.6f}'.format(display_name(file), 'Eopt', bbe.scf_energy))
                else:
                    xyz.write_text('{:<39}'.format(display_name(file)))
                if bbe.atom_types and bbe.cartesians:
                    xyz.write_coords(bbe.atom_types, bbe.cartesians)

            # Check for possible error in Gaussian calculation of linear molecules which can return 2 rotational constants instead of 3
            if bbe.linear_warning:
                log.write("\nx  " + '{:<39}'.format(display_name(file)))
                log.write('          ----   Caution! Potential invalid calculation of linear molecule from Gaussian')
            else:
                if hasattr(bbe, "gibbs_free_energy"):
                    if options.spc is not False:
                        if bbe.sp_energy != '!':
                            log.write("\no  ")
                            log.write('{:<39}'.format(display_name(file)), thermodata=True)
                            log.write(' {:13.6f}'.format(bbe.sp_energy), thermodata=True)
                        if bbe.sp_energy == '!':
                            log.write("\nx  ")
                            log.write('{:<39}'.format(display_name(file)), thermodata=True)
                            log.write(' {:>13}'.format('----'), thermodata=True)
                    else:
                        log.write("\no  ")
                        log.write('{:<39}'.format(display_name(file)), thermodata=True)
                # Gaussian SPC file handling
                if hasattr(bbe, "scf_energy") and not hasattr(bbe, "gibbs_free_energy"):
                    log.write("\nx  " + '{:<39}'.format(display_name(file)))
                # ORCA spc files
                elif not hasattr(bbe, "scf_energy") and not hasattr(bbe, "gibbs_free_energy"):
                    log.write("\nx  " + '{:<39}'.format(display_name(file)))
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
                            display_name(file).lower():
                        log.write("  Solvent: {:4.2f}M ".format(media_conc))

            # Append requested options to end of output
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
        # Cluster files if requested
        if clustering:
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
    return stars, dup_list, total_cpu_time, add_days


def print_temperature_interval(files, options, log, stars, gas_phase, media_conc=None):
    """Run variable-temperature analysis and print results. Returns (interval_bbe_data, interval, file_list)."""
    log.write("\n\n   Variable-Temperature analysis of the enthalpy, entropy and the entropy at a constant pressure between")
    temperature_interval = [float(temp) for temp in options.temperature_interval.split(',')]
    # If no temperature step was defined, divide the region into 10
    if len(temperature_interval) == 2:
        temperature_interval.append((temperature_interval[1] - temperature_interval[0]) / 10.0)
    interval = range(int(temperature_interval[0]), int(temperature_interval[1] + 1),
                     int(temperature_interval[2]))
    log.write("\n   T init:  %.1f,  T final:  %.1f,  T interval: %.1f" % (
        temperature_interval[0], temperature_interval[1], temperature_interval[2]))
    if options.QH:
        qh_print_format = "\n\n   {:<39} {:>13} {:>24} {:>13} {:>10} {:>10} {:>13} {:>13}"
        if options.spc:
            log.write(qh_print_format.format("Structure", "Temp/K", "H_SPC", "qh-H_SPC", "T.S", "T.qh-S",
                                             "G(T)_SPC", "qh-G(T)_SPC"), thermodata=True)
        else:
            log.write(qh_print_format.format("Structure", "Temp/K", "H", "qh-H", "T.S", "T.qh-S", "G(T)",
                                             "qh-G(T)"), thermodata=True)
    else:
        print_format_3 = '\n\n   {:<39} {:>13} {:>24} {:>10} {:>10} {:>13} {:>13}'
        if options.spc:
            log.write(print_format_3.format("Structure", "Temp/K", "H_SPC", "T.S", "T.qh-S", "G(T)_SPC",
                                            "qh-G(T)_SPC"), thermodata=True)
        else:
            log.write(print_format_3.format("Structure", "Temp/K", "H", "T.S", "T.qh-S", "G(T)", "qh-G(T)"),
                      thermodata=True)

    try:
        from .media import solvents
    except ImportError:
        try:
            from media import solvents
        except ImportError:
            solvents = {}

    interval_bbe_data = []
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
            bbe = calc_bbe(file, options.QS, options.QH, options.S_freq_cutoff, options.H_freq_cutoff, temp,
                           conc, options.freq_scale_factor, options.freespace, options.spc, options.invert,
                           inertia=options.inertia, g4=options.g4)
            interval_bbe_data[h].append(bbe)
            linear_warning.append(bbe.linear_warning)
            if linear_warning == [['Warning! Potential invalid calculation of linear molecule from Gaussian.']]:
                log.write("\nx  ")
                log.write('{:<39}'.format(display_name(file)), thermodata=True)
                log.write('             Warning! Potential invalid calculation of linear molecule from Gaussian ...')
            else:
                # Gaussian spc files
                if hasattr(bbe, "scf_energy") and not hasattr(bbe, "gibbs_free_energy"):
                    log.write("\nx  " + '{:<39}'.format(display_name(file)))
                # ORCA spc files
                elif not hasattr(bbe, "scf_energy") and not hasattr(bbe, "gibbs_free_energy"):
                    log.write("\nx  " + '{:<39}'.format(display_name(file)))
                if not hasattr(bbe, "gibbs_free_energy"):
                    log.write("Warning! Couldn't find frequency information ...")
                else:
                    log.write("\no  ")
                    log.write('{:<39} {:13.1f}'.format(display_name(file), temp),
                              thermodata=True)
                    # if not options.media:
                    if all(getattr(bbe, attrib) for attrib in
                           ["enthalpy", "entropy", "qh_entropy", "gibbs_free_energy", "qh_gibbs_free_energy"]):
                        if options.QH:
                            log.write(' {:24.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(
                                bbe.enthalpy, bbe.qh_enthalpy, (temp * bbe.entropy),
                                (temp * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy),
                                thermodata=True)
                        else:
                            log.write(' {:24.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(bbe.enthalpy, (
                                    temp * bbe.entropy), (temp * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy),
                                      thermodata=True)
                    if options.media is not False and options.media.lower() in solvents and options.media.lower() == \
                            display_name(file).lower():
                        log.write("  Solvent: {:4.2f}M ".format(media_conc))

        log.write("\n" + stars + "\n")

    return interval_bbe_data, interval, list(files)


def print_pes_results(files, thermo_data, options, log, stars, clustering, clusters, dup_list,
                      interval_bbe_data=None, interval=None, file_list=None):
    """Tabulate relative PES values, Boltzmann weighting, and selectivity."""
    if options.gconf:
        log.write('\n   Gconf correction requested to be applied to below relative values using quasi-harmonic Boltzmann factors\n')
    for key in thermo_data:
        if not hasattr(thermo_data[key], "qh_gibbs_free_energy"):
            pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
            sys.exit(pes_error)
        if not hasattr(thermo_data[key], "sp_energy") and options.spc is not False:
            pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
            sys.exit(pes_error)
    # Interval applied to PES
    if options.temperature_interval:
        if options.gconf:
            log.write('\n   Gconf correction requested to be applied to below relative values using quasi-harmonic Boltzmann factors\n')
        for key in thermo_data:
            if not hasattr(thermo_data[key], "qh_gibbs_free_energy"):
                pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
                sys.exit(pes_error)
            if not hasattr(thermo_data[key], "sp_energy") and options.spc is not False:
                pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
                sys.exit(pes_error)
        # Interval applied to PES
        if options.temperature_interval:
            stars = stars + '*' * 22
            interval_thermo_data = []
            for i in range(len(interval)):
                bbe_vals = []
                for j in range(len(interval_bbe_data)):
                    bbe_vals.append(interval_bbe_data[j][i])
                interval_thermo_data.append(dict(zip(file_list, bbe_vals)))
            j = 0
            for i in interval:
                temp = float(i)
                pes = get_pes(options.pes, interval_thermo_data[j], log, temp, options.gconf, options.QH)
                for k, path in enumerate(pes.path):
                    if options.QH:
                        zero_vals = [pes.spc_zero[k][0], pes.e_zero[k][0], pes.zpe_zero[k][0], pes.h_zero[k][0],
                                     pes.qh_zero[k][0], temp * pes.ts_zero[k][0], temp * pes.qhts_zero[k][0],
                                     pes.g_zero[k][0], pes.qhg_zero[k][0]]
                    else:
                        zero_vals = [pes.spc_zero[k][0], pes.e_zero[k][0], pes.zpe_zero[k][0], pes.h_zero[k][0],
                                     temp * pes.ts_zero[k][0], temp * pes.qhts_zero[k][0], pes.g_zero[k][0],
                                     pes.qhg_zero[k][0]]
                    if pes.boltz:
                        e_sum, h_sum, g_sum, qhg_sum = 0.0, 0.0, 0.0, 0.0
                        sels = []
                        for m, e_abs in enumerate(pes.e_abs[k]):
                            if options.QH:
                                species = [pes.spc_abs[k][m], pes.e_abs[k][m], pes.zpe_abs[k][m], pes.h_abs[k][m],
                                           pes.qh_abs[k][m], temp * pes.s_abs[k][m], temp * pes.qs_abs[k][m],
                                           pes.g_abs[k][m], pes.qhg_abs[k][m]]
                            else:
                                species = [pes.spc_abs[k][m], pes.e_abs[k][m], pes.zpe_abs[k][m], pes.h_abs[k][m],
                                           temp * pes.s_abs[k][m], temp * pes.qs_abs[k][m], pes.g_abs[k][m],
                                           pes.qhg_abs[k][m]]
                            relative = [species[x] - zero_vals[x] for x in range(len(zero_vals))]
                            e_sum += math.exp(-relative[1] * J_TO_AU / GAS_CONSTANT / temp)
                            h_sum += math.exp(-relative[3] * J_TO_AU / GAS_CONSTANT / temp)
                            g_sum += math.exp(-relative[7] * J_TO_AU / GAS_CONSTANT / temp)
                            qhg_sum += math.exp(-relative[8] * J_TO_AU / GAS_CONSTANT / temp)
                    if options.spc is False:
                        log.write("\n   " + '{:<40}'.format("RXN: " + path + " (" + pes.units + ")  at T: " + str(temp)))
                        if options.QH:
                            log.write('{:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} '
                                      '{:>13}'.format(" DE", "DZPE", "DH", "qh-DH", "T.DS", "T.qh-DS", "DG(T)",
                                                      "qh-DG(T)"), thermodata=True)
                        else:
                            log.write('{:>13} {:>10} {:>13} {:>10} {:>10} {:>13} '
                                      '{:>13}'.format(" DE", "DZPE", "DH", "T.DS", "T.qh-DS", "DG(T)", "qh-DG(T)"),
                                      thermodata=True)
                    else:
                        log.write("\n   " + '{:<40}'.format("RXN: " + path + " (" + pes.units + ")  at T: " +
                                                            str(temp)))
                        if options.QH:
                            log.write('{:>13} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>14} '
                                      '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "qh-DH_SPC", "T.DS",
                                                      "T.qh-DS", "DG(T)_SPC", "qh-DG(T)_SPC"), thermodata=True)
                        else:
                            log.write('{:>13} {:>13} {:>10} {:>13} {:>10} {:>10} {:>14} '
                                      '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "T.DS", "T.qh-DS",
                                                      "DG(T)_SPC", "qh-DG(T)_SPC"), thermodata=True)
                    log.write("\n" + stars)

                    for m, e_abs in enumerate(pes.e_abs[k]):
                        if options.QH:
                            species = [pes.spc_abs[k][m], pes.e_abs[k][m], pes.zpe_abs[k][m], pes.h_abs[k][m],
                                       pes.qh_abs[k][m], temp * pes.s_abs[k][m], temp * pes.qs_abs[k][m],
                                       pes.g_abs[k][m], pes.qhg_abs[k][m]]
                        else:
                            species = [pes.spc_abs[k][m], pes.e_abs[k][m], pes.zpe_abs[k][m], pes.h_abs[k][m],
                                       temp * pes.s_abs[k][m], temp * pes.qs_abs[k][m], pes.g_abs[k][m],
                                       pes.qhg_abs[k][m]]
                        relative = [species[x] - zero_vals[x] for x in range(len(zero_vals))]
                        if pes.units == 'kJ/mol':
                            formatted_list = [J_TO_AU / 1000.0 * x for x in relative]
                        else:
                            formatted_list = [KCAL_TO_AU * x for x in relative]  # Defaults to kcal/mol
                        log.write("\no  ")
                        if options.spc is False:
                            formatted_list = formatted_list[1:]
                            if options.QH:
                                format_1 = '{:<39} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} {:13.1f}'
                                format_2 = '{:<39} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} {:13.2f}'
                            else:
                                format_1 = '{:<39} {:13.1f} {:10.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} {:13.1f}'
                                format_2 = '{:<39} {:13.2f} {:10.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} {:13.2f}'
                            if pes.dec == 1:
                                log.write(format_1.format(pes.species[k][m], *formatted_list), thermodata=True)
                            if pes.dec == 2:
                                log.write(format_2.format(pes.species[k][m], *formatted_list), thermodata=True)
                        else:
                            if options.QH:
                                if pes.dec == 1:
                                    log.write('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} {:13.1f}'.format(
                                            pes.species[k][m], *formatted_list), thermodata=True)
                                if pes.dec == 2:
                                    log.write('{:<39} {:13.1f} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} {:13.2f}'.format(
                                            pes.species[k][m], *formatted_list), thermodata=True)
                            else:
                                if pes.dec == 1:
                                    log.write('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} {:13.1f}'.format(
                                            pes.species[k][m], *formatted_list), thermodata=True)
                                if pes.dec == 2:
                                    log.write('{:<39} {:13.2f} {:13.2f} {:10.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} {:13.2f}'.format(
                                            pes.species[k][m], *formatted_list), thermodata=True)
                        if pes.boltz:
                            boltz = [math.exp(-relative[1] * J_TO_AU / GAS_CONSTANT / options.temperature) / e_sum,
                                     math.exp(-relative[3] * J_TO_AU / GAS_CONSTANT / options.temperature) / h_sum,
                                     math.exp(-relative[6] * J_TO_AU / GAS_CONSTANT / options.temperature) / g_sum,
                                     math.exp(-relative[7] * J_TO_AU / GAS_CONSTANT / options.temperature) / qhg_sum]
                            selectivity = [boltz[x] * 100.0 for x in range(len(boltz))]
                            log.write("\n  " + '{:<39} {:13.2f}%{:24.2f}%{:35.2f}%{:13.2f}%'.format('', *selectivity))
                            sels.append(selectivity)
                        formatted_list = [round(formatted_list[x], 6) for x in range(len(formatted_list))]
                    if pes.boltz == 'ee' and len(sels) == 2:
                        ee = [sels[0][x] - sels[1][x] for x in range(len(sels[0]))]
                        if options.spc is False:
                            log.write("\n" + stars + "\n   " + '{:<39} {:13.1f}%{:24.1f}%{:35.1f}%{:13.1f}%'.format('ee (%)',
                                                                                                              *ee))
                        else:
                            log.write("\n" + stars + "\n   " + '{:<39} {:27.1f} {:24.1f} {:35.1f} {:13.1f} '.format('ee (%)',
                                                                                                              *ee))
                    log.write("\n" + stars + "\n")
                j += 1
        else:
            pes = get_pes(options.pes, thermo_data, log, options.temperature, options.gconf, options.QH)
            # Output the relative energy data
            for i, path in enumerate(pes.path):
                if options.QH:
                    zero_vals = [pes.spc_zero[i][0], pes.e_zero[i][0], pes.zpe_zero[i][0], pes.h_zero[i][0],
                                 pes.qh_zero[i][0], options.temperature * pes.ts_zero[i][0],
                                 options.temperature * pes.qhts_zero[i][0], pes.g_zero[i][0], pes.qhg_zero[i][0]]
                else:
                    zero_vals = [pes.spc_zero[i][0], pes.e_zero[i][0], pes.zpe_zero[i][0], pes.h_zero[i][0],
                                 options.temperature * pes.ts_zero[i][0], options.temperature * pes.qhts_zero[i][0],
                                 pes.g_zero[i][0], pes.qhg_zero[i][0]]
                if pes.boltz:
                    e_sum, h_sum, g_sum, qhg_sum = 0.0, 0.0, 0.0, 0.0
                    sels = []
                    for j, e_abs in enumerate(pes.e_abs[i]):
                        if options.QH:
                            species = [pes.spc_abs[i][j], pes.e_abs[i][j], pes.zpe_abs[i][j], pes.h_abs[i][j],
                                       pes.qh_abs[i][j], options.temperature * pes.s_abs[i][j],
                                       options.temperature * pes.qs_abs[i][j], pes.g_abs[i][j], pes.qhg_abs[i][j]]
                        else:
                            species = [pes.spc_abs[i][j], pes.e_abs[i][j], pes.zpe_abs[i][j], pes.h_abs[i][j],
                                       options.temperature * pes.s_abs[i][j], options.temperature * pes.qs_abs[i][j],
                                       pes.g_abs[i][j], pes.qhg_abs[i][j]]
                        relative = [species[x] - zero_vals[x] for x in range(len(zero_vals))]
                        e_sum += math.exp(-relative[1] * J_TO_AU / GAS_CONSTANT / options.temperature)
                        h_sum += math.exp(-relative[3] * J_TO_AU / GAS_CONSTANT / options.temperature)
                        g_sum += math.exp(-relative[7] * J_TO_AU / GAS_CONSTANT / options.temperature)
                        qhg_sum += math.exp(-relative[8] * J_TO_AU / GAS_CONSTANT / options.temperature)

                if options.spc is False:
                    log.write("\n   " + '{:<40}'.format("RXN: " + path + " (" + pes.units + ") ", ))
                    if options.QH:
                        log.write('{:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} '
                                  '{:>13}'.format(" DE", "DZPE", "DH", "qh-DH", "T.DS", "T.qh-DS", "DG(T)", "qh-DG(T)"),
                                  thermodata=True)
                    else:
                        log.write('{:>13} {:>10} {:>13} {:>10} {:>10} {:>13} '
                                  '{:>13}'.format(" DE", "DZPE", "DH", "T.DS", "T.qh-DS", "DG(T)", "qh-DG(T)"),
                                  thermodata=True)
                else:
                    log.write("\n   " + '{:<40}'.format("RXN: " + path + " (" + pes.units + ") ", ))
                    if options.QH:
                        log.write('{:>13} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>14} '
                                  '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "qh-DH_SPC", "T.DS", "T.qh-DS",
                                                  "DG(T)_SPC", "qh-DG(T)_SPC"), thermodata=True)
                    else:
                        log.write('{:>13} {:>13} {:>10} {:>13} {:>10} {:>10} {:>14} '
                                  '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "T.DS", "T.qh-DS", "DG(T)_SPC",
                                                  "qh-DG(T)_SPC"), thermodata=True)
                log.write("\n" + stars)

                for j, e_abs in enumerate(pes.e_abs[i]):
                    if options.QH:
                        species = [pes.spc_abs[i][j], pes.e_abs[i][j], pes.zpe_abs[i][j], pes.h_abs[i][j],
                                   pes.qh_abs[i][j], options.temperature * pes.s_abs[i][j],
                                   options.temperature * pes.qs_abs[i][j], pes.g_abs[i][j], pes.qhg_abs[i][j]]
                    else:
                        species = [pes.spc_abs[i][j], pes.e_abs[i][j], pes.zpe_abs[i][j], pes.h_abs[i][j],
                                   options.temperature * pes.s_abs[i][j], options.temperature * pes.qs_abs[i][j],
                                   pes.g_abs[i][j], pes.qhg_abs[i][j]]
                    relative = [species[x] - zero_vals[x] for x in range(len(zero_vals))]
                    if pes.units == 'kJ/mol':
                        formatted_list = [J_TO_AU / 1000.0 * x for x in relative]
                    else:
                        formatted_list = [KCAL_TO_AU * x for x in relative]  # Defaults to kcal/mol
                    log.write("\no  ")
                    if options.spc is False:
                        formatted_list = formatted_list[1:]
                        if options.QH:
                            if pes.dec == 1:
                                log.write('{:<39} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} '
                                          '{:13.1f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                            if pes.dec == 2:
                                log.write('{:<39} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} '
                                          '{:13.2f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                        else:
                            if pes.dec == 1:
                                log.write('{:<39} {:13.1f} {:10.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} '
                                          '{:13.1f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                            if pes.dec == 2:
                                log.write('{:<39} {:13.2f} {:10.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} '
                                          '{:13.2f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                    else:
                        if options.QH:
                            if pes.dec == 1:
                                log.write('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} '
                                          '{:13.1f} {:13.1f}'.format(pes.species[i][j], *formatted_list),
                                          thermodata=True)
                            if pes.dec == 2:
                                log.write('{:<39} {:13.1f} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} '
                                          '{:13.2f} {:13.2f}'.format(pes.species[i][j], *formatted_list),
                                          thermodata=True)
                        else:
                            if pes.dec == 1:
                                log.write('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} '
                                          '{:13.1f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                            if pes.dec == 2:
                                log.write('{:<39} {:13.2f} {:13.2f} {:10.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} '
                                          '{:13.2f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                    if pes.boltz:
                        boltz = [math.exp(-relative[1] * J_TO_AU / GAS_CONSTANT / options.temperature) / e_sum,
                                 math.exp(-relative[3] * J_TO_AU / GAS_CONSTANT / options.temperature) / h_sum,
                                 math.exp(-relative[6] * J_TO_AU / GAS_CONSTANT / options.temperature) / g_sum,
                                 math.exp(-relative[7] * J_TO_AU / GAS_CONSTANT / options.temperature) / qhg_sum]
                        selectivity = [boltz[x] * 100.0 for x in range(len(boltz))]
                        log.write("\n  " + '{:<39} {:13.2f}%{:24.2f}%{:35.2f}%{:13.2f}%'.format('', *selectivity))
                        sels.append(selectivity)
                    formatted_list = [round(formatted_list[x], 6) for x in range(len(formatted_list))]
                if pes.boltz == 'ee' and len(sels) == 2:
                    ee = [sels[0][x] - sels[1][x] for x in range(len(sels[0]))]
                    if options.spc is False:
                        log.write("\n" + stars + "\n   " + '{:<39} {:13.1f}%{:24.1f}%{:35.1f}%{:13.1f}%'.format('ee (%)', *ee))
                    else:
                        log.write("\n" + stars + "\n   " + '{:<39} {:27.1f} {:24.1f} {:35.1f} {:13.1f} '.format('ee (%)', *ee))
                log.write("\n" + stars + "\n")

    # Compute enantiomeric excess
    if options.ee is not False:
        selec_stars = "   " + '*' * 109
        boltz_facs, weighted_free_energy, boltz_sum = get_boltz(files, thermo_data, clustering, clusters,
                                                                options.temperature, dup_list)
        ee, er, ratio, dd_free_energy, failed, preference = get_selectivity(options.ee, files, boltz_facs, boltz_sum,
                                                                            options.temperature, log, dup_list)
        if not failed:
            log.write("\n   " + '{:<39} {:>13} {:>13} {:>13} {:>13} {:>13}'.format("Selectivity", "Excess (%)", "Ratio (%)", "Ratio", "Major Iso", "ddG"), thermodata=True)
            log.write("\n" + selec_stars)
            log.write('\no {:<40} {:13.2f} {:>13} {:>13} {:>13} {:13.2f}'.format('', ee, er, ratio, preference,
                                                                                 dd_free_energy), thermodata=True)
            log.write("\n" + selec_stars + "\n")
    # Graph reaction profiles
    if options.graph is not False:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            log.write("\n\n   Warning! matplotlib module is not installed, reaction profile will not be graphed.")
            log.write("\n   To install matplotlib, run the following commands: \n\t   python -m pip install -U pip" +
                      "\n\t   python -m pip install -U matplotlib\n\n")
        for key in thermo_data:
            if not hasattr(thermo_data[key], "qh_gibbs_free_energy"):
                pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
                sys.exit(pes_error)
            if not hasattr(thermo_data[key], "sp_energy") and options.spc is not False:
                pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
                sys.exit(pes_error)

        graph_data = get_pes(options.graph, thermo_data, log, options.temperature, options.gconf, options.QH)
        graph_reaction_profile(graph_data, log, options, plt)


def main():
    """CLI entry point: parse arguments, compute thermochemistry, and print results."""
    options, files, command, clustering, clusters = parse_arguments()

    # Set up stars separator based on QH option
    if options.QH:
        stars = "   " + "*" * 142
    else:
        stars = "   " + "*" * 128

    # If necessary, create an xyz file for Cartesians
    xyz = None
    if options.xyz:
        xyz = xyz_out("Goodvibes", "xyz", "output")

    # Start logger
    log = Logger("GoodVibes", options.output, options.csv)

    # Print banner
    start = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    log.write("   GoodVibes v" + __version__ + " " + start + "\n   Citation: " + goodvibes_ref + "\n")

    # Check if user has specified any files
    if len(files) == 0:
        sys.exit("\nPlease provide GoodVibes with calculation output files on the command line.\n"
                 "For help, use option '-h'\n")
    if clustering:
        command += '(clustering active)'
    log.write('\n' + command + '\n\n')
    if options.temperature_interval is False:
        log.write("   Temperature = " + str(options.temperature) + " Kelvin")

    # Concentration / pressure
    if options.conc:
        gas_phase = False
        log.write("   Concentration = " + str(options.conc) + " mol/L")
    else:
        gas_phase = True
        options.conc = ATMOS / (GAS_CONSTANT * options.temperature)
        log.write("   Pressure = 1 atm")
    log.write('\n   All energetic values below shown in Hartree unless otherwise specified.')

    # Collect file data and validate
    files, l_o_t, s_m, orientation, grid = collect_and_validate_files(files, options, log)

    # Resolve frequency scaling factor
    resolve_scaling_factor(files, options, l_o_t, log)

    # Validate options and configure
    ssymm_option, vmm_option = validate_and_configure(options, s_m, log)

    # Compute thermochemistry for all files
    thermo_data, bbe_vals, media_conc = compute_thermochemistry(files, options, ssymm_option, vmm_option, log)

    interval_bbe_data, interval, file_list = None, None, None
    dup_list = []

    # Standard mode: single temperature
    if options.temperature_interval is False:
        stars, dup_list, total_cpu_time, add_days = print_results(
            files, thermo_data, options, log, stars, clustering, clusters, xyz=xyz, media_conc=media_conc)

        # Perform checks for consistent options
        if options.check:
            check_files(log, files, thermo_data, options, stars, l_o_t, s_m, orientation, grid)

    # Variable temperature analysis
    elif options.temperature_interval:
        interval_bbe_data, interval, file_list = print_temperature_interval(
            files, options, log, stars, gas_phase, media_conc=media_conc)
        total_cpu_time, add_days = datetime(100, 1, 1, 0, 0, 0, 0), 0

    # Print CPU usage if requested
    if options.cputime:
        log.write('   {:<13} {:>2} {:>4} {:>2} {:>3} {:>2} {:>4} {:>2} '
                  '{:>4}\n'.format('TOTAL CPU', total_cpu_time.day + add_days - 1, 'days', total_cpu_time.hour, 'hrs',
                                   total_cpu_time.minute, 'mins', total_cpu_time.second, 'secs'))

    # Tabulate relative values (PES)
    if options.pes:
        print_pes_results(files, thermo_data, options, log, stars, clustering, clusters, dup_list,
                          interval_bbe_data=interval_bbe_data, interval=interval, file_list=file_list)

    # Close the log
    log.finalize()
    if xyz:
        xyz.finalize()


if __name__ == "__main__":
    main()
