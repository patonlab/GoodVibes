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
###########  Last modified:  May 27, 2021                 ############
####################################################################"""

import cclib, fnmatch, math, os.path, sys, time
from datetime import datetime, timedelta
from glob import glob
from argparse import ArgumentParser
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Importing regardless of relative import
from goodvibes.vib_scale_factors import scaling_data_dict, scaling_data_dict_mod, scaling_refs
from goodvibes.media import solvents
import goodvibes.pes as pes
import goodvibes.io as io
import goodvibes.thermo as thermo

SUPPORTED_EXTENSIONS = set(('.out', '.log'))

# PHYSICAL CONSTANTS                                      UNITS
GAS_CONSTANT = 8.3144621  # J / K / mol
ATMOS = 101.325  # UNIT CONVERSION
J_TO_AU = 4.184 * 627.509541 * 1000.0  # UNIT CONVERSION
KCAL_TO_AU = 627.509541  # UNIT CONVERSION

alphabet = 'abcdefghijklmnopqrstuvwxyz'


def all_same(items):
    """Returns bool for checking if all items in a list are the same."""
    return all(x == items[0] for x in items)


def add_time(tm, cpu):
    """Calculate elapsed time."""
    [days, hrs, mins, secs, msecs] = cpu
    fulldate = datetime(100, 1, tm.day, tm.hour, tm.minute, tm.second, tm.microsecond)
    fulldate = fulldate + timedelta(days=days, hours=hrs, minutes=mins, seconds=secs, microseconds=msecs * 1000)
    return fulldate


def calc_cpu(thermo_data, log):
    # Initialize the total CPU time
    add_days = 0
    cpu = datetime(100, 1, 1, 00, 00, 00, 00)

    for file, bbe in thermo_data.items():
        if hasattr(bbe, "cpu"):
            if bbe.cpu != None:
                cpu = add_time(cpu, bbe.cpu)

    if cpu.month > 1: add_days += 31 * (cpu.month -1)
    else: add_days = 0

    log.write('   {:<13} {:>2} {:>4} {:>2} {:>3} {:>2} {:>4} {:>2} '
              '{:>4}\n'.format('TOTAL CPU', cpu.day + add_days - 1, 'days', cpu.hour, 'hrs',
                               cpu.minute, 'mins', cpu.second, 'secs'))


def get_vib_scale_factor(files, level_of_theory, log, freq_scale_factor=False, mm_freq_scale_factor=False):
    ''' Attempt to automatically obtain frequency scale factor
    Application of freq scale factors requires all outputs to be same level of theory'''

    if freq_scale_factor is not False:
        if 'ONIOM' not in level_of_theory[0]:
            log.write("\n\n   User-defined vibrational scale factor " + str(freq_scale_factor) + " for " +
                      level_of_theory[0] + " level of theory")
        else:
            log.write("\n\n   User-defined vibrational scale factor " + str(freq_scale_factor) +
                      " for QM region")

    else:
        # Look for vibrational scaling factor automatically
        if all_same(level_of_theory):
            level = level_of_theory[0].upper()

            for data in (scaling_data_dict, scaling_data_dict_mod):
                if level in data:

                    freq_scale_factor = data[level].zpe_fac
                    ref = scaling_refs[data[level].zpe_ref]
                    log.write("\n\no  Found vibrational scaling factor of {:.3f} for {} level of theory\n"
                              "   {}".format(freq_scale_factor, level_of_theory[0], ref))
                    break
        else:  # Print files and different levels of theory found
            files_l_o_t, levels_l_o_t, filtered_calcs_l_o_t = [], [], []
            for file in files:
                files_l_o_t.append(file)
            for i in level_of_theory:
                levels_l_o_t.append(i)
            filtered_calcs_l_o_t.append(files_l_o_t)
            filtered_calcs_l_o_t.append(levels_l_o_t)
            #print(filtered_calcs_l_o_t)
            #io.print_check_fails(log, filtered_calcs_l_o_t[1], filtered_calcs_l_o_t[0], "levels of theory")

    # Exit program if molecular mechanics scaling factor is given and all files are not ONIOM calculations
    if mm_freq_scale_factor is not False:
        if all_same(l_o_t) and 'ONIOM' in l_o_t[0]:
            log.write("\n\no  User-defined vibrational scale factor " +
                      str(mm_freq_scale_factor) + " for MM region of " + l_o_t[0])
            log.write("\n   REF: {}".format(oniom_scale_ref))
        else:
            sys.exit("\n   Option --vmm is only for use in ONIOM calculation output files.\n   "
                     " help use option '-h'\n")

    if freq_scale_factor is False:
        freq_scale_factor = 1.0  # If no scaling factor is found use 1.0
        if all_same(level_of_theory):
            log.write("\n   Using vibrational scale factor {} for {} level of "
                      "theory".format(freq_scale_factor, level_of_theory[0]))
        else:
            log.write("\n   Using vibrational scale factor {}: differing levels of theory "
                      "detected.".format(freq_scale_factor))

    return freq_scale_factor, mm_freq_scale_factor


def get_selectivity(pattern, files, boltz_facs, temperature, log):
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
    a_files, b_files, a_sum, b_sum, failed, pref = [], [], 0.0, 0.0, False, ''

    try:
        [a_regex,b_regex] = pattern.split(':')
        [a_regex,b_regex] = [a_regex.strip(), b_regex.strip()]

        A = ''.join(a for a in a_regex if a.isalnum())
        B = ''.join(b for b in b_regex if b.isalnum())

        a_files = fnmatch.filter(files, a_regex)
        b_files = fnmatch.filter(files, b_regex)

    except: pass

    if len(a_files) == 0 or len(b_files) == 0:
        log.write("\n   Warning! Filenames have not been formatted correctly for determining selectivity\n")
        log.write("   Make sure the filename contains either " + A + " or " + B + "\n")
        sys.exit("   Please edit either your filenames or selectivity pattern argument and try again\n")

    # Grab Boltzmann sums
    for file in files:
        if file in a_files:
            a_sum += boltz_facs[file]
        elif file in b_files:
            b_sum += boltz_facs[file]

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
        except ZeroDivisionError: ratio = '1:0'
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
    if ee > 99.99: ee = 99.99

    try:
        dd_free_energy = GAS_CONSTANT / J_TO_AU * temperature * math.log((50 + abs(ee) / 2.0) / (50 - abs(ee) / 2.0)) * KCAL_TO_AU
    except ZeroDivisionError:
        dd_free_energy = 0.0

    if not failed:
        selec_stars = "   " + '*' * 109
        log.write("\n   " + '{:<39} {:>13} {:>13} {:>13} {:>13} {:>13}'.format("Selectivity", "Excess (%)", "Ratio (%)", "Ratio", "Major", "DDG kcal/mol"), thermodata=True)
        log.write("\n" + selec_stars)
        log.write('\no {:<40} {:13.2f} {:>13} {:>13} {:>13} {:13.2f}'.format('', ee, r, ratio, pref, dd_free_energy), thermodata=True)
        log.write("\n" + selec_stars + "\n")

    return [A, B], [a_files, b_files], ee, r, ratio, dd_free_energy, pref


def get_boltz(thermo_data, clustering, clusters, temperature, log):
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
    boltz_facs, weighted_free_energy, e_rel, e_min = {}, {}, {}, sys.float_info.max

    for file in thermo_data:  # Need the most stable structure
        bbe = thermo_data[file]
        if hasattr(bbe, "qh_gibbs_free_energy"):
            if bbe.qh_gibbs_free_energy != None:
                if bbe.qh_gibbs_free_energy < e_min:
                    e_min = bbe.qh_gibbs_free_energy

    if clustering:
        for n, cluster in enumerate(clusters):
            boltz_facs['cluster-' + alphabet[n].upper()] = 0.0
            weighted_free_energy['cluster-' + alphabet[n].upper()] = 0.0
    # Calculate E_rel and Boltzmann factors
    for file in thermo_data:
        bbe = thermo_data[file]
        if hasattr(bbe, "qh_gibbs_free_energy"):
            if bbe.qh_gibbs_free_energy != None:
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

    # normalize
    boltz_total = sum(boltz_facs.values(), 0.0)
    boltz_facs = {k: v / boltz_total for k, v in boltz_facs.items()}
    return boltz_facs, weighted_free_energy


def check_dup(files, thermo_data, log, e_cutoff = 1e-4, ro_cutoff = 0.1):
    """
    Check for duplicate species from among all files based on energy, rotational constants and frequencies

    Energy cutoff = 1 milliHartree
    RMS Rotational Constant cutoff = 1kHz
    RMS Freq cutoff = 10 wavenumbers
    """
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

    for [dup,original] in dup_list:
        log.write("\n!  {} is a duplicate of {} and is removed from analysis".format(dup, original))
        files.remove(dup)
        del thermo_data[dup]
    return files, thermo_data


def sort_by_stability(thermo_data, value):
    ''' order the dictionary object of thermochemical data by energy, enthalpy or quasi-harmnonic Gibbs energy'''
    try:
        if value == "E": sorted_thermo_data = dict(sorted(thermo_data.items(), key=lambda item: item[1].scf_energy))
        elif value == "H": sorted_thermo_data = dict(sorted(thermo_data.items(), key=lambda item: item[1].enthalpy))
        elif value == "G": sorted_thermo_data = dict(sorted(thermo_data.items(), key=lambda item: item[1].qh_gibbs_free_energy))
        return sorted_thermo_data
    except:
        return thermo_data


def get_output_files(args, spc = False, spcdir = '.', clustering = False, cosmo = False):
    ''' retrieve filenames for analysis'''
    clusters, nclust = [], -1
    files, sp_files = [], []

    for elem in args:

        if clustering:
            if elem == 'clust:':
                clusters.append([]); nclust += 0
        try:
            if os.path.splitext(elem)[1].lower() in SUPPORTED_EXTENSIONS:  # Look for file names

                for file in glob(elem):
                    if spc is False or spc == 'link':
                        sp_files.append(None)
                        if file is not cosmo:
                            files.append(file)
                        if clustering:
                            clusters[nclust].append(file)
                    else:
                        if file.find('_' + spc + ".") == -1:
                            files.append(file)
                            if clustering:
                                clusters[nclust].append(file)
                            name, ext = os.path.splitext(file)
                            spcfile = spcdir + '/' + Path(name).stem + '_' + spc
                            if os.path.exists(spcfile + '.log'): sp_files.append(spcfile+'.log')
                            if os.path.exists(spcfile + '.out'): sp_files.append(spcfile+'.out')
                            if not (os.path.exists(spcfile + '.log') or os.path.exists(spcfile + '.out')) and spc != 'link':
                                sys.exit("\nError! SPC calculation file '{}' not found! Make sure files are named with "
                                         "the convention: 'filename_spc' or specify link job.\nFor help, use option '-h'\n"
                                         "".format(spcfile))
        except IndexError: pass
    return files, sp_files, clusters


def filter_output_files(files, log, spc = False, sp_files = None):
    # Grab level of theory, solvation model, check for Normal Termination
    l_o_t, s_m, progress, spc_progress, orientation, grid = [], [], {}, {}, {}, {}
    for i, file in enumerate(files):
        lot_sm_prog = io.read_initial(file)
        l_o_t.append(lot_sm_prog[0])
        s_m.append(lot_sm_prog[1])
        progress[file] = lot_sm_prog[2]
        orientation[file] = lot_sm_prog[3]
        grid[file] = lot_sm_prog[4]
        #check spc files for normal termination
        if spc is not False and spc != 'link':
            lot_sm_prog = io.read_initial(sp_files[i])
            spc_progress[sp_files[i]] = lot_sm_prog[2]

    remove_key = []
    # Remove problem files and print errors
    for i, key in enumerate(files):
        if progress[key] == 'Error':
            log.write("\nx  Warning! Error termination found in file {}. This file will be omitted from further "
                      "calculations.".format(key))
            remove_key.append([i, key])
        elif progress[key] == 'Incomplete':
            log.write("\nx  Warning! File {} may not have terminated normally or the calculation may still be "
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

    return files, l_o_t, s_m


'''
IDEA!

def cc_parser(file, sp_file=None):
    does the json file exist:

        if yes = read it and create the cclib object (assume that all the data is there becuase we already created it!)

        if no:
            parse the output file to create the cclib object - we also have to augment with some extra stuff, then save as json

            use cclib first
            we need to know what program it is and then try out own gaussianparser

'''

def cc_parser(file, sp_file=None):

    try: data = cclib.io.ccread(file)
    except: data = None

    try: sp_data = cclib.io.ccread(sp_file)
    except: sp_data = None

    ## adding essential ingredients not in standard cclib parse
    outfile = open(file, "r")
    outlines = outfile.readlines()

    level, basis, program, keyword_line, solvation_model = 'none', 'none', 'none', 'none', 'none'
    a, repeated_theory = 0, 0
    remove_key = []
    if data:
        for i,line in enumerate(outlines):
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
            if "Q-Chem, Inc." in line:
                program = "Q-Chem"
                break
        #NWChem and Orca specific parsing
        if program == 'NWChem' or program == 'Orca':
            keyword_line_1 = "gas phase"
            keyword_line_2 = ''
            keyword_line_3 = ''


        for i,line in enumerate(outlines):
            if line.find('Rotational constants (GHZ):') > -1:
                try:
                    roconst = [float(line.strip().replace(':', ' ').split()[3]),
                                    float(line.strip().replace(':', ' ').split()[4]),
                                    float(line.strip().replace(':', ' ').split()[5])]
                except ValueError:
                    if line.find('********') > -1:
                        roconst = [float(line.strip().replace(':', ' ').split()[4]),
                                        float(line.strip().replace(':', ' ').split()[5])]
                data.roconsts = roconst

            if line.find('Rotational temperature ') > -1:
                rotemp = [float(line.strip().split()[3])]
                data.rotemps = rotemp

            if line.find('Rotational temperatures') > -1:
                try:
                    rotemp = [float(line.strip().split()[3]), float(line.strip().split()[4]),
                                float(line.strip().split()[5])]
                except ValueError:
                    if line.find('********') > -1:
                        rotemp = [float(line.strip().split()[4]), float(line.strip().split()[5])]
                data.rotemps = rotemp

            if 'Rotational symmetry number' in line:
                data.symmno = int(line.strip().split()[3].split(".")[0])

            if 'Molecular mass:' in line:
                data.mass = float(line.strip().split()[2])
            
            # Level of Theory and Progress Info
            if line.strip().find('External calculation') > -1:
                level, basis = 'ext', 'ext'
            if '\\Freq\\' in line.strip() and repeated_theory == 0:
                try:
                    level, basis = (line.strip().split("\\")[4:6])
                    repeated_theory = 1
                except IndexError:
                    pass
            elif '|Freq|' in line.strip() and repeated_theory == 0:
                try:
                    level, basis = (line.strip().split("|")[4:6])
                    repeated_theory = 1
                except IndexError:
                    pass
            if '\\SP\\' in line.strip() and repeated_theory == 0:
                try:
                    level, basis = (line.strip().split("\\")[4:6])
                    repeated_theory = 1
                except IndexError:
                    pass
            elif '|SP|' in line.strip() and repeated_theory == 0:
                try:
                    level, basis = (line.strip().split("|")[4:6])
                    repeated_theory = 1
                except IndexError:
                    pass
            if 'DLPNO BASED TRIPLES CORRECTION' in line.strip():
                level = 'DLPNO-CCSD(T)'
            if 'Estimated CBS total energy' in line.strip():
                try:
                    basis = ("Extrapol." + line.strip().split()[4])
                except IndexError:
                    pass
            # Remove the restricted R or unrestricted U label
            if level[0] in ('R', 'U'):
                level = level[1:]

            """ integrate this into ^^ loop"""
            if program == 'NWChem':
                if line.strip().startswith("xc "):
                    level=line.strip().split()[1]
                if line.strip().startswith("* library "):
                    basis = line.strip().replace("* library ",'')
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
            

            if program == 'Q-Chem':
                if 'Total energy in the final basis set' in line.strip(): 
                    progress = 'Normal'
                if 'Error' in line.strip(): 
                    progress = 'Error'
                solvation_model = 'unknown'

            # Grab solvation models - Gaussian files
            if program == 'Gaussian':
                if '#' in line.strip() and a == 0:
                    for j in range(i,i+10):
                        if '--' in outlines[j].strip():
                            a = a + 1
                            break
                        if a != 0:
                            break
                        else:
                            for k in range(len(outlines[j].strip().split("\n"))):
                                keyword_line += outlines[j].strip().split("\n")[k]
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
            if program == 'Orca':
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
        
        level_of_theory = '/'.join([level, basis])
        if program == "NWChem" or program == "Orca": 
            solvation_model = keyword_line_1 + keyword_line_2 + keyword_line_3

        data.l_o_t = level_of_theory
        data.solvation_model = solvation_model
        data.progress = progress
        # if progress == 'Error':
        #     log.write("\nx  Warning! Error termination found in file {}. This file will be omitted from further "
        #               "calculations.".format(key))
        #     remove_key.append([i, key])
        # elif progress[key] == 'Incomplete':
        #     log.write("\nx  Warning! File {} may not have terminated normally or the calculation may still be "
        #               "running. This file will be omitted from further calculations.".format(key))
        #     remove_key.append([i, key])

    return data, sp_data


class GV_options:
    def parse_args(self, argv=None):
        # Get command line inputs. Use -h to list all possible arguments and default values
        parser = ArgumentParser()
        parser.add_argument("-q", dest="Q", action="store_true", default=False,
                            help="Quasi-harmonic entropy correction and enthalpy correction applied (default S=Grimme, "
                                 "H=Head-Gordon)")
        parser.add_argument("--qs", dest="QS", default="grimme", type=str.lower,
                            choices=('grimme', 'truhlar'),
                            help="Type of quasi-harmonic entropy correction (Grimme or Truhlar) (default Grimme)", )
        parser.add_argument("--qh", dest="QH", action="store_true", default=False,
                            help="Type of quasi-harmonic enthalpy correction (Head-Gordon)")
        parser.add_argument("--fcut", dest="freq_cutoff", default=None, type=float,
                            help="Cut-off frequency for both entropy and enthalpy (wavenumbers) (default = 100)", )
        parser.add_argument("--fs", dest="S_freq_cutoff", default=100.0, type=float,
                            help="Cut-off frequency for entropy (wavenumbers) (default = 100)")
        parser.add_argument("--fh", dest="H_freq_cutoff", default=100.0, type=float,
                            help="Cut-off frequency for enthalpy (wavenumbers) (default = 100)")
        parser.add_argument("-t", dest="temperature", default=298.15, type=float,
                            help="Temperature (K) (default 298.15)")
        parser.add_argument("-c", dest="conc", default=False, type=float,
                            help="Concentration (mol/l) (default 1 atm)")
        parser.add_argument("--ti", dest="temperature_interval", default=False,
                            help="Initial temp, final temp, step size (K)")
        parser.add_argument("-v", dest="freq_scale_factor", default=False, type=float,
                            help="Frequency scaling factor. If not set, try to find a suitable value in database. "
                                 "If not found, use 1.0")
        parser.add_argument("--vmm", dest="mm_freq_scale_factor", default=False, type=float,
                            help="Additional frequency scaling factor used in ONIOM calculations")
        parser.add_argument("--spc", dest="spc", type=str, default=False,
                            help="Indicates single point corrections (default False)")
        parser.add_argument("--spcdir", dest="spcdir", type=str, default='.',
                            help="Directory containing single point corrections (default .)")
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
        parser.add_argument("--freespace", dest="freespace", default=None, type=str,
                            help="Solvent (H2O, toluene, DMF, AcOH, chloroform) (default none)")
        parser.add_argument("--dedup", dest="duplicate", action="store_true", default=False,
                            help="Remove duplicate structures from thermochemical analysis")
        parser.add_argument("--sort", dest="sort", action="store", default=False,
                            help="Sort structures by relative stability (E, H or G)")
        parser.add_argument("--cosmo", dest="cosmo", default=False,
                            help="Filename of a COSMO-RS .tab output file")
        parser.add_argument("--cosmo_int", dest="cosmo_int", default=False,
                            help="Filename of a COSMO-RS .tab output file along with a temperature range (K): "
                                 "file.tab,'Initial_T, Final_T'")
        parser.add_argument("--output", dest="output", default="output",
                            help="Change the default name of the output file to GoodVibes_\"output\".dat")
        parser.add_argument("--pes", dest="pes", default=False,
                            help="Tabulate relative values")
        parser.add_argument("--nogconf", dest="gconf", action="store_false", default=True,
                            help="Calculate a free-energy correction related to multi-configurational space (default "
                                 "calculate Gconf)")
        parser.add_argument("--sel", dest="ee", default=False, type=str,
                            help="Tabulate selectivity values (excess, ratio) from a mixture using regex "
                                 "types such as *_R*,*_S*")
        parser.add_argument("--selplot", dest="selplot", action="store_true", default=False,
                            help="Plot relative energies in selectivity prediction")
        parser.add_argument("--check", dest="check", action="store_true", default=False,
                            help="Checks if calculations were done with the same program, level of theory and solvent, "
                                 "as well as detects potential duplicates")
        parser.add_argument("--media", dest="media", default=False,
                            help="Entropy correction for standard concentration of solvents")
        parser.add_argument("--custom_ext", type=str, default='',
                            help="List of additional file extensions to support, beyond .log or .out, use separated by "
                                 "commas (ie, '.qfi, .gaussian'). It can also be specified with environment variable "
                                 "GOODVIBES_CUSTOM_EXT")
        parser.add_argument("--graph", dest='graph', default=False,
                            help="Graph a reaction profile based on free energies calculated. ")
        parser.add_argument("--ssymm", dest='ssymm', action="store_true", default=False,
                            help="Turn on the symmetry correction.")
        parser.add_argument("--bav", dest='inertia', default="global",type=str,choices=['global','conf'],
                            help="Choice of how the moment of inertia is computed. Options = 'global' or 'conf'."
                                "'global' will use the same moment of inertia for all input molecules of 10*10-44,"
                                "'conf' will compute moment of inertia from parsed rotational constants from each Gaussian output file.")
        parser.add_argument("--g4", dest="g4", action="store_true", default=False,
                            help="Use this option when using G4 calculations in Gaussian")
        parser.add_argument("--gtype", dest="gtype", action="store", default="G",
                            help="Use this option to request plotting of either relative E, H or G values")

        # Parse Arguments
        (self.options, self.args) = parser.parse_known_args()

        # If requested, turn on head-gordon enthalpy correction
        if self.options.Q: self.options.QH = True

        # If not at standard temp, need to correct the molarity of 1 atmosphere (assuming pressure is still 1 atm)
        if self.options.conc:
            self.options.gas_phase = False
        else:
            self.options.gas_phase = True
            self.options.conc = thermo.ATMOS / (thermo.GAS_CONSTANT * self.options.temperature)

        self.options.start = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
        self.options.command = 'o  Requested: '

        self.options.clustering = False
        # figure out whether conformer clustering is required
        if argv != None:
            if 'clust:' in argv:
                self.options.clustering = True; self.options.boltz = True
            # Keep track of the manually requested options
            for arg in argv:
                if arg != 'clust:' and os.path.splitext(arg)[1].lower() not in SUPPORTED_EXTENSIONS :
                    self.options.command += arg + ' '
            if self.options.clustering: self.options.command += '(clustering active)'

        if self.options.freq_cutoff:
            self.options.S_freq_cutoff = self.options.freq_cutoff
            self.options.H_freq_cutoff = self.options.freq_cutoff


def main():
    # Fetch default parameters and any specified at the command line
    gv = GV_options()
    gv.parse_args(sys.argv[1:])
    options, args = gv.options, gv.args

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

    # Get the filenames from the command line prompt
    files, sp_files, clusters = get_output_files(sys.argv[1:], options.spc, options.spcdir, options.clustering, options.cosmo)

    # Check if user has specified any files, if not quit now
    if len(files) == 0:
        sys.exit("\nNo valid output files specified.\nFor help, use option '-h'\n")

    # Loop over all specified output files and compute thermochemistry
    file_list = sorted (files, key = lambda x: ( isinstance (x, str ), x)) #alphanumeric sorting
    sp_files = sorted (sp_files, key = lambda x: ( isinstance (x, str ), x)) #alphanumeric sorting
    
    bbe_vals = [[]] * len(file_list) # initialize a list that will be populated with thermochemical values

    # Initial read of files
    ''' IDEA!
    move cclib parsing to here and skip the method below
    would need to remove bad outputs etc
    data, sp_data = cc_parser(file, sp_file)
    '''
    #files, l_o_t, s_m = filter_output_files(files, log, options.spc, sp_files)
    data_list = [(cc_parser(file, sp_file)) for file, sp_file in zip(files, sp_files)]

    l_o_t = [data[0].l_o_t for data in data_list]

    # scaling vibrational Frequencies
    options.freq_scale_factor, options.mm_freq_scale_factor =  get_vib_scale_factor(file_list, l_o_t, log, options.freq_scale_factor, options.mm_freq_scale_factor)

    #for i, (file, sp_file) in enumerate(zip(file_list, sp_files)):
    for i, (data,sp_data) in enumerate(data_list):
        #data, sp_data = cc_parser(file, sp_file)
        file = file_list[i]
        d3_term = 0.0 # computes D3 term if requested
        cosmo_option = None # computes COSMO term if requested

        bbe_vals[i] = thermo.calc_bbe(file, data, sp_data, options.QS, options.QH, options.S_freq_cutoff, options.H_freq_cutoff, options.temperature,
                       options.conc, options.freq_scale_factor, options.freespace, options.spc, options.invert,
                       d3_correction = d3_term, cosmo = cosmo_option, ssymm = options.ssymm, mm_freq_scale_factor = options.mm_freq_scale_factor, inertia = options.inertia)

    # Creates a new dictionary object thermo_data, which attaches the bbe data to each file-name
    thermo_data = dict(zip(file_list, bbe_vals))  # The collected thermochemical data for all files

    # Standard goodvibes analysis (single temperature) requested
    gv_intro = io.intro(options, log)

    if options.sort: # Sort by energy/enthalpy/gibbs energy if requested
        thermo_data = sort_by_stability(thermo_data, options.sort)

    if options.duplicate: # Look for duplicates or enantiomers if requested
        file_list, thermo_data = check_dup(file_list, thermo_data, log)

    if options.xyz or options.sdf: # If necessary, create a file with Cartesians
        io.write_structures("Goodvibes_output", file_list, xyz = options.xyz, sdf = options.sdf)

    if options.check: # Perform checks for consistent options provided in calculation files (level of theory)
        io.check_files(thermo_data, options, log)

    if options.boltz: # Compute Boltzmann factors
        boltz_facs, weighted_free_energy = get_boltz(thermo_data, options.clustering, clusters, options.temperature, log)
    else: boltz_facs = None

    # Printing absolute values
    gv_summary = io.summary(thermo_data, options, log, boltz_facs, clusters)

    if options.cputime: # Print CPU usage if requested
        cpu = calc_cpu(thermo_data, log)

    if options.ee: # Compute selectivity
        [a_name, b_name], [a_files, b_files], ee, er, ratio, dd_free_energy, preference = get_selectivity(options.ee, file_list, boltz_facs, options.temperature, log)
        if options.selplot is not False: pes.sel_striplot(a_name, b_name, a_files, b_files, thermo_data, plt)

    if options.pes: # Tabulate relative values
        species, table = pes.tabulate(thermo_data, options, log, show=True)

    if options.graph: # Graph reaction profiles
        graph_data = pes.get_pes(options.pes, thermo_data, log, options.temperature, options.gconf, options.QH)
        pes.graph_reaction_profile(graph_data, options, log, plt, options.gtype)

    log.finalize()
if __name__ == "__main__":
    main()
