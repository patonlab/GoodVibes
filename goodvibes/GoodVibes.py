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


#######################################################################
###########  Authors:     Rob Paton, Guilian Luchini,       ###########
###########               Juan V. Alegre-Requena,           ###########
###########               Ignacio Funes-Ardoiz              ###########
###########  Last modified:  July 5, 2022                 ###########
####################################################################"""

import math, os.path, sys, time, json, cclib, fnmatch
from datetime import datetime, timedelta
from glob import glob
from argparse import ArgumentParser
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("\n\n   Warning! matplotlib module is not installed, reaction profile will not be graphed.")
    print("\n   To install matplotlib, run the following commands: \n\t   python -m pip install -U pip" +
              "\n\t   python -m pip install -U matplotlib\n\n")

# Importing regardless of relative import
from goodvibes.vib_scale_factors import scaling_data_dict, scaling_data_dict_mod, scaling_refs
from goodvibes.media import solvents
import goodvibes.pes as pes
import goodvibes.io as io
import goodvibes.thermo as thermo

SUPPORTED_EXTENSIONS = set(('.out', '.log', '.json'))

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


def get_vib_scale_factor(files, level_of_theory, log, freq_scale_factor=False):
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

    if freq_scale_factor is False:
        freq_scale_factor = 1.0  # If no scaling factor is found use 1.0
        if all_same(level_of_theory):
            log.write("\n   Using vibrational scale factor {} for {} level of "
                      "theory".format(freq_scale_factor, level_of_theory[0]))
        else:
            log.write("\n   Using vibrational scale factor {}: differing levels of theory "
                      "detected.".format(freq_scale_factor))

    return freq_scale_factor


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


def get_boltz(thermo_data, clustering, clusters, temperature):
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
    return boltz_facs, weighted_free_energy, boltz_total


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


def get_json_data(file,cclib_data):
    '''
    Get metadata and GoodVibes data for the json file (for older versions of cclib)
    '''
    
    outfile = open(file, "r")
    outlines = outfile.readlines()
    outfile.close()

    # initial loop just to detect the QM program
    for i,line in enumerate(outlines):
        # get program
        if line.strip() == "Cite this work as:":
            cclib_data['metadata'] = {}
            qm_program = outlines[i+1]

            cclib_data['metadata']['QM program'] = qm_program[1:-2]
            for j in range(i,i+60):
                if '**********' in outlines[j]:
                    run_date = outlines[j+2].strip()
                    cclib_data['metadata']['run date'] = run_date
                    break
            break

        elif '* O   R   C   A *' in line:
            for j in range(i,i+100):
                if 'Program Version' in line.strip():
                    cclib_data['metadata'] = {}
                    version_program = "ORCA version " + line.split()[2]
                    cclib_data['metadata']['QM program'] = version_program
                    break

        elif "NWChem" in line:
            if 'nwchem branch' in line.strip():
                cclib_data['metadata'] = {}
                cclib_data['metadata']['QM program'] = "NWChem version " + line.split()[3]
                break

    if cclib_data['metadata']['QM program'].lower().find('gaussian') > -1:

        cclib_data['properties']['rotational'] = {}
        for i,line in enumerate(outlines):
            # Extract memory
            if '%mem' in line:
                mem = line.strip().split('=')[-1]
                cclib_data['metadata']['memory'] = mem

            # Extract number of processors
            elif '%nprocs' in line:
                nprocs = int(line.strip().split('=')[-1])
                cclib_data['metadata']['processors'] = nprocs

            # Extract keywords line, solvation, dispersion and calculation type
            elif '#' in line and not hasattr(cclib_data, 'keywords_line'):
                keywords_line = ''
                for j in range(i,i+10):
                    if '----------' in outlines[j]:
                        break
                    else:
                        keywords_line += outlines[j].rstrip("\n")[1:]
                cclib_data['metadata']['keywords line'] = keywords_line[2:]
                qm_solv,qm_disp = 'gas_phase','none'
                calc_type = 'ground_state'
                calcfc_found, ts_found = False, False
                for keyword in keywords_line.split():
                    if keyword.lower().find('opt') > -1:
                        if keyword.lower().find('calcfc') > -1:
                            calcfc_found = True
                        if keyword.lower().find('ts') > -1:
                            ts_found = True
                    elif keyword.lower().startswith('scrf'):
                        qm_solv = keyword
                    elif keyword.lower().startswith('emp'):
                        qm_disp = keyword
                if calcfc_found and ts_found:
                    calc_type = 'transition_state'
                cclib_data['metadata']['solvation'] = qm_solv
                cclib_data['metadata']['dispersion model'] = qm_disp
                cclib_data['metadata']['ground or transition state'] = calc_type

            # Basis set name
            elif line[1:15] == "Standard basis":
                cclib_data['metadata']['basis set'] = line.split()[2]
            elif  "General basis read from cards" in line.strip():
                cclib_data['metadata']['basis set'] = 'User-Specified General Basis'

            # functional
            if not hasattr(cclib_data, 'BOMD') and line[1:9] == 'SCF Done':
                t1 = line.split()[2]
                if t1 == 'E(RHF)':
                    cclib_data['metadata']['functional'] = 'HF'
                else:
                    cclib_data['metadata']['functional'] = t1[t1.index("(") + 2:t1.rindex(")")]
                break

        for i in reversed(range(0,len(outlines)-50)):
            # Grab molecular mass
            if 'Molecular mass:' in outlines[i]:
                cclib_data['properties']['molecular mass'] = float(outlines[i].strip().split()[2])
            # Extract <S**2> before and after spin annihilation
            if 'S**2 before annihilation' in outlines[i]:
                cclib_data['properties']['S2 after annihilation'] = float(outlines[i].strip().split()[-1])
                cclib_data['properties']['S2 before annihilation'] = float(outlines[i].strip().split()[-3][:-1])
            # Extract symmetry point group
            elif 'Full point group' in outlines[i]:
                cclib_data['properties']['rotational']['symmetry point group'] = outlines[i].strip().split()[3]
                break
            # For time dependent (TD) calculations
            elif 'E(TD-HF/TD-DFT)' in outlines[i]:
                td_e = float(line.strip().split()[-1])
                cclib_data['properties']['energy']['TD energy'] = cclib.parser.utils.convertor(td_e, "hartree", "eV")
            # For G4 calculations look for G4 energies (Gaussian16a bug prints G4(0 K) as DE(HF)) --Brian modified to work for G16c-where bug is fixed.
            elif line.strip().startswith('E(ZPE)='): #Overwrite DFT ZPE with G4 ZPE
                zero_point_corr = float(line.strip().split()[1])
            elif line.strip().startswith('G4(0 K)'):
                G4_energy = float(line.strip().split()[2])
                G4_energy -= zero_point_corr #Remove G4 ZPE
                cclib_data['properties']['energy']['G4 energy'] = cclib.parser.utils.convertor(G4_energy, "hartree", "eV")
            # For ONIOM calculations use the extrapolated value rather than SCF value
            elif "ONIOM: extrapolated energy" in line.strip():
                oniom_e = float(line.strip().split()[4])
                cclib_data['properties']['energy']['ONIOM energy'] = cclib.parser.utils.convertor(oniom_e, "hartree", "eV")
            # Extract symmetry number, rotational constants and rotational temperatures
            elif 'Rotational symmetry number' in outlines[i]:
                cclib_data['properties']['rotational']['symmetry number'] = int(outlines[i].strip().split()[3].split(".")[0])
                
            elif outlines[i].find('Rotational constants (GHZ):') > -1:
                try:
                    roconst = [float(outlines[i].strip().replace(':', ' ').split()[3]),
                                    float(outlines[i].strip().replace(':', ' ').split()[4]),
                                    float(outlines[i].strip().replace(':', ' ').split()[5])]
                except ValueError:
                    if outlines[i].find('********') > -1:
                        roconst = [float(outlines[i].strip().replace(':', ' ').split()[4]),
                                        float(outlines[i].strip().replace(':', ' ').split()[5])]
                cclib_data['properties']['rotational']['rotational constants'] = roconst

            elif outlines[i].find('Rotational temperature ') > -1:
                rotemp = [float(outlines[i].strip().split()[3])]
                cclib_data['properties']['rotational']['rotational temperatures'] = rotemp

            elif outlines[i].find('Rotational temperatures') > -1:
                try:
                    rotemp = [float(outlines[i].strip().split()[3]), float(outlines[i].strip().split()[4]),
                                float(outlines[i].strip().split()[5])]
                except ValueError:
                    if outlines[i].find('********') > -1:
                        rotemp = [float(outlines[i].strip().split()[4]), float(outlines[i].strip().split()[5])]
                cclib_data['properties']['rotational']['rotational temperatures'] = rotemp

    elif cclib_data['metadata']['QM program'].lower().find('orca') > -1:
        for i in reversed(range(0,outlines)):
            if outlines[i][:25] == 'FINAL SINGLE POINT ENERGY':
                # in eV to match the format from cclib
                cclib_data['properties']['energy']['final single point energy'] = cclib.parser.utils.convertor(float(outlines[i].split()[-1]), "hartree", "eV")
                break

    elif cclib_data['metadata']['QM program'].lower().find('nwchem') > -1:
        # reversed loop to save time
        # this part misses a break in one of the properties to speed up the loop (i.e. after all the properties are read)
        for i in reversed(range(0,outlines)):
            # Grab rational symmetry number
            if line.strip().find('symmetry #') != -1:
                cclib_data['properties']['rotational']['symmetry number'] = int(line.strip().split()[-1][0:-1])
            # Grab point group
            elif line.strip().find('symmetry detected') != -1:
                cclib_data['properties']['rotational']['symmetry point group'] = line.strip().split()[0]
            # Grab rotational constants (convert cm-1 to GHz)
            elif line.strip().startswith('A=') or line.strip().startswith('B=') or line.strip().startswith('C=') :
                letter=line.strip()[0]
                h = 0
                if letter == 'A':
                    h = 0
                elif letter == 'B':
                    h = 1
                elif letter == 'C':
                    h = 2
                roconst[h]=float(line.strip().split()[1])*29.9792458
                rotemp[h]=float(line.strip().split()[4])

        # this part misses a break in one of the properties to speed up the loop (i.e. after all the properties are read)
        for i in range(0,outlines):
            if line.strip().startswith("xc "):
                cclib_data['metadata']['functional'] = line.strip().split()[1]
            if line.strip().startswith("* library "):
                cclib_data['metadata']['basis set'] = line.strip().replace("* library ",'')

            # need to include tags for NWChem solvation

            if outlines[i].strip().find('disp vdw 3') > -1:
                cclib_data['metadata']['dispersion model'] = "D3"
            if outlines[i].strip().find('disp vdw 4') > -1:
                cclib_data['metadata']['dispersion model'] = "D3BJ"

        if 'dispersion model' not in cclib_data['metadata']:
            cclib_data['metadata']['dispersion model'] = "none"
    
    if cclib_data != {}:
        with open(f'{file.split(".")[0]}.json', 'w') as outfile:
            json.dump(cclib_data, outfile, indent=1)

    return cclib_data


def cclib_init(file_fun,progress_fun,calc_type):
    json_file = f'{file_fun.split(".")[0]}.json'
    # if the corresponding json file exists, read it instead of creating the file again
    if json_file not in glob('*.json'):
        data = cclib.io.ccread(file_fun)
        text = cclib.io.ccwrite(outputtype='json',ccobj=data)
        f = open(json_file, "w")
        f.write(text)
        f.close()

    cclib_data,progress_fun[file_fun] = {},''
    try:
        with open(json_file) as file:
            cclib_data = json.load(file)
    except FileNotFoundError:
        progress_fun[file_fun] = 'Error'

    # add parameters that might be missing from cclib (depends on the cclib version)
    if not hasattr(cclib_data, 'metadata') and 'metadata' not in cclib_data and progress_fun[file_fun] != 'error':
        cclib_data = get_json_data(file_fun,cclib_data)
    
    if calc_type == 'freq':
        # calculations with 1 atom
        if cclib_data['properties']['number of atoms'] == 1:
            cclib_data['vibrations'] = {'frequencies': [], 'displacement': []}
            outfile = open(json_file, "w")
            json.dump(cclib_data, outfile, indent=1)
            outfile.close()

        # other calculations
        if 'vibrations' in cclib_data:
            progress_fun[file_fun] = 'Normal'

        # general errors
        else:
            progress_fun[file_fun] = 'Error'

    elif calc_type == 'spc':
        if 'total' in cclib_data['properties']['energy']:
            progress_fun[file_fun] = 'Normal'
        else:
            progress_fun[file_fun] = 'Error'
    
    return cclib_data,progress_fun


def sort_by_stability(thermo_data, value):
    ''' order the dictionary object of thermochemical data by energy, enthalpy or quasi-harmonic Gibbs energy'''
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
    l_o_t, s_m, progress, spc_progress = [], [], {}, {}
    for i,file in enumerate(files):
        cclib_data,progress = cclib_init(file,progress,'freq')
        level_of_theory = '/'.join([cclib_data['metadata']['functional'] , cclib_data['metadata']['basis set'] ])
        l_o_t.append(level_of_theory)
        s_m.append(cclib_data['metadata']['solvation'])
        #check spc files for normal termination
        if spc is not False:
            if sp_files:
                spc_file = sp_files[i]
            else:
                name, ext = os.path.splitext(file)
                if os.path.exists(name + '_' + spc + '.log'):
                    spc_file = name + '_' + spc + '.log'
                elif os.path.exists(name + '_' + spc + '.out'):
                    spc_file = name + '_' + spc + '.out'

            cclib_data,spc_progress = cclib_init(spc_file,spc_progress,'spc')


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
                log.write("\n\nx  ERROR! Error termination found in file {} calculations.".format(key))
            elif spc_progress[key] == 'Incomplete':
                log.write("\n\nx  ERROR! File {} may not have terminated normally or the "
                    "calculation may still be running.".format(key))

    for [i, key] in list(reversed(remove_key)):
        files.remove(key)
        del l_o_t[i]
        del s_m[i]

    return files, cclib_data, l_o_t, s_m


class GV_options:
    def parse_args(self, argv=None):
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
        parser.add_argument("--freq", dest="freq_cutoff", default=None, type=float, metavar="FREQ_CUTOFF",
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
        parser.add_argument("--spc", dest="spc", type=str, default=False, metavar="SPC",
                            help="Indicates single point corrections (default False)")
        parser.add_argument("--spcdir", dest="spcdir", type=str, default='.', metavar="SPCDIR",
                            help="Directory containing single point corrections (default .)")
        parser.add_argument("--boltz", dest="boltz", action="store_true", default=False,
                            help="Show Boltzmann factors")
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
        parser.add_argument("--freespace", dest="freespace", default=None, type=str, metavar="FREESPACE",
                            help="Solvent (H2O, toluene, DMF, AcOH, chloroform) (default none)")
        parser.add_argument("--dedup", dest="duplicate", action="store_true", default=False,
                            help="Remove duplicate structures from thermochemical analysis")
        parser.add_argument("--sort", dest="sort", action="store", default=False,
                            help="Sort structures by relative stability (E, H or G)")
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
        parser.add_argument("--sel", dest="ee", default=False, type=str,
                            help="Tabulate selectivity values (excess, ratio) from a mixture using regex "
                                 "types such as *_R*,*_S*")
        parser.add_argument("--selplot", dest="selplot", action="store_true", default=False,
                            help="Plot relative energies in selectivity prediction")
        parser.add_argument("--media", dest="media", default=False, metavar="MEDIA",
                            help="Entropy correction for standard concentration of solvents")
        parser.add_argument("--custom_ext", type=str, default='',
                            help="List of additional file extensions to support, beyond .log or .out, use separated by "
                                 "commas (ie, '.qfi, .gaussian'). It can also be specified with environment variable "
                                 "GOODVIBES_CUSTOM_EXT")
        parser.add_argument("--graph", dest='graph', default=False, metavar="GRAPH",
                            help="Graph a reaction profile based on free energies calculated. ")
        parser.add_argument("--nosymm", dest='nosymm', action="store_true", default=False,
                            help="Disable symmetry correction.")
        parser.add_argument("--bav", dest='inertia', default="global",type=str,choices=['global','conf'],
                            help="Choice of how the moment of inertia is computed. Options = 'global' or 'conf'."
                                "'global' will use the same moment of inertia for all input molecules of 10*10-44,"
                                "'conf' will compute moment of inertia from parsed rotational constants from each Gaussian output file.")
        parser.add_argument("--g4", dest="g4", action="store_true", default=False,
                            help="Use this option when using G4 calculations in Gaussian")
        parser.add_argument("--gtype", dest="gtype", action="store", default="G",
                            help="Use this option to request plotting of either relative E, H or G values")
        parser.add_argument("--noStrans", dest="noStrans", action="store_true", default=False,
                            help="Use this option to supress translational entropy")
        parser.add_argument("--noEtrans", dest="noEtrans", action="store_true", default=False,
                            help="Use this option to supress translational energy (affecting enthalpy)")
                        
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

    if len(files) == 0:
        sys.exit("\nNo valid output files specified.\nFor help, use option '-h'\n")

    # Initial read of files
    files, cclib_data, l_o_t, s_m = filter_output_files(files, log, options.spc, sp_files)

    # Check if user has specified any files, if not quit now
    if len(files) == 0:
        sys.exit("\nNo valid output files specified.\nFor help, use option '-h'\n")

    # Loop over all specified output files and compute thermochemistry
    file_list = sorted (files, key = lambda x: ( isinstance (x, str ), x)) #alphanumeric sorting
    bbe_vals = [[]] * len(file_list) # initialize a list that will be populated with thermochemical values

    # scaling vibrational Frequencies
    options.freq_scale_factor =  get_vib_scale_factor(file_list, l_o_t, log, options.freq_scale_factor)

    #set frequency cutoff values if requested
    if options.freq_cutoff:
        options.S_freq_cutoff = options.freq_cutoff
        options.H_freq_cutoff = options.freq_cutoff

    for i, file in enumerate(file_list):

        with open(f'{file.split(".")[0]}.json') as json_file:
            cclib_data = json.load(json_file)

        d3_term = 0.0 # computes D3 term if requested
        cosmo_option = None # computes COSMO term if requested

        bbe_vals[i] = thermo.calc_bbe(file, cclib_data, sp_files[i], options.QS, options.QH, options.S_freq_cutoff, options.H_freq_cutoff, options.temperature,
                       options.conc, options.freq_scale_factor, options.freespace, options.spc, options.invert,
                       d3_correction = d3_term, cosmo = cosmo_option, nosymm = options.nosymm, inertia = options.inertia, 
                       g4 = options.g4, noStrans=options.noStrans, noEtrans=options.noEtrans)

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

    if options.boltz is not False: # Compute Boltzmann factors
        boltz_facs, weighted_free_energy, boltz_sum = get_boltz(thermo_data, options.clustering, clusters, options.temperature)
    else: boltz_facs = None

    # Printing absolute values
    gv_summary = io.summary(thermo_data, options, log, boltz_facs, clusters)

    if options.ee is not False: # Compute selectivity
        [a_name, b_name], [a_files, b_files], ee, er, ratio, dd_free_energy, preference = get_selectivity(options.ee, file_list, boltz_facs, options.temperature, log)
        if options.selplot is not False: pes.sel_striplot(a_name, b_name, a_files, b_files, thermo_data, plt)

    if options.pes: # Tabulate relative values
        species, table = pes.tabulate(thermo_data, options, log, show=True)

    if options.graph is not False: # Graph reaction profiles
        graph_data = pes.get_pes(options.pes, thermo_data, log, options.temperature, options.gconf, options.QH)
        pes.graph_reaction_profile(graph_data, options, log, plt, options.gtype)

    log.finalize()
if __name__ == "__main__":
    main()