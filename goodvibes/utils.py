'''Useful funtions for GoodVibes'''
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import datetime
import math
import sys
import numpy as np
import pymsym

from goodvibes.vib_scale_factors import scaling_data_dict, scaling_data_dict_mod, scaling_refs

# PHYSICAL CONSTANTS & UNITS
GAS_CONSTANT = 8.3144621  # J / K / mol
ATMOS = 101.325  # UNIT CONVERSION
J_TO_AU = 4.184 * 627.509541 * 1000.0  # UNIT CONVERSION
KCAL_TO_AU = 627.509541  # UNIT CONVERSION
PLANCK_CONSTANT = 6.62606957e-34  # J * s
BOLTZMANN_CONSTANT = 1.3806488e-23  # J / K
SPEED_OF_LIGHT = 2.99792458e10  # cm / s
AVOGADRO_CONSTANT = 6.0221415e23  # 1 / mol
AMU_TO_KG = 1.66053886E-27  # UNIT CONVERSION
GHZ_TO_K = 0.0479924341590786 # UNIT CONVERSION
EV_TO_H = 1.0 / 27.21138505 # UNIT CONVERSION

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
         "D8d": 16, "S4": 4, "S6": 6, "S8": 8, "T": 6, "Th": 12, "Td": 12, "O": 12, "Oh": 24, "C*v": 1, "D*h": 2,
         "I": 30, "Ih": 60, "Kh": 1}

def detect_symm(species_list):
    ''' Detects the point group and symmetry number of a species using pymsym.'''
    for species in species_list:
        try:
            species.point_group = pymsym.get_point_group(species.atomnos, species.atomcoords[-1])
            species.symm_no = pymsym.get_symmetry_number(species.atomnos, species.atomcoords[-1])
        except ValueError:
            species.point_group = 'C1'
            species.symm_no = 1

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

def get_vib_scaling(log, model_chemistry, freq_scale_factor, mm_freq_scale_factor=False):
    '''Attempt to automatically obtain frequency scale factor,
    Application of freq scale factors requires all outputs to be same level of theory'''
    # if the user has defined a scaling factor, use that where possible
    if freq_scale_factor is not False:
        if model_chemistry != 'mixed':
            # user manually defines vibrational scaling factor
            if 'ONIOM' not in model_chemistry:
                log.write("\n\n   User-defined vibrational scale factor " + str(freq_scale_factor) + " for " +
                        model_chemistry + " level of theory\n\n")
            else:
                log.write("\n\n   User-defined vibrational scale factor " + str(freq_scale_factor) +
                        " for QM region of " + model_chemistry+"\n\n")
        else:
            freq_scale_factor = 1.0
            log.write("\n\n   Vibrational scale factor " + str(freq_scale_factor) +
                        " applied due to mixed levels of theory\n")

    # Otherwise, try to find a suitable value in the database
    if freq_scale_factor is False:

        for data in (scaling_data_dict, scaling_data_dict_mod):

            if model_chemistry.upper() in data:
                freq_scale_factor = data[model_chemistry.upper()].zpe_fac
                ref = scaling_refs[data[model_chemistry.upper()].zpe_ref]
                log.write(f"\n\no  Found vibrational scaling factor of {freq_scale_factor:.3f} for {model_chemistry} level of theory\n"
                        "   {ref}\n\n")
                break

    if freq_scale_factor is False: # if no scaling factor is found, use 1.0
        freq_scale_factor = 1.0
        log.write("\n\n   Vibrational scale factor " + str(freq_scale_factor) +
                        " applied throughout\n\n")

    # Exit program if molecular mechanics scaling factor is given and all files are not ONIOM calculations
    if mm_freq_scale_factor is not False:
        if 'ONIOM' in model_chemistry:
            log.write("\n\n   User-defined vibrational scale factor " +
                    str(mm_freq_scale_factor) + " for MM region of " + model_chemistry)
            log.write(f"\n   REF: {SIMON_REF}\n")
        else:
            sys.exit("\n\n   Option --vmm is only for use in ONIOM calculation output files.\n   "
                    " help use option '-h'\n")

    return freq_scale_factor

def get_cpu_time(species_list):
    '''Calculate the total CPU time for all calculations'''    
    total_cpu_time = datetime.timedelta(hours=0)

    for species in species_list:
        try:
            for cpu_time in species.metadata['cpu_time']:
                total_cpu_time += cpu_time
        except KeyError:
            pass
    return total_cpu_time

def check_dup(thermo_data, e_cutoff=1e-4, ro_cutoff=0.1):
    """
    Check for duplicate species from among all files based on energy, rotational constants and frequencies
    Defaults
    Energy cutoff = 1 microHartree
    RMS Rotational Constant cutoff = 1kHz
    """

    for i, thermo in enumerate(thermo_data):
        ref_name = thermo.name
        ref_energy = thermo.scf_energy

        try:
            ref_rotconsts = thermo.rotconsts
        except AttributeError:
            ref_rotconsts = None

        for j in range(i+1, len(thermo_data)):
            energy = thermo_data[j].scf_energy

            try:
                rotconsts = thermo_data[j].rotconsts
            except AttributeError:
                rotconsts = None

            e_diff = abs(ref_energy - energy)
            if e_diff < e_cutoff: # only check this if the energies are similar
                if ref_rotconsts is not None and rotconsts is not None and len(ref_rotconsts) == len(rotconsts):
                    ro_diff = np.linalg.norm(np.array(ref_rotconsts) - np.array(rotconsts))
                    if ro_diff < ro_cutoff:
                        #print(f'!  {ref_name} and {name} have similar energies: {e_diff}')
                        #print(f'!  {ref_name} and {name} have similar rotational constants: {ro_diff}')
                        thermo_data[j].duplicate_of = ref_name

    for thermo in thermo_data:
        if hasattr(thermo, 'duplicate_of'):
            print(f'!  {thermo.name} is a duplicate of {thermo.duplicate_of}: removing...')
            thermo_data.remove(thermo)

def sort_conformers(thermo_data, use_gibbs=True):
    '''Sort conformers based on energy or Gibbs free energy.'''
    if use_gibbs:
        thermo_data.sort(key=lambda x: x.qh_gibbs_free_energy if x.qh_gibbs_free_energy is not None else 0)
    else:
        thermo_data.sort(key=lambda x: x.scf_energy if x.scf_energy is not None else 0)

def get_boltz_facs(thermo_data, temperature=298.15, use_gibbs=True):
    """
    Obtain Boltzmann factors, Boltzmann sums, and weighted free energy values.
    The assumption is that duplicates have already been removed!
    Used for selectivity and boltzmann requested options.

    Parameters:
    thermo_data (dict): dict of calc_bbe objects with thermodynamic data to use for Boltzmann averaging.
    temperature (float): temperature to compute Boltzmann populations at
    use_gibbs (bool): use Gibbs free energy instead of energy for Boltzmann averaging.

    Returns:boltz_facs, weighted_free_energy, boltz_sum
    dict: dictionary of files with corresponding Boltzmann factors.
    dict: dictionary of files with corresponding weighted Gibbs free energy.
    float: Boltzmann sum computed from Boltzmann factors and Gibbs free energy.
    """

    glob_min, boltz_sum = 0.0, 0.0

    for thermo in thermo_data:  # Need the most stable structure
        if use_gibbs is True:
            if hasattr(thermo, "qh_gibbs_free_energy"):
                glob_min = min(glob_min, thermo.qh_gibbs_free_energy)
        else:
            if hasattr(thermo, "scf_energy"):
                glob_min = min(glob_min, thermo.scf_energy)

    # Calculate G_rel and Boltzmann factors
    for thermo in thermo_data:
        if use_gibbs is True:
            if hasattr(thermo, "qh_gibbs_free_energy"):
                thermo.g_rel = thermo.qh_gibbs_free_energy - glob_min # in Hartree
                thermo.boltz_fac = math.exp(-thermo.g_rel * J_TO_AU / GAS_CONSTANT / temperature)
                boltz_sum += thermo.boltz_fac
            else:
                thermo.boltz_fac = np.nan

        else:
            if hasattr(thermo, "scf_energy"):
                thermo.e_rel = thermo.scf_energy - glob_min # in Hartree
                thermo.boltz_fac = math.exp(-thermo.e_rel * J_TO_AU / GAS_CONSTANT / temperature)
                boltz_sum += thermo.boltz_fac

    for thermo in thermo_data:
        thermo.boltz_fac = thermo.boltz_fac / boltz_sum

def get_selectivity(pattern, thermo_data, temperature=298.15):
    """
    Calculate selectivity as enantioselectivity/diastereomeric ratio.

    Parameters:
    pattern (str): pattern to recognize for selectivity calculation, i.e. "R":"S".
    thermo_data (str): objects to use for selectivity calculation.
    temperature (float)

    Returns:
    float: enantiomeric/diasteriomeric ratio.
    str: pattern used to identify ratio.
    float: Gibbs free energy barrier.
    bool: flag for failed selectivity calculation.
    str: preferred enantiomer/diastereomer configuration.
    """

    [a_regex,b_regex] = pattern.split(':')
    [a_regex,b_regex] = [a_regex.strip(), b_regex.strip()]

    species_a, species_b = [], []
    for thermo in thermo_data:
        if a_regex in thermo.name:
            species_a.append(thermo)
        if b_regex in thermo.name:
            species_b.append(thermo)

    if len(species_a) + len(species_b) != len(thermo_data):
        print(f'\n\nx  Selectivity pattern {pattern} leads to groups {a_regex} with {len(species_a)} and {b_regex} with {len(species_b)} species')
        print(f'   However, there are {len(thermo_data)} species to match in total! Try a different pattern...\n')
        sys.exit()

    # Add up Boltzmann Factors for the two groups
    a_sum, b_sum = 0, 0
    for thermo in species_a:
        a_sum += thermo.boltz_fac
    for thermo in species_b:
        b_sum += thermo.boltz_fac

    # Ratio
    a_round = round(a_sum * 100)
    b_round = round(b_sum * 100)
    ratio = str(a_round) + ':' + str(b_round)
    excess = (a_sum - b_sum) / (a_sum + b_sum) * 100.0

    try:
        ddg = GAS_CONSTANT / J_TO_AU * temperature * math.log((50 + abs(excess) / 2.0) / (50 - abs(excess) / 2.0)) * KCAL_TO_AU * np.sign(excess)

    except ZeroDivisionError:
        ddg = 0.0
    return excess, ratio, ddg
