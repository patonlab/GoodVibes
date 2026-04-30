"""Boltzmann weighting and selectivity calculations for GoodVibes."""
import logging
import math
import os.path
import sys
from glob import glob
from .constants import GAS_CONSTANT, J_TO_AU, KCAL_TO_AU
from .sort import SORT_KEYS

log = logging.getLogger('goodvibes')


def get_selectivity(pattern, files, boltz_facs, temperature, dup_list):
    """
    Calculate selectivity as enantioselectivity/diastereomeric ratio.

    Parameters:
    pattern (str): pattern to recognize for selectivity calculation, i.e. "R":"S".
    files (str): files to use for selectivity calculation.
    boltz_facs (dict): normalized Boltzmann populations for each file.
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

    parts = pattern.split(':')
    if len(parts) != 2:
        raise ValueError(
            f"Invalid selectivity pattern '{pattern}'. "
            "Expected format: 'pattern_a:pattern_b' with exactly one colon."
        )
    [a_regex, b_regex] = [parts[0].strip(), parts[1].strip()]
    if not a_regex or not b_regex:
        raise ValueError(
            f"Invalid selectivity pattern '{pattern}'. "
            "Both patterns before and after ':' must be non-empty."
        )

    A = ''.join(a for a in a_regex if a.isalnum())
    B = ''.join(b for b in b_regex if b.isalnum())

    if len(dirs) > 1 or dirs[0] != '':
        for dir in dirs:
            a_files.extend(glob(dir+'/'+a_regex))
            b_files.extend(glob(dir+'/'+b_regex))
    else:
        a_files.extend(glob(a_regex))
        b_files.extend(glob(b_regex))


    if not a_files or not b_files:
        log.info("\n   Warning! Filenames have not been formatted correctly for determining selectivity\n")
        log.info("   Make sure the filename contains either " + A + " or " + B + "\n")
        sys.exit("   Please edit either your filenames or selectivity pattern argument and try again\n")
    # Grab Boltzmann sums
    for file in files:
        duplicate = False
        if dup_list:
            for dup in dup_list:
                if dup[0] == file:
                    duplicate = True
        if not duplicate:
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
        log.info("\n   Warning! No files found for an enantioselectivity analysis, adjust the stereodetermining step name and try again.\n")
        failed = True
    ee = abs(ee)
    # Detect singularity: ee >= 100 or denominator approaches zero
    if abs(ee) >= 100.0 or abs(50 - abs(ee) / 2.0) < 1e-10:
        dd_free_energy = math.copysign(float('inf'), ee)
    else:
        dd_free_energy = GAS_CONSTANT / J_TO_AU * temperature * math.log((50 + abs(ee) / 2.0) / (50 - abs(ee) / 2.0)) * KCAL_TO_AU
    return ee, r, ratio, dd_free_energy, failed, pref


def get_boltz(thermo_data, temperature, dup_list, key='gibbs'):
    """Compute normalized Boltzmann populations from thermo_data.

    Duplicates in dup_list are excluded from the population.

    Parameters:
        thermo_data (dict): file path → calc_bbe mapping.
        temperature (float): temperature in Kelvin for Boltzmann weighting.
        dup_list (list): pairs [file_i, file_j] to exclude as duplicates.
        key (str): energy attribute to weight by — 'energy' (scf_energy) or
            'gibbs' (qh_gibbs_free_energy). Default: 'gibbs'.

    Returns:
        dict: boltz_facs — normalized Boltzmann populations keyed by file path
        (values sum to 1.0).
    """
    attr = SORT_KEYS[key]
    files = list(thermo_data)
    boltz_facs, e_min, boltz_sum = {}, sys.float_info.max, 0.0

    for file in files:  # Need the most stable structure
        val = getattr(thermo_data[file], attr, None)
        if val is not None and val < e_min:
            e_min = val

    # Calculate E_rel and Boltzmann factors
    for file in files:
        duplicate = False
        if dup_list:
            for dup in dup_list:
                if dup[0] == file:
                    duplicate = True
        if not duplicate:
            val = getattr(thermo_data[file], attr, None)
            if val is not None:
                boltz_facs[file] = math.exp(-(val - e_min) * J_TO_AU / GAS_CONSTANT / temperature)
                boltz_sum += boltz_facs[file]

    # Normalize to populations that sum to 1.0
    if boltz_sum > 0:
        for file in boltz_facs:
            boltz_facs[file] /= boltz_sum

    return boltz_facs
