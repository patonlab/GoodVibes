"""Boltzmann weighting and selectivity calculations for GoodVibes."""
import logging
import math
import os.path
import sys
from glob import glob
from string import ascii_lowercase as alphabet

from .constants import GAS_CONSTANT, J_TO_AU, KCAL_TO_AU

log = logging.getLogger('goodvibes')


def get_selectivity(pattern, files, boltz_facs, boltz_sum, temperature, dup_list):
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
        log.info("\n   Warning! No files found for an enantioselectivity analysis, adjust the stereodetermining step name and try again.\n")
        failed = True
    ee = abs(ee)
    try:
        dd_free_energy = GAS_CONSTANT / J_TO_AU * temperature * math.log((50 + abs(ee) / 2.0) / (50 - abs(ee) / 2.0)) * KCAL_TO_AU
    except ZeroDivisionError:
        dd_free_energy = 0.0
    return ee, r, ratio, dd_free_energy, failed, pref


def get_boltz(thermo_data, clustering, clusters, temperature, dup_list):
    """Compute Boltzmann factors, weighted free energies, and the partition sum.

    Uses quasi-harmonic Gibbs free energies from thermo_data. Duplicates in
    dup_list are excluded from the population.

    Parameters:
        thermo_data (dict): file path → calc_bbe mapping.
        clustering (bool): whether to aggregate by cluster.
        clusters (list): cluster definitions (groups of file paths), or None.
        temperature (float): temperature in Kelvin for Boltzmann weighting.
        dup_list (list): pairs [file_i, file_j] to exclude as duplicates.

    Returns:
        tuple: (boltz_facs, weighted_free_energy, boltz_sum) — dicts keyed by
        file path and the total Boltzmann sum.
    """
    files = list(thermo_data)
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
        if dup_list:
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
