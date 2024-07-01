# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import os.path
import math
import sys
import numpy as np
import datetime
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
AMU_to_KG = 1.66053886E-27  # UNIT CONVERSION
GHz_to_K = 0.0479924341590786 # UNIT CONVERSION
eV_to_Hartree = 1.0 / 27.21138505 # UNIT CONVERSION

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
        except:
            species.point_group = 'C1'
            species.symm_no = 1

def all_same(items):
    """Returns bool for checking if all items in a list are the same."""
    return all(x == items[0] for x in items)

def add_time(tm, cpu):
    """Calculate elapsed time."""
    [days, hrs, mins, secs, msecs] = cpu
    fulldate = datetime(100, 1, tm.day, tm.hour, tm.minute, tm.second, tm.microsecond)
    fulldate = fulldate + timedelta(days=days, hours=hrs, minutes=mins, seconds=secs, microseconds=msecs * 1000)
    return fulldate

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
                log.write("\n\no  Found vibrational scaling factor of {:.3f} for {} level of theory\n"
                        "   {}\n\n".format(freq_scale_factor, model_chemistry, ref))
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
            log.write("\n   REF: {}\n".format(SIMON_REF_scale_ref))
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
        except:
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
            except:
                ref_rotconsts = None

            for j in range(i+1, len(thermo_data)):
                name = thermo_data[j].name
                energy = thermo_data[j].scf_energy
                
                try:
                    rotconsts = thermo_data[j].rotconsts
                except:
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
        if use_gibbs: 
            thermo_data.sort(key=lambda x: x.qh_gibbs_free_energy if x.qh_gibbs_free_energy is not None else 0)
        else:
            thermo_data.sort(key=lambda x: x.scf_energy if x.scf_energy is not None else 0)
    
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

def check_files(log, files, thermo_data, options):
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

def get_boltz_facs(thermo_data, temperature, use_gibbs=True):
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
                if thermo.qh_gibbs_free_energy < glob_min:
                    glob_min = thermo.qh_gibbs_free_energy
        else:
            if hasattr(thermo, "scf_energy"):
                if thermo.scf_energy < glob_min:
                    glob_min = thermo.scf_energy

    # Calculate G_rel and Boltzmann factors
    for thermo in thermo_data:
        if use_gibbs is True:
            if hasattr(thermo, "qh_gibbs_free_energy"):
                thermo.g_rel = thermo.qh_gibbs_free_energy - glob_min # in Hartree
                thermo.boltz_fac = math.exp(-thermo.g_rel * J_TO_AU / GAS_CONSTANT / temperature)
                boltz_sum += thermo.boltz_fac
        else:
            if hasattr(thermo, "scf_energy"):
                thermo.e_rel = thermo.scf_energy - glob_min # in Hartree
                thermo.boltz_fac = math.exp(-thermo.e_rel * J_TO_AU / GAS_CONSTANT / temperature)
                boltz_sum += thermo.boltz_fac

    for thermo in thermo_data:
        thermo.boltz_fac = thermo.boltz_fac / boltz_sum

def get_selectivity(pattern, thermo_data, temperature):
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
    A_round = round(a_sum * 100)
    B_round = round(b_sum * 100)
    ratio = str(A_round) + ':' + str(B_round)
    excess = (a_sum - b_sum) / (a_sum + b_sum) * 100.0

    try:
        ddg = GAS_CONSTANT / J_TO_AU * temperature * math.log((50 + abs(excess) / 2.0) / (50 - abs(excess) / 2.0)) * KCAL_TO_AU * np.sign(excess)

    except ZeroDivisionError:
        ddg = 0.0
    return excess, ratio, ddg