"""File validation and consistency checks for GoodVibes."""
import logging
import os.path
import sys

from .sort import deduplicate
from .utils import all_same
from .io import parse_qcdata, read_initial, find_spc_file
from .io import level_of_theory as read_level_of_theory

log = logging.getLogger('goodvibes')


def _normalize_solvation(value):
    """Return (sort_key, display) for a solvation_model field.

    Gaussian non-gas-phase stores ``[sorted_key, display]``; every other
    program (and Gaussian gas-phase, plus the empty default) stores a plain
    string. Normalizes both shapes so comparisons and display use the right
    value rather than indexing into a bare string.
    """
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return value[0], value[1]
    s = value if value else 'gas phase'
    return s, s


def print_check_fails(check_attribute, file, attribute, option2=None):
    """
    Report groups of files that share differing attribute values.
    
    Groups files by the values in check_attribute (or by (value, option2_value) when option2 is provided) and logs each distinct group. For each group the doc prints the attribute value(s) and up to the first three filenames; if more files are present the remaining count is reported.
    
    Parameters:
        check_attribute (list): Attribute values aligned with the files list (one value per file).
        file (list): File identifiers aligned with check_attribute.
        attribute (str): Human-readable name of the attribute being checked (e.g., "levels of theory").
        option2 (list, optional): Secondary attribute values aligned with the files list; when provided groups are keyed by (check_attribute[i], option2[i]).
    """
    unique_attr = {}
    for i, attr in enumerate(check_attribute):
        if option2 is not None:
            attr = (attr, option2[i])
        if attr not in unique_attr:
            unique_attr[attr] = [file[i]]
        else:
            unique_attr[attr].append(file[i])
    log.info("\n\n   Caution! Multiple {} found: ".format(attribute))
    for attr in unique_attr:
        if option2 is not None:
            try:
                negative = float(attr[0]) < 0
            except (TypeError, ValueError):
                negative = False
            if negative:
                log.info('\n       {} {}: '.format(attr[0], attr[1]))
            else:
                log.info('\n        {} {}: '.format(attr[0], attr[1]))
        else:
            log.info('\n   ✔ {}: '.format(attr))
        filenames = unique_attr[attr]
        if len(filenames) > 3:
            others = len(filenames) - 3
            log.info(f'{filenames[0]}, {filenames[1]}, {filenames[2]}, and {others} others')
        else:
            log.info(', '.join(filenames))


def collect_and_validate_files(files, options):
    """
    Read initial metadata for each output file, remove files that terminated with 'Error' or 'Incomplete', and verify SPC file termination when SPC mode is enabled.
    
    When a file's initial read reports progress 'Error' or 'Incomplete', that file is removed from the returned lists and a warning is logged. If SPC mode (options.spc) is set and not 'link', the function attempts to locate corresponding SPC files; if any discovered SPC file reports 'Error' or 'Incomplete' the program exits. Returned lists remain aligned: the returned level_of_theory and solvation_model correspond to the returned files order.
    
    Parameters:
        files (list): Paths to output files to validate.
        options (Namespace): Parsed CLI options. Uses the `spc` attribute to control SPC lookup behavior.
    
    Returns:
        tuple: (files, level_of_theory, solvation_model) — the filtered file paths, and parallel lists of each file's level of theory and solvation model.
    """
    level_of_theory, solvation_model, progress, spc_progress = [], [], {}, {}
    for file in files:
        lot_sm_prog = read_initial(file)
        level_of_theory.append(lot_sm_prog[0])
        solvation_model.append(lot_sm_prog[1])
        progress[file] = lot_sm_prog[2]
        # Check spc files for normal termination
        if options.spc is not None and options.spc != 'link':
            name, _ = os.path.splitext(file)
            spc_file = find_spc_file(name, options.spc)
            if spc_file is not None:
                lot_sm_prog = read_initial(spc_file)
                spc_progress[spc_file] = lot_sm_prog[2]

    remove_key = []
    # Remove problem files and print errors. Only the first ! warning gets
    # a blank line before it; subsequent warnings stack tightly.
    first_warning = True

    def _warn_prefix():
        nonlocal first_warning
        prefix = "\n\n" if first_warning else "\n"
        first_warning = False
        return prefix

    for i, key in enumerate(files):
        if progress[key] == 'Error':
            log.info(_warn_prefix() + "   ! Error termination found in {}: omitted from further "
                      "calculations.".format(key))
            remove_key.append([i, key])
        elif progress[key] == 'Incomplete':
            log.info(_warn_prefix() + "   ! {} may not have terminated normally or the calculation may still be "
                      "running: omitted from further calculations.".format(key))
            remove_key.append([i, key])
    # Check spc files for normal termination
    if spc_progress:
        for key in spc_progress:
            if spc_progress[key] == 'Error':
                sys.exit(_warn_prefix() + "   ! Error termination found in file {} calculations.".format(key))
            elif spc_progress[key] == 'Incomplete':
                sys.exit(_warn_prefix() + "   ! File {} may not have terminated normally or the "
                    "calculation may still be running.".format(key))

    for [i, key] in list(reversed(remove_key)):
        files.remove(key)
        del level_of_theory[i]
        del solvation_model[i]
    if not files:
        sys.exit("\n\nPlease try again with normally terminated output files.\nFor help, use option '-h'\n")

    return files, level_of_theory, solvation_model


def check_files(thermo_data, options, level_of_theory):
    """Run consistency checks across all calculation output files.

    Checks: program version, solvation model, level of theory, charge/multiplicity,
    standard concentration, linear molecule frequencies, TS imaginary frequencies,
    empirical dispersion, and (if --spc) single-point correction consistency.

    Parameters:
        thermo_data (dict): file path → calc_bbe mapping.
        options (Namespace): parsed CLI options. Uses: conc, spc, duplicate.
        level_of_theory (list): level of theory strings, one per file.
    """
    files = list(thermo_data)
    STARS = "   " + "*" * 128
    log.info("\n   Checks for thermochemistry calculations (frequency calculations):")
    log.info("\n" + STARS)
    # Check program used and version
    version_check = [thermo_data[key].version_program for key in thermo_data]
    file_check = [thermo_data[key].file for key in thermo_data]
    if all_same(version_check):
        log.info("\no  Using {} in all calculations.".format(version_check[0]))
    else:
        print_check_fails(version_check, file_check, "programs or versions")

    # Check level of theory
    if all_same(level_of_theory):
        log.info("\no  Using {} in all calculations.".format(level_of_theory[0]))
    else:
        print_check_fails(level_of_theory, file_check, "levels of theory")

    # Check for solvent models
    solvation_pairs = [_normalize_solvation(thermo_data[key].solvation_model) for key in thermo_data]
    sort_keys = [p[0] for p in solvation_pairs]
    solvent_check = [p[1] for p in solvation_pairs]
    if all_same(sort_keys):
        log.info("\no  Using {} in all calculations.".format(solvent_check[0]))
    else:
        print_check_fails(solvent_check, file_check, "solvation models")

    # Check for -c 1 when solvent is added
    if all_same(sort_keys):
        if solvent_check[0] == "gas phase" and options.conc is None:
            log.info("\no  Using a standard concentration of 1 atm for gas phase.")
        elif solvent_check[0] == "gas phase" and options.conc is not None:
            log.info("\n   x Caution! Standard concentration is not 1 atm for gas phase (using {} M).".format(options.conc))
        elif solvent_check[0] != "gas phase" and options.conc is None:
            log.info("\n   x Using a standard concentration of 1 atm for solvent phase (option -c 1 should be included for 1 M).")
        elif solvent_check[0] != "gas phase" and str(options.conc) == str(1.0):
            log.info("\no  Using a standard concentration of 1 M for solvent phase.")
        elif solvent_check[0] != "gas phase" and options.conc is not None and str(options.conc) != str(1.0):
            log.info("\n   x Caution! Standard concentration is not 1 M for solvent phase (using {} M).".format(options.conc))
    if not all_same(sort_keys) and "gas phase" in solvent_check:
        log.info("\n   x Caution! The right standard concentration cannot be determined because the calculations use a combination of gas and solvent phases.")
    if not all_same(sort_keys) and "gas phase" not in solvent_check:
        log.info("\n   x Caution! Different solvents used, fix this issue and use option -c 1 for a standard concentration of 1 M.")

    # Check charge and multiplicity
    charge_check = [thermo_data[key].charge for key in thermo_data]
    multiplicity_check = [thermo_data[key].multiplicity for key in thermo_data]
    if all_same(charge_check) and all_same(multiplicity_check):
        log.info("\no  Using charge {} and multiplicity {} in all calculations.".format(charge_check[0],
                                                                                         multiplicity_check[0]))
    else:
        print_check_fails(charge_check, file_check, "charge and multiplicity", multiplicity_check)

    # Check for duplicate structures
    dup_list = deduplicate(thermo_data,
                           e_cutoff=getattr(options, 'e_cutoff', 0.05),
                           ro_cutoff=getattr(options, 'ro_cutoff', 0.01),
                           rmsd_cutoff=getattr(options, 'rmsd_cutoff', None))
    if not dup_list:
        log.info("\no  No duplicates or enantiomers found")
    else:
        log.info("\n   x Caution! Possible duplicates or enantiomers found:")
        for dup in dup_list:
            log.info('\n        {} and {}'.format(dup[0], dup[1]))

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
    linear_correct_print = ', '.join(linear_mol_correct)
    linear_wrong_print = ', '.join(linear_mol_wrong)
    if not linear_mol_correct:
        if not linear_mol_wrong:
            log.info("\n-  No linear molecules found.")
        if linear_mol_wrong:
            log.info("\n   x Caution! Potential linear molecules with wrong number of frequencies found "
                      "(correct number = 3N-5) -{}.".format(linear_wrong_print))
    elif linear_mol_correct:
        if not linear_mol_wrong:
            log.info("\no  All the linear molecules have the correct number of frequencies -{}.".format(linear_correct_print))
        if linear_mol_wrong:
            log.info("\n   x Caution! Potential linear molecules with wrong number of frequencies found -{}. Correct "
                      "number of frequencies (3N-5) found in other calculations -{}.".format(linear_wrong_print,
                                                                                             linear_correct_print))

    # Checks whether any TS have > 1 imaginary frequency and any GS have any imaginary frequencies
    for file in files:
        bbe = thermo_data[file]
        if bbe.job_type.find('TS') > -1 and len(bbe.im_frequency_wn) != 1:
            log.info("\n   x Caution! TS {} does not have 1 imaginary frequency greater than -50 wavenumbers.".format(file))
        if bbe.job_type.find('GS') > -1 and bbe.job_type.find('TS') == -1 and bbe.im_frequency_wn:
            log.info("\n   x Caution: GS {} has 1 or more imaginary frequencies greater than -50 wavenumbers.".format(file))

    # Check for empirical dispersion
    dispersion_check = [thermo_data[key].empirical_dispersion or 'No empirical dispersion detected'
                        for key in thermo_data]
    if all_same(dispersion_check):
        if dispersion_check[0] == 'No empirical dispersion detected':
            log.info("\n-  No empirical dispersion detected in any of the calculations.")
        else:
            log.info("\no  Using " + dispersion_check[0] + " in all calculations.")
    else:
        print_check_fails(dispersion_check, file_check, "dispersion models")
    log.info("\n" + STARS + "\n")

    # Check for single-point corrections
    if options.spc is not None:
        log.info("\n   Checks for single-point corrections:")
        log.info("\n" + STARS)
        names_spc, version_check_spc = [], []
        for file in files:
            name, _ = os.path.splitext(file)
            spc_file = find_spc_file(name, options.spc)
            if spc_file is not None:
                names_spc.append(spc_file)

        # Check SPC program versions
        version_check_spc = [thermo_data[key].sp_version_program for key in thermo_data]
        if all_same(version_check_spc):
            log.info("\no  Using {} in all the single-point corrections.".format(version_check_spc[0]))
        else:
            print_check_fails(version_check_spc, file_check, "programs or versions")

        # Check SPC solvation
        spc_solv_pairs = [_normalize_solvation(thermo_data[key].sp_solvation_model) for key in thermo_data]
        spc_sort_keys = [p[0] for p in spc_solv_pairs]
        spc_solvent_check = [p[1] for p in spc_solv_pairs]
        if all_same(spc_sort_keys):
            log.info("\no  Using {} in all single-point corrections.".format(spc_solvent_check[0]))
        else:
            print_check_fails(spc_solvent_check, file_check, "solvation models")

        # Check SPC level of theory (the `level_of_theory` parameter holds the
        # freq-level strings, so the SPC files are re-read via io.level_of_theory)
        l_o_t_spc = [read_level_of_theory(name) for name in names_spc]
        if len(names_spc) == len(files):
            if all_same(l_o_t_spc):
                log.info("\no  Using {} in all the single-point corrections.".format(l_o_t_spc[0]))
            else:
                print_check_fails(l_o_t_spc, names_spc, "levels of theory")
        else:
            log.info("\n   x One or more single-point correction files are missing; "
                     "unable to check levels of theory.")

        # Check SPC charge and multiplicity
        charge_spc_check = [thermo_data[key].sp_charge for key in thermo_data]
        multiplicity_spc_check = [thermo_data[key].sp_multiplicity for key in thermo_data]
        if all_same(charge_spc_check) and all_same(multiplicity_spc_check):
            log.info("\no  Using charge and multiplicity {} {} in all the single-point corrections.".format(
                charge_spc_check[0], multiplicity_spc_check[0]))
        else:
            print_check_fails(charge_spc_check, file_check, "charge and multiplicity", multiplicity_spc_check)

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
                log.info("\no  No potential differences found between frequency and single-point geometries (based on input coordinates).")
            else:
                spc_mismatching_1 = spc_mismatching[:84]
                spc_mismatching_2 = spc_mismatching[85:]
                log.info("\n   x " + spc_mismatching_1 + spc_mismatching_2 + '.')
        else:
            log.info("\n   x One or more geometries from single-point corrections are missing.")

        # Check for SPC dispersion models
        dispersion_check_spc = [thermo_data[key].sp_empirical_dispersion or 'No empirical dispersion detected'
                                for key in thermo_data]
        if all_same(dispersion_check_spc):
            if dispersion_check_spc[0] == 'No empirical dispersion detected':
                log.info("\n-  No empirical dispersion detected in any of the calculations.")
            else:
                log.info("\no  Using " + dispersion_check_spc[0] + " in all the single-point calculations.")
        else:
            print_check_fails(dispersion_check_spc, file_check, "dispersion models")
        log.info("\n" + STARS + "\n")
