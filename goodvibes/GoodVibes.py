#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Quasi-harmonic thermochemical corrections from electronic structure calculations."""
from __future__ import print_function, absolute_import

import os.path
import sys
from datetime import datetime
from glob import glob
from argparse import ArgumentParser

# Importing regardless of relative import
try:
    from .vib_scale_factors import scaling_data_dict, scaling_refs, canonicalize_level
    from .io import read_initial, xyz_out
    from .thermo import calc_bbe, get_free_space
    from .media import solvents
    from .constants import (
        __version__, SUPPORTED_EXTENSIONS, GAS_CONSTANT, ATMOS, J_TO_AU, KCAL_TO_AU,
        grimme_mRRHO_ref, grimme_msRRHO_ref, truhlar_ref, head_gordon_ref,
        goodvibes_ref, csd_ref, oniom_scale_ref,
    )
    from .pes import get_pes
    from .utils import all_same, Logger, add_time, display_name
    from .validation import check_files, print_check_fails
    from .output import print_results, print_temperature_interval, print_pes_results
except ImportError:
    from vib_scale_factors import scaling_data_dict, scaling_refs, canonicalize_level
    from io import read_initial, xyz_out
    from thermo import calc_bbe, get_free_space
    from media import solvents
    from constants import (  # noqa: F401
        __version__, SUPPORTED_EXTENSIONS, GAS_CONSTANT, ATMOS, J_TO_AU, KCAL_TO_AU,
        grimme_mRRHO_ref, grimme_msRRHO_ref, truhlar_ref, head_gordon_ref,
        goodvibes_ref, csd_ref, oniom_scale_ref,
    )
    from pes import get_pes  # noqa: F401
    from utils import all_same, Logger, add_time, display_name  # noqa: F401
    from validation import check_files, print_check_fails
    from output import print_results, print_temperature_interval, print_pes_results


def parse_arguments():
    """Parse command-line arguments and return (options, args, command, clustering, clusters)."""
    files = []
    clusters = []
    command = 'o  Requested: '
    clustering = False
    # Get command line inputs. Use -h to list all possible arguments and default values
    parser = ArgumentParser(
        prog="goodvibes",
        description="Compute quasi-harmonic thermochemical corrections from quantum chemistry output files.")
    parser.add_argument("--bav", dest='inertia', default="global", type=str, choices=['global', 'conf'],
                        help="Moment of inertia for free-rotor entropy: 'global' uses Bav = 10e-44 kg m^2 "
                             "for all molecules, 'conf' computes from rotational constants per file "
                             "(default: global)")
    parser.add_argument("--boltz", dest="boltz", action="store_true", default=False,
                        help="Print Boltzmann-weighted populations for each structure")
    parser.add_argument("--check", dest="check", action="store_true", default=False,
                        help="Verify all files use the same program, level of theory, and solvation model; "
                             "also flag potential duplicates")
    parser.add_argument("--conc", dest="conc", default=False, type=float, metavar="CONC",
                        help="Concentration in mol/L for solution-phase entropy; gas phase (1 atm) if not set")
    parser.add_argument("--cpu", dest="cputime", action="store_true", default=False,
                        help="Print total CPU time from output files")
    parser.add_argument("--csv", dest="csv", action="store_true", default=False,
                        help="Write output in comma-separated value (.csv) format")
    parser.add_argument("--custom_ext", type=str, default='', metavar="exts",
                        help="Additional file extensions to accept, comma-separated "
                             "(e.g. '.qfi,.gaussian'); also settable via GOODVIBES_CUSTOM_EXT env var")
    parser.add_argument("--dedup", dest="duplicate", action="store_true", default=False,
                        help="Remove duplicate structures based on energy, rotational constants, and stoichiometry")
    parser.add_argument("--ee", dest="ee", default=False, type=str, metavar="patterns",
                        help="Compute selectivity (ee, ratio) from a conformer mixture; provide glob patterns "
                             "for two species (e.g. '*_R*,*_S*')")
    parser.add_argument("-f", "--tau", dest="freq_cutoff", default=100, type=float, metavar="FREQ_CUTOFF",
                        help="Frequency cut-off for both entropy and enthalpy in cm-1 (default: 100)")
    parser.add_argument("--fh", dest="H_freq_cutoff", default=100.0, type=float, metavar="H_FREQ_CUTOFF",
                        help="Frequency cut-off for enthalpy only in cm-1; overrides -f for H (default: 100)")
    parser.add_argument("--freespace", dest="freespace", default=None, type=str, metavar="SOLVENT",
                        help="Apply free-space correction for the given solvent (e.g. H2O, toluene, DMF, "
                             "AcOH, chloroform)")
    parser.add_argument("--fs", dest="S_freq_cutoff", default=100.0, type=float, metavar="S_FREQ_CUTOFF",
                        help="Frequency cut-off for entropy only in cm-1; overrides -f for S (default: 100)")
    parser.add_argument("--graph", dest='graph', default=False, metavar="file",
                        help="Graph a reaction profile from free energies; provide the PES YAML file")
    parser.add_argument("--imag", dest="imag_freq", action="store_true", default=False,
                        help="Print imaginary frequencies for each structure")
    parser.add_argument("--invert", dest="invert", nargs='?', const=True, default=False,
                        help="Invert small imaginary frequencies (> -50 cm-1) to positive values; "
                             "optionally provide a custom threshold in cm-1")
    parser.add_argument("--media", dest="media", default=False, metavar="solvent",
                        help="Apply standard-state concentration correction for the given solvent "
                             "(e.g. H2O corrects to 55.34 M)")
    parser.add_argument("--nogconf", dest="gconf", action="store_false", default=True,
                        help="Disable the Gconf correction for multi-conformer ensembles (enabled by default)")
    parser.add_argument("--output", dest="output", default="output", metavar="name",
                        help="Base name for the output file, written as GoodVibes_OUTPUT.dat (default: output)")
    parser.add_argument("--pes", dest="pes", default=False, metavar="file",
                        help="YAML file defining a reaction pathway for tabulating relative energies")
    parser.add_argument("-q", dest="Q", action="store_true", default=False,
                        help="Apply both quasi-harmonic entropy (Grimme mRRHO) and enthalpy (Head-Gordon) corrections")
    parser.add_argument("--qh", dest="QH", action="store_true", default=False,
                        help="Apply Head-Gordon quasi-harmonic enthalpy correction")
    parser.add_argument("--qs", dest="QS", default="grimme", type=str.lower, metavar="QS",
                        choices=('grimme', 'truhlar'),
                        help="Quasi-harmonic entropy method: 'grimme' for mRRHO free-rotor interpolation, "
                             "'truhlar' for frequency raising (default: grimme)")
    parser.add_argument("--spc", dest="spc", type=str, default=False, metavar="suffix",
                        help="Single-point correction suffix: reads energy from FILE_SPC.ext "
                             "(e.g. --spc TZ reads from FILE_TZ.log)")
    parser.add_argument("--ssymm", dest='ssymm', action="store_true", default=False,
                        help="Apply external symmetry correction to entropy using point-group detection (pymsym)")
    parser.add_argument("--temp", dest="temperature", default=298.15, type=float, metavar="TEMP",
                        help="Temperature in Kelvin (default: 298.15)")
    parser.add_argument("--ti", dest="temperature_interval", default=False, metavar="TI",
                        help="Temperature interval as START,END,STEP in Kelvin (e.g. '300,1000,100')")
    parser.add_argument("-v", dest="freq_scale_factor", default=False, type=float, metavar="SCALE_FACTOR",
                        help="Vibrational frequency scaling factor; auto-detected from level of theory if not set, "
                             "falls back to 1.0")
    parser.add_argument("--vmm", dest="mm_freq_scale_factor", default=False, type=float, metavar="MM_SCALE_FACTOR",
                        help="Frequency scaling factor for the MM region in ONIOM calculations")
    parser.add_argument("--xyz", dest="xyz", action="store_true", default=False,
                        help="Write optimized Cartesian coordinates to a .xyz file")
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
                sys.exit("\nx  ERROR! Error termination found in file {} calculations.".format(key))
            elif spc_progress[key] == 'Incomplete':
                sys.exit("\nx  ERROR! File {} may not have terminated normally or the "
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
            log.write("\n   User-defined vibrational scale factor " + str(options.freq_scale_factor) + " for " +
                      l_o_t[0] + " level of theory")
        else:
            log.write("\n   User-defined vibrational scale factor " + str(options.freq_scale_factor) +
                      " for QM region of " + l_o_t[0])
    else:
        # Look for vibrational scaling factor automatically
        if all_same(l_o_t):
            level = canonicalize_level(l_o_t[0])
            if level in scaling_data_dict:
                options.freq_scale_factor = scaling_data_dict[level].harm_fac
                ref = scaling_refs[scaling_data_dict[level].harm_ref]
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
            log.write("\n   Caution! Implicit solvation (SMD/CPCM) detected. Enthalpic and entropic terms cannot be "
                      "safely separated. Use them at your own risk!")
            printed_solv_warn = True

    if options.freq_cutoff != 100.0:
        options.S_freq_cutoff = options.freq_cutoff
        options.H_freq_cutoff = options.freq_cutoff

    # Summary of the quasi-harmonic treatment; print out the relevant reference

    if options.QS == "grimme":
        if options.freq_scale_factor == 1.0:
            log.write("\n\n   Using mRRHO entropies with a frequency cut-off value (tau) of " + str(options.S_freq_cutoff) + " cm-1")
            qs_ref = grimme_mRRHO_ref
        else:
            log.write("\n\n   Using msRRHO entropies with a frequency cut-off value (tau) of " + str(options.S_freq_cutoff) + " cm-1")
            qs_ref = grimme_msRRHO_ref

    elif options.QS == "truhlar":
        log.write("\n\n   Using an RRHO treatment where low frequencies are adjusted to " + str(options.S_freq_cutoff) + " cm-1")
        qs_ref = truhlar_ref
    else:
        log.fatal("\n   FATAL ERROR: Unknown quasi-harmonic model " + options.QS + " specified (QS must = grimme or truhlar).")
    log.write("\n   REF: " + qs_ref)

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
            if options.media.lower() in solvents and options.media.lower() == display_name(file).lower():
                mweight = solvents[options.media.lower()][0]
                density = solvents[options.media.lower()][1]
                conc = (density * 1000) / mweight
                media_conc = conc
        bbe = calc_bbe(file, options.QS, options.QH, options.S_freq_cutoff, options.H_freq_cutoff, options.temperature,
                       conc, options.freq_scale_factor, options.freespace, options.spc, options.invert,
                       ssymm=ssymm_option, mm_freq_scale_factor=vmm_option,
                       inertia=options.inertia)

        # Populate bbe_vals with indivual bbe entries for each file
        bbe_vals.append(bbe)

    # Creates a new dictionary object thermo_data, which attaches the bbe data to each file-name
    file_list = list(files)
    thermo_data = dict(zip(file_list, bbe_vals))  # The collected thermochemical data for all files
    return thermo_data, bbe_vals, media_conc


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
        xyz = xyz_out("GoodVibes", "xyz", "output")

    # Start logger
    log = Logger("GoodVibes", options.output, options.csv)

    # Print banner
    log.write(
              "      ________   ________   ________    _______   ________   ________   ________   ________   ________ \n"
              "     \u2571        \u2572 \u2571        \u2572 \u2571        \u2572 _\u2571       \u2572 \u2571    \u2571   \u2572 \u2571        \u2572 \u2571       \u2571  \u2571        \u2572 \u2571        \u2572\n"
              "    \u2571   G   __\u2571\u2571    O    \u2571\u2571    O    \u2571\u2571    D    \u2571\u2571    V    \u2571_\u2571   I   \u2571 \u2571    B   \u2572 \u2571    E    \u2571\u2571    S   _\u2571\n"
              "   \u2571       \u2571 \u2571\u2571         \u2571\u2571         \u2571\u2571         \u2571 \u2572        \u2571\u2571         \u2571\u2571         \u2571\u2571        _\u2571\u2571-  v" + __version__ + "  \u2571 \n"
              "   \u2572________\u2571 \u2572________\u2571 \u2572________\u2571 \u2572________\u2571   \u2572______\u2571 \u2572________\u2571 \u2572________\u2571 \u2572________\u2571 \u2572________\u2571\n"
              "\n   Citation: " + goodvibes_ref + "\n")

    # Check if user has specified any files
    if len(files) == 0:
        sys.exit("\nPlease provide GoodVibes with calculation output files on the command line.\n"
                 "For help, use option '-h'\n")
    if clustering:
        command += '(clustering active)'
    log.write('\n' + command + '\n')
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
