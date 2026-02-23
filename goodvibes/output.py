"""Output formatting and printing functions for GoodVibes."""
import logging
import math
import os.path
import sys
from datetime import datetime

from .utils import display_name, add_time
from .selectivity import get_selectivity
from .constants import GAS_CONSTANT, ATMOS, J_TO_AU, KCAL_TO_AU

from .pes import get_pes, graph_reaction_profile
from .thermo import calc_bbe
from .media import solvents

log = logging.getLogger('goodvibes')


def print_cpu_time(thermo_data, exclude=None):
    """Sum and print total CPU time (including SPC) across all files.

    Parameters:
        thermo_data (dict): file path → calc_bbe mapping.
        exclude (str, optional): glob pattern — skip matching files from the total.
    """
    from fnmatch import fnmatch
    total_cpu_time, add_days = datetime(100, 1, 1, 0, 0, 0, 0), 0
    for file, bbe in thermo_data.items():
        if exclude and fnmatch(file, exclude):
            continue
        if hasattr(bbe, "cpu") and bbe.cpu is not None:
            total_cpu_time = add_time(total_cpu_time, bbe.cpu)
        if hasattr(bbe, "sp_cpu") and bbe.sp_cpu is not None:
            total_cpu_time = add_time(total_cpu_time, bbe.sp_cpu)
        if total_cpu_time.month > 1:
            add_days += 31
    log.info(f'   {"TOTAL CPU":<13} {total_cpu_time.day + add_days - 1:>2} {"days":>4} '
              f'{total_cpu_time.hour:>2} {"hrs":>3} {total_cpu_time.minute:>2} {"mins":>4} '
              f'{total_cpu_time.second:>2} {"secs":>4}\n')


def print_results(thermo_data, options, media_conc=None,
                  dup_list=None, boltz_facs=None, boltz_sum=None):
    """Print the single-temperature thermochemistry results table.

    Outputs energies, enthalpies, entropies and free energies for each file.
    Optionally includes Boltzmann weighting, imaginary frequencies, symmetry
    point groups, CPU times, and media concentration annotations.

    Parameters:
        thermo_data (dict): file path → calc_bbe mapping.
        options (Namespace): parsed CLI options. Uses: dp, QH, invert, spc,
            imag_freq, boltz, symm, duplicate, temperature, cputime, media.
        media_conc (float, optional): neat solvent concentration for display.
        dup_list (list, optional): pairs of duplicate/enantiomer file paths.
            Computed by deduplicate() in the orchestrator. Defaults to [].
        boltz_facs (dict, optional): Boltzmann factors keyed by file path.
            Computed by get_boltz() in the orchestrator. None when --boltz is off.
        boltz_sum (float, optional): total Boltzmann partition sum.
            None when --boltz is off.
    """
    files = list(thermo_data)
    # Decimal places for energy output (default 6)
    dp = getattr(options, 'dp', 6)
    # Extra width needed when dp > 6
    dw = dp - 6
    # Set up stars separator based on QH option and decimal places
    if options.QH:
        stars = "   " + "*" * (142 + 8 * dw)  # 8 energy columns with QH
    else:
        stars = "   " + "*" * (128 + 7 * dw)  # 7 energy columns without QH
    # Format helpers: w=wide (13+dw), n=narrow (10+dw)
    ef = '{{:{w}.{d}f}}'.format(w=13 + dw, d=dp)  # energy format e.g. {:13.6f} or {:15.8f}
    nf = '{{:{w}.{d}f}}'.format(w=10 + dw, d=dp)  # narrow format e.g. {:10.6f} or {:12.8f}

    # Check if user has chosen to make any low lying imaginary frequencies positive
    inverted_freqs, inverted_files = [], []
    for file in files:
        if thermo_data[file].inverted_freqs:
            inverted_freqs.append(thermo_data[file].inverted_freqs)
            inverted_files.append(file)
    if options.invert is not None:
        for i, file in enumerate(inverted_files):
            if len(inverted_freqs[i]) == 1:
                log.info("\n\n   The following frequency was made positive and used in calculations: " +
                          str(inverted_freqs[i][0]) + " from " + file)
            elif len(inverted_freqs[i]) > 1:
                log.info("\n\n   The following frequencies were made positive and used in calculations: " +
                          str(inverted_freqs[i]) + " from " + file)

    # Adjust printing according to options requested
    if options.spc is not None:
        stars += '*' * (14 + dw)
    # Width of the energy data columns (stars minus leading spaces and name)
    data_width = len(stars) - 3 - 39
    if options.imag_freq is True:
        stars += '*' * 9
    if options.boltz is True:
        stars += '*' * 7
    if options.symm or options.pg:
        stars += '*' * 13

    total_cpu_time, add_days = datetime(100, 1, 1, 00, 00, 00, 00), 0

    ew = 13 + dw  # energy column width for headers
    nw = 10 + dw  # narrow column width for headers

    if options.spc is None:
        log.info("\n\n   ")
        if options.QH:
            log.info(('{{:<39}} {{:>{ew}}} {{:>{nw}}} {{:>{ew}}} {{:>{ew}}} {{:>{nw}}} {{:>{nw}}} {{:>{ew}}} '
                       '{{:>{ew}}}').format(ew=ew, nw=nw).format(
                "Structure", "E", "ZPE", "H", "qh-H", "T.S", "T.qh-S", "G(T)", "qh-G(T)"),
)
        else:
            log.info(('{{:<39}} {{:>{ew}}} {{:>{nw}}} {{:>{ew}}} {{:>{nw}}} {{:>{nw}}} {{:>{ew}}} '
                       '{{:>{ew}}}').format(ew=ew, nw=nw).format(
                "Structure", "E", "ZPE", "H", "T.S", "T.qh-S", "G(T)", "qh-G(T)"))
    else:
        log.info("\n\n   ")
        if options.QH:
            log.info(('{{:<39}} {{:>{ew}}} {{:>{ew}}} {{:>{nw}}} {{:>{ew}}} {{:>{ew}}} {{:>{nw}}} {{:>{nw}}} {{:>{ew}}} '
                       '{{:>{ew}}}').format(ew=ew, nw=nw).format(
                "Structure", "E_SPC", "E", "ZPE", "H_SPC", "qh-H_SPC", "T.S", "T.qh-S",
                "G(T)_SPC", "qh-G(T)_SPC"))
        else:
            log.info(('{{:<39}} {{:>{ew}}} {{:>{ew}}} {{:>{nw}}} {{:>{ew}}} {{:>{nw}}} {{:>{nw}}} {{:>{ew}}} '
                       '{{:>{ew}}}').format(ew=ew, nw=nw).format(
                "Structure", "E_SPC", "E", "ZPE", "H_SPC", "T.S", "T.qh-S", "G(T)_SPC",
                "qh-G(T)_SPC"))
    if options.boltz is True:
        log.info('{:>7}'.format("Boltz"))
    if options.imag_freq is True:
        log.info('{:>9}'.format("im freq"))
    if options.symm or options.pg:
        log.info('{:>13}'.format("Point Group"))
    log.info("\n" + stars + "")

    # Default dup_list if not passed by caller
    if dup_list is None:
        dup_list = []

    for file in files:  # Loop over the output files and compute thermochemistry
        duplicate = False
        if dup_list:
            for dup in dup_list:
                if dup[0] == file:
                    duplicate = True
                    log.info('\nx  {} is a duplicate or enantiomer of {}'.format(dup[0].rsplit('.', 1)[0],
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

            # Check for possible error in Gaussian calculation of linear molecules which can return 2 rotational constants instead of 3
            if bbe.linear_warning:
                log.info("\nx  " + '{:<39}'.format(display_name(file)))
                log.info('{:<{w}}'.format('          ----   Caution! Potential invalid calculation of linear molecule in Gaussian', w=data_width))
            else:
                if hasattr(bbe, "gibbs_free_energy"):
                    if options.spc is not None:
                        if bbe.sp_energy != '!':
                            log.info("\no  ")
                            log.info('{:<39}'.format(display_name(file)))
                            log.info((' ' + ef).format(bbe.sp_energy))
                        if bbe.sp_energy == '!':
                            log.info("\nx  ")
                            log.info('{:<39}'.format(display_name(file)))
                            log.info(' {:>13}'.format('----'))
                    else:
                        log.info("\no  ")
                        log.info('{:<39}'.format(display_name(file)))
                # Gaussian SPC file handling
                if bbe.scf_energy is not None and not hasattr(bbe, "gibbs_free_energy"):
                    log.info("\nx  " + '{:<39}'.format(display_name(file)))
                # ORCA spc files
                elif bbe.scf_energy is None and not hasattr(bbe, "gibbs_free_energy"):
                    log.info("\nx  " + '{:<39}'.format(display_name(file)))
                if bbe.scf_energy is not None:
                    log.info((' ' + ef).format(bbe.scf_energy))
                # No freqs found
                if not hasattr(bbe, "gibbs_free_energy"):
                    log.info("   Warning! Couldn't find frequency information ...")
                else:
                    if all(getattr(bbe, attrib) for attrib in
                           ["enthalpy", "entropy", "qh_entropy", "gibbs_free_energy", "qh_gibbs_free_energy"]):
                        if options.QH:
                            log.info((' ' + nf + ' ' + ef + ' ' + ef + ' ' + nf + ' ' + nf + ' ' + ef + ' ' + ef).format(
                                bbe.zpe, bbe.enthalpy, bbe.qh_enthalpy, (options.temperature * bbe.entropy),
                                (options.temperature * bbe.qh_entropy), bbe.gibbs_free_energy,
                                bbe.qh_gibbs_free_energy))
                        else:
                            log.info((' ' + nf + ' ' + ef + ' ' + nf + ' ' + nf + ' ' + ef + ' '
                                       + ef).format(bbe.zpe, bbe.enthalpy,
                                                     (options.temperature * bbe.entropy),
                                                     (options.temperature * bbe.qh_entropy),
                                                     bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy),
                )

                    if options.media is not None and options.media.lower() in solvents and options.media.lower() == \
                            display_name(file).lower():
                        log.info("  Solvent: {:4.2f}M ".format(media_conc))

            # Append requested options to end of output
            if options.boltz is True:
                log.info('{:7.3f}'.format(boltz_facs[file] / boltz_sum))
            if options.imag_freq is True and hasattr(bbe, "im_frequency_wn"):
                for freq in bbe.im_frequency_wn:
                    log.info('{:9.2f}'.format(freq))
            if options.symm or options.pg:
                if hasattr(bbe, "point_group") and bbe.point_group:
                    log.info('{:>13}'.format(bbe.point_group))
                else:
                    log.info('{:>37}'.format('---'))

    log.info("\n" + stars + "\n")


def print_temperature_interval(thermo_data, options, gas_phase, media_conc=None, qcdata_cache=None):
    """Recompute thermochemistry across a temperature range and print results.

    Re-runs calc_bbe at each temperature step for every file, printing enthalpy,
    entropy, and free energy at each point.

    Parameters:
        thermo_data (dict): file path → calc_bbe mapping (used for file list).
        options (Namespace): parsed CLI options. Uses: dp, QH, QS, S_freq_cutoff,
            H_freq_cutoff, spc, conc, freq_scale_factor, freespace, invert,
            inertia, media, temperature_interval.
        gas_phase (bool): whether all calculations are gas-phase (affects concentration).
        media_conc (float, optional): neat solvent concentration for display.
        qcdata_cache (dict, optional): pre-parsed QCData keyed by basename.

    Returns:
        tuple: (interval_bbe_data, interval, file_list).
    """
    files = list(thermo_data)
    dp = getattr(options, 'dp', 6)
    dw = dp - 6
    ef = '{{:{w}.{d}f}}'.format(w=13 + dw, d=dp)
    nf = '{{:{w}.{d}f}}'.format(w=10 + dw, d=dp)
    hf = '{{:{w}.{d}f}}'.format(w=24 + dw, d=dp)  # wide H column in temp interval
    ew = 13 + dw
    nw = 10 + dw
    hw = 24 + dw
    # Set up stars separator based on QH option and decimal places
    if options.QH:
        stars = "   " + "*" * (142 + 8 * dw)  # 8 energy columns with QH
    else:
        stars = "   " + "*" * (128 + 7 * dw)  # 7 energy columns without QH

    log.info("\n\n   Variable-Temperature analysis of the enthalpy, entropy and the entropy at a constant pressure between")
    temperature_interval = [float(temp) for temp in options.temperature_interval.split(',')]
    # If no temperature step was defined, divide the region into 10
    if len(temperature_interval) == 2:
        temperature_interval.append((temperature_interval[1] - temperature_interval[0]) / 10.0)
    interval = range(int(temperature_interval[0]), int(temperature_interval[1] + 1),
                     int(temperature_interval[2]))
    log.info("\n   T init:  %.1f,  T final:  %.1f,  T interval: %.1f" % (
        temperature_interval[0], temperature_interval[1], temperature_interval[2]))
    if options.QH:
        qh_print_format = ('\n\n   {{:<39}} {{:>13}} {{:>{hw}}} {{:>{ew}}} {{:>{nw}}} {{:>{nw}}} '
                           '{{:>{ew}}} {{:>{ew}}}').format(hw=hw, ew=ew, nw=nw)
        if options.spc:
            log.info(qh_print_format.format("Structure", "Temp/K", "H_SPC", "qh-H_SPC", "T.S", "T.qh-S",
                                             "G(T)_SPC", "qh-G(T)_SPC"))
        else:
            log.info(qh_print_format.format("Structure", "Temp/K", "H", "qh-H", "T.S", "T.qh-S", "G(T)",
                                             "qh-G(T)"))
    else:
        print_format_3 = ('\n\n   {{:<39}} {{:>13}} {{:>{hw}}} {{:>{nw}}} {{:>{nw}}} '
                          '{{:>{ew}}} {{:>{ew}}}').format(hw=hw, nw=nw, ew=ew)
        if options.spc:
            log.info(print_format_3.format("Structure", "Temp/K", "H_SPC", "T.S", "T.qh-S", "G(T)_SPC",
                                            "qh-G(T)_SPC"))
        else:
            log.info(print_format_3.format("Structure", "Temp/K", "H", "T.S", "T.qh-S", "G(T)", "qh-G(T)"),
)

    interval_bbe_data = []
    for h, file in enumerate(files):  # Temperature interval
        log.info("\n" + stars)
        interval_bbe_data.append([])
        for temp in interval:  # Iterate through the temperature range
            if gas_phase:
                conc = ATMOS / GAS_CONSTANT / temp
            else:
                conc = options.conc
            linear_warning = []
            # Look up cached QCData if available
            cached_qcdata = None
            if qcdata_cache is not None:
                key = os.path.splitext(os.path.basename(file))[0]
                cached_qcdata = qcdata_cache.get(key)
            bbe = calc_bbe(file, options.QS, options.QH, options.S_freq_cutoff, options.H_freq_cutoff, temp,
                           conc, options.freq_scale_factor, options.freespace, options.spc, options.invert,
                           inertia=options.inertia, qcdata=cached_qcdata)
            interval_bbe_data[h].append(bbe)
            linear_warning.append(bbe.linear_warning)
            if linear_warning == [['Warning! Potential invalid calculation of linear molecule from Gaussian.']]:
                log.info("\nx  ")
                log.info('{:<39}'.format(display_name(file)))
                log.info('             Warning! Potential invalid calculation of linear molecule from Gaussian ...')
            else:
                # Gaussian spc files
                if bbe.scf_energy is not None and not hasattr(bbe, "gibbs_free_energy"):
                    log.info("\nx  " + '{:<39}'.format(display_name(file)))
                # ORCA spc files
                elif bbe.scf_energy is None and not hasattr(bbe, "gibbs_free_energy"):
                    log.info("\nx  " + '{:<39}'.format(display_name(file)))
                if not hasattr(bbe, "gibbs_free_energy"):
                    log.info("Warning! Couldn't find frequency information ...")
                else:
                    log.info("\no  ")
                    log.info('{:<39} {:13.1f}'.format(display_name(file), temp),
        )
                    # if not options.media:
                    if all(getattr(bbe, attrib) for attrib in
                           ["enthalpy", "entropy", "qh_entropy", "gibbs_free_energy", "qh_gibbs_free_energy"]):
                        if options.QH:
                            log.info((' ' + hf + ' ' + ef + ' ' + nf + ' ' + nf + ' ' + ef + ' ' + ef).format(
                                bbe.enthalpy, bbe.qh_enthalpy, (temp * bbe.entropy),
                                (temp * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy),
          )
                        else:
                            log.info((' ' + hf + ' ' + nf + ' ' + nf + ' ' + ef + ' ' + ef).format(bbe.enthalpy, (
                                    temp * bbe.entropy), (temp * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy),
                )
                    if options.media is not None and options.media.lower() in solvents and options.media.lower() == \
                            display_name(file).lower():
                        log.info("  Solvent: {:4.2f}M ".format(media_conc))

        log.info("\n" + stars + "\n")

    return interval_bbe_data, interval, list(files)


def print_pes_results(thermo_data, options, dup_list,
                      boltz_facs=None, boltz_sum=None,
                      interval_bbe_data=None, interval=None, file_list=None):
    """Print relative PES energies from a YAML-defined reaction pathway.

    Reads the PES definition, computes relative energies with optional Boltzmann
    weighting and conformational corrections, and prints formatted tables.
    Optionally computes enantioselectivity and generates reaction profile graphs.

    Parameters:
        thermo_data (dict): file path → calc_bbe mapping.
        options (Namespace): parsed CLI options. Uses: dp, QH, spc, gconf,
            temperature_interval, pes, temperature, ee, graph.
        dup_list (list): pairs of duplicate/enantiomer file paths.
        boltz_facs (dict, optional): Boltzmann factors keyed by file path.
            Computed by get_boltz() in the orchestrator. None when --ee is off.
        boltz_sum (float, optional): total Boltzmann partition sum.
            None when --ee is off.
        interval_bbe_data (list, optional): per-file, per-temperature calc_bbe data.
        interval (range, optional): temperature steps for variable-T PES.
        file_list (list, optional): file list from temperature interval analysis.
    """
    files = list(thermo_data)
    dp = getattr(options, 'dp', 6)
    dw = dp - 6
    # Set up stars separator based on QH option and decimal places
    if options.QH:
        stars = "   " + "*" * (142 + 8 * dw)  # 8 energy columns with QH
    else:
        stars = "   " + "*" * (128 + 7 * dw)  # 7 energy columns without QH

    # Validate thermodynamic data once
    for key in thermo_data:
        if not hasattr(thermo_data[key], "qh_gibbs_free_energy"):
            pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
            sys.exit(pes_error)
        if not hasattr(thermo_data[key], "sp_energy") and options.spc is not None:
            pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
            sys.exit(pes_error)

    if options.gconf:
        log.info('\n   Gconf correction requested to be applied to below relative values using quasi-harmonic Boltzmann factors\n')

    # Interval applied to PES
    if options.temperature_interval:
            stars = stars + '*' * 22
            interval_thermo_data = [dict(zip(file_list, bbe_vals))
                                    for bbe_vals in zip(*interval_bbe_data)]
            j = 0
            for i in interval:
                temp = float(i)
                pes = get_pes(options.pes, interval_thermo_data[j], temp, options.gconf, options.QH)
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
                            relative = [s - z for s, z in zip(species, zero_vals)]
                            e_sum += math.exp(-relative[1] * J_TO_AU / GAS_CONSTANT / temp)
                            h_sum += math.exp(-relative[3] * J_TO_AU / GAS_CONSTANT / temp)
                            g_sum += math.exp(-relative[7] * J_TO_AU / GAS_CONSTANT / temp)
                            qhg_sum += math.exp(-relative[8] * J_TO_AU / GAS_CONSTANT / temp)
                    if options.spc is None:
                        log.info("\n   " + '{:<40}'.format("RXN: " + path + " (" + pes.units + ")  at T: " + str(temp)))
                        if options.QH:
                            log.info('{:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} '
                                      '{:>13}'.format(" DE", "DZPE", "DH", "qh-DH", "T.DS", "T.qh-DS", "DG(T)",
                                                      "qh-DG(T)"))
                        else:
                            log.info('{:>13} {:>10} {:>13} {:>10} {:>10} {:>13} '
                                      '{:>13}'.format(" DE", "DZPE", "DH", "T.DS", "T.qh-DS", "DG(T)", "qh-DG(T)"),
                )
                    else:
                        log.info("\n   " + '{:<40}'.format("RXN: " + path + " (" + pes.units + ")  at T: " +
                                                            str(temp)))
                        if options.QH:
                            log.info('{:>13} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>14} '
                                      '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "qh-DH_SPC", "T.DS",
                                                      "T.qh-DS", "DG(T)_SPC", "qh-DG(T)_SPC"))
                        else:
                            log.info('{:>13} {:>13} {:>10} {:>13} {:>10} {:>10} {:>14} '
                                      '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "T.DS", "T.qh-DS",
                                                      "DG(T)_SPC", "qh-DG(T)_SPC"))
                    log.info("\n" + stars)

                    for m, e_abs in enumerate(pes.e_abs[k]):
                        if options.QH:
                            species = [pes.spc_abs[k][m], pes.e_abs[k][m], pes.zpe_abs[k][m], pes.h_abs[k][m],
                                       pes.qh_abs[k][m], temp * pes.s_abs[k][m], temp * pes.qs_abs[k][m],
                                       pes.g_abs[k][m], pes.qhg_abs[k][m]]
                        else:
                            species = [pes.spc_abs[k][m], pes.e_abs[k][m], pes.zpe_abs[k][m], pes.h_abs[k][m],
                                       temp * pes.s_abs[k][m], temp * pes.qs_abs[k][m], pes.g_abs[k][m],
                                       pes.qhg_abs[k][m]]
                        relative = [s - z for s, z in zip(species, zero_vals)]
                        if pes.units == 'kJ/mol':
                            formatted_list = [J_TO_AU / 1000.0 * x for x in relative]
                        else:
                            formatted_list = [KCAL_TO_AU * x for x in relative]  # Defaults to kcal/mol
                        log.info("\no  ")
                        if options.spc is None:
                            formatted_list = formatted_list[1:]
                            if options.QH:
                                format_1 = '{:<39} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} {:13.1f}'
                                format_2 = '{:<39} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} {:13.2f}'
                            else:
                                format_1 = '{:<39} {:13.1f} {:10.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} {:13.1f}'
                                format_2 = '{:<39} {:13.2f} {:10.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} {:13.2f}'
                            if pes.dec == 1:
                                log.info(format_1.format(pes.species[k][m], *formatted_list))
                            if pes.dec == 2:
                                log.info(format_2.format(pes.species[k][m], *formatted_list))
                        else:
                            if options.QH:
                                if pes.dec == 1:
                                    log.info('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} {:13.1f}'.format(
                                            pes.species[k][m], *formatted_list))
                                if pes.dec == 2:
                                    log.info('{:<39} {:13.1f} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} {:13.2f}'.format(
                                            pes.species[k][m], *formatted_list))
                            else:
                                if pes.dec == 1:
                                    log.info('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} {:13.1f}'.format(
                                            pes.species[k][m], *formatted_list))
                                if pes.dec == 2:
                                    log.info('{:<39} {:13.2f} {:13.2f} {:10.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} {:13.2f}'.format(
                                            pes.species[k][m], *formatted_list))
                        if pes.boltz:
                            boltz = [math.exp(-relative[1] * J_TO_AU / GAS_CONSTANT / options.temperature) / e_sum,
                                     math.exp(-relative[3] * J_TO_AU / GAS_CONSTANT / options.temperature) / h_sum,
                                     math.exp(-relative[6] * J_TO_AU / GAS_CONSTANT / options.temperature) / g_sum,
                                     math.exp(-relative[7] * J_TO_AU / GAS_CONSTANT / options.temperature) / qhg_sum]
                            selectivity = [b * 100.0 for b in boltz]
                            log.info("\n  " + '{:<39} {:13.2f}%{:24.2f}%{:35.2f}%{:13.2f}%'.format('', *selectivity))
                            sels.append(selectivity)
                        formatted_list = [round(v, 6) for v in formatted_list]
                    if pes.boltz == 'ee' and len(sels) == 2:
                        ee = [a - b for a, b in zip(sels[0], sels[1])]
                        if options.spc is None:
                            log.info("\n" + stars + "\n   " + '{:<39} {:13.1f}%{:24.1f}%{:35.1f}%{:13.1f}%'.format('ee (%)',
                                                                                                              *ee))
                        else:
                            log.info("\n" + stars + "\n   " + '{:<39} {:27.1f} {:24.1f} {:35.1f} {:13.1f} '.format('ee (%)',
                                                                                                              *ee))
                    log.info("\n" + stars + "\n")
                j += 1
    else:
        pes = get_pes(options.pes, thermo_data, options.temperature, options.gconf, options.QH)
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
                    relative = [s - z for s, z in zip(species, zero_vals)]
                    e_sum += math.exp(-relative[1] * J_TO_AU / GAS_CONSTANT / options.temperature)
                    h_sum += math.exp(-relative[3] * J_TO_AU / GAS_CONSTANT / options.temperature)
                    g_sum += math.exp(-relative[7] * J_TO_AU / GAS_CONSTANT / options.temperature)
                    qhg_sum += math.exp(-relative[8] * J_TO_AU / GAS_CONSTANT / options.temperature)

            if options.spc is None:
                log.info("\n   " + '{:<40}'.format("RXN: " + path + " (" + pes.units + ") ", ))
                if options.QH:
                    log.info('{:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} '
                              '{:>13}'.format(" DE", "DZPE", "DH", "qh-DH", "T.DS", "T.qh-DS", "DG(T)", "qh-DG(T)"),
        )
                else:
                    log.info('{:>13} {:>10} {:>13} {:>10} {:>10} {:>13} '
                              '{:>13}'.format(" DE", "DZPE", "DH", "T.DS", "T.qh-DS", "DG(T)", "qh-DG(T)"),
        )
            else:
                log.info("\n   " + '{:<40}'.format("RXN: " + path + " (" + pes.units + ") ", ))
                if options.QH:
                    log.info('{:>13} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>14} '
                              '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "qh-DH_SPC", "T.DS", "T.qh-DS",
                                              "DG(T)_SPC", "qh-DG(T)_SPC"))
                else:
                    log.info('{:>13} {:>13} {:>10} {:>13} {:>10} {:>10} {:>14} '
                              '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "T.DS", "T.qh-DS", "DG(T)_SPC",
                                              "qh-DG(T)_SPC"))
            log.info("\n" + stars)

            for j, e_abs in enumerate(pes.e_abs[i]):
                if options.QH:
                    species = [pes.spc_abs[i][j], pes.e_abs[i][j], pes.zpe_abs[i][j], pes.h_abs[i][j],
                               pes.qh_abs[i][j], options.temperature * pes.s_abs[i][j],
                               options.temperature * pes.qs_abs[i][j], pes.g_abs[i][j], pes.qhg_abs[i][j]]
                else:
                    species = [pes.spc_abs[i][j], pes.e_abs[i][j], pes.zpe_abs[i][j], pes.h_abs[i][j],
                               options.temperature * pes.s_abs[i][j], options.temperature * pes.qs_abs[i][j],
                               pes.g_abs[i][j], pes.qhg_abs[i][j]]
                relative = [s - z for s, z in zip(species, zero_vals)]
                if pes.units == 'kJ/mol':
                    formatted_list = [J_TO_AU / 1000.0 * x for x in relative]
                else:
                    formatted_list = [KCAL_TO_AU * x for x in relative]  # Defaults to kcal/mol
                log.info("\no  ")
                if options.spc is None:
                    formatted_list = formatted_list[1:]
                    if options.QH:
                        if pes.dec == 1:
                            log.info('{:<39} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} '
                                      '{:13.1f}'.format(pes.species[i][j], *formatted_list))
                        if pes.dec == 2:
                            log.info('{:<39} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} '
                                      '{:13.2f}'.format(pes.species[i][j], *formatted_list))
                    else:
                        if pes.dec == 1:
                            log.info('{:<39} {:13.1f} {:10.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} '
                                      '{:13.1f}'.format(pes.species[i][j], *formatted_list))
                        if pes.dec == 2:
                            log.info('{:<39} {:13.2f} {:10.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} '
                                      '{:13.2f}'.format(pes.species[i][j], *formatted_list))
                else:
                    if options.QH:
                        if pes.dec == 1:
                            log.info('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} '
                                      '{:13.1f} {:13.1f}'.format(pes.species[i][j], *formatted_list),
                )
                        if pes.dec == 2:
                            log.info('{:<39} {:13.1f} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} '
                                      '{:13.2f} {:13.2f}'.format(pes.species[i][j], *formatted_list),
                )
                    else:
                        if pes.dec == 1:
                            log.info('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} '
                                      '{:13.1f}'.format(pes.species[i][j], *formatted_list))
                        if pes.dec == 2:
                            log.info('{:<39} {:13.2f} {:13.2f} {:10.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} '
                                      '{:13.2f}'.format(pes.species[i][j], *formatted_list))
                if pes.boltz:
                    boltz = [math.exp(-relative[1] * J_TO_AU / GAS_CONSTANT / options.temperature) / e_sum,
                             math.exp(-relative[3] * J_TO_AU / GAS_CONSTANT / options.temperature) / h_sum,
                             math.exp(-relative[6] * J_TO_AU / GAS_CONSTANT / options.temperature) / g_sum,
                             math.exp(-relative[7] * J_TO_AU / GAS_CONSTANT / options.temperature) / qhg_sum]
                    selectivity = [b * 100.0 for b in boltz]
                    log.info("\n  " + '{:<39} {:13.2f}%{:24.2f}%{:35.2f}%{:13.2f}%'.format('', *selectivity))
                    sels.append(selectivity)
                formatted_list = [round(v, 6) for v in formatted_list]
            if pes.boltz == 'ee' and len(sels) == 2:
                ee = [a - b for a, b in zip(sels[0], sels[1])]
                if options.spc is None:
                    log.info("\n" + stars + "\n   " + '{:<39} {:13.1f}%{:24.1f}%{:35.1f}%{:13.1f}%'.format('ee (%)', *ee))
                else:
                    log.info("\n" + stars + "\n   " + '{:<39} {:27.1f} {:24.1f} {:35.1f} {:13.1f} '.format('ee (%)', *ee))
            log.info("\n" + stars + "\n")

    # Compute enantiomeric excess
    if options.ee is not None:
        selec_stars = "   " + '*' * 109
        ee, er, ratio, dd_free_energy, failed, preference = get_selectivity(options.ee, files, boltz_facs, boltz_sum,
                                                                            options.temperature, dup_list)
        if not failed:
            log.info("\n   " + '{:<39} {:>13} {:>13} {:>13} {:>13} {:>13}'.format("Selectivity", "Excess (%)", "Ratio (%)", "Ratio", "Major Iso", "ddG"))
            log.info("\n" + selec_stars)
            log.info('\no {:<40} {:13.2f} {:>13} {:>13} {:>13} {:13.2f}'.format('', ee, er, ratio, preference,
                                                                                 dd_free_energy))
            log.info("\n" + selec_stars + "\n")
    # Graph reaction profiles
    if options.graph is not None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            log.info("\n\n   Warning! matplotlib module is not installed, reaction profile will not be graphed.")
            log.info("\n   To install matplotlib, run the following commands: \n\t   python -m pip install -U pip" +
                      "\n\t   python -m pip install -U matplotlib\n\n")
        for key in thermo_data:
            if not hasattr(thermo_data[key], "qh_gibbs_free_energy"):
                pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
                sys.exit(pes_error)
            if not hasattr(thermo_data[key], "sp_energy") and options.spc is not None:
                pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
                sys.exit(pes_error)

        graph_data = get_pes(options.graph, thermo_data, options.temperature, options.gconf, options.QH)
        graph_reaction_profile(graph_data, options, plt)
