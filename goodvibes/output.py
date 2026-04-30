"""Output formatting and printing functions for GoodVibes."""
import logging
import math
import os.path
import sys
from datetime import datetime

from .utils import display_name, add_time, get_console_stdout, get_console_dat
from .selectivity import get_selectivity
from .constants import GAS_CONSTANT, ATMOS, J_TO_AU, KCAL_TO_AU

from .pes import get_pes, graph_reaction_profile
from .thermo import calc_bbe
from .media import solvents

try:
    from rich.table import Table
    from rich import box as rich_box
except ImportError:
    Table = None
    rich_box = None

log = logging.getLogger('goodvibes')


def _print_rich_table(table: "Table") -> None:
    """Print a Rich Table to both stdout and .dat file consoles."""
    if Table is not None:
        get_console_stdout().print(table)
        get_console_dat().print(table)


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


def _build_results_table(options) -> "Table":
    """Build a Rich Table with columns for thermochemistry results.

    Column layout is determined by the options (QH, spc, boltz, symm, imag_freq).
    Returns a Table configured but with no rows yet.
    """
    if Table is None:
        return None

    dp = getattr(options, 'dp', 6)
    dw = dp - 6
    ew = 13 + dw  # wide numeric column
    nw = 10 + dw  # narrow numeric column

    table = Table(
        box=rich_box.SIMPLE_HEAD,
        show_header=True,
        show_edge=False,
        pad_edge=False,
        highlight=False,
        header_style="",  # Disable bold header styling
    )

    # Status column (o/x markers)
    table.add_column("", width=3)
    table.add_column("Structure", min_width=39, no_wrap=True)

    if options.spc is not None:
        table.add_column("E_SPC", min_width=ew, justify="right")
    table.add_column("E", min_width=ew, justify="right")
    table.add_column("ZPE", min_width=nw, justify="right")

    if options.QH:
        table.add_column("H", min_width=ew, justify="right")
        table.add_column("qh-H", min_width=ew, justify="right")
    else:
        table.add_column("H", min_width=ew, justify="right")

    table.add_column("T.S", min_width=nw, justify="right")
    table.add_column("T.qh-S", min_width=nw, justify="right")

    if options.spc is not None:
        table.add_column("G(T)_SPC", min_width=ew, justify="right")
        table.add_column("qh-G(T)_SPC", min_width=ew, justify="right")
    else:
        table.add_column("G(T)", min_width=ew, justify="right")
        table.add_column("qh-G(T)", min_width=ew, justify="right")

    if options.boltz:
        table.add_column("Boltz", min_width=7, justify="right")
    if options.symm or options.pg:
        table.add_column("Point Group", min_width=13, justify="right")
    if options.imag_freq is True:
        table.add_column("im freq", min_width=9, justify="right")

    return table


def print_results(thermo_data, options, media_conc=None,
                  dup_list=None, boltz_facs=None):
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
    """
    files = list(thermo_data)
    dp = getattr(options, 'dp', 6)
    dw = dp - 6

    # Format helpers for legacy fallback
    ef = '{{:{w}.{d}f}}'.format(w=13 + dw, d=dp)
    nf = '{{:{w}.{d}f}}'.format(w=10 + dw, d=dp)

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

    total_cpu_time, add_days = datetime(100, 1, 1, 00, 00, 00, 00), 0

    # Build the Rich table
    table = _build_results_table(options)
    if table is None:
        # Fallback if Rich is not available (shouldn't happen, but safe)
        log.info("\n\nWarning: Rich not available, falling back to ASCII output")
        return

    # Default dup_list if not passed by caller
    if dup_list is None:
        dup_list = []

    for file in files:
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
            if options.cputime:
                if hasattr(bbe, "cpu") and bbe.cpu is not None:
                    total_cpu_time = add_time(total_cpu_time, bbe.cpu)
                if hasattr(bbe, "sp_cpu") and bbe.sp_cpu is not None:
                    total_cpu_time = add_time(total_cpu_time, bbe.sp_cpu)
            if total_cpu_time.month > 1:
                add_days += 31

            # Handle linear molecule warning separately (not a table row)
            if bbe.linear_warning:
                log.info('\nx  {} — Caution! Potential invalid linear molecule calculation in Gaussian'.format(
                    display_name(file)))
                continue

            # Build row for this file
            row = []
            marker = "o" if hasattr(bbe, "gibbs_free_energy") else "x"
            name = display_name(file)

            # No gibbs free energy computed
            if not hasattr(bbe, "gibbs_free_energy"):
                row = [marker, name, "Warning! Couldn't find frequency information"]
                # Pad with empty cells to match other rows
                num_cols = len(table.columns) - 2  # -2 for marker and name
                row.extend([""] * num_cols)
                table.add_row(*row)
                continue

            row.append(marker)
            row.append(name)

            # E_SPC column (if applicable)
            if options.spc is not None:
                if bbe.sp_energy != '!':
                    row.append(f"{bbe.sp_energy:.{dp}f}")
                else:
                    row.append("----")

            # E column
            if bbe.scf_energy is not None:
                row.append(f"{bbe.scf_energy:.{dp}f}")
            else:
                row.append("----")

            # ZPE column
            row.append(f"{bbe.zpe:.{dp-3}f}")  # narrow format

            # H and optionally qh-H
            row.append(f"{bbe.enthalpy:.{dp}f}")
            if options.QH:
                row.append(f"{bbe.qh_enthalpy:.{dp}f}")

            # T.S and T.qh-S
            row.append(f"{options.temperature * bbe.entropy:.{dp-3}f}")  # narrow format
            row.append(f"{options.temperature * bbe.qh_entropy:.{dp-3}f}")  # narrow format

            # G(T) and qh-G(T) (or with _SPC suffix)
            row.append(f"{bbe.gibbs_free_energy:.{dp}f}")
            row.append(f"{bbe.qh_gibbs_free_energy:.{dp}f}")

            # Optional columns
            if options.boltz:
                row.append(f"{boltz_facs[file]:.3f}")
            if options.symm or options.pg:
                pg = getattr(bbe, "point_group", None) or "---"
                row.append(pg)
            if options.imag_freq is True:
                if hasattr(bbe, "im_frequency_wn") and bbe.im_frequency_wn:
                    freq_str = " ".join(f"{f:.2f}" for f in bbe.im_frequency_wn)
                    row.append(freq_str)
                else:
                    row.append("")

            table.add_row(*row)

            # Media annotation (print after table, not in it)
            if options.media is not None and options.media.lower() in solvents and options.media.lower() == \
                    display_name(file).lower():
                log.info("  Solvent: {:4.2f}M ".format(media_conc))

    # Print the table
    _print_rich_table(table)


def print_temperature_interval(thermo_data, options, media_conc=None, qcdata_cache=None):
    """Recompute thermochemistry across a temperature range and print results.

    Re-runs calc_bbe at each temperature step for every file, printing enthalpy,
    entropy, and free energy at each point.

    Parameters:
        thermo_data (dict): file path → calc_bbe mapping (used for file list).
        options (Namespace): parsed CLI options. Uses: dp, QH, QS, S_freq_cutoff,
            H_freq_cutoff, spc, conc, freq_scale_factor, freespace, invert,
            inertia, media, temperature_interval.
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
            conc = options.conc if options.conc else ATMOS / GAS_CONSTANT / temp
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
                      boltz_facs=None,
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
        ee, er, ratio, dd_free_energy, failed, preference = get_selectivity(options.ee, files, boltz_facs,
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
