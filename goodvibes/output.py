"""Output formatting and printing functions for GoodVibes."""
import math
import sys
from datetime import datetime
from string import ascii_lowercase as alphabet

from .utils import display_name, add_time
from .validation import check_dup
from .selectivity import get_boltz, get_selectivity
from .constants import GAS_CONSTANT, ATMOS, J_TO_AU, KCAL_TO_AU

try:
    from .pes import get_pes, graph_reaction_profile
    from .thermo import calc_bbe
    from .media import solvents
except ImportError:
    from pes import get_pes, graph_reaction_profile
    from thermo import calc_bbe
    from media import solvents


def print_results(files, thermo_data, options, log, stars, clustering, clusters, xyz=None, media_conc=None):
    """Print single-temperature thermochemistry output table."""
    # Check if user has chosen to make any low lying imaginary frequencies positive
    inverted_freqs, inverted_files = [], []
    for file in files:
        if len(thermo_data[file].inverted_freqs) > 0:
            inverted_freqs.append(thermo_data[file].inverted_freqs)
            inverted_files.append(file)
    if options.invert is not False:
        for i, file in enumerate(inverted_files):
            if len(inverted_freqs[i]) == 1:
                log.write("\n\n   The following frequency was made positive and used in calculations: " +
                          str(inverted_freqs[i][0]) + " from " + file)
            elif len(inverted_freqs[i]) > 1:
                log.write("\n\n   The following frequencies were made positive and used in calculations: " +
                          str(inverted_freqs[i]) + " from " + file)

    # Adjust printing according to options requested
    if options.spc is not False:
        stars += '*' * 14
    if options.imag_freq is True:
        stars += '*' * 9
    if options.boltz is True:
        stars += '*' * 7
    if options.ssymm is True:
        stars += '*' * 13

    total_cpu_time, add_days = datetime(100, 1, 1, 00, 00, 00, 00), 0

    if options.spc is False:
        log.write("\n\n   ")
        if options.QH:
            log.write('{:<39} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} '
                      '{:>13}'.format("Structure", "E", "ZPE", "H", "qh-H", "T.S", "T.qh-S", "G(T)", "qh-G(T)"),
                      thermodata=True)
        else:
            log.write('{:<39} {:>13} {:>10} {:>13} {:>10} {:>10} {:>13} {:>13}'.format("Structure", "E", "ZPE", "H",
                                                                                       "T.S", "T.qh-S", "G(T)",
                                                                                       "qh-G(T)"), thermodata=True)
    else:
        log.write("\n\n   ")
        if options.QH:
            log.write('{:<39} {:>13} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} '
                      '{:>13}'.format("Structure", "E_SPC", "E", "ZPE", "H_SPC", "qh-H_SPC", "T.S", "T.qh-S",
                                      "G(T)_SPC", "qh-G(T)_SPC"), thermodata=True)
        else:
            log.write('{:<39} {:>13} {:>13} {:>10} {:>13} {:>10} {:>10} {:>13} '
                      '{:>13}'.format("Structure", "E_SPC", "E", "ZPE", "H_SPC", "T.S", "T.qh-S", "G(T)_SPC",
                                      "qh-G(T)_SPC"), thermodata=True)
    if options.boltz is True:
        log.write('{:>7}'.format("Boltz"), thermodata=True)
    if options.imag_freq is True:
        log.write('{:>9}'.format("im freq"), thermodata=True)
    if options.ssymm:
        log.write('{:>13}'.format("Point Group"), thermodata=True)
    log.write("\n" + stars + "")

    # Look for duplicates or enantiomers
    if options.duplicate:
        dup_list = check_dup(files, thermo_data)
    else:
        dup_list = []

    # Boltzmann factors and averaging over clusters
    boltz_facs, boltz_sum = None, None
    if options.boltz:
        boltz_facs, weighted_free_energy, boltz_sum = get_boltz(files, thermo_data, clustering, clusters,
                                                                options.temperature, dup_list)

    for file in files:  # Loop over the output files and compute thermochemistry
        duplicate = False
        if len(dup_list) != 0:
            for dup in dup_list:
                if dup[0] == file:
                    duplicate = True
                    log.write('\nx  {} is a duplicate or enantiomer of {}'.format(dup[0].rsplit('.', 1)[0],
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

            if xyz:  # Write Cartesians
                xyz.write_text(str(len(bbe.atom_types)))
                if bbe.scf_energy is not None:
                    xyz.write_text(
                        '{:<39} {:>13} {:13.6f}'.format(display_name(file), 'Eopt', bbe.scf_energy))
                else:
                    xyz.write_text('{:<39}'.format(display_name(file)))
                if bbe.atom_types and bbe.cartesians:
                    xyz.write_coords(bbe.atom_types, bbe.cartesians)

            # Check for possible error in Gaussian calculation of linear molecules which can return 2 rotational constants instead of 3
            if bbe.linear_warning:
                log.write("\nx  " + '{:<39}'.format(display_name(file)))
                log.write('          ----   Caution! Potential invalid calculation of linear molecule from Gaussian')
            else:
                if hasattr(bbe, "gibbs_free_energy"):
                    if options.spc is not False:
                        if bbe.sp_energy != '!':
                            log.write("\no  ")
                            log.write('{:<39}'.format(display_name(file)), thermodata=True)
                            log.write(' {:13.6f}'.format(bbe.sp_energy), thermodata=True)
                        if bbe.sp_energy == '!':
                            log.write("\nx  ")
                            log.write('{:<39}'.format(display_name(file)), thermodata=True)
                            log.write(' {:>13}'.format('----'), thermodata=True)
                    else:
                        log.write("\no  ")
                        log.write('{:<39}'.format(display_name(file)), thermodata=True)
                # Gaussian SPC file handling
                if bbe.scf_energy is not None and not hasattr(bbe, "gibbs_free_energy"):
                    log.write("\nx  " + '{:<39}'.format(display_name(file)))
                # ORCA spc files
                elif bbe.scf_energy is None and not hasattr(bbe, "gibbs_free_energy"):
                    log.write("\nx  " + '{:<39}'.format(display_name(file)))
                if bbe.scf_energy is not None:
                    log.write(' {:13.6f}'.format(bbe.scf_energy), thermodata=True)
                # No freqs found
                if not hasattr(bbe, "gibbs_free_energy"):
                    log.write("   Warning! Couldn't find frequency information ...")
                else:
                    if all(getattr(bbe, attrib) for attrib in
                           ["enthalpy", "entropy", "qh_entropy", "gibbs_free_energy", "qh_gibbs_free_energy"]):
                        if options.QH:
                            log.write(' {:10.6f} {:13.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(
                                bbe.zpe, bbe.enthalpy, bbe.qh_enthalpy, (options.temperature * bbe.entropy),
                                (options.temperature * bbe.qh_entropy), bbe.gibbs_free_energy,
                                bbe.qh_gibbs_free_energy), thermodata=True)
                        else:
                            log.write(' {:10.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} '
                                      '{:13.6f}'.format(bbe.zpe, bbe.enthalpy,
                                                        (options.temperature * bbe.entropy),
                                                        (options.temperature * bbe.qh_entropy),
                                                        bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy),
                                      thermodata=True)

                    if options.media is not False and options.media.lower() in solvents and options.media.lower() == \
                            display_name(file).lower():
                        log.write("  Solvent: {:4.2f}M ".format(media_conc))

            # Append requested options to end of output
            if options.boltz is True:
                log.write('{:7.3f}'.format(boltz_facs[file] / boltz_sum), thermodata=True)
            if options.imag_freq is True and hasattr(bbe, "im_frequency_wn"):
                for freq in bbe.im_frequency_wn:
                    log.write('{:9.2f}'.format(freq), thermodata=True)
            if options.ssymm:
                if hasattr(bbe, "qh_gibbs_free_energy"):
                    log.write('{:>13}'.format(bbe.point_group))
                else:
                    log.write('{:>37}'.format('---'))
        # Cluster files if requested
        if clustering:
            dashes = "-" * (len(stars) - 3)
            for n, cluster in enumerate(clusters):
                for id, structure in enumerate(cluster):
                    if structure == file:
                        if id == len(cluster) - 1:
                            log.write("\n   " + dashes)
                            log.write("\n   " + '{name:<{var_width}} {gval:13.6f} {weight:6.2f}'.format(
                                name='Boltzmann-weighted Cluster ' + alphabet[n].upper(), var_width=len(stars) - 24,
                                gval=weighted_free_energy['cluster-' + alphabet[n].upper()] / boltz_facs[
                                    'cluster-' + alphabet[n].upper()],
                                weight=100 * boltz_facs['cluster-' + alphabet[n].upper()] / boltz_sum),
                                      thermodata=True)
                            log.write("\n   " + dashes)
    log.write("\n" + stars + "\n")
    return stars, dup_list, total_cpu_time, add_days


def print_temperature_interval(files, options, log, stars, gas_phase, media_conc=None):
    """Run variable-temperature analysis and print results. Returns (interval_bbe_data, interval, file_list)."""
    log.write("\n\n   Variable-Temperature analysis of the enthalpy, entropy and the entropy at a constant pressure between")
    temperature_interval = [float(temp) for temp in options.temperature_interval.split(',')]
    # If no temperature step was defined, divide the region into 10
    if len(temperature_interval) == 2:
        temperature_interval.append((temperature_interval[1] - temperature_interval[0]) / 10.0)
    interval = range(int(temperature_interval[0]), int(temperature_interval[1] + 1),
                     int(temperature_interval[2]))
    log.write("\n   T init:  %.1f,  T final:  %.1f,  T interval: %.1f" % (
        temperature_interval[0], temperature_interval[1], temperature_interval[2]))
    if options.QH:
        qh_print_format = "\n\n   {:<39} {:>13} {:>24} {:>13} {:>10} {:>10} {:>13} {:>13}"
        if options.spc:
            log.write(qh_print_format.format("Structure", "Temp/K", "H_SPC", "qh-H_SPC", "T.S", "T.qh-S",
                                             "G(T)_SPC", "qh-G(T)_SPC"), thermodata=True)
        else:
            log.write(qh_print_format.format("Structure", "Temp/K", "H", "qh-H", "T.S", "T.qh-S", "G(T)",
                                             "qh-G(T)"), thermodata=True)
    else:
        print_format_3 = '\n\n   {:<39} {:>13} {:>24} {:>10} {:>10} {:>13} {:>13}'
        if options.spc:
            log.write(print_format_3.format("Structure", "Temp/K", "H_SPC", "T.S", "T.qh-S", "G(T)_SPC",
                                            "qh-G(T)_SPC"), thermodata=True)
        else:
            log.write(print_format_3.format("Structure", "Temp/K", "H", "T.S", "T.qh-S", "G(T)", "qh-G(T)"),
                      thermodata=True)

    interval_bbe_data = []
    for h, file in enumerate(files):  # Temperature interval
        log.write("\n" + stars)
        interval_bbe_data.append([])
        for i in range(len(interval)):  # Iterate through the temperature range
            temp = interval[i]
            if gas_phase:
                conc = ATMOS / GAS_CONSTANT / temp
            else:
                conc = options.conc
            linear_warning = []
            bbe = calc_bbe(file, options.QS, options.QH, options.S_freq_cutoff, options.H_freq_cutoff, temp,
                           conc, options.freq_scale_factor, options.freespace, options.spc, options.invert,
                           inertia=options.inertia)
            interval_bbe_data[h].append(bbe)
            linear_warning.append(bbe.linear_warning)
            if linear_warning == [['Warning! Potential invalid calculation of linear molecule from Gaussian.']]:
                log.write("\nx  ")
                log.write('{:<39}'.format(display_name(file)), thermodata=True)
                log.write('             Warning! Potential invalid calculation of linear molecule from Gaussian ...')
            else:
                # Gaussian spc files
                if bbe.scf_energy is not None and not hasattr(bbe, "gibbs_free_energy"):
                    log.write("\nx  " + '{:<39}'.format(display_name(file)))
                # ORCA spc files
                elif bbe.scf_energy is None and not hasattr(bbe, "gibbs_free_energy"):
                    log.write("\nx  " + '{:<39}'.format(display_name(file)))
                if not hasattr(bbe, "gibbs_free_energy"):
                    log.write("Warning! Couldn't find frequency information ...")
                else:
                    log.write("\no  ")
                    log.write('{:<39} {:13.1f}'.format(display_name(file), temp),
                              thermodata=True)
                    # if not options.media:
                    if all(getattr(bbe, attrib) for attrib in
                           ["enthalpy", "entropy", "qh_entropy", "gibbs_free_energy", "qh_gibbs_free_energy"]):
                        if options.QH:
                            log.write(' {:24.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(
                                bbe.enthalpy, bbe.qh_enthalpy, (temp * bbe.entropy),
                                (temp * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy),
                                thermodata=True)
                        else:
                            log.write(' {:24.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(bbe.enthalpy, (
                                    temp * bbe.entropy), (temp * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy),
                                      thermodata=True)
                    if options.media is not False and options.media.lower() in solvents and options.media.lower() == \
                            display_name(file).lower():
                        log.write("  Solvent: {:4.2f}M ".format(media_conc))

        log.write("\n" + stars + "\n")

    return interval_bbe_data, interval, list(files)


def print_pes_results(files, thermo_data, options, log, stars, clustering, clusters, dup_list,
                      interval_bbe_data=None, interval=None, file_list=None):
    """Tabulate relative PES values, Boltzmann weighting, and selectivity."""
    if options.gconf:
        log.write('\n   Gconf correction requested to be applied to below relative values using quasi-harmonic Boltzmann factors\n')
    for key in thermo_data:
        if not hasattr(thermo_data[key], "qh_gibbs_free_energy"):
            pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
            sys.exit(pes_error)
        if not hasattr(thermo_data[key], "sp_energy") and options.spc is not False:
            pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
            sys.exit(pes_error)
    # Interval applied to PES
    if options.temperature_interval:
        if options.gconf:
            log.write('\n   Gconf correction requested to be applied to below relative values using quasi-harmonic Boltzmann factors\n')
        for key in thermo_data:
            if not hasattr(thermo_data[key], "qh_gibbs_free_energy"):
                pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
                sys.exit(pes_error)
            if not hasattr(thermo_data[key], "sp_energy") and options.spc is not False:
                pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
                sys.exit(pes_error)
        # Interval applied to PES
        if options.temperature_interval:
            stars = stars + '*' * 22
            interval_thermo_data = []
            for i in range(len(interval)):
                bbe_vals = []
                for j in range(len(interval_bbe_data)):
                    bbe_vals.append(interval_bbe_data[j][i])
                interval_thermo_data.append(dict(zip(file_list, bbe_vals)))
            j = 0
            for i in interval:
                temp = float(i)
                pes = get_pes(options.pes, interval_thermo_data[j], log, temp, options.gconf, options.QH)
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
                            relative = [species[x] - zero_vals[x] for x in range(len(zero_vals))]
                            e_sum += math.exp(-relative[1] * J_TO_AU / GAS_CONSTANT / temp)
                            h_sum += math.exp(-relative[3] * J_TO_AU / GAS_CONSTANT / temp)
                            g_sum += math.exp(-relative[7] * J_TO_AU / GAS_CONSTANT / temp)
                            qhg_sum += math.exp(-relative[8] * J_TO_AU / GAS_CONSTANT / temp)
                    if options.spc is False:
                        log.write("\n   " + '{:<40}'.format("RXN: " + path + " (" + pes.units + ")  at T: " + str(temp)))
                        if options.QH:
                            log.write('{:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} '
                                      '{:>13}'.format(" DE", "DZPE", "DH", "qh-DH", "T.DS", "T.qh-DS", "DG(T)",
                                                      "qh-DG(T)"), thermodata=True)
                        else:
                            log.write('{:>13} {:>10} {:>13} {:>10} {:>10} {:>13} '
                                      '{:>13}'.format(" DE", "DZPE", "DH", "T.DS", "T.qh-DS", "DG(T)", "qh-DG(T)"),
                                      thermodata=True)
                    else:
                        log.write("\n   " + '{:<40}'.format("RXN: " + path + " (" + pes.units + ")  at T: " +
                                                            str(temp)))
                        if options.QH:
                            log.write('{:>13} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>14} '
                                      '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "qh-DH_SPC", "T.DS",
                                                      "T.qh-DS", "DG(T)_SPC", "qh-DG(T)_SPC"), thermodata=True)
                        else:
                            log.write('{:>13} {:>13} {:>10} {:>13} {:>10} {:>10} {:>14} '
                                      '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "T.DS", "T.qh-DS",
                                                      "DG(T)_SPC", "qh-DG(T)_SPC"), thermodata=True)
                    log.write("\n" + stars)

                    for m, e_abs in enumerate(pes.e_abs[k]):
                        if options.QH:
                            species = [pes.spc_abs[k][m], pes.e_abs[k][m], pes.zpe_abs[k][m], pes.h_abs[k][m],
                                       pes.qh_abs[k][m], temp * pes.s_abs[k][m], temp * pes.qs_abs[k][m],
                                       pes.g_abs[k][m], pes.qhg_abs[k][m]]
                        else:
                            species = [pes.spc_abs[k][m], pes.e_abs[k][m], pes.zpe_abs[k][m], pes.h_abs[k][m],
                                       temp * pes.s_abs[k][m], temp * pes.qs_abs[k][m], pes.g_abs[k][m],
                                       pes.qhg_abs[k][m]]
                        relative = [species[x] - zero_vals[x] for x in range(len(zero_vals))]
                        if pes.units == 'kJ/mol':
                            formatted_list = [J_TO_AU / 1000.0 * x for x in relative]
                        else:
                            formatted_list = [KCAL_TO_AU * x for x in relative]  # Defaults to kcal/mol
                        log.write("\no  ")
                        if options.spc is False:
                            formatted_list = formatted_list[1:]
                            if options.QH:
                                format_1 = '{:<39} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} {:13.1f}'
                                format_2 = '{:<39} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} {:13.2f}'
                            else:
                                format_1 = '{:<39} {:13.1f} {:10.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} {:13.1f}'
                                format_2 = '{:<39} {:13.2f} {:10.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} {:13.2f}'
                            if pes.dec == 1:
                                log.write(format_1.format(pes.species[k][m], *formatted_list), thermodata=True)
                            if pes.dec == 2:
                                log.write(format_2.format(pes.species[k][m], *formatted_list), thermodata=True)
                        else:
                            if options.QH:
                                if pes.dec == 1:
                                    log.write('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} {:13.1f}'.format(
                                            pes.species[k][m], *formatted_list), thermodata=True)
                                if pes.dec == 2:
                                    log.write('{:<39} {:13.1f} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} {:13.2f}'.format(
                                            pes.species[k][m], *formatted_list), thermodata=True)
                            else:
                                if pes.dec == 1:
                                    log.write('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} {:13.1f}'.format(
                                            pes.species[k][m], *formatted_list), thermodata=True)
                                if pes.dec == 2:
                                    log.write('{:<39} {:13.2f} {:13.2f} {:10.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} {:13.2f}'.format(
                                            pes.species[k][m], *formatted_list), thermodata=True)
                        if pes.boltz:
                            boltz = [math.exp(-relative[1] * J_TO_AU / GAS_CONSTANT / options.temperature) / e_sum,
                                     math.exp(-relative[3] * J_TO_AU / GAS_CONSTANT / options.temperature) / h_sum,
                                     math.exp(-relative[6] * J_TO_AU / GAS_CONSTANT / options.temperature) / g_sum,
                                     math.exp(-relative[7] * J_TO_AU / GAS_CONSTANT / options.temperature) / qhg_sum]
                            selectivity = [boltz[x] * 100.0 for x in range(len(boltz))]
                            log.write("\n  " + '{:<39} {:13.2f}%{:24.2f}%{:35.2f}%{:13.2f}%'.format('', *selectivity))
                            sels.append(selectivity)
                        formatted_list = [round(formatted_list[x], 6) for x in range(len(formatted_list))]
                    if pes.boltz == 'ee' and len(sels) == 2:
                        ee = [sels[0][x] - sels[1][x] for x in range(len(sels[0]))]
                        if options.spc is False:
                            log.write("\n" + stars + "\n   " + '{:<39} {:13.1f}%{:24.1f}%{:35.1f}%{:13.1f}%'.format('ee (%)',
                                                                                                              *ee))
                        else:
                            log.write("\n" + stars + "\n   " + '{:<39} {:27.1f} {:24.1f} {:35.1f} {:13.1f} '.format('ee (%)',
                                                                                                              *ee))
                    log.write("\n" + stars + "\n")
                j += 1
    else:
        pes = get_pes(options.pes, thermo_data, log, options.temperature, options.gconf, options.QH)
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
                    relative = [species[x] - zero_vals[x] for x in range(len(zero_vals))]
                    e_sum += math.exp(-relative[1] * J_TO_AU / GAS_CONSTANT / options.temperature)
                    h_sum += math.exp(-relative[3] * J_TO_AU / GAS_CONSTANT / options.temperature)
                    g_sum += math.exp(-relative[7] * J_TO_AU / GAS_CONSTANT / options.temperature)
                    qhg_sum += math.exp(-relative[8] * J_TO_AU / GAS_CONSTANT / options.temperature)

            if options.spc is False:
                log.write("\n   " + '{:<40}'.format("RXN: " + path + " (" + pes.units + ") ", ))
                if options.QH:
                    log.write('{:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} '
                              '{:>13}'.format(" DE", "DZPE", "DH", "qh-DH", "T.DS", "T.qh-DS", "DG(T)", "qh-DG(T)"),
                              thermodata=True)
                else:
                    log.write('{:>13} {:>10} {:>13} {:>10} {:>10} {:>13} '
                              '{:>13}'.format(" DE", "DZPE", "DH", "T.DS", "T.qh-DS", "DG(T)", "qh-DG(T)"),
                              thermodata=True)
            else:
                log.write("\n   " + '{:<40}'.format("RXN: " + path + " (" + pes.units + ") ", ))
                if options.QH:
                    log.write('{:>13} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>14} '
                              '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "qh-DH_SPC", "T.DS", "T.qh-DS",
                                              "DG(T)_SPC", "qh-DG(T)_SPC"), thermodata=True)
                else:
                    log.write('{:>13} {:>13} {:>10} {:>13} {:>10} {:>10} {:>14} '
                              '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "T.DS", "T.qh-DS", "DG(T)_SPC",
                                              "qh-DG(T)_SPC"), thermodata=True)
            log.write("\n" + stars)

            for j, e_abs in enumerate(pes.e_abs[i]):
                if options.QH:
                    species = [pes.spc_abs[i][j], pes.e_abs[i][j], pes.zpe_abs[i][j], pes.h_abs[i][j],
                               pes.qh_abs[i][j], options.temperature * pes.s_abs[i][j],
                               options.temperature * pes.qs_abs[i][j], pes.g_abs[i][j], pes.qhg_abs[i][j]]
                else:
                    species = [pes.spc_abs[i][j], pes.e_abs[i][j], pes.zpe_abs[i][j], pes.h_abs[i][j],
                               options.temperature * pes.s_abs[i][j], options.temperature * pes.qs_abs[i][j],
                               pes.g_abs[i][j], pes.qhg_abs[i][j]]
                relative = [species[x] - zero_vals[x] for x in range(len(zero_vals))]
                if pes.units == 'kJ/mol':
                    formatted_list = [J_TO_AU / 1000.0 * x for x in relative]
                else:
                    formatted_list = [KCAL_TO_AU * x for x in relative]  # Defaults to kcal/mol
                log.write("\no  ")
                if options.spc is False:
                    formatted_list = formatted_list[1:]
                    if options.QH:
                        if pes.dec == 1:
                            log.write('{:<39} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} '
                                      '{:13.1f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                        if pes.dec == 2:
                            log.write('{:<39} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} '
                                      '{:13.2f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                    else:
                        if pes.dec == 1:
                            log.write('{:<39} {:13.1f} {:10.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} '
                                      '{:13.1f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                        if pes.dec == 2:
                            log.write('{:<39} {:13.2f} {:10.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} '
                                      '{:13.2f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                else:
                    if options.QH:
                        if pes.dec == 1:
                            log.write('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} '
                                      '{:13.1f} {:13.1f}'.format(pes.species[i][j], *formatted_list),
                                      thermodata=True)
                        if pes.dec == 2:
                            log.write('{:<39} {:13.1f} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} '
                                      '{:13.2f} {:13.2f}'.format(pes.species[i][j], *formatted_list),
                                      thermodata=True)
                    else:
                        if pes.dec == 1:
                            log.write('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} '
                                      '{:13.1f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                        if pes.dec == 2:
                            log.write('{:<39} {:13.2f} {:13.2f} {:10.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} '
                                      '{:13.2f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                if pes.boltz:
                    boltz = [math.exp(-relative[1] * J_TO_AU / GAS_CONSTANT / options.temperature) / e_sum,
                             math.exp(-relative[3] * J_TO_AU / GAS_CONSTANT / options.temperature) / h_sum,
                             math.exp(-relative[6] * J_TO_AU / GAS_CONSTANT / options.temperature) / g_sum,
                             math.exp(-relative[7] * J_TO_AU / GAS_CONSTANT / options.temperature) / qhg_sum]
                    selectivity = [boltz[x] * 100.0 for x in range(len(boltz))]
                    log.write("\n  " + '{:<39} {:13.2f}%{:24.2f}%{:35.2f}%{:13.2f}%'.format('', *selectivity))
                    sels.append(selectivity)
                formatted_list = [round(formatted_list[x], 6) for x in range(len(formatted_list))]
            if pes.boltz == 'ee' and len(sels) == 2:
                ee = [sels[0][x] - sels[1][x] for x in range(len(sels[0]))]
                if options.spc is False:
                    log.write("\n" + stars + "\n   " + '{:<39} {:13.1f}%{:24.1f}%{:35.1f}%{:13.1f}%'.format('ee (%)', *ee))
                else:
                    log.write("\n" + stars + "\n   " + '{:<39} {:27.1f} {:24.1f} {:35.1f} {:13.1f} '.format('ee (%)', *ee))
            log.write("\n" + stars + "\n")

    # Compute enantiomeric excess
    if options.ee is not False:
        selec_stars = "   " + '*' * 109
        boltz_facs, weighted_free_energy, boltz_sum = get_boltz(files, thermo_data, clustering, clusters,
                                                                options.temperature, dup_list)
        ee, er, ratio, dd_free_energy, failed, preference = get_selectivity(options.ee, files, boltz_facs, boltz_sum,
                                                                            options.temperature, log, dup_list)
        if not failed:
            log.write("\n   " + '{:<39} {:>13} {:>13} {:>13} {:>13} {:>13}'.format("Selectivity", "Excess (%)", "Ratio (%)", "Ratio", "Major Iso", "ddG"), thermodata=True)
            log.write("\n" + selec_stars)
            log.write('\no {:<40} {:13.2f} {:>13} {:>13} {:>13} {:13.2f}'.format('', ee, er, ratio, preference,
                                                                                 dd_free_energy), thermodata=True)
            log.write("\n" + selec_stars + "\n")
    # Graph reaction profiles
    if options.graph is not False:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            log.write("\n\n   Warning! matplotlib module is not installed, reaction profile will not be graphed.")
            log.write("\n   To install matplotlib, run the following commands: \n\t   python -m pip install -U pip" +
                      "\n\t   python -m pip install -U matplotlib\n\n")
        for key in thermo_data:
            if not hasattr(thermo_data[key], "qh_gibbs_free_energy"):
                pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
                sys.exit(pes_error)
            if not hasattr(thermo_data[key], "sp_energy") and options.spc is not False:
                pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
                sys.exit(pes_error)

        graph_data = get_pes(options.graph, thermo_data, log, options.temperature, options.gconf, options.QH)
        graph_reaction_profile(graph_data, log, options, plt)
