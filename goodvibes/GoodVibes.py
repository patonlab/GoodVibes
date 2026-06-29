#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Quasi-harmonic thermochemical corrections for electronic structure calculations."""
import os.path
import sys
from glob import glob
from argparse import ArgumentParser

from .vib_scale_factors import scaling_data_dict, scaling_refs, canonicalize_level
from .io import write_xyz, load_cache, save_cache, qcdata_to_dict, find_spc_file
from .thermo import calc_bbe, get_free_space, FREESPACE_SOLVENTS
from .media import solvents, compute_media_conc, lookup_solvent
from .constants import (
    SUPPORTED_EXTENSIONS, GAS_CONSTANT, ATMOS,
    grimme_mRRHO_ref, grimme_msRRHO_ref, truhlar_ref, head_gordon_ref,
    oniom_scale_ref, gv_banner
)
import logging
from .utils import all_same, setup_logging, fatal, natural_key
from .validation import collect_and_validate_files, check_files, print_check_fails
from .sort import deduplicate, sort_thermo
from .selectivity import (get_boltz, parse_label_args, load_label_yaml,
                            assign_files_to_labels, compute_selectivity,
                            compute_selectivity_scan,
                            compute_selectivity_lowest_only,
                            compute_selectivity_lowest_only_scan)
from .output import (print_results, print_temperature_interval,
                      print_pes_results, print_cpu_time, write_json_results,
                      print_selectivity_results)

log = logging.getLogger('goodvibes')

# Above this many input files, the command echo collapses the file list to a
# "<N files>" placeholder instead of listing every path.
COMMAND_ECHO_FILE_LIMIT = 10


def parse_arguments():
    """Parse command-line arguments and return (options, args)."""

    # Get command line inputs. Use -h to list all possible arguments and default values
    parser = ArgumentParser(
        prog="goodvibes",
        description="Compute quasi-harmonic thermochemical corrections from quantum chemistry output files.")

    state = parser.add_argument_group("Temperature & state")
    state.add_argument("--temp", dest="temperature", default=298.15, type=float, metavar="TEMP",
                       help="Temperature in Kelvin (default: 298.15)")
    state.add_argument("--ti", dest="temperature_interval", default=None, metavar="TI",
                       help="Temperature interval as START,END,STEP in Kelvin (e.g. '300,1000,100')")
    state.add_argument("-c", "--conc", dest="conc", default=None, type=float, metavar="CONC",
                       help="Concentration in mol/L for solution-phase entropy; gas phase (1 atm) if not set")
    state.add_argument("--media", dest="media", default=None, metavar="solvent",
                       help="Apply standard-state concentration correction for the given solvent "
                            "(e.g. H2O corrects to 55.34 M)")
    state.add_argument("--freespace", dest="freespace", default=None, type=str, metavar="SOLVENT",
                       help="Apply free-space correction for the given solvent (e.g. H2O, toluene, DMF, "
                            "AcOH, chloroform)")

    qh = parser.add_argument_group("Quasi-harmonic corrections")
    qh.add_argument("-q", dest="Q", action="store_true", default=False,
                    help="Apply both quasi-harmonic entropy (Grimme mRRHO) and enthalpy (Head-Gordon) corrections")
    qh.add_argument("--qh", dest="QH", action="store_true", default=False,
                    help="Apply Head-Gordon quasi-harmonic enthalpy correction")
    qh.add_argument("--qs", dest="QS", default="grimme", type=str.lower, metavar="QS",
                    choices=('grimme', 'truhlar'),
                    help="Quasi-harmonic entropy method: 'grimme' for mRRHO free-rotor interpolation, "
                         "'truhlar' for frequency raising (default: grimme)")
    qh.add_argument("-f", "--tau", dest="freq_cutoff", default=100, type=float, metavar="FREQ_CUTOFF",
                    help="Frequency cut-off for both entropy and enthalpy in cm-1 (default: 100)")
    qh.add_argument("--fh", dest="H_freq_cutoff", default=100.0, type=float, metavar="H_FREQ_CUTOFF",
                    help="Frequency cut-off for enthalpy only in cm-1; overrides -f for H (default: 100)")
    qh.add_argument("--fs", dest="S_freq_cutoff", default=100.0, type=float, metavar="S_FREQ_CUTOFF",
                    help="Frequency cut-off for entropy only in cm-1; overrides -f for S (default: 100)")
    qh.add_argument("--bav", dest='inertia', default="global", type=str, choices=['global', 'conf'],
                    help="Moment of inertia for free-rotor entropy: 'global' uses Bav = 10e-44 kg m^2 "
                         "for all molecules, 'conf' computes from rotational constants per file "
                         "(default: global)")

    freq = parser.add_argument_group("Frequency handling")
    freq.add_argument("-v", "--vscal", dest="freq_scale_factor", default=None, type=float, metavar="SCALE_FACTOR",
                      help="Vibrational frequency scaling factor for the partition-function frequencies "
                           "(H_vib, S_vib); auto-detected from level of theory via Truhlar's harm_fac if "
                           "not set, falls back to 1.0")
    freq.add_argument("--zpe-vscal", dest="zpe_scale_factor", default=None, type=float, metavar="ZPE_SCALE_FACTOR",
                      help="Separate scaling factor for the zero-point energy (ZPE); auto-detected from "
                           "level of theory via Truhlar's zpe_fac if not set. If --vscal is set but "
                           "--zpe-vscal is not, ZPE inherits --vscal (back-compat).")
    freq.add_argument("--vmm", dest="mm_freq_scale_factor", default=None, type=float, metavar="MM_SCALE_FACTOR",
                      help="Frequency scaling factor for the MM region in ONIOM calculations")
    freq.add_argument("--invert", dest="invert", nargs='?', const=True, default=None, type=float,
                      help="Invert small imaginary frequencies (> -50 cm-1) to positive values; "
                           "optionally provide a custom threshold in cm-1")
    freq.add_argument("--imag", dest="imag_freq", action="store_true", default=False,
                      help="Print imaginary frequencies for each structure")

    sym = parser.add_argument_group("Symmetry")
    sym.add_argument("--symm", dest='symm', action="store_true", default=False,
                     help="Apply external symmetry correction to entropy using point-group detection (pymsym)")
    sym.add_argument("--pg", dest='pg', action="store_true", default=False,
                     help="Display point group from output file (no entropy correction applied)")

    sort_grp = parser.add_argument_group("Sorting & deduplication")
    sort_grp.add_argument("--sort", dest="sort", default=None, nargs='?', const='gibbs',
                          type=str, choices=['energy', 'gibbs'], metavar="KEY",
                          help="Sort output by energy ('energy' for SCF, 'gibbs' for quasi-harmonic G; default when flag used: gibbs)")
    sort_grp.add_argument("--boltz", dest="boltz", default=None, nargs='?', const='gibbs',
                          type=str, choices=['energy', 'gibbs'], metavar="KEY",
                          help="Print Boltzmann-weighted populations ('energy' for SCF, 'gibbs' for quasi-harmonic G; "
                               "default when flag used: gibbs)")
    sort_grp.add_argument("--dedup", dest="duplicate", action="store_true", default=False,
                          help="Remove duplicate structures based on energy, rotational constants, and stoichiometry")
    sort_grp.add_argument("--e_cutoff", dest="e_cutoff", default=0.05, type=float, metavar="KCAL",
                          help="Energy cutoff for duplicate detection in kcal/mol (default: 0.05)")
    sort_grp.add_argument("--ro_cutoff", dest="ro_cutoff", default=0.01, type=float, metavar="FRAC",
                          help="Rotational constant cutoff for duplicate detection as a fraction (default: 0.01 = 1%%)")
    sort_grp.add_argument("--rmsd_cutoff", dest="rmsd_cutoff", default=None, type=float, metavar="ANGS",
                          help="RMSD cutoff for duplicate detection in Angstrom (default: off; 0.125 matches CREST)")

    sel = parser.add_argument_group("Selectivity & PES")
    sel.add_argument("--label", dest="labels", default=None, action='append', metavar="NAME=PATTERN",
                     help="Repeatable: assign files matching fnmatch PATTERN to species NAME "
                          "for selectivity analysis. Two or more species supported "
                          "(N=2: excess + ΔΔG; N>2: populations only). Example: "
                          "--label R='*P_R_*' --label S='*P_S_*'.")
    sel.add_argument("--selectivity", dest="selectivity_spec", default=None, metavar="FILE.yaml",
                     help="YAML spec for selectivity analysis. Top-level key 'labels' "
                          "maps species -> fnmatch pattern, or 'files' maps species -> "
                          "explicit list of files. Use instead of --label for many "
                          "species or shareable specs.")
    sel.add_argument("--ee", dest="ee", default=None, type=str, metavar="patterns",
                     help="DEPRECATED — use --label/--selectivity. Compute 2-species "
                          "selectivity from a colon-delimited glob pair (e.g. '*_R*:*_S*'). "
                          "Will be removed in v5.0.")
    sel.add_argument("--pes", dest="pes", default=None, metavar="file",
                     help="YAML file defining a reaction pathway for tabulating relative energies")
    sel.add_argument("--graph", dest='graph', default=None, metavar="file",
                     help="Graph a reaction profile from free energies; provide the PES YAML file")
    sel.add_argument("--nogconf", dest="gconf", action="store_false", default=True,
                     help="Disable the Gconf correction for multi-conformer ensembles (enabled by default)")
    sel.add_argument("--lowest-only", dest="lowest_only", action="store_true", default=False,
                     help="PES tables: use only each species' lowest qh-G conformer "
                          "(overrides --nogconf and the default Gconf correction)")

    inp = parser.add_argument_group("Input")
    inp.add_argument("--custom_ext", type=str, default='', metavar="exts",
                     help="Additional file extensions to accept, comma-separated "
                          "(e.g. '.qfi,.gaussian'); also settable via GOODVIBES_CUSTOM_EXT env var")
    inp.add_argument("--exclude", dest="exclude", default=None, type=str, metavar="PATTERN",
                     help="Glob pattern to exclude files from the analysis (e.g. '*_TZ.out')")
    inp.add_argument("--spc", dest="spc", type=str, default=None, metavar="suffix",
                     help="Single-point correction suffix: reads energy from FILE_SPC.ext "
                          "(e.g. --spc TZ reads from FILE_TZ.log)")
    inp.add_argument("--import", dest="import_path", default=None, type=str, metavar="PATH",
                     help="Read pre-parsed data from a v1.0 JSON file (or a legacy "
                          "--cache-save envelope) instead of re-parsing the input "
                          "QC outputs. Useful for fast re-analysis at a different "
                          "temperature/concentration without re-reading large outputs.")
    inp.add_argument("--cache-read", dest="cache_read", default=None, type=str, metavar="FILE",
                     help="(deprecated, use --import instead) Read pre-parsed data from a JSON file.")

    out = parser.add_argument_group("Output & reports")
    out.add_argument("--output", dest="output", default="output", metavar="name",
                     help="Base name for the output file, written as GoodVibes_OUTPUT.dat (default: output)")
    out.add_argument("--dp", dest="dp", default=6, type=int, metavar="DP",
                     help="Number of decimal places for energy values in output (default: 6)")
    out.add_argument("--json", dest="json_path", default=None, metavar="PATH",
                     help="Write structured results to PATH as JSON (every parsed and computed "
                          "field per file plus the run options). Schema is preview/v0.1.")
    out.add_argument("--csv", dest="csv_path", default=None, metavar="PATH",
                     help="Write per-file thermochemistry to PATH as CSV (one row per "
                          "structure; columns mirror ThermoResult). Requires pandas; "
                          "install with `pip install goodvibes[full]`.")
    out.add_argument("--parquet", dest="parquet_path", default=None, metavar="PATH",
                     help="Write per-file thermochemistry to PATH as Parquet (same "
                          "columns as --csv). Requires pandas + a Parquet engine "
                          "(pyarrow); install with `pip install goodvibes[full]`.")
    out.add_argument("--xyz", dest="xyz", action="store_true", default=False,
                     help="Write optimized Cartesian coordinates to a .xyz file")
    out.add_argument("--strip-plot", dest="strip_plot_path", default=None, metavar="PATH",
                     help="Save a per-species ΔG strip plot to PATH (PNG/PDF/SVG by "
                          "extension). Requires --label or --selectivity to define "
                          "buckets; matplotlib via `pip install goodvibes[plot]`.")
    out.add_argument("--pes-plot", dest="pes_plot_path", default=None, metavar="PATH",
                     help="Save a clean reaction-profile diagram to PATH (PNG/PDF/SVG "
                          "by extension). Requires --pes; uses goodvibes.plot.plot_pes "
                          "(simpler than the legacy --graph FILE.yaml, no styling YAML "
                          "needed). matplotlib via `pip install goodvibes[plot]`.")
    out.add_argument("--export", dest="export_path", default=None, type=str, metavar="PATH",
                     help="Write the unified v1.0 JSON payload to PATH "
                          "(per-file QCData + thermo + run options + selectivity / pes "
                          "blocks). Identical to --json. The same file can be re-loaded "
                          "via --import to skip parsing on a follow-up run.")
    out.add_argument("--cache-save", dest="cache_save", default=None, type=str, metavar="FILE",
                     help="(deprecated, use --export instead) Save parsed data to a JSON file.")
    out.add_argument("--cpu", dest="cputime", action="store_true", default=False,
                     help="Print total CPU time from output files")
    out.add_argument("--check", dest="check", action="store_true", default=False,
                     help="Verify all files use the same program, level of theory, and solvation model; "
                          "also flag potential duplicates")

    perf = parser.add_argument_group("Performance")
    perf.add_argument("--jobs", dest="jobs", type=int, default=1, metavar="N",
                      help="Parse and compute thermochemistry in parallel across N "
                           "worker processes (default 1, sequential). 0 = use all "
                           "available CPU cores.")
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
    elif options.invert is not None and options.invert > 0:
        options.invert = -1 * options.invert

    # Build the command echo from the original CLI invocation so it can be
    # logged to the .dat file (users copy/paste it to reproduce runs).
    # File-path positional args are kept verbatim when there are only a few,
    # but collapsed to a "<N files>" placeholder past a threshold so the line
    # stays readable for runs over dozens of .log/.out files.
    file_args, flag_args = [], []
    for arg in sys.argv[1:]:
        if os.path.splitext(arg)[1].lower() in SUPPORTED_EXTENSIONS:
            file_args.append(arg)
        else:
            flag_args.append(arg)
    if len(file_args) > COMMAND_ECHO_FILE_LIMIT:
        file_parts = [f'<{len(file_args)} files>']
    else:
        file_parts = file_args
    parts = ['o  Requested: goodvibes', *file_parts, *flag_args]
    options.command = ' '.join(parts)

    # Collect filenames from the unrecognized positional arguments
    files = []
    for elem in args:
        if os.path.splitext(elem)[1].lower() not in SUPPORTED_EXTENSIONS:
            continue
        for file in glob(elem):
            if options.spc is not None and options.spc != 'link' and f'_{options.spc}.' in file:
                continue
            files.append(file)
            if options.spc is not None and options.spc != 'link':
                name, _ = os.path.splitext(file)
                if find_spc_file(name, options.spc) is None:
                    sys.exit("\nError! No SPC calculation file '{0}_{1}.(log|out)' found! "
                             "The --spc suffix must match the SPC filename exactly "
                             "(e.g. --spc def2_TZVP for filename_def2_TZVP.log), or specify "
                             "link job.\nFor help, use option '-h'\n".format(name, options.spc))

    # Exclude files matching the --exclude glob pattern
    if options.exclude:
        from fnmatch import fnmatch
        files = [f for f in files if not fnmatch(f, options.exclude)]
    # Natural-sort so file order is deterministic across shells (some glob
    # implementations don't sort) and conf_10 follows conf_9, not conf_1.
    files.sort(key=natural_key)
    return options, files


def resolve_scaling_factor(files, options, level_of_theory):
    """Look up or validate the vibrational frequency scaling factor.

    If the user provided --freq_scale_factor, log it. Otherwise, attempt automatic
    lookup from the Truhlar database based on level of theory. Warns if multiple
    levels of theory are found.

    Parameters:
        files (list): output file paths.
        options (Namespace): parsed CLI options. Uses: freq_scale_factor, mm_freq_scale_factor, boltz, ee.
        level_of_theory (list): level of theory strings, one per file.
    """
    if options.freq_scale_factor is not None:
        if 'ONIOM' not in level_of_theory[0]:
            log.info(f"\n   User-defined vibrational scale factor {options.freq_scale_factor} for {level_of_theory[0]} level of theory")
        else:
            log.info(f"\n   User-defined vibrational scale factor {options.freq_scale_factor} for QM region of {level_of_theory[0]}")
    else:
        # Look for vibrational scaling factor automatically. Truhlar's
        # database provides separate harm_fac (for partition functions)
        # and zpe_fac (for ZPE). When auto-detecting, use both; when the
        # user supplied --vscal explicitly, ZPE inherits it unless they
        # also passed --zpe-vscal (handled below).
        if all_same(level_of_theory):
            level = canonicalize_level(level_of_theory[0])
            if level in scaling_data_dict:
                entry = scaling_data_dict[level]
                options.freq_scale_factor = entry.harm_fac
                if options.zpe_scale_factor is None:
                    options.zpe_scale_factor = entry.zpe_fac
                ref = scaling_refs[entry.harm_ref]
                log.info("\n\no  Found vibrational scaling factors of {:.3f} (harmonic, H/S) and "
                         "{:.3f} (ZPE) for {} level of theory\n   {}".format(
                              entry.harm_fac, entry.zpe_fac, level_of_theory[0], ref))

    # Warn if different levels of theory are found
    if not all_same(level_of_theory):
        print_check_fails(list(level_of_theory), list(files), "levels of theory")

    # Exit program if a comparison of Boltzmann factors is requested and level of theory is not uniform across all files
    if not all_same(level_of_theory) and (options.boltz or options.ee is not None):
        sys.exit("\n\n   ✗ FATAL ERROR: Boltzmann factors require all species computed at the same level of theory\n")

    # Exit program if molecular mechanics scaling factor is given and all files are not ONIOM calculations
    if options.mm_freq_scale_factor is not None:
        if all_same(level_of_theory) and 'ONIOM' in level_of_theory[0]:
            log.info(f"\n\n   User-defined vibrational scale factor {options.mm_freq_scale_factor} for MM region of {level_of_theory[0]}")
            log.info("\n   {}".format(oniom_scale_ref))
        else:
            sys.exit("\n   Option --vmm is only for use in ONIOM calculation output files.\n   "
                     " help use option '-h'\n")

    if options.freq_scale_factor is None:
        options.freq_scale_factor = 1.0  # If no scaling factor is found use 1.0


def warn_orca_prescaled(files):
    """Warn about ORCA files where pre-scaled frequencies were un-scaled."""
    for file in files:
        if file.endswith('.out'):
            try:
                # ORCA can emit non-UTF-8 bytes (paths, internal metadata);
                # match the io.py parsers and replace bad bytes rather than
                # crash on UnicodeDecodeError.
                with open(file, encoding='utf-8', errors='replace') as f:
                    for line in f:
                        if 'Scaling factor for frequencies' in line and 'already applied' in line:
                            try:
                                factor = float(line.split('=')[1].split('(')[0].strip())
                            except (IndexError, ValueError):
                                factor = None
                            if factor is not None and factor != 1.0:
                                log.info("\n   Caution! ORCA applied a frequency scaling factor of {:.6f} in {}. "
                                          "Frequencies have been un-scaled to avoid double-scaling.".format(
                                              factor, os.path.basename(file)))
                            break
            except (IOError, OSError):
                pass


def validate_and_configure(options, solvation_model):
    """Validate solvent, print QH/QS configuration, and return (symm_option, vmm_option)."""
    # Checks to see whether the available free space of a requested solvent is defined
    if options.freespace is not None:
        freespace = get_free_space(options.freespace)
        if freespace != 1000.0:
            log.info(f"\n   Specified solvent {options.freespace}: free volume {freespace / 10.0:.3f} (mol/l) corrects the translational entropy")

    # Check for implicit solvation
    if any('smd' in i.lower() or 'cpcm' in i.lower() for i in solvation_model):
        log.info("\n   Caution! Implicit solvation (SMD/CPCM) detected. Enthalpic and entropic terms cannot be "
                  "safely separated. Use them at your own risk!")

    if options.freq_cutoff != 100.0:
        options.S_freq_cutoff = options.freq_cutoff
        options.H_freq_cutoff = options.freq_cutoff

    # Summary of the quasi-harmonic treatment; print out the relevant reference

    if options.QS == "grimme":
        if options.freq_scale_factor == 1.0:
            log.info(f"\n\n   ✔ Using mRRHO entropies with a frequency cut-off value (tau) of {options.S_freq_cutoff} cm-1")
            qs_ref = grimme_mRRHO_ref
        else:
            log.info(f"\n\n   ✔ Using msRRHO entropies with a frequency cut-off value (tau) of {options.S_freq_cutoff} cm-1")
            qs_ref = grimme_msRRHO_ref

    elif options.QS == "truhlar":
        log.info(f"\n\n   Using an RRHO treatment where low frequencies are adjusted to {options.S_freq_cutoff} cm-1")
        qs_ref = truhlar_ref
    else:
        fatal("\n   ✗ FATAL ERROR: Unknown quasi-harmonic model " + options.QS + " specified (QS must = grimme or truhlar).")
    log.info("\n     " + qs_ref)

    # Check if qh-H correction should be applied
    if options.QH:
        log.info(f"\n\n   ✔ Enthalpy quasi-harmonic treatment: frequency cut-off value of {options.H_freq_cutoff} wavenumbers will be applied.")
        log.info("\n     QH = Head-Gordon: Using an RRHO treatement with an approximation term for vibrational energy.")
        qh_ref = head_gordon_ref
        log.info("\n     " + qh_ref + '\n')

    # Check if entropy symmetry correction should be applied
    if options.symm:
        log.info('\n   ✔ Point groups and symmetry numbers auto-detected using pymsym: https://github.com/corinwagen/pymsym')
    else:
        log.info('\n   ✔ Point group symmetry parsed directly from outputs - caution if not provided or incorrect! Use --symm to auto-detect with pymsym')

    # Whether single-point energies are to be used
    if options.spc:
        log.info("\n   Combining final single point energy with thermal corrections.")
    # Solvent correction message
    if options.media:
        log.info(f"\n   ✔ Applying standard concentration correction (based on density at 20C) to {options.media} solvent.")


def _calc_bbe_worker(args):
    """Top-level worker for ProcessPoolExecutor (must be picklable, hence
    not a closure). Takes (file, conc, qcdata, opts_dict) and returns
    a calc_bbe instance via the v5.0 from_options entry point."""
    from .thermo import ThermoOptions
    file, conc, cached_qcdata, opts = args
    options = ThermoOptions(
        QS=opts['QS'], QH=opts['QH'],
        s_freq_cutoff=opts['S_freq_cutoff'],
        h_freq_cutoff=opts['H_freq_cutoff'],
        temperature=opts['temperature'],
        concentration=conc,
        freq_scale_factor=opts['freq_scale_factor'],
        zpe_scale_factor=opts.get('zpe_scale_factor'),
        solv=opts['freespace'],
        spc=opts['spc'], invert=opts['invert'],
        symm=opts['symm'],
        mm_freq_scale_factor=opts['mm_freq_scale_factor'],
        inertia=opts['inertia'],
    )
    return calc_bbe.from_options(cached_qcdata if cached_qcdata is not None else file, options)


def compute_thermochem(files, options, qcdata_cache=None):
    """Run calc_bbe for each file and collect results.

    For files matching the --media solvent name, the neat solvent concentration
    is used instead of options.conc. Parallelized across `options.jobs`
    worker processes when `>1`; sequential otherwise.

    Parameters:
        files (list): output file paths.
        options (Namespace): parsed CLI options. Uses: QS, QH, S_freq_cutoff,
            H_freq_cutoff, temperature, conc, freq_scale_factor, freespace,
            spc, invert, symm, mm_freq_scale_factor, inertia, media, jobs.
        qcdata_cache (dict, optional): pre-parsed QCData keyed by basename.

    Returns:
        dict: file path → calc_bbe mapping (thermo_data).
    """
    # Build per-file argument tuples sequentially (cheap; only the actual
    # calc_bbe work is parallelized).
    opts_dict = {
        'QS': options.QS, 'QH': options.QH,
        'S_freq_cutoff': options.S_freq_cutoff,
        'h_freq_cutoff': options.H_freq_cutoff,
        'H_freq_cutoff': options.H_freq_cutoff,
        'temperature': options.temperature,
        'freq_scale_factor': options.freq_scale_factor,
        'zpe_scale_factor': getattr(options, 'zpe_scale_factor', None),
        'freespace': options.freespace,
        'spc': options.spc, 'invert': options.invert,
        'symm': options.symm,
        'mm_freq_scale_factor': options.mm_freq_scale_factor,
        'inertia': options.inertia,
    }
    default_conc = options.conc if options.conc else ATMOS / (GAS_CONSTANT * options.temperature)
    per_file_args = []
    for file in files:
        cached_qcdata = None
        if qcdata_cache is not None:
            key = os.path.splitext(os.path.basename(file))[0]
            cached_qcdata = qcdata_cache.get(key)
        conc = default_conc
        if options.media:
            media_conc = compute_media_conc(options.media, file)
            if media_conc is not None:
                conc = media_conc
        per_file_args.append((file, conc, cached_qcdata, opts_dict))

    jobs = getattr(options, 'jobs', 1) or 1
    if jobs <= 0:
        jobs = os.cpu_count() or 1

    if jobs == 1 or len(files) <= 1:
        bbe_vals = [_calc_bbe_worker(args) for args in per_file_args]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            bbe_vals = list(ex.map(_calc_bbe_worker, per_file_args))

    return dict(zip(files, bbe_vals))


def main():
    """CLI entry point: parse arguments, compute thermochemistry, and print results."""
    options, files = parse_arguments()

    # Start logger
    setup_logging("GoodVibes", options.output)

    # Print banner
    log.info(gv_banner)

    # Echo the command line into stdout *and* the .dat file so the run is
    # self-documenting and reproducible (issue #103).
    log.info(getattr(options, 'command', 'o  Requested: goodvibes'))

    # Fail fast on unknown solvent names — let the user fix the typo before we
    # parse a thousand files.
    if options.media is not None:
        try:
            lookup_solvent(options.media)
        except ValueError as exc:
            fatal(f"\n   ✗ FATAL ERROR: {exc}")
    if options.freespace is not None and options.freespace not in FREESPACE_SOLVENTS:
        fatal(
            f"\n   ✗ FATAL ERROR: Unknown --freespace solvent {options.freespace!r}. "
            f"Supported (case-sensitive): {', '.join(FREESPACE_SOLVENTS)}."
        )

    # Selectivity spec: parse --label and --selectivity into a label_patterns
    # / label_files dict before any heavy work, so typos fail immediately.
    label_patterns, label_files = None, None
    if options.labels and options.selectivity_spec:
        fatal("\n   ✗ FATAL ERROR: --label and --selectivity are mutually exclusive.")
    if options.ee is not None and (options.labels or options.selectivity_spec):
        fatal("\n   ✗ FATAL ERROR: --ee is incompatible with --label / --selectivity. "
              "Use one selectivity API at a time.")
    if options.labels:
        try:
            label_patterns = parse_label_args(options.labels)
        except ValueError as exc:
            fatal(f"\n   ✗ FATAL ERROR: {exc}")
    elif options.selectivity_spec:
        try:
            mode, spec = load_label_yaml(options.selectivity_spec)
        except (ValueError, RuntimeError, FileNotFoundError, OSError) as exc:
            fatal(f"\n   ✗ FATAL ERROR: {exc}")
        if mode == 'patterns':
            label_patterns = spec
        else:
            label_files = spec

    # --cache-read is deprecated in v5.0 in favor of --import; redirect with a
    # one-time warning. --cache-save likewise → --export.
    if options.cache_read is not None:
        import warnings
        warnings.warn(
            "--cache-read is deprecated; use --import instead. Both accept "
            "v1.0 JSON payloads and legacy cache envelopes.",
            DeprecationWarning, stacklevel=2,
        )
        if options.import_path is None:
            options.import_path = options.cache_read
    if options.cache_save is not None:
        import warnings
        warnings.warn(
            "--cache-save is deprecated; use --export instead (or --json, "
            "which is the same writer).",
            DeprecationWarning, stacklevel=2,
        )
        if options.export_path is None:
            options.export_path = options.cache_save
        options.cache_save = None        # legacy save_cache block below skipped
    # --export is just a synonym for --json; if both are set, --json wins so
    # explicit --json scripts keep working unchanged.
    if options.export_path is not None and options.json_path is None:
        options.json_path = options.export_path

    # Load QCData cache early if requested (needed to determine file list in cache-only mode)
    qcdata_cache = None
    cache_only = False
    if options.import_path:
        qcdata_cache = load_cache(options.import_path)
        if not files:
            # Cache-only mode: derive file list from cache entries
            cache_only = True
            files = [key + '.cache' for key in qcdata_cache]

    # Check if user has specified any files
    if not files:
        sys.exit("\n!  Specify at least one output files on the command line ¯\\_(ツ)_/¯ \n")

    if options.temperature_interval is None:
        log.info(f"   Temperature = {options.temperature} Kelvin")

    # Concentration / pressure
    if options.conc:
        log.info(f"   Concentration = {options.conc} mol/L")
    else:
        log.info("   Pressure = 1 atm")

    if cache_only:
        # Cache-only mode: skip file validation (no output files on disk)
        level_of_theory, solvation_model = [], []
        for key, qcdata in qcdata_cache.items():
            level_of_theory.append('')
            # solvation_model may be a string or a list [sorted, display] depending on parser
            sm = qcdata.solvation_model
            if isinstance(sm, list):
                sm = sm[0]
            solvation_model.append(sm)
        if options.freq_scale_factor is None:
            options.freq_scale_factor = 1.0
        log.info("\n   Reading from QCData cache: " + options.import_path)
    else:
        # Collect file data and validate
        files, level_of_theory, solvation_model = collect_and_validate_files(files, options)
        # Resolve frequency scaling factor
        resolve_scaling_factor(files, options, level_of_theory)
        if options.import_path:
            log.info("\n   Loaded QCData cache from " + options.import_path)

    # Warn about ORCA files if pre-scaled frequencies were un-scaled
    warn_orca_prescaled(files)

    # Validate options, configure and print
    validate_and_configure(options, solvation_model)

    # Compute thermochemistry for all files
    thermo_data = compute_thermochem(files, options, qcdata_cache=qcdata_cache)

    # Media concentration for display in output (the per-file conc override is handled in compute_thermochem)
    media_conc = None
    if options.media and options.media.lower() in solvents:
        mweight, density = solvents[options.media.lower()]
        media_conc = (density * 1000) / mweight

    # Save QCData cache if requested
    if options.cache_save:
        cache_dict = {}
        for file, bbe in thermo_data.items():
            key = os.path.splitext(os.path.basename(file))[0]
            if hasattr(bbe, 'xyz') and bbe.xyz is not None:
                cache_dict[key] = qcdata_to_dict(bbe.xyz)
        save_cache(cache_dict, options.cache_save)
        log.info("\n   Saved QCData cache to " + options.cache_save)

    # Sort output if requested
    if options.sort:
        thermo_data = sort_thermo(thermo_data, options.sort)

    # Deduplicate structures if requested (needed for both standard and PES output)
    dup_list = deduplicate(thermo_data, e_cutoff=options.e_cutoff,
                           ro_cutoff=options.ro_cutoff,
                           rmsd_cutoff=options.rmsd_cutoff) if options.duplicate else []

    # Compute Boltzmann factors once (used by --boltz display and --ee selectivity)
    boltz_facs = None
    if options.boltz or options.ee is not None:
        boltz_key = options.boltz if options.boltz else 'gibbs'
        boltz_facs = get_boltz(thermo_data, options.temperature, dup_list, key=boltz_key)

    # Selectivity (new --label / --selectivity API). Computed once for
    # single-T mode; one entry per temperature in --ti scan mode. Two
    # methods are computed side-by-side: Boltzmann-averaged across all
    # conformers in each species, and lowest-conformer-only.
    selectivity_results = None
    selectivity_lowest_results = None
    if label_patterns is not None or label_files is not None:
        if label_patterns is not None:
            files_per_label = assign_files_to_labels(list(thermo_data), label_patterns)
        else:
            files_per_label = label_files
        sel_key = options.boltz if options.boltz else 'gibbs'
        try:
            if options.temperature_interval is None:
                selectivity_results = [compute_selectivity(
                    thermo_data, files_per_label, options.temperature,
                    dup_list=dup_list, key=sel_key)]
                selectivity_lowest_results = [compute_selectivity_lowest_only(
                    thermo_data, files_per_label, options.temperature,
                    dup_list=dup_list, key=sel_key)]
            else:
                # Mirror print_temperature_interval's parsing of --ti.
                ti = [float(x) for x in options.temperature_interval.split(',')]
                if len(ti) == 2:
                    ti.append((ti[1] - ti[0]) / 10.0)
                temps = list(range(int(ti[0]), int(ti[1]) + 1, int(ti[2])))
                selectivity_results = compute_selectivity_scan(
                    thermo_data, files_per_label, temps,
                    dup_list=dup_list, key=sel_key)
                selectivity_lowest_results = compute_selectivity_lowest_only_scan(
                    thermo_data, files_per_label, temps,
                    dup_list=dup_list, key=sel_key)
        except ValueError as exc:
            fatal(f"\n   ✗ FATAL ERROR: {exc}")

    # Variables populated by temperature-interval mode, used by PES
    interval_bbe_data, interval, file_list = None, None, None

    # Standard mode: single temperature
    if options.temperature_interval is None:
        print_results(thermo_data, options, media_conc=media_conc,
                      dup_list=dup_list, boltz_facs=boltz_facs)
        if selectivity_results is not None:
            print_selectivity_results({
                'Boltzmann-averaged': selectivity_results,
                'Lowest conformer only': selectivity_lowest_results,
            })

    # PES: build the v4.2 model once for single-T mode so we can pass it
    # to both the Rich table renderer and the JSON writer. T-interval mode
    # still flows through the legacy print_pes_results below.
    pes_result = None
    if options.pes and options.temperature_interval is None:
        from .pes_loader import load_pes
        pes_result = load_pes(options.pes, thermo_data,
                              temperatures=[options.temperature])

    # Structured (JSON) output — v1.0 stable schema. Additive; runs
    # alongside the .dat output.
    if options.json_path:
        # JSON write fires in both single-T and T-interval modes. In
        # T-interval mode the per-file thermo numbers reflect the base
        # temperature only; the temperature_interval value is recorded
        # in `options` for downstream consumers to scan.
        media_conc_per_file = {}
        if options.media:
            for file in thermo_data:
                mc = compute_media_conc(options.media, file)
                if mc is not None:
                    media_conc_per_file[file] = mc
        write_json_results(thermo_data, options, options.json_path,
                           media_conc_per_file=media_conc_per_file,
                           boltz_facs=boltz_facs,
                           selectivity_results=selectivity_results,
                           selectivity_lowest_results=selectivity_lowest_results,
                           pes_result=pes_result)

    # CSV / Parquet exports (single-T only; one row per structure,
    # ThermoResult columns). Pandas is in the optional `[full]` extras;
    # Parquet additionally needs pyarrow. Build the ThermoResult list
    # once and reuse for both formats.
    if (options.csv_path or options.parquet_path) and options.temperature_interval is None:
        from .api import bbe_to_result, to_dataframe, to_parquet
        try:
            results = [bbe_to_result(bbe, file) for file, bbe in thermo_data.items()]
            if options.csv_path:
                to_dataframe(results).to_csv(options.csv_path, index=False)
                log.info(f"\n   ✔ CSV written to {options.csv_path}")
            if options.parquet_path:
                to_parquet(results, options.parquet_path)
                log.info(f"\n   ✔ Parquet written to {options.parquet_path}")
        except ImportError as exc:
            fatal(str(exc))

    # Selectivity strip plot — needs a SelectivityResult and a
    # {file: qh_g} lookup. matplotlib via `goodvibes[plot]`.
    if options.strip_plot_path:
        if selectivity_results is None or not selectivity_results:
            fatal("--strip-plot requires --label or --selectivity to define species buckets.")
        try:
            from .plot import plot_selectivity_strip
        except ImportError as exc:
            fatal(str(exc))
        # Use the single-T (or first-T-in-scan) selectivity result.
        sel = selectivity_results[0]
        thermo_lookup = {f: bbe.qh_gibbs_free_energy for f, bbe in thermo_data.items()}
        ax = plot_selectivity_strip(sel, thermo_lookup)
        ax.figure.savefig(options.strip_plot_path, dpi=200, bbox_inches="tight")
        log.info(f"\n   ✔ Strip plot written to {options.strip_plot_path}\n")

    # PES profile diagram — clean v5.0 plot driven by PESResult.
    # Reuses the pes_result that was built earlier for the Rich
    # tables / JSON in single-T mode. matplotlib via `goodvibes[plot]`.
    if options.pes_plot_path:
        if pes_result is None:
            fatal("--pes-plot requires --pes (a reaction pathway YAML) to define the profile.")
        try:
            from .plot import plot_pes
        except ImportError as exc:
            fatal(str(exc))
        ax = plot_pes(pes_result)
        ax.figure.savefig(options.pes_plot_path, dpi=200, bbox_inches="tight")
        log.info(f"\n   ✔ PES plot written to {options.pes_plot_path}\n")

    # Legacy --graph FILE.yaml: the busier matplotlib reaction profile
    # (bezier connectors, jitter overlays, YAML-driven styling). The
    # styling YAML can be the same file as --pes — both blocks
    # (`--- # PES` and `--- # FORMAT`) coexist. Wired here in main()
    # because v4.2's single-T path doesn't call print_pes_results,
    # which is where this used to live.
    if options.graph is not None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            fatal("--graph requires matplotlib. Install with `pip install goodvibes[plot]`.")
        from .pes import get_pes, graph_reaction_profile
        graph_data = get_pes(options.graph, thermo_data,
                             options.temperature, options.gconf, options.QH)
        graph_reaction_profile(graph_data, options, plt)

    # Perform checks for consistent options
    if options.check:
        check_files(thermo_data, options, level_of_theory)

    # Variable temperature analysis
    elif options.temperature_interval:
        print_temperature_interval(thermo_data, options, media_conc=media_conc, qcdata_cache=qcdata_cache)
        if selectivity_results is not None:
            print_selectivity_results({
                'Boltzmann-averaged': selectivity_results,
                'Lowest conformer only': selectivity_lowest_results,
            })

    # Print CPU usage if requested
    if options.cputime:
        print_cpu_time(thermo_data, options.exclude)

    # Tabulate relative values (PES). New Rich table renderer for the
    # single-T case; T-interval still uses the legacy text path.
    if options.pes:
        if options.temperature_interval is None:
            from .output import print_pes_tables
            print_pes_tables(pes_result, options, temperature=options.temperature)
        else:
            print_pes_results(thermo_data, options, dup_list,
                              boltz_facs=boltz_facs,
                              interval_bbe_data=interval_bbe_data,
                              interval=interval, file_list=file_list)

    # Close the log
    logging.shutdown()

    # Write xyz file if requested
    if options.xyz:
        write_xyz("GoodVibes_"+options.output+".xyz", files, thermo_data)
        log.info("   ✔ Cartesian coordinates written to GoodVibes_"+options.output+".xyz\n")


if __name__ == "__main__":
    main()
