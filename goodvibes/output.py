"""Output formatting and printing functions for GoodVibes."""
import json
import logging
import math
import os.path
import sys
from datetime import datetime, timedelta, timezone

from .utils import display_name, get_console_stdout, get_console_dat
from .selectivity import get_selectivity
from .constants import GAS_CONSTANT, ATMOS, J_TO_AU, KCAL_TO_AU, __version__
from .io import qcdata_to_dict

from .pes import get_pes
from .thermo import calc_bbe
from .media import solvents


# --------------------------------------------------------------------------
# Structured (JSON) output
# --------------------------------------------------------------------------
# Schema is now stable from v1.0 onward — see goodvibes.schema for the
# single source of truth (TypedDicts, version constants, validator).
# Re-exported here as a module attribute for back-compat with anything
# that imports `JSON_SCHEMA_VERSION` directly.
from .schema import SCHEMA_VERSION as JSON_SCHEMA_VERSION

# Run-level option keys that should appear in the JSON header, mapped from
# their Namespace attribute names. Any options not in this list are dropped
# (avoids leaking nuisance flags like --output, --custom_ext, etc.).
_JSON_OPTION_KEYS = (
    'temperature', 'temperature_interval', 'conc',
    'QS', 'QH', 'freq_cutoff', 'S_freq_cutoff', 'H_freq_cutoff',
    'freq_scale_factor', 'media', 'freespace', 'spc',
    'invert', 'symm', 'duplicate', 'boltz', 'inertia',
    'mm_freq_scale_factor',
)

# calc_bbe attributes whose names match exactly between the object and the
# JSON output. Optional — None when calc_bbe didn't compute them (SP-only).
_THERMO_FIELDS = (
    'scf_energy', 'sp_energy', 'zpe', 'zero_point_corr',
    'enthalpy', 'qh_enthalpy',
    'entropy', 'qh_entropy',
    'gibbs_free_energy', 'qh_gibbs_free_energy',
    'frequency_wn', 'im_frequency_wn', 'inverted_freqs',
    'point_group', 'symmno', 'linear_mol',
    'multiplicity', 'job_type', 'applied_freq_scale_factor',
)


def _options_to_json(options):
    """Project the argparse Namespace down to the JSON-relevant subset."""
    out = {}
    for key in _JSON_OPTION_KEYS:
        out[key] = getattr(options, key, None)
    return out


def _bbe_to_json(bbe, file_path, media_conc=None, boltz_factor=None):
    """Serialize one calc_bbe instance + its underlying QCData to a dict."""
    entry = {
        'file': os.path.abspath(file_path),
        'name': display_name(file_path),
    }
    qcdata = getattr(bbe, 'xyz', None)
    if qcdata is not None:
        d = qcdata_to_dict(qcdata)
        d.pop('_cache_version', None)
        entry['qcdata'] = d
    thermo = {}
    for field_name in _THERMO_FIELDS:
        thermo[field_name] = getattr(bbe, field_name, None)
    entry['thermo'] = thermo
    entry['media_conc'] = media_conc
    entry['boltzmann_factor'] = boltz_factor
    return entry


def print_selectivity_results(results_by_method):
    """Print one or more selectivity tables, one block per method.

    Parameters:
        results_by_method (dict[str, list[SelectivityResult]]): ordered
            mapping of method name (e.g. "Boltzmann-averaged",
            "Lowest conformer only") to its list of SelectivityResults.
            Single-temperature mode passes a one-element list per method;
            scan mode passes one per temperature.
    """
    if not results_by_method or Table is None:
        return
    for method, results in results_by_method.items():
        if not results:
            continue
        if len(results) == 1:
            _print_selectivity_single(results[0], method=method)
        else:
            _print_selectivity_scan(results, method=method)


def _format_ratio(populations, labels, scale=100):
    """Format populations as integer-percent ratio string ('60:40' or '40:30:20:10')."""
    rounded = [round(populations[label] * scale) for label in labels]
    return ':'.join(str(v) for v in rounded)


def _print_selectivity_single(result, method=""):
    """Per-species table + summary footer for a single SelectivityResult."""
    HA_TO_KCAL = 627.509541
    suffix = f", {method}" if method else ""
    log.info(f"\n   Selectivity{suffix} ({result.key}, T = {result.temperature:.2f} K)")

    # Per-species table: leading marker column (★ for major) | Species |
    # Files | Population (%) | ΔΔG (kcal/mol, vs. major).
    table = Table(
        box=rich_box.SIMPLE_HEAD,
        show_header=True,
        show_edge=False,
        pad_edge=False,
        highlight=False,
        header_style="",
    )
    table.add_column("", width=3)             # marker col, mirrors print_results
    table.add_column("Species", no_wrap=True)
    table.add_column("Files", justify="right")
    table.add_column("Population (%)", justify="right")
    table.add_column("ΔΔG (kcal/mol)", justify="right")

    # ΔΔG = -RT ln(p / p_max); the major species is 0 by construction,
    # others are positive (less stable). Skip when p == 0 (printed as "—").
    ref_pop = max(result.populations.values())
    rt_kcal = 8.3144621 * result.temperature / 1000.0 / 4.184  # kcal/mol
    for label in result.labels:
        p = result.populations[label]
        n_files = len(result.files_per_label.get(label, []))
        marker = "★" if label == result.preferred else " "
        if p > 0:
            dG = -rt_kcal * math.log(p / ref_pop)
            # The major species's dG can be -0.0 from float math; force +0.000.
            if abs(dG) < 1e-9:
                dG = 0.0
            dG_str = f"{dG:.3f}"
        else:
            dG_str = "—"
        table.add_row(marker, label, str(n_files), f"{p * 100:.2f}", dG_str)

    _print_rich_table(table)

    # Summary line: ratio (always); excess + ΔΔG for N=2.
    n = len(result.labels)
    ratio_str = _format_ratio(result.populations, result.labels)
    pieces = [f"Ratio {':'.join(result.labels)} = {ratio_str}"]
    pieces.append(f"Major: {result.preferred}")
    if n == 2 and result.ee is not None:
        pieces.append(f"excess = {result.ee:.2f}%")
    if n == 2 and result.ddG is not None:
        pieces.append(f"ΔΔG = {result.ddG * HA_TO_KCAL:.2f} kcal/mol")
    log.info("\n   " + "   ".join(pieces) + "\n")


def _print_selectivity_scan(results, method=""):
    """One row per temperature: excess/ΔΔG for N=2, populations only for N>2."""
    HA_TO_KCAL = 627.509541
    labels = results[0].labels
    n = len(labels)
    suffix = f", {method}" if method else ""
    log.info(f"\n   Selectivity scan{suffix} ({results[0].key})")

    table = Table(
        box=rich_box.SIMPLE_HEAD,
        show_header=True,
        show_edge=False,
        pad_edge=False,
        highlight=False,
        header_style="",
    )
    table.add_column("", width=3)             # leading indent, mirrors single-T
    table.add_column("T (K)", justify="right")
    for label in labels:
        table.add_column(f"{label} (%)", justify="right")
    if n == 2:
        table.add_column("excess (%)", justify="right")
        table.add_column("ΔΔG (kcal/mol)", justify="right")
    table.add_column("Major", justify="center")

    for r in results:
        row = ["", f"{r.temperature:g}"]
        for label in labels:
            row.append(f"{r.populations[label] * 100:.2f}")
        if n == 2:
            row.append(f"{r.ee:.2f}" if r.ee is not None else "—")
            ddG_kc = r.ddG * HA_TO_KCAL if r.ddG is not None else None
            row.append(f"{ddG_kc:.2f}" if ddG_kc is not None else "—")
        row.append(r.preferred)
        table.add_row(*row)

    _print_rich_table(table)
    log.info("\n")


def _selectivity_to_json(selectivity_results):
    """Serialize a list of SelectivityResult instances for the JSON payload.

    For N=2 each entry includes ee + ddG; for N>2 those fields are null
    and consumers derive any ratios they need from `populations`.
    """
    if not selectivity_results:
        return None
    first = selectivity_results[0]
    return {
        'labels': list(first.labels),
        'key': first.key,
        'results': [
            {
                'temperature': r.temperature,
                'populations': dict(r.populations),
                'raw_boltzmann': dict(r.raw_boltzmann),
                'preferred': r.preferred,
                'ee': r.ee,
                'ddG': r.ddG,
                'files_per_label': {k: list(v) for k, v in r.files_per_label.items()},
            }
            for r in selectivity_results
        ],
    }


def _pes_to_json(result, temperature):
    """Serialize a PESResult into the v0.4 JSON `pes` block.

    Per-pathway, per-point: the original label, the species breakdown
    (with coefficient and resolved files), and the relative thermo bundle
    (Hartree internally, converted to user units in the `units` field).
    """
    if result is None or not result.pathways:
        return None
    pathways_out = []
    units_factor = result.options.to_user_units(1.0)   # Hartree → kcal/mol or kJ/mol
    lowest_only = getattr(result.options, 'lowest_only', False)
    rollup_kw = dict(gconf=result.options.gconf, QH=result.options.QH,
                     lowest_only=lowest_only)
    for pathway in result.pathways:
        zero_th = pathway.zero.thermo(temperature, **rollup_kw)
        points_out = []
        for point in pathway.points:
            pt_th = point.thermo(temperature, **rollup_kw)
            rel = pt_th - zero_th
            points_out.append({
                'label': point.label,
                'species': [
                    {
                        'coefficient': coeff,
                        'name': cset.name,
                        'files': list(cset.files),
                    }
                    for coeff, cset in point.species
                ],
                'relative': {
                    'scf': rel.scf_energy * units_factor,
                    'zpe': rel.zpe * units_factor,
                    'h': rel.enthalpy * units_factor,
                    'qh_h': rel.qh_enthalpy * units_factor,
                    'ts': temperature * rel.entropy * units_factor,
                    'qh_ts': temperature * rel.qh_entropy * units_factor,
                    'g': rel.gibbs * units_factor,
                    'qh_g': rel.qh_gibbs * units_factor,
                    'spc': (rel.sp_energy * units_factor) if rel.sp_energy is not None else None,
                },
            })
        pathways_out.append({
            'name': pathway.name,
            'temperature': temperature,
            'units': result.options.units,
            'zero_label': pathway.zero.label,
            'points': points_out,
        })
    return {'pathways': pathways_out}


def write_json_results(thermo_data, options, path, media_conc_per_file=None,
                       boltz_facs=None, selectivity_results=None,
                       selectivity_lowest_results=None,
                       pes_result=None):
    """Write structured run results to *path* as JSON.

    Captures every parsed QCData field plus computed thermo per file, the
    schema version, the goodvibes version, and the run options. Schema is
    preview/v0.x — fields may shift before v5.0 stabilizes it.

    Parameters:
        thermo_data (dict): file path -> calc_bbe.
        options (Namespace): the parsed CLI options.
        path (str): output JSON file path.
        media_conc_per_file (dict, optional): file path -> neat solvent
            concentration applied to that specific file (None for files
            where --media didn't match).
        boltz_facs (dict, optional): file path -> Boltzmann factor.
        selectivity_results (list[SelectivityResult], optional): one entry
            per temperature; included as a top-level "selectivity" block.
    """
    media_conc_per_file = media_conc_per_file or {}
    boltz_facs = boltz_facs or {}
    payload = {
        'schema_version': JSON_SCHEMA_VERSION,
        'goodvibes_version': __version__,
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'options': _options_to_json(options),
        'results': [
            _bbe_to_json(
                bbe, file,
                media_conc=media_conc_per_file.get(file),
                boltz_factor=boltz_facs.get(file),
            )
            for file, bbe in thermo_data.items()
        ],
    }
    sel_block = _selectivity_to_json(selectivity_results)
    if sel_block is not None:
        payload['selectivity'] = sel_block
    sel_low_block = _selectivity_to_json(selectivity_lowest_results)
    if sel_low_block is not None:
        payload['selectivity_lowest'] = sel_low_block
    pes_block = _pes_to_json(pes_result, getattr(options, 'temperature', 298.15))
    if pes_block is not None:
        payload['pes'] = pes_block
    # default=str catches anything stringifiable that json doesn't natively
    # know about (e.g., the '!' sentinel value calc_bbe uses for failed SPC).
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=str)
    log.info(f"\n   ✔ Structured results written to {path} (schema v{JSON_SCHEMA_VERSION}).")

try:
    from rich.table import Table
    from rich import box as rich_box
except ImportError:
    Table = None
    rich_box = None

log = logging.getLogger('goodvibes')


def _print_rich_table(table: "Table") -> None:
    """Print a Rich Table to both stdout and .dat file consoles.

    Renders to a buffer per console, strips Rich's trailing blank padding
    row (`box.SIMPLE` always adds one), and writes directly so the
    separator that follows lands immediately under the data.
    """
    if Table is None:
        return
    import re
    from io import StringIO
    from rich.console import Console
    ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGK]")

    def _render(console):
        buf = StringIO()
        Console(
            file=buf,
            force_terminal=console.is_terminal,
            color_system=console.color_system,
            width=console.width,
        ).print(table)
        text = buf.getvalue().rstrip("\n")
        lines = text.split("\n")
        # Drop trailing visually-blank lines (Rich SIMPLE-box padding).
        while lines and not ANSI_RE.sub("", lines[-1]).strip():
            lines.pop()
        # Leading "\n\n" gives exactly one blank line before the table:
        # the logger uses terminator='' so the prior message has no trailing
        # \n; the first \n closes that line, the second is the blank line.
        return "\n\n" + "\n".join(lines) + "\n"

    for console in (get_console_stdout(), get_console_dat()):
        console.file.write(_render(console))
        console.file.flush()

    # Measure visible width via a plain render (no escape codes).
    stdout_console = get_console_stdout()
    plain_buf = StringIO()
    Console(file=plain_buf, color_system=None, width=stdout_console.width).print(table)
    line_width = max(
        (len(line.rstrip()) for line in plain_buf.getvalue().split("\n")),
        default=0,
    )
    log.info("─" * line_width)
    log.info("\n")


# ---------------------------------------------------------------------------
# PES tables (v4.2 Rich renderer; see Sub-plan B in ROADMAP.md)
# ---------------------------------------------------------------------------

def _pes_column_spec(spc_used: bool, QH: bool):
    """Column headers + which ThermoVector field each one reads.

    Returns a list of (header, field, scale_by_T) tuples.  `field` is the
    attribute on a ThermoVector; `scale_by_T` indicates the column is T·S
    rather than raw S/H/G (i.e. needs an extra multiplication at render).
    Without --spc the energy/H/G columns use their plain names; with
    --spc, H and G are SPC-substituted in calc_bbe so the labels are
    annotated `_SPC` to signal that to the reader (the values themselves
    are taken from the same fields).
    """
    cols = []
    if spc_used:
        cols.append(("ΔE_SPC", "sp_energy", False))
    cols.append(("ΔE", "scf_energy", False))
    cols.append(("ΔZPE", "zpe", False))
    h_label = "ΔH_SPC" if spc_used else "ΔH"
    cols.append((h_label, "enthalpy", False))
    if QH:
        qh_h_label = "Δqh-H_SPC" if spc_used else "Δqh-H"
        cols.append((qh_h_label, "qh_enthalpy", False))
    cols.append(("T·ΔS", "entropy", True))
    cols.append(("T·Δqh-S", "qh_entropy", True))
    g_label = "ΔG(T)_SPC" if spc_used else "ΔG(T)"
    cols.append((g_label, "gibbs", False))
    qhg_label = "Δqh-G(T)_SPC" if spc_used else "Δqh-G(T)"
    cols.append((qhg_label, "qh_gibbs", False))
    return cols


def _build_pes_table(pathway, options, temperature, pes_options) -> "Table":
    """Build a Rich Table for one Pathway at one temperature."""
    spc_used = bool(getattr(options, 'spc', None))
    cols = _pes_column_spec(spc_used, options.QH)
    # Concentration: when --conc isn't set the default is gas-phase 1 atm,
    # so report "p = 1 atm" rather than its numeric P/RT equivalent.
    user_conc = getattr(options, 'conc', None)
    state_str = f"c = {user_conc:.4g} mol/L" if user_conc else "p = 1 atm"
    if getattr(pes_options, 'lowest_only', False):
        mode_str = " — lowest conformer per species"
    elif not pes_options.gconf:
        mode_str = " — Boltzmann-averaged (no gconf)"
    else:
        mode_str = ""
    title = (
        f"RXN: {pathway.name}  ({pes_options.units})  "
        f"at T = {temperature:g} K, {state_str}{mode_str}"
    )
    table = Table(title=title, box=rich_box.SIMPLE, header_style="bold")
    table.add_column("", justify="left", no_wrap=True)        # leading marker (matches selectivity tables)
    table.add_column("Species", justify="left", no_wrap=True)
    for header, _, _ in cols:
        table.add_column(header, justify="right")

    units_factor = pes_options.to_user_units(1.0)
    fmt = f"{{:.{pes_options.decimals}f}}"
    rollup_kw = dict(
        gconf=pes_options.gconf,
        QH=pes_options.QH,
        lowest_only=getattr(pes_options, 'lowest_only', False),
    )
    zero_th = pathway.zero.thermo(temperature, **rollup_kw)
    for point in pathway.points:
        pt_th = point.thermo(temperature, **rollup_kw)
        rel = pt_th - zero_th
        row = ["", point.label]
        for _header, field, scale_by_T in cols:
            value = getattr(rel, field)
            if value is None:
                row.append("—")
                continue
            if scale_by_T:
                value = temperature * value
            row.append(fmt.format(value * units_factor))
        table.add_row(*row)
    return table


def print_pes_tables(result, options, temperature=None):
    """Render PES results as Rich tables (one per pathway).

    Parameters:
        result: PESResult instance (from goodvibes.pes_loader.load_pes).
        options: argparse Namespace; reads .QH, .spc, .gconf.
        temperature: K. Defaults to result.temperatures[0].
    """
    if Table is None:                              # pragma: no cover
        log.warning("Rich not installed; falling back to legacy text PES output.")
        return
    if temperature is None:
        temperature = result.temperatures[0] if result.temperatures else 298.15
    # Sync run-time flags into the model's options so the column spec and
    # rollup math agree with what the user requested on the CLI.
    result.options.gconf = bool(getattr(options, 'gconf', True))
    result.options.QH = bool(getattr(options, 'QH', False))
    result.options.spc_used = bool(getattr(options, 'spc', None))
    result.options.lowest_only = bool(getattr(options, 'lowest_only', False))
    if result.options.lowest_only:
        log.info("\n   Lowest conformer per species (no Boltzmann averaging, no gconf)")
    elif options.gconf:
        log.info("\n   Gconf correction applied to relative values")
    for pathway in result.pathways:
        table = _build_pes_table(pathway, options, temperature, result.options)
        _print_rich_table(table)


def print_cpu_time(thermo_data, exclude=None):
    """
    Aggregate and log total CPU time across the provided thermochemistry results.
    
    Parameters:
        thermo_data (dict): Mapping of file path to calc_bbe-like objects whose `cpu` and optional `sp_cpu` attributes contribute to the total.
        exclude (str, optional): Glob pattern; files matching this pattern are omitted from the summed total.
    """
    from fnmatch import fnmatch

    def _to_timedelta(cpu):
        d, h, m, s, ms = cpu
        return timedelta(days=d, hours=h, minutes=m, seconds=s, microseconds=ms * 1000)

    total = timedelta()
    has_orca_scaled = False
    for file, bbe in thermo_data.items():
        if exclude and fnmatch(file, exclude):
            continue
        if hasattr(bbe, "cpu") and bbe.cpu is not None:
            total += _to_timedelta(bbe.cpu)
        if hasattr(bbe, "sp_cpu") and bbe.sp_cpu is not None:
            total += _to_timedelta(bbe.sp_cpu)
        # Detect ORCA wall-time-scaled-by-nprocs entries via QCData.
        qc = getattr(bbe, "xyz", None)
        if qc is not None and getattr(qc, "program", None) == "Orca" and getattr(qc, "nprocs", 1) > 1:
            has_orca_scaled = True
    total_secs = int(total.total_seconds())
    days, rem = divmod(total_secs, 86400)
    hrs, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    log.info(f'   {"TOTAL CPU":<13} {days:>2} {"days":>4} '
              f'{hrs:>2} {"hrs":>3} {mins:>2} {"mins":>4} '
              f'{secs:>2} {"secs":>4}')
    if has_orca_scaled:
        log.info("\n   † ORCA wall-time × nprocs counted as CPU time (ORCA does not "
                 "report a summed CPU time).\n")
    else:
        log.info("\n")


def _build_results_table(options) -> "Table":
    """
    Create a Rich Table configured for thermochemistry result columns based on the provided options.
    
    Columns included depend on the options flags: QH, spc, boltz, symm/pg, and imag_freq. Column widths and numeric precision are derived from options.dp when present.
    
    Parameters:
        options: An object with attributes used to determine the table layout (commonly includes
            `dp`, `QH`, `spc`, `boltz`, `symm`, `pg`, and `imag_freq`).
    
    Returns:
        table (Table | None): A configured rich.table.Table ready for rows, or `None` if Rich is not available.
    """
    if Table is None:
        return None

    dp = getattr(options, 'dp', 6)
    dw = dp - 6
    ew = 11 + dw  # wide numeric column (energies)
    sw = 2 + dp   # small column (ZPE, T.S, T.qh-S - always 0.xxx)

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
    table.add_column("Structure", no_wrap=True)

    if options.spc is not None:
        table.add_column("E_SPC", min_width=ew, justify="right")
    table.add_column("E", min_width=ew, justify="right")
    table.add_column("ZPE", min_width=sw, justify="right")

    h_col = "H_SPC" if options.spc is not None else "H"
    if options.QH:
        table.add_column(h_col, min_width=ew, justify="right")
        table.add_column("qh-H", min_width=ew, justify="right")
    else:
        table.add_column(h_col, min_width=ew, justify="right")

    table.add_column("T.S", min_width=sw, justify="right")
    table.add_column("T.qh-S", min_width=sw, justify="right")

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
        table.add_column("im freq", min_width=8, justify="right")
    if _selectivity_active(options):
        table.add_column("Grel (kcal/mol)", min_width=8, justify="right")

    return table


def _selectivity_active(options):
    """True when any selectivity flag is on (--label / --selectivity / --ee)."""
    return bool(getattr(options, 'labels', None)
                or getattr(options, 'selectivity_spec', None)
                or getattr(options, 'ee', None))


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

    # Surface files lacking frequency info up front (one summary line, not
    # one per file). Truncated to first 3 + count when it gets long.
    no_freq_files = [
        display_name(f) for f in files
        if not hasattr(thermo_data[f], "gibbs_free_energy")
    ]
    if no_freq_files:
        if len(no_freq_files) > 3:
            n_more = len(no_freq_files) - 3
            list_str = ', '.join(no_freq_files[:3]) + f', and {n_more} others'
        else:
            list_str = ', '.join(no_freq_files)
        log.info(f'\n\n   ! No frequency information found: {list_str}')

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

    # Build the Rich table
    table = _build_results_table(options)
    if table is None:
        # Fallback if Rich is not available (shouldn't happen, but safe)
        log.info("\n\nWarning: Rich not available, falling back to ASCII output")
        return

    # Default dup_list if not passed by caller
    if dup_list is None:
        dup_list = []

    # Reference qh-G for the optional Grel column (selectivity mode only).
    min_qhg = None
    if _selectivity_active(options):
        dup_files = {dup[0] for dup in dup_list}
        candidates = [
            thermo_data[f].qh_gibbs_free_energy
            for f in files
            if f not in dup_files
            and hasattr(thermo_data[f], "gibbs_free_energy")
            and not getattr(thermo_data[f], "linear_warning", False)
        ]
        if candidates:
            min_qhg = min(candidates)

    for file in files:
        duplicate = False
        if dup_list:
            for dup in dup_list:
                if dup[0] == file:
                    duplicate = True
                    log.info('\n   x {} is a duplicate or enantiomer of {}'.format(dup[0].rsplit('.', 1)[0],
                                                                                  dup[1].rsplit('.', 1)[0]))
                    break
        if not duplicate:
            bbe = thermo_data[file]
            # Handle linear molecule warning separately (not a table row)
            if bbe.linear_warning:
                log.info('\n   x {} — Caution! Potential invalid linear molecule calculation in Gaussian'.format(
                    display_name(file)))
                continue

            # Build row for this file
            row = []
            marker = "o" if hasattr(bbe, "gibbs_free_energy") else "x"
            name = display_name(file)

            # No gibbs free energy — already reported in the summary above.
            if not hasattr(bbe, "gibbs_free_energy"):
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
            row.append(f"{bbe.zpe:.{dp}f}")

            # H and optionally qh-H
            row.append(f"{bbe.enthalpy:.{dp}f}")
            if options.QH:
                row.append(f"{bbe.qh_enthalpy:.{dp}f}")

            # T.S and T.qh-S
            row.append(f"{options.temperature * bbe.entropy:.{dp}f}")
            row.append(f"{options.temperature * bbe.qh_entropy:.{dp}f}")

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
            if min_qhg is not None:
                grel = (bbe.qh_gibbs_free_energy - min_qhg) * 627.509541
                row.append(f"{grel:.3f}")

            table.add_row(*row)

            # Media annotation (print after table, not in it)
            if options.media is not None and options.media.lower() in solvents and options.media.lower() == \
                    display_name(file).lower():
                log.info("  Solvent: {:4.2f}M ".format(media_conc))

    # Print the table
    _print_rich_table(table)


def print_temperature_interval(thermo_data, options, media_conc=None, qcdata_cache=None):
    """
    Recompute thermochemical properties over a temperature interval, log formatted results for each structure and return the computed per-temperature data.
    
    For each file in `thermo_data` this function evaluates thermochemistry at every temperature in the interval (derived from options.temperature_interval), logs a human-readable table of H, T·S, G(T) and their quasi-harmonic variants when requested, and collects the resulting computed objects.
    
    Parameters:
        thermo_data (dict): Mapping of file path → previously computed thermochemistry object (used to determine file order).
        options (Namespace): Parsed CLI/options object. Uses these attributes: temperature_interval, dp, QH, QS, S_freq_cutoff, H_freq_cutoff, spc, conc, freq_scale_factor, freespace, invert, inertia, media.
        media_conc (float, optional): Solvent concentration (M) to annotate solvent-matching structures when media is enabled.
        qcdata_cache (dict, optional): Optional mapping from file basename → pre-parsed QCData to pass through to the recomputation routine.
    
    Returns:
        tuple: (interval_bbe_data, interval, file_list)
            - interval_bbe_data (list[list]): Outer list indexed by file; each inner list contains the recomputed thermochemistry objects (one per temperature).
            - interval (range): Python range object describing the temperatures iterated.
            - file_list (list): List of file paths in the order processed.
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
                log.info("\n   x ")
                log.info('{:<39}'.format(display_name(file)))
                log.info('             Warning! Potential invalid calculation of linear molecule from Gaussian ...')
            else:
                # Gaussian spc files
                if bbe.scf_energy is not None and not hasattr(bbe, "gibbs_free_energy"):
                    log.info("\n   x " + '{:<39}'.format(display_name(file)))
                # ORCA spc files
                elif bbe.scf_energy is None and not hasattr(bbe, "gibbs_free_energy"):
                    log.info("\n   x " + '{:<39}'.format(display_name(file)))
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
    # --graph rendering moved up to GoodVibes.main() in v5.0 so it
    # fires in single-T mode too (v4.2's print_pes_tables doesn't go
    # through this function). The legacy T-interval path still calls
    # print_pes_results, but main() handles the graph either way.
