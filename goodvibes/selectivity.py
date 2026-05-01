"""Boltzmann weighting and selectivity calculations for GoodVibes."""
import fnmatch
import logging
import math
import os.path
import sys
import warnings
from dataclasses import dataclass
from glob import glob
from typing import Dict, List, Optional

from .constants import GAS_CONSTANT, J_TO_AU, KCAL_TO_AU
from .sort import SORT_KEYS

log = logging.getLogger('goodvibes')


# ---------------------------------------------------------------------------
# Structured selectivity result (v4.2+)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SelectivityResult:
    """N-species selectivity outcome at one temperature.

    Numeric data only — formatting strings (e.g. "60:40", "1.5:1") are
    derived in the print layer from `populations`. Pairwise data are not
    stored here: for N=2, ee + ddG suffice; for N>2, downstream consumers
    derive any ratios they need from `populations`.
    """
    temperature: float                          # Kelvin
    key: str                                    # 'gibbs' | 'energy'
    labels: List[str]                           # ordered species names
    files_per_label: Dict[str, List[str]]       # species -> file paths
    populations: Dict[str, float]               # normalized: Σ = 1.0
    raw_boltzmann: Dict[str, float]             # un-normalized e^(-ΔG/RT)
    preferred: str                              # max-population label
    ee: Optional[float] = None                  # 2-label only: (a-b)*100, in %
    ddG: Optional[float] = None                 # 2-label only: ΔΔG‡ in Hartree


# ---------------------------------------------------------------------------
# Label-spec parsing
# ---------------------------------------------------------------------------

def parse_label_args(label_args):
    """Parse repeatable --label NAME=PATTERN args into an ordered dict.

    Parameters:
        label_args (list[str] or None): values from argparse for --label.

    Returns:
        dict[str, str]: ordered mapping label -> fnmatch glob pattern.
    """
    if not label_args:
        return {}
    spec = {}
    for arg in label_args:
        if '=' not in arg:
            raise ValueError(
                f"Invalid --label {arg!r}: expected NAME=PATTERN "
                "(e.g. --label R='*P_R_*')."
            )
        name, _, pattern = arg.partition('=')
        name = name.strip()
        pattern = pattern.strip()
        if not name or not pattern:
            raise ValueError(
                f"Invalid --label {arg!r}: NAME and PATTERN must both be "
                "non-empty."
            )
        if name in spec:
            raise ValueError(f"Duplicate --label name {name!r}.")
        spec[name] = pattern
    return spec


def load_label_yaml(path):
    """Load a selectivity YAML file.

    Supports two shapes (mutually exclusive):
        labels: { R: '*P_R_*', S: '*P_S_*' }            # fnmatch patterns
        files:  { R: [a.log, b.log], S: [c.log, ...] }  # explicit file lists

    Returns:
        (mode, dict): mode is 'patterns' or 'files'; dict maps label to its
        spec (string pattern, or list of file paths).
    """
    try:
        import yaml
    except ImportError:
        raise RuntimeError(
            "PyYAML is required for --selectivity. Install with "
            "`pip install pyyaml` or use --label instead."
        )
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if 'labels' in data and 'files' in data:
        raise ValueError(
            f"{path}: provide either 'labels' (patterns) OR 'files' "
            "(explicit lists), not both."
        )
    if 'labels' in data:
        return 'patterns', dict(data['labels'])
    if 'files' in data:
        return 'files', {k: list(v) for k, v in data['files'].items()}
    raise ValueError(
        f"{path}: top-level key 'labels' or 'files' is required."
    )


def assign_files_to_labels(files, label_patterns):
    """Group file paths by fnmatch against each label's pattern.

    Patterns are compared against the basename of each file; a file matches
    at most one label (first match in label-spec order wins).

    Parameters:
        files (Iterable[str]): file paths to species.
        label_patterns (dict): ordered label -> fnmatch glob pattern.

    Returns:
        dict[str, list[str]]: label -> list of matching file paths.
    """
    speciess = {label: [] for label in label_patterns}
    for file in files:
        base = os.path.basename(file)
        for label, pattern in label_patterns.items():
            if fnmatch.fnmatch(base, pattern):
                speciess[label].append(file)
                break
    return speciess


# ---------------------------------------------------------------------------
# Core compute
# ---------------------------------------------------------------------------

def _excluded_files(dup_list):
    """Set of file paths to exclude from sums based on a dup_list.

    Convention from `sort.deduplicate`: each pair is [duplicate, canonical].
    The first element is the redundant copy that should be dropped; the
    second is the kept structure. Excluding only the first member matches
    the existing behavior of `get_boltz` and `get_selectivity` and avoids
    over-counting losses.
    """
    excluded = set()
    if not dup_list:
        return excluded
    for pair in dup_list:
        if pair:
            excluded.add(pair[0])
    return excluded


def compute_selectivity(thermo_data, files_per_label, temperature,
                        dup_list=None, key='gibbs'):
    """Compute populations and (for N=2) ee + ΔΔG‡ for a labeled species set.

    Parameters:
        thermo_data (dict): file path -> calc_bbe.
        files_per_label (dict): ordered label -> list of file paths.
        temperature (float): Kelvin.
        dup_list (list, optional): pairs of duplicate files to exclude
            from sums.
        key (str): 'gibbs' (qh_gibbs_free_energy) or 'energy' (scf_energy).

    Returns:
        SelectivityResult.

    Raises:
        ValueError: if any label has no files, or if no files have a usable
            energy attribute.
    """
    attr = SORT_KEYS[key]
    excluded = _excluded_files(dup_list)
    labels = list(files_per_label.keys())
    if len(labels) < 2:
        raise ValueError("Selectivity needs at least two labels.")

    empty = [label for label in labels if not files_per_label[label]]
    if empty:
        raise ValueError(
            f"No files matched the following label(s): {', '.join(empty)}. "
            "Check the patterns or file lists in your selectivity spec."
        )

    # Find the global minimum energy (across all labeled files we'll keep) so
    # we can shift before exponentiating to avoid float overflow / underflow.
    e_min = math.inf
    for label in labels:
        for file in files_per_label[label]:
            if file in excluded:
                continue
            bbe = thermo_data.get(file)
            if bbe is None:
                continue
            val = getattr(bbe, attr, None)
            if val is not None and val < e_min:
                e_min = val
    if not math.isfinite(e_min):
        raise ValueError(
            "No files in any label had a usable energy attribute "
            f"({attr}); cannot compute selectivity."
        )

    raw = {label: 0.0 for label in labels}
    rt = GAS_CONSTANT * temperature  # J/mol
    for label in labels:
        for file in files_per_label[label]:
            if file in excluded:
                continue
            bbe = thermo_data.get(file)
            if bbe is None:
                continue
            val = getattr(bbe, attr, None)
            if val is None:
                continue
            raw[label] += math.exp(-(val - e_min) * J_TO_AU / rt)

    total = sum(raw.values())
    if total == 0.0:
        raise ValueError(
            "Boltzmann sums are zero across all labels; cannot compute "
            "selectivity."
        )

    populations = {label: raw[label] / total for label in labels}
    preferred = max(populations, key=populations.get)

    ee = None
    ddG = None
    if len(labels) == 2:
        a, b = labels
        pa = populations[a]
        pb = populations[b]
        ee = abs(pa - pb) * 100.0
        # ΔΔG‡ = RT ln(p_major / p_minor), in Hartree, positive by convention
        # (the gap between major and minor TS). None when one species is
        # empty enough that the ratio diverges.
        if pa > 0 and pb > 0:
            ddG = rt * math.log(max(pa, pb) / min(pa, pb)) / J_TO_AU
        else:
            ddG = None

    return SelectivityResult(
        temperature=temperature,
        key=key,
        labels=labels,
        files_per_label={label: list(files_per_label[label]) for label in labels},
        populations=populations,
        raw_boltzmann=raw,
        preferred=preferred,
        ee=ee,
        ddG=ddG,
    )


def compute_selectivity_scan(thermo_data, files_per_label, temperatures,
                             dup_list=None, key='gibbs'):
    """Compute a SelectivityResult at each temperature in `temperatures`.

    Convenience wrapper that pairs naturally with --ti temperature
    intervals; the species grouping is fixed across temperatures.
    """
    return [
        compute_selectivity(thermo_data, files_per_label, T,
                            dup_list=dup_list, key=key)
        for T in temperatures
    ]


def _lowest_per_label(thermo_data, files_per_label, dup_list, key):
    """Return a {label: [single_lowest_file]} reduction of files_per_label.

    For each species, picks the conformer with the lowest energy attribute,
    skipping files in dup_list[k][0] and files with no usable energy.
    """
    attr = SORT_KEYS[key]
    excluded = _excluded_files(dup_list)
    minimal = {}
    for label, files in files_per_label.items():
        best, best_val = None, math.inf
        for f in files:
            if f in excluded:
                continue
            bbe = thermo_data.get(f)
            if bbe is None:
                continue
            val = getattr(bbe, attr, None)
            if val is None:
                continue
            if val < best_val:
                best, best_val = f, val
        minimal[label] = [best] if best is not None else []
    return minimal


def compute_selectivity_lowest_only(thermo_data, files_per_label, temperature,
                                     dup_list=None, key='gibbs'):
    """Selectivity using only the most stable conformer per species.

    For each label, picks the file with the lowest `key` (default
    qh_gibbs_free_energy), drops the rest, and runs the standard
    Boltzmann calc on the resulting 1-conformer-per-species set. The
    result complements compute_selectivity by showing how much of the
    selectivity comes from conformer mixing versus the gap between the
    lowest TSs.
    """
    minimal = _lowest_per_label(thermo_data, files_per_label, dup_list, key)
    return compute_selectivity(thermo_data, minimal, temperature,
                                 dup_list=dup_list, key=key)


def compute_selectivity_lowest_only_scan(thermo_data, files_per_label,
                                           temperatures, dup_list=None,
                                           key='gibbs'):
    """Lowest-only selectivity at each temperature in `temperatures`."""
    minimal = _lowest_per_label(thermo_data, files_per_label, dup_list, key)
    return [compute_selectivity(thermo_data, minimal, T,
                                  dup_list=dup_list, key=key)
            for T in temperatures]


# ---------------------------------------------------------------------------
# Boltzmann weighting (unchanged)
# ---------------------------------------------------------------------------

def get_boltz(thermo_data, temperature, dup_list, key='gibbs'):
    """Produce normalized Boltzmann populations across all files.

    Used by `--boltz` for the per-file population display and by the
    legacy `--ee` selectivity flow (which still calls `get_selectivity`).

    Parameters:
        thermo_data (dict): file path -> calc_bbe.
        temperature (float): Kelvin.
        dup_list (list): pairs [file_i, file_j]; entries whose path is the
            FIRST member of any pair are excluded (legacy behavior).
        key (str): 'gibbs' or 'energy'.

    Returns:
        dict[str, float]: file path -> normalized population (Σ = 1.0).
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


# ---------------------------------------------------------------------------
# Legacy --ee shim (deprecated; remove in v5.0)
# ---------------------------------------------------------------------------

def get_selectivity(pattern, files, boltz_facs, temperature, dup_list):
    """DEPRECATED — legacy 'a:b' colon-pattern selectivity.

    Forwards to `compute_selectivity` with a 2-label spec built from the
    glob pattern. Emits a DeprecationWarning. Use --label / --selectivity
    going forward.

    Returns the legacy 6-tuple: (ee, er, ratio, dd_free_energy, failed,
    pref) so the existing CLI print path keeps working unchanged.
    """
    warnings.warn(
        "get_selectivity / --ee is deprecated; use --label / "
        "--selectivity instead. Will be removed in v5.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    parts = pattern.split(':')
    if len(parts) != 2:
        raise ValueError(
            f"Invalid selectivity pattern '{pattern}'. "
            "Expected format: 'pattern_a:pattern_b' with exactly one colon."
        )
    a_regex, b_regex = parts[0].strip(), parts[1].strip()
    if not a_regex or not b_regex:
        raise ValueError(
            f"Invalid selectivity pattern '{pattern}'. "
            "Both patterns before and after ':' must be non-empty."
        )

    A = ''.join(a for a in a_regex if a.isalnum())
    B = ''.join(b for b in b_regex if b.isalnum())

    # Legacy behavior: glob the filesystem (one or more parent dirs), not
    # fnmatch against `files`.
    dirs = list(set(os.path.dirname(f) for f in files))
    a_files, b_files = [], []
    if len(dirs) > 1 or (dirs and dirs[0] != ''):
        for d in dirs:
            a_files.extend(glob(d + '/' + a_regex))
            b_files.extend(glob(d + '/' + b_regex))
    else:
        a_files.extend(glob(a_regex))
        b_files.extend(glob(b_regex))

    if not a_files or not b_files:
        log.info("\n   Warning! Filenames have not been formatted correctly for determining selectivity\n")
        log.info("   Make sure the filename contains either " + A + " or " + B + "\n")
        sys.exit("   Please edit either your filenames or selectivity pattern argument and try again\n")

    # Restrict to files we have thermo data for, and that haven't been
    # marked as duplicates (legacy: only dup[0] is excluded here).
    excluded = set(dup[0] for dup in (dup_list or []))
    a_files = [f for f in a_files if f in boltz_facs and f not in excluded]
    b_files = [f for f in b_files if f in boltz_facs and f not in excluded]
    a_sum = sum(boltz_facs[f] for f in a_files)
    b_sum = sum(boltz_facs[f] for f in b_files)

    A_round = round(a_sum * 100)
    B_round = round(b_sum * 100)
    er = f'{A_round}:{B_round}'
    failed = False
    if a_sum > b_sum:
        pref = A
        try:
            r = a_sum / b_sum
            ratio = f'{round(r, 1)}:1' if r < 3 else f'{round(r)}:1'
        except ZeroDivisionError:
            ratio = '1:0'
    else:
        pref = B
        try:
            r = b_sum / a_sum
            ratio = f'1:{round(r, 1)}' if r < 3 else f'1:{round(r)}'
        except ZeroDivisionError:
            ratio = '0:1'

    ee = (a_sum - b_sum) * 100.0
    if ee == 0:
        log.info("\n   Warning! No files found for an enantioselectivity analysis, adjust the stereodetermining step name and try again.\n")
        failed = True
    ee = abs(ee)
    try:
        dd_free_energy = GAS_CONSTANT / J_TO_AU * temperature * math.log((50 + ee / 2.0) / (50 - ee / 2.0)) * KCAL_TO_AU
    except (ZeroDivisionError, ValueError):
        dd_free_energy = 0.0
    return ee, er, ratio, dd_free_energy, failed, pref
