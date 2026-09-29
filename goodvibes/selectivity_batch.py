"""Selectivity for many reactions at once, with sensitivity sweeps.

``compute_selectivity_batch`` evaluates a set of selectivity jobs (each a
mapping of competing labels to their conformers) at several temperatures
and, optionally, over a range of quasi-harmonic entropy cutoffs and
conformer energy windows. It returns one row per job and condition, as a
pandas DataFrame or as plain records, ready for a prediction pipeline.
``summarize_selectivity`` turns the sweep into the statement a paper
needs: "ee +92 % (88 to 94 % over s_freq_cutoff 50–150 cm⁻¹)".

Every structure is parsed once; each condition re-evaluates it from its
parsed data (``ComputedEntry``), so a sweep costs no extra parsing.
"""
from __future__ import annotations

import glob
import math
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .constants import KCAL_TO_AU

__all__ = ["compute_selectivity_batch", "summarize_selectivity"]

_GLOB_CHARS = set("*?[")


def _structures(spec, where: str) -> List[Any]:
    """The structures a job label names: a glob pattern or path, a
    ConformerSet, or a sequence of paths, patterns, ThermoResults, QCData
    or ComputedEntries."""
    from .pes_model import ConformerSet
    if isinstance(spec, ConformerSet):
        if not spec.entries:
            raise ValueError(f"{where}: the ConformerSet carries no parsed structures to re-evaluate")
        return list(spec.entries)
    items = [spec] if isinstance(spec, str) or not isinstance(spec, Sequence) else list(spec)
    out: List[Any] = []
    for item in items:
        if isinstance(item, str) and _GLOB_CHARS & set(item):
            matches = sorted(glob.glob(item))
            if not matches:
                raise ValueError(f"{where}: no files match {item!r}")
            out.extend(matches)
        else:
            out.append(item)
    if not out:
        raise ValueError(f"{where}: no structures")
    return out


def _entries(jobs: Mapping[str, Mapping[str, Any]], workers: int, thermo_options: dict):
    """{job: {label: [ComputedEntry]}}, parsing each path once."""
    from .api import ThermoResult, compute_batch
    from .io import QCData
    from .pes_model import ComputedEntry
    specs = {job: {label: _structures(spec, f"job {job!r}, label {label!r}") for label, spec in labels.items()}
             for job, labels in jobs.items()}
    paths = sorted({item for labels in specs.values() for items in labels.values()
                    for item in items if isinstance(item, str)})
    parsed = dict(zip(paths, compute_batch(paths, jobs=workers, **thermo_options))) if paths else {}

    def entry(item):
        if isinstance(item, ComputedEntry):
            return item
        if isinstance(item, str):
            return ComputedEntry.from_result(parsed[item])
        if isinstance(item, ThermoResult):
            return ComputedEntry.from_result(item)
        if isinstance(item, QCData):
            return ComputedEntry.from_result(compute_batch([item], **thermo_options)[0])
        raise TypeError(f"cannot use a {type(item).__name__} as a structure (a path, ThermoResult, QCData, "
                        "ComputedEntry or ConformerSet is expected)")

    out = {}
    for job, labels in specs.items():
        if len(labels) < 2:
            raise ValueError(f"job {job!r}: a selectivity needs at least two labels")
        out[job] = {label: [entry(item) for item in items] for label, items in labels.items()}
    return out


def compute_selectivity_batch(
    jobs: Mapping[str, Mapping[str, Any]],
    temperatures: Sequence[float] = (298.15,),
    *,
    quantity: str = "qh_gibbs",
    s_freq_cutoffs: Optional[Sequence[float]] = None,
    conformer_windows: Optional[Sequence[float]] = None,
    records: bool = False,
    workers: int = 1,
    **thermo_options: Any,
):
    """Selectivities of many jobs over temperatures, entropy cutoffs and
    conformer windows.

    Parameters:
        jobs: ``{job: {label: structures}}``, the labels of each job in
            the order that sets the sign of ``ee`` (positive when the
            first label is major). ``structures`` is a glob pattern or
            path, a ``ConformerSet``, or a list of paths, patterns,
            ``ThermoResult``, ``QCData`` or ``ComputedEntry`` objects.
        temperatures: K.
        quantity: the registry quantity the conformers are weighted by
            (``'qh_gibbs'``; ``'electronic'`` for energy-only ensembles
            such as ``read_xyz_frames`` frames).
        s_freq_cutoffs: quasi-harmonic entropy cutoffs (cm⁻¹) to sweep;
            the nominal cutoff (``s_freq_cutoff`` in ``thermo_options``,
            100 by default) is always included.
        conformer_windows: energy windows (kcal/mol) to sweep: a label
            keeps the conformers within the window of its lowest one at
            that condition (0 keeps only the lowest). All conformers
            (window None, the nominal) are always included.
        records: return a list of dicts instead of a pandas DataFrame.
        workers: processes used to parse the files (``compute_batch``'s
            ``jobs``).
        thermo_options: ``compute_thermo`` keywords (``QS``, ``QH``,
            ``spc``, ``freq_scale_factor``, ...), applied to every file.

    Returns:
        One row per job, temperature, cutoff and window: ``job``,
        ``temperature``, ``s_freq_cutoff``, ``conformer_window``,
        ``nominal`` (the nominal cutoff and all conformers), ``quantity``,
        ``labels`` (comma-joined), ``major``, ``ee`` (signed, two labels),
        ``ddG`` (kcal/mol, major over runner-up), ``ratio``, and per label
        ``population[<label>]`` (0–1) and ``n[<label>]`` (conformers
        kept). Rows follow the job order, then temperature, cutoff and
        window (None first).
    """
    from .quantities import resolve_quantity
    from .selectivity import selectivity_from_energies
    qid = resolve_quantity(quantity).id
    nominal_cutoff = float(thermo_options.get("s_freq_cutoff", 100.0))
    cutoffs = sorted({nominal_cutoff, *(float(c) for c in (s_freq_cutoffs or ()))})
    windows: List[Optional[float]] = [None] + sorted({float(w) for w in (conformer_windows or ())})
    if any(w < 0 for w in windows if w is not None):
        raise ValueError("conformer windows must be ≥ 0 kcal/mol")
    temps = [float(t) for t in temperatures]
    if not temps or any(t <= 0 for t in temps):
        raise ValueError("temperatures must be positive")

    entries = _entries(jobs, workers, thermo_options)
    rows: List[Dict[str, Any]] = []
    for job, labels in entries.items():
        for T in temps:
            for cutoff in cutoffs:
                values = {}
                for label, ents in labels.items():
                    vals = []
                    for e in ents:
                        opts = replace(e.options, s_freq_cutoff=cutoff)
                        v = e.thermo(T, opts).get(qid, T)
                        if v is None:
                            raise ValueError(f"job {job!r}, label {label!r}: {e.file or 'a structure'} has no "
                                             f"{qid!r} (an energy-only structure needs quantity='electronic')")
                        vals.append(v)
                    values[label] = vals
                for window in windows:
                    kept = values if window is None else {
                        label: [v for v in vals if v - min(vals) <= window / KCAL_TO_AU + 1e-12]
                        for label, vals in values.items()}
                    r = selectivity_from_energies(kept, T, quantity=qid)
                    row: Dict[str, Any] = {
                        "job": job, "temperature": T, "s_freq_cutoff": cutoff, "conformer_window": window,
                        "nominal": cutoff == nominal_cutoff and window is None, "quantity": qid,
                        "labels": ",".join(r.labels), "major": r.major, "ee": r.ee_signed,
                        "ddG": r.ddG * KCAL_TO_AU if r.ddG is not None else None, "ratio": r.ratio,
                    }
                    for label in r.labels:
                        row[f"population[{label}]"] = r.populations[label]
                        row[f"n[{label}]"] = len(kept[label])
                    rows.append(row)
    if records:
        return rows
    try:
        import pandas as pd
    except ImportError:                                     # pragma: no cover - pandas is in [full]
        raise ImportError("compute_selectivity_batch returns a pandas DataFrame; install pandas "
                          "(pip install goodvibes[full]) or pass records=True") from None
    return pd.DataFrame(rows)


def summarize_selectivity(table, value: str = "ee", *, decimals: int = 0) -> List[Dict[str, Any]]:
    """The nominal value of each job at each temperature and its range over
    the sweep: ``{job, temperature, value, nominal, low, high, text}``,
    with ``text`` such as ``"ee +92 % (88 to 94 % over s_freq_cutoff
    50–150 cm⁻¹, conformer window 0–2 kcal/mol)"``.

    ``table`` is what ``compute_selectivity_batch`` returned (DataFrame or
    records); ``value`` is a numeric column (``ee``, ``ddG``, ``ratio`` or
    a ``population[<label>]``).
    """
    rows = table.to_dict("records") if hasattr(table, "to_dict") else list(table)
    units = {"ee": " %", "ddG": " kcal/mol", "ratio": ":1"}.get(value, "")
    groups: Dict[tuple, List[dict]] = {}
    for row in rows:
        groups.setdefault((row["job"], row["temperature"]), []).append(row)
    out = []
    for (job, T), group in groups.items():
        nominal = next((r for r in group if r["nominal"]), None)
        vals = [r[value] for r in group if _number(r.get(value))]
        if nominal is None or not _number(nominal.get(value)) or not vals:
            continue
        low, high = min(vals), max(vals)
        nom = nominal[value]
        sign = "+" if value == "ee" else ""
        text = f"{value} {nom:{sign}.{decimals}f}{units}"
        swept = []
        cutoffs = sorted({r["s_freq_cutoff"] for r in group})
        if len(cutoffs) > 1:
            swept.append(f"s_freq_cutoff {cutoffs[0]:g}–{cutoffs[-1]:g} cm⁻¹")
        windows = sorted({r["conformer_window"] for r in group if _number(r["conformer_window"])})
        if windows:
            swept.append(f"conformer window {windows[0]:g}–{windows[-1]:g} kcal/mol"
                         if len(windows) > 1 else f"conformer window {windows[0]:g} kcal/mol")
        if swept and f"{low:.{decimals}f}" != f"{high:.{decimals}f}":     # a range the precision shows
            text += f" ({low:.{decimals}f} to {high:.{decimals}f}{units} over {', '.join(swept)})"
        out.append({"job": job, "temperature": T, "value": value, "nominal": nom,
                    "low": low, "high": high, "text": text})
    return out


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v)
