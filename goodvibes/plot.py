"""Visualization for GoodVibes (v5.0 ROADMAP item 14).

Pure plotting layer that renders the v4.2+ structured types
(`PESResult`, `SelectivityResult`, `ThermoResult`) to matplotlib axes.
matplotlib is an optional dependency; install with
`pip install goodvibes[plot]` (or `pip install matplotlib` directly).

All functions accept an optional `ax=None`; when None, they create a
fresh `plt.subplots()` figure and return the axes. The `plt` import
is deferred so just `import goodvibes.plot` doesn't fail when
matplotlib is missing — only the call-site fails, with a clear message.

Public API:
    plot_pes(pes_result, ax=None, **kw)              — reaction profile
    plot_selectivity_strip(selectivity,
                           thermo_lookup, ax=None)   — per-species scatter
    plot_boltzmann_histogram(results, ax=None)       — population bars
    plot_temperature_scan(results_per_T, ax=None)    — thermo vs T

The first two are implemented; the latter two are stubs that raise
`NotImplementedError` to lock in the API while leaving room for v5.1.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

from .constants import KCAL_TO_AU


def _import_matplotlib():
    """Import matplotlib.pyplot with a clear error if it isn't installed."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:                         # pragma: no cover
        raise ImportError(
            "goodvibes.plot requires matplotlib; install with "
            "`pip install goodvibes[plot]` or `pip install matplotlib`."
        ) from exc


# ---------------------------------------------------------------------------
# Selectivity strip plot
# ---------------------------------------------------------------------------

def plot_selectivity_strip(
    selectivity: Any,
    thermo_lookup: Union[Mapping[str, float], Callable[[str], float]],
    *,
    ax=None,
    units: str = "kcal/mol",
    show_boltz_mean: bool = True,
    show_lowest: bool = True,
    title: Optional[str] = None,
    jitter: float = 0.04,
    seed: int = 0,
):
    """Per-species strip plot of conformer ΔG values.

    Visualizes how much of the apparent selectivity is driven by the
    lowest conformer of each species vs. conformer mixing across the
    full ensemble. Each species is one column; conformer ΔG values
    (relative to the lowest across all species) are dots; the
    Boltzmann-weighted mean and the lowest-conformer ΔG are drawn as
    horizontal bars per species.

    Parameters:
        selectivity: a `SelectivityResult` (from
            `goodvibes.selectivity.compute_selectivity`) — provides the
            label order, populations, and `files_per_label`.
        thermo_lookup: either a `{file_path: qh_gibbs_free_energy}`
            mapping (Hartree) or a callable that takes a path and
            returns the same. Used to read the per-conformer ΔG.
        ax: optional matplotlib Axes. New figure created if None.
        units: 'kcal/mol' (default) or 'kJ/mol'.
        show_boltz_mean: draw a horizontal bar at each species'
            Boltzmann-weighted mean ΔG.
        show_lowest: draw a horizontal bar at each species' lowest ΔG.
        title: figure title; auto-generated from the result's
            temperature and key when None.
        jitter: horizontal spread of conformer dots within each
            species column (fraction of column width).
        seed: RNG seed for the jitter (deterministic plots).

    Returns:
        The matplotlib Axes the strip plot was drawn on.
    """
    import math
    import random

    plt = _import_matplotlib()

    # Convert hartree → user units.
    if units == "kJ/mol":
        from .constants import J_TO_AU
        scale = J_TO_AU / 1000.0
    elif units == "kcal/mol":
        scale = KCAL_TO_AU
    else:
        raise ValueError(f"units must be 'kcal/mol' or 'kJ/mol', got {units!r}")

    # Resolve thermo_lookup to a callable.
    if isinstance(thermo_lookup, Mapping):
        _lookup = thermo_lookup.__getitem__
    else:
        _lookup = thermo_lookup

    # Collect ΔG values per species (relative to global minimum).
    per_species: Dict[str, list] = {}
    all_g = []
    for label in selectivity.labels:
        files = selectivity.files_per_label.get(label, [])
        gs = [_lookup(f) for f in files]
        per_species[label] = gs
        all_g.extend(gs)
    if not all_g:
        raise ValueError("plot_selectivity_strip: no conformers to plot")
    g_min = min(all_g)
    rel_per_species = {
        label: [(g - g_min) * scale for g in gs]
        for label, gs in per_species.items()
    }

    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, 1.2 * len(selectivity.labels) + 1.5), 4))

    rng = random.Random(seed)
    bar_half = 0.25

    for x, label in enumerate(selectivity.labels):
        rels = rel_per_species[label]
        if not rels:
            continue
        # Jittered scatter
        xs = [x + (rng.random() - 0.5) * 2 * jitter for _ in rels]
        ax.scatter(xs, rels, alpha=0.6, edgecolor="black", linewidth=0.5)
        # Lowest-conformer marker
        if show_lowest:
            lowest = min(rels)
            ax.hlines(lowest, x - bar_half, x + bar_half,
                      colors="C1", linewidth=2.0, zorder=3,
                      label="lowest" if x == 0 else None)
        # Boltzmann mean marker — derived from populations to avoid
        # re-deriving the math here.
        if show_boltz_mean:
            T = selectivity.temperature
            # Boltzmann weights inferred from populations (scaled by RT).
            # populations[label] is the fraction *of total* in that label,
            # so within-species we need to recompute. Use the conformers
            # themselves: w_i ∝ exp(-ΔG_i / RT), normalised within species.
            RT_kcal = 8.3144621 * T / 1000.0 / 4.184
            if units == "kJ/mol":
                RT = 8.3144621 * T / 1000.0
            else:
                RT = RT_kcal
            ws = [math.exp(-r / RT) for r in rels]
            norm = sum(ws)
            if norm > 0:
                mean = sum(r * w / norm for r, w in zip(rels, ws))
                ax.hlines(mean, x - bar_half, x + bar_half,
                          colors="C0", linewidth=2.0, linestyle="--", zorder=3,
                          label="Boltzmann mean" if x == 0 else None)

    ax.set_xticks(range(len(selectivity.labels)))
    ax.set_xticklabels(selectivity.labels)
    ax.set_ylabel(f"ΔG ({units})")
    if title is None:
        title = (f"Selectivity strip ({selectivity.key}, "
                 f"T = {selectivity.temperature:.2f} K)")
    ax.set_title(title)
    if show_boltz_mean or show_lowest:
        ax.legend(loc="best", fontsize="small")
    return ax


# ---------------------------------------------------------------------------
# PES profile (clean rewrite consuming PESResult)
# ---------------------------------------------------------------------------

def plot_pes(
    pes_result: Any,
    *,
    ax=None,
    show_conformers: bool = False,
    thermo_lookup: Optional[Union[Mapping[str, float], Callable[[str], float]]] = None,
    title: Optional[str] = None,
    pathway_index: int = 0,
):
    """Plot one pathway from a `PESResult` as a reaction profile.

    Renders the relative qh-G of each point as a step plot. With
    `show_conformers=True`, individual conformers are also scattered
    around each step (requires `thermo_lookup` to read per-conformer
    qh-G values).

    Parameters:
        pes_result: a `PESResult` from `goodvibes.pes_loader.load_pes`.
        ax: optional matplotlib Axes.
        show_conformers: scatter individual conformer ΔG around each
            step. Useful for showing the spread within a step relative
            to the Boltzmann-averaged value the line connects.
        thermo_lookup: required when `show_conformers=True`; maps each
            conformer file path → qh_gibbs_free_energy (Hartree).
        title: figure title; defaults to the pathway name + units.
        pathway_index: which pathway in the result to draw (0-based).
            Multi-pathway plots will get separate calls.

    Returns:
        The matplotlib Axes the profile was drawn on.
    """
    plt = _import_matplotlib()

    pathway = pes_result.pathways[pathway_index]
    pes_options = pes_result.options
    T = pes_result.temperatures[0] if pes_result.temperatures else 298.15
    units_factor = pes_options.to_user_units(1.0)

    rels = pathway.relative(
        T,
        gconf=pes_options.gconf,
        QH=pes_options.QH,
        lowest_only=pes_options.lowest_only,
    )
    qhg = [r.qh_gibbs * units_factor for r in rels]

    if ax is None:
        _, ax = plt.subplots(figsize=(max(5, 0.9 * len(pathway.points) + 1), 4))

    xs = list(range(len(pathway.points)))
    # Step plot — flat line at each point's level, vertical jumps between.
    for i in range(len(xs)):
        ax.hlines(qhg[i], xs[i] - 0.3, xs[i] + 0.3, colors="C0", linewidth=2.5)
    for i in range(len(xs) - 1):
        ax.plot([xs[i] + 0.3, xs[i + 1] - 0.3], [qhg[i], qhg[i + 1]],
                color="C0", linewidth=1.0)

    if show_conformers:
        if thermo_lookup is None:
            raise ValueError(
                "plot_pes(show_conformers=True) requires thermo_lookup "
                "(a {file: qh_g_in_Hartree} mapping or callable)."
            )
        if isinstance(thermo_lookup, Mapping):
            _lookup = thermo_lookup.__getitem__
        else:
            _lookup = thermo_lookup
        zero_th = pathway.zero.thermo(
            T, gconf=pes_options.gconf, QH=pes_options.QH,
            lowest_only=pes_options.lowest_only,
        )
        zero_qhg = zero_th.qh_gibbs
        import random
        rng = random.Random(0)
        for i, point in enumerate(pathway.points):
            for _coeff, cset in point.species:
                if cset.is_single:
                    continue
                for bbe in cset.bbes:
                    rel = (bbe.qh_gibbs_free_energy - zero_qhg) * units_factor
                    x = xs[i] + (rng.random() - 0.5) * 0.3
                    ax.scatter([x], [rel], alpha=0.4, s=18, color="black", zorder=2)

    ax.set_xticks(xs)
    ax.set_xticklabels([p.label for p in pathway.points],
                       rotation=15, ha="right", fontsize="small")
    ax.set_ylabel(f"Δqh-G ({pes_options.units})")
    if title is None:
        title = f"{pathway.name}  (T = {T:g} K)"
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    return ax


# ---------------------------------------------------------------------------
# Stubs for v5.1+ work
# ---------------------------------------------------------------------------

def plot_boltzmann_histogram(
    thermo_results: Sequence[Any],
    *,
    ax=None,
    temperature: float = 298.15,
):
    """Bar chart of per-conformer Boltzmann populations.

    Not yet implemented — slated for v5.1 alongside the Ensemble
    container that gives this a natural data source.
    """
    raise NotImplementedError(
        "plot_boltzmann_histogram is reserved for v5.1; "
        "follow ROADMAP.md item 14."
    )


def plot_temperature_scan(
    results_per_T: Sequence[tuple],
    *,
    ax=None,
):
    """Plot thermochemistry quantities (qh-G, S, H) vs temperature.

    Not yet implemented — slated for v5.1 alongside a structured
    representation of `--ti` output.
    """
    raise NotImplementedError(
        "plot_temperature_scan is reserved for v5.1; "
        "follow ROADMAP.md item 14."
    )
