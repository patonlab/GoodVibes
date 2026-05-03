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
    pathway_index: Optional[Union[int, Sequence[int]]] = None,
    connector_style: str = "bezier",
    colors: Optional[Sequence[str]] = None,
    show_conformers: bool = False,
    thermo_lookup: Optional[Union[Mapping[str, float], Callable[[str], float]]] = None,
    title: Optional[str] = None,
    label_points: bool = False,
):
    """Plot one or more pathways from a `PESResult` as a reaction profile.

    Multi-pathway YAMLs (e.g. comparing R vs S transition states across
    a shared set of points) are overlaid on the same axes by default,
    each pathway in its own color from matplotlib's color cycle.

    Parameters:
        pes_result: a `PESResult` from `goodvibes.pes_loader.load_pes`.
        ax: optional matplotlib Axes; new figure if None.
        pathway_index: which pathway(s) to draw.
            None (default) → all pathways overlaid.
            int → just that pathway.
            list of ints → that subset.
        connector_style: 'bezier' (default; smooth curved jumps between
            steps, like the legacy --graph plot) or 'linear' (straight
            line between step ends).
        colors: per-pathway colors (sequence of matplotlib color specs:
            names, hex, rgb tuples). Falls back to the default color
            cycle when None.
        show_conformers: scatter individual conformer ΔG around each
            step (single-pathway only when used; for multi-pathway
            select one via `pathway_index=N`).
        thermo_lookup: required when `show_conformers=True`; maps each
            conformer file path → qh_gibbs_free_energy (Hartree).
        title: figure title; defaults to the pathway names + temperature.
        label_points: annotate each point's ΔqhG value above its
            horizontal bar.

    Returns:
        The matplotlib Axes the profile(s) were drawn on.
    """
    plt = _import_matplotlib()
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches

    if connector_style not in ("bezier", "linear"):
        raise ValueError(
            f"connector_style must be 'bezier' or 'linear', got {connector_style!r}"
        )

    # Resolve which pathways to draw.
    if pathway_index is None:
        indices = list(range(len(pes_result.pathways)))
    elif isinstance(pathway_index, int):
        indices = [pathway_index]
    else:
        indices = list(pathway_index)
    if not indices:
        raise ValueError("plot_pes: no pathways to draw")
    pathways = [pes_result.pathways[i] for i in indices]

    # All chosen pathways must share the same number of points so the
    # x-axis matches; mixed lengths would need a separate axes per
    # pathway (legacy --graph supports this; deferred to v5.1).
    n_points_per_path = {len(p.points) for p in pathways}
    if len(n_points_per_path) > 1:
        raise ValueError(
            "plot_pes: all selected pathways must have the same number "
            "of points; got "
            + ", ".join(f"{p.name}={len(p.points)}" for p in pathways)
        )
    n_points = pathways[0].points.__len__()

    pes_options = pes_result.options
    T = pes_result.temperatures[0] if pes_result.temperatures else 298.15
    units_factor = pes_options.to_user_units(1.0)
    rollup_kw = dict(
        gconf=pes_options.gconf, QH=pes_options.QH,
        lowest_only=pes_options.lowest_only,
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(max(5, 0.9 * n_points + 1), 4))

    # Resolve the per-pathway color cycle.
    if colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
        colors_resolved = [cycle[i % len(cycle)] for i in range(len(pathways))]
    else:
        if len(colors) < len(pathways):
            raise ValueError(
                f"plot_pes: need at least {len(pathways)} colors, got {len(colors)}"
            )
        colors_resolved = list(colors)

    xs = list(range(n_points))
    bar_half = 0.3
    Path = mpath.Path

    # Draw each pathway as: hlines at each level + curved/linear
    # connectors between consecutive levels.
    for path, color in zip(pathways, colors_resolved):
        rels = path.relative(T, **rollup_kw)
        qhg = [r.qh_gibbs * units_factor for r in rels]

        # Step bars at each level.
        for i in range(n_points):
            ax.hlines(qhg[i], xs[i] - bar_half, xs[i] + bar_half,
                      colors=color, linewidth=2.5,
                      label=path.name if i == 0 else None,
                      zorder=3)
            if label_points:
                ax.annotate(f"{qhg[i]:.1f}", (xs[i], qhg[i]),
                            xytext=(0, 6), textcoords="offset points",
                            ha="center", fontsize="x-small", color=color)

        # Connectors between consecutive levels.
        for i in range(n_points - 1):
            x0, x1 = xs[i] + bar_half, xs[i + 1] - bar_half
            y0, y1 = qhg[i], qhg[i + 1]
            if connector_style == "bezier":
                # Cubic Bezier with horizontal handles — smooth S-curve.
                xm = (x0 + x1) / 2
                patch = mpatches.PathPatch(
                    Path([(x0, y0), (xm, y0), (xm, y1), (x1, y1)],
                         [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]),
                    fc="none", color=color, linewidth=1.0, zorder=2,
                )
                ax.add_patch(patch)
            else:
                ax.plot([x0, x1], [y0, y1], color=color,
                        linewidth=1.0, zorder=2)

    # Optional per-conformer scatter (single-pathway only — the spread
    # is per-pathway; for multi-pathway choose one via `pathway_index`).
    if show_conformers:
        if thermo_lookup is None:
            raise ValueError(
                "plot_pes(show_conformers=True) requires thermo_lookup "
                "(a {file: qh_g_in_Hartree} mapping or callable)."
            )
        if len(pathways) > 1:
            raise ValueError(
                "plot_pes(show_conformers=True) requires a single "
                "pathway; pass `pathway_index=N` to pick one."
            )
        path = pathways[0]
        if isinstance(thermo_lookup, Mapping):
            _lookup = thermo_lookup.__getitem__
        else:
            _lookup = thermo_lookup
        zero_qhg = path.zero.thermo(T, **rollup_kw).qh_gibbs
        import random
        rng = random.Random(0)
        for i, point in enumerate(path.points):
            for _coeff, cset in point.species:
                if cset.is_single:
                    continue
                for bbe in cset.bbes:
                    rel = (bbe.qh_gibbs_free_energy - zero_qhg) * units_factor
                    x = xs[i] + (rng.random() - 0.5) * 0.3
                    ax.scatter([x], [rel], alpha=0.4, s=18,
                               color="black", zorder=4)

    # x-axis: point labels from the first pathway (all share the same
    # length; labels may differ across pathways, so we just take one).
    ax.set_xticks(xs)
    ax.set_xticklabels([p.label for p in pathways[0].points],
                       rotation=15, ha="right", fontsize="small")
    ax.set_ylabel(f"Δqh-G ({pes_options.units})")
    if title is None:
        names = ", ".join(p.name for p in pathways)
        title = f"{names}  (T = {T:g} K)"
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    if len(pathways) > 1:
        ax.legend(loc="best", fontsize="small")
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
