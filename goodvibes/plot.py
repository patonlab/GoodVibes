"""Visualization for GoodVibes.

Pure plotting layer that renders the v4.2+ structured types
(`PESResult`, `SelectivityResult`, `ThermoResult`) to matplotlib axes.
matplotlib is an optional dependency; install with
`pip install goodvibes[plot]` (or `pip install matplotlib` directly).

All functions accept an optional `ax=None`; when None, they create a
fresh `plt.subplots()` figure and return the axes. The `plt` import
is deferred so just `import goodvibes.plot` doesn't fail when
matplotlib is missing — only the call-site fails, with a clear message.

Public API:
    plot_profile(pes_result, series=..., ...)        — reaction profile (ProfileAxes)
    plot_pes(pes_result, ax=None, **kw)              — 4.2-4.5 shim over plot_profile
    plot_selectivity_strip(selectivity,
                           thermo_lookup, ax=None)   — per-species scatter
    plot_boltzmann_histogram(results, ax=None)       — population bars
    plot_temperature_scan(results_per_T, ax=None)    — thermo vs T

The first two are implemented; the latter two are stubs that raise
`NotImplementedError` to lock in the API while leaving room for v5.1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from .constants import canonical_units, hartree_factor


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
    title: Optional[str] = None,
    jitter: float = 0.04,
    seed: int = 0,
):
    """Per-species strip plot of conformer ΔG values.

    Each species is one column; conformer ΔG values (relative to the
    lowest across all species) are scattered as dots.

    Parameters:
        selectivity: a `SelectivityResult` (from
            `goodvibes.selectivity.compute_selectivity`) — provides the
            label order, populations, and `files_per_label`.
        thermo_lookup: either a `{file_path: qh_gibbs_free_energy}`
            mapping (Hartree) or a callable that takes a path and
            returns the same. Used to read the per-conformer ΔG.
        ax: optional matplotlib Axes. New figure created if None.
        units: 'kcal/mol' (default), 'kJ/mol', 'eV' or 'hartree'.
        title: figure title; auto-generated from the result's
            temperature and key when None.
        jitter: horizontal spread of conformer dots within each
            species column (fraction of column width).
        seed: RNG seed for the jitter (deterministic plots).

    Returns:
        The matplotlib Axes the strip plot was drawn on.
    """
    import random

    plt = _import_matplotlib()

    # Convert hartree → user units (raises ValueError on an unknown unit).
    units = canonical_units(units)
    scale = hartree_factor(units)

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

    n_labels = len(selectivity.labels)
    if ax is None:
        # Compact aspect: ~0.8" per species column with a small base
        # margin for the y-axis tick labels. Override by passing
        # `ax=` from your own `plt.subplots(figsize=...)` if you want
        # a different shape.
        _, ax = plt.subplots(figsize=(max(3.0, 0.8 * n_labels + 1.4), 4))

    rng = random.Random(seed)

    for x, label in enumerate(selectivity.labels):
        rels = rel_per_species[label]
        if not rels:
            continue
        xs = [x + (rng.random() - 0.5) * 2 * jitter for _ in rels]
        ax.scatter(xs, rels, alpha=0.6, edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(n_labels))
    ax.set_xticklabels(selectivity.labels)
    # Tighten the x-axis to the column footprint so dots don't float
    # in oversized whitespace; matplotlib's default autoscale pads
    # noticeably for small-N scatter.
    ax.set_xlim(-0.5, n_labels - 0.5)
    ax.set_ylabel(rf"$G_\mathrm{{rel}}$ ({units})")
    if title is None:
        title = (f"Selectivity strip ({selectivity.key}, "
                 f"T = {selectivity.temperature:.2f} K)")
    ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# Reaction profile (plot_profile) and the plot_pes shim
# ---------------------------------------------------------------------------

def _rollup_vector(cset, T, rollup_kw):
    """The species-level ThermoVector Point.thermo uses for `cset`."""
    return cset.rollup(T, **rollup_kw)


_LINESTYLES = ("-", "--", ":", "-.")


@dataclass
class ProfileAxes:
    """What ``plot_profile`` drew, with the numbers behind it.

    ``levels`` is {series id: {pathway name: {point label: value}}} in
    ``units``: the same evaluation the bars were drawn from, so a table
    written from it cannot disagree with the figure. ``order`` is the
    merged x order (point labels) and ``x`` maps a label to its position.
    """
    figure: Any
    axes: List[Any]
    levels: Dict[str, Dict[str, Dict[str, Optional[float]]]]
    units: str
    order: List[str]
    x: Dict[str, float]
    series: List[Any]
    pathways: List[Any]
    colors: Dict[str, Any]
    linestyles: Dict[str, str]
    layout: str = "overlay"

    @property
    def ax(self):
        """The (first) matplotlib Axes."""
        return self.axes[0]

    def axes_for(self, pathway) -> Any:
        """The Axes a pathway (name or object) was drawn on."""
        name = pathway if isinstance(pathway, str) else pathway.name
        names = [p.name for p in self.pathways]
        if name not in names:
            raise KeyError(f"no pathway {name!r} in this figure")
        return self.axes[names.index(name)] if self.layout == "panels" else self.axes[0]

    def level(self, pathway, point: str, series=None) -> Optional[float]:
        """A drawn level in ``units``. ``series`` defaults to the first."""
        name = pathway if isinstance(pathway, str) else pathway.name
        sid = self._series_id(series)
        return self.levels[sid][name].get(point)

    def _series_id(self, series) -> str:
        if series is None:
            return self.series[0].id
        return series if isinstance(series, str) else series.id

    def annotate_barrier(self, pathway, src: str, dst: str, series=None, *,
                         fmt: str = "{:+.1f}", offset: float = 0.3, color=None, ax=None):
        """Mark the difference dst − src on one pathway with a vertical
        double-headed arrow beside ``dst`` and the value in ``units``.
        Returns the matplotlib annotation."""
        name = pathway if isinstance(pathway, str) else pathway.name
        sid = self._series_id(series)
        y0 = self.levels[sid][name].get(src)
        y1 = self.levels[sid][name].get(dst)
        if y0 is None or y1 is None:
            raise ValueError(f"annotate_barrier: no level for {src!r} -> {dst!r} in series {sid!r}")
        axis = ax if ax is not None else self.axes_for(name)
        # Beside dst, on the side facing the profile: to the right, unless dst
        # is the last column (the label would leave the axes).
        last = self.x[dst] >= max(self.x.values())
        xa = self.x[dst] - offset if last else self.x[dst] + offset
        col = color if color is not None else self.colors.get(name, "k")
        axis.annotate("", xy=(xa, y1), xytext=(xa, y0),
                      arrowprops=dict(arrowstyle="<->", color=col, linewidth=0.8, shrinkA=0, shrinkB=0))
        return axis.annotate(fmt.format(y1 - y0), (xa, (y0 + y1) / 2), xytext=(-3 if last else 3, 0),
                             textcoords="offset points", ha="right" if last else "left", va="center",
                             fontsize="x-small", color=col)

    def save(self, *paths: str, dpi: int = 200, bbox_inches: str = "tight", **kw) -> None:
        """Write the figure to each path (format by extension)."""
        for path in paths:
            self.figure.savefig(path, dpi=dpi, bbox_inches=bbox_inches, **kw)

    def close(self) -> None:
        _import_matplotlib().close(self.figure)


def _resolve_pathways(pes_result, pathways):
    if pathways is None:
        return list(pes_result.pathways)
    if isinstance(pathways, (int, str)):
        pathways = [pathways]
    return [pes_result.pathway(p) if isinstance(p, (int, str)) else p for p in pathways]


def _resolve_series(pes_result, series, quantity, temperatures):
    """The Series objects to draw (see plot_profile)."""
    from .pes_model import Series
    if series is not None:
        if isinstance(series, (str, Series)):
            series = [series]
        out = []
        for s in series:
            if isinstance(s, Series):
                out.append(s)
            else:
                match = [x for x in pes_result.series if x.id == s]
                if not match:
                    raise KeyError(f"no series {s!r} in the result "
                                   f"(available: {[x.id for x in pes_result.series]})")
                out.append(match[0])
        if quantity is not None or temperatures is not None:
            raise ValueError("plot_profile: pass either series= or quantity=/temperatures=, not both")
        return out
    if pes_result.series and quantity is None and temperatures is None:
        return list(pes_result.series)
    return pes_result.default_series(quantity or "qh_gibbs", temperatures)


def _resolve_colors(plt, pathways, colors):
    if colors is None:
        if len(pathways) == 1:
            return {pathways[0].name: "k"}
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
        return {p.name: cycle[i % len(cycle)] for i, p in enumerate(pathways)}
    if isinstance(colors, Mapping):
        missing = [p.name for p in pathways if p.name not in colors]
        if missing:
            raise ValueError(f"plot_profile: no color for pathway(s) {missing}")
        return {p.name: colors[p.name] for p in pathways}
    colors = list(colors)
    if len(colors) < len(pathways):
        raise ValueError(
            f"plot_profile: need at least {len(pathways)} colors, got {len(colors)}"
        )
    return {p.name: c for p, c in zip(pathways, colors)}


def _draw_connector(ax, x0, y0, x1, y1, *, style, color, linestyle, linewidth, zorder=2):
    import matplotlib.patches as mpatches
    import matplotlib.path as mpath
    Path = mpath.Path
    if style == "bezier":
        # Cubic Bezier with horizontal handles, control points at the
        # midpoint x and y0 / y1: the curve passes under each bar (drawn
        # on top) so the bar reads as a plateau on a continuous line.
        xm = (x0 + x1) / 2
        patch = mpatches.PathPatch(
            Path([(x0, y0), (xm, y0), (xm, y1), (x1, y1)],
                 [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]),
            fc="none", color=color, linewidth=linewidth, linestyle=linestyle, zorder=zorder,
        )
        ax.add_patch(patch)
    elif style == "step":
        xm = (x0 + x1) / 2
        ax.plot([x0, xm, xm, x1], [y0, y0, y1, y1], color=color,
                linewidth=linewidth, linestyle=linestyle, zorder=zorder)
    else:
        ax.plot([x0, x1], [y0, y1], color=color, linewidth=linewidth,
                linestyle=linestyle, zorder=zorder)


def plot_profile(
    pes_result: Any,
    *,
    series=None,
    quantity: Optional[str] = None,
    temperatures: Optional[Sequence[float]] = None,
    pathways=None,
    layout: str = "overlay",
    ax=None,
    style: Optional[Mapping[str, Any]] = None,
    colors=None,
    show_conformers: bool = False,
    label_points: bool = False,
    title: Optional[str] = None,
    order: Optional[Sequence[str]] = None,
) -> ProfileAxes:
    """Draw a reaction profile from a ``PESResult``.

    Every pathway is drawn on a shared x axis whose order is the merge of
    the pathways' point sequences (``PESResult.merged_order``), so
    pathways of different lengths, or branches sharing a reactant, line
    up by point label. Colour encodes the pathway; linestyle encodes the
    series (a quantity at a temperature, computed from the model or
    declared). Transition-state points (``role='ts'``) carry their value
    label above the bar, other points below; a ``barrierless`` edge is a
    dotted connector; an edge of kind ``none`` draws no connector.

    Parameters:
        pes_result: a `PESResult` (``goodvibes.load_pes`` or built by hand).
        series: what to draw. A ``Series`` / list of ``Series`` (computed or
            declared), or series ids from ``pes_result.series``. Default:
            ``pes_result.series`` when it declares any, else one computed
            series of ``quantity`` per temperature.
        quantity: registry id or alias for the implicit series (default
            'qh_gibbs'); the y-label follows it.
        temperatures: temperatures of the implicit series (default: the
            result's); several give a temperature overlay.
        pathways: which pathways to draw (names, indices or objects);
            default all.
        layout: 'overlay' (all pathways on one axes) or 'panels' (one
            axes per pathway, shared y).
        ax: an Axes to draw on (overlay only); a new figure otherwise.
        style: {'connector': 'bezier' | 'linear' | 'step', 'bar_half':
            float, 'decimals': int, 'figsize': (w, h), 'linestyles': [...]}.
        colors: per-pathway colours, a sequence in pathway order or a
            {name: colour} mapping. Default: black for one pathway, the
            matplotlib cycle otherwise.
        show_conformers: scatter each conformer at level + (conformer −
            species rollup) in the plotted quantity (computed series only).
        label_points: print each level's value next to its bar.
        title: figure title (default: pathway names, and the temperature
            when there is one).
        order: explicit x order of point labels (overrides the result's).

    Returns:
        A ``ProfileAxes`` with the figure, axes and the drawn levels.
    """
    from .quantities import resolve_quantity
    plt = _import_matplotlib()

    style = dict(style or {})
    connector = style.get("connector", "bezier")
    if connector not in ("bezier", "linear", "step"):
        raise ValueError(f"style['connector'] must be 'bezier', 'linear' or 'step', got {connector!r}")
    if layout not in ("overlay", "panels"):
        raise ValueError(f"layout must be 'overlay' or 'panels', got {layout!r}")
    bar_half = float(style.get("bar_half", 0.15))
    decimals = int(style.get("decimals", 1))
    linestyles = list(style.get("linestyles", _LINESTYLES))
    bar_lw, connector_lw = 1.5, 1.0

    paths = _resolve_pathways(pes_result, pathways)
    if not paths:
        raise ValueError("plot_profile: no pathways to draw")
    series_list = _resolve_series(pes_result, series, quantity, temperatures)
    if not series_list:
        raise ValueError("plot_profile: no series to draw")
    units = pes_result.options.units

    # x order and positions
    if order is not None:
        order = list(order)
    elif pes_result.order:
        order = list(pes_result.order)
    else:
        from .pes_model import merge_point_order
        order = merge_point_order([p.labels for p in paths])
    for p in paths:
        missing = [lab for lab in p.labels if lab not in order]
        if missing:
            raise ValueError(f"plot_profile: point(s) {missing} of pathway {p.name!r} are not in the x order")
    xpos = {label: float(i) for i, label in enumerate(order)}
    n_points = len(order)

    # Evaluate every series on every pathway (user units)
    levels: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for s in series_list:
        levels[s.id] = {}
        for p in paths:
            vals = s.evaluate(pes_result, p)
            if not s.declared and any(v is None for v in vals.values()):
                raise ValueError(
                    f"plot_profile: quantity {s.quantity!r} is not available for every point "
                    f"of pathway {p.name!r} (no single-point energy?)")
            levels[s.id][p.name] = vals

    colors_by_path = _resolve_colors(plt, paths, colors)
    # Series without an explicit linestyle take the next style from the cycle
    # that no other series asked for, so defaults never repeat a chosen one.
    _aliases = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}
    explicit = {_aliases.get(s.style["linestyle"], s.style["linestyle"])
                for s in series_list if "linestyle" in s.style}
    free = [ls for ls in linestyles if _aliases.get(ls, ls) not in explicit] or linestyles
    ls_by_series = {}
    n_default = 0
    for s in series_list:
        if "linestyle" in s.style:
            ls_by_series[s.id] = s.style["linestyle"]
        else:
            ls_by_series[s.id] = free[n_default % len(free)]
            n_default += 1

    # Figure / axes
    figsize = style.get("figsize")
    if layout == "panels":
        if ax is not None:
            raise ValueError("plot_profile: ax= cannot be combined with layout='panels'")
        if figsize is None:
            figsize = (max(5, 0.9 * n_points + 1), 3.2 * len(paths))
        fig, axes = plt.subplots(len(paths), 1, sharey=True, sharex=True, figsize=figsize, squeeze=False)
        axes = [a[0] for a in axes]
    else:
        if ax is None:
            if figsize is None:
                figsize = (max(5, 0.9 * n_points + 1), 4)
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        axes = [ax]

    def _axis_for(i):
        return axes[i] if layout == "panels" else axes[0]

    rng = None
    for pi, path in enumerate(paths):
        axis = _axis_for(pi)
        color = colors_by_path[path.name]
        for si, s in enumerate(series_list):
            ls = ls_by_series[s.id]
            label_shift = 6 + 8 * si          # stack value labels when several series share a bar
            lv = levels[s.id][path.name]
            # bars
            for point in path.points:
                y = lv.get(point.label)
                if y is None:
                    continue
                x = xpos[point.label]
                if s.declared:
                    axis.hlines(y, x - bar_half, x + bar_half, colors=color, linewidth=bar_lw,
                                linestyle=ls, zorder=3)
                    axis.plot([x], [y], marker="o", markersize=4, markerfacecolor="white",
                              markeredgecolor=color, linestyle="none", zorder=4)
                else:
                    axis.hlines(y, x - bar_half, x + bar_half, colors=color, linewidth=bar_lw, zorder=3)
                if label_points:
                    above = point.is_ts
                    axis.annotate(f"{y:.{decimals}f}", (x, y),
                                  xytext=(0, label_shift if above else -label_shift), textcoords="offset points",
                                  ha="center", va="bottom" if above else "top",
                                  fontsize="x-small", color=color)
            # connectors along the edges
            for edge in path.edges:
                if edge.kind == "none":
                    continue
                y0, y1 = lv.get(edge.src), lv.get(edge.dst)
                if y0 is None or y1 is None:
                    continue
                edge_ls = ":" if edge.kind == "barrierless" else ls
                _draw_connector(axis, xpos[edge.src], y0, xpos[edge.dst], y1,
                                style=connector, color=color, linestyle=edge_ls,
                                linewidth=connector_lw)
            # conformer dots: level + (conformer − species rollup) in the plotted quantity
            if show_conformers and not s.declared:
                import random
                rng = rng or random.Random(0)
                T = s.temperature if s.temperature is not None else pes_result.temperature
                qid = resolve_quantity(s.quantity).id
                factor = hartree_factor(units)
                rollup_kw = pes_result.options.rollup_kw
                for point in path.points:
                    base = lv.get(point.label)
                    if base is None:
                        continue
                    for _coeff, cset in point.species:
                        if cset.is_single:
                            continue
                        rollup = _rollup_vector(cset, T, rollup_kw).get(qid, T)
                        for vec in cset.vectors(T):
                            conf = vec.get(qid, T)
                            if conf is None or rollup is None:
                                continue
                            y = base + (conf - rollup) * factor
                            x = xpos[point.label] + (rng.random() - 0.5) * 0.3
                            axis.scatter([x], [y], alpha=0.4, s=18, color=color, zorder=4)

    # Axis furniture
    display = {}
    for p in paths:
        for point in p.points:
            display.setdefault(point.label, point.display_label)
    quantities = {s.quantity for s in series_list}
    if len(quantities) == 1:
        ylabel = f"{resolve_quantity(next(iter(quantities))).label} ({units})"
    else:
        ylabel = f"relative energy ({units})"
    for i, axis in enumerate(axes):
        axis.set_xticks(list(range(n_points)))
        axis.set_xticklabels([display.get(lab, lab) for lab in order],
                             rotation=15, ha="right", fontsize="small")
        axis.set_ylabel(ylabel)
        axis.margins(y=0.1)      # room for the value labels above TS bars and below minima
        axis.minorticks_on()
        axis.tick_params(axis='x', which='minor', bottom=False, top=False)
        axis.tick_params(axis='y', which='both', labelright=True, right=True)
        if layout == "panels":
            axis.set_title(paths[i].name, fontsize="small", loc="left")

    if title is None:
        names = ", ".join(p.name for p in paths)
        temps = {s.temperature if s.temperature is not None else pes_result.temperature
                 for s in series_list if not s.declared}
        if len(temps) == 1:
            title = f"{names}  (T = {next(iter(temps)):g} K)"
        else:
            title = names
    if layout == "panels":
        fig.suptitle(title)
    else:
        axes[0].set_title(title)

    # Legend: pathway colours (when more than one pathway on an axes) and
    # series linestyles (when more than one series). Series labels already
    # name the quantity / temperature / method they represent.
    from matplotlib.lines import Line2D
    handles = []
    if layout == "overlay" and len(paths) > 1:
        handles += [Line2D([], [], color=colors_by_path[p.name], label=p.name) for p in paths]
    if len(series_list) > 1:
        for s in series_list:
            handles.append(Line2D([], [], color="k", linestyle=ls_by_series[s.id],
                                  marker="o" if s.declared else None, markerfacecolor="white",
                                  label=s.label))
    if handles:
        axes[0].legend(handles=handles, loc="best", fontsize="small")

    return ProfileAxes(
        figure=fig, axes=axes, levels=levels, units=units, order=order, x=xpos,
        series=series_list, pathways=paths, colors=colors_by_path,
        linestyles=ls_by_series, layout=layout,
    )


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
    quantity: str = "qh_gibbs",
):
    """Plot one or more pathways from a `PESResult` as a reaction profile.

    A thin wrapper over :func:`plot_profile` kept for the 4.2–4.5 API; it
    returns the matplotlib Axes. New code should call ``plot_profile``,
    which also returns the drawn levels and supports several series
    (temperatures, quantities, declared values) on one axes.

    Parameters:
        pes_result: a `PESResult` from `goodvibes.pes_loader.load_pes`.
        ax: optional matplotlib Axes; new figure if None.
        pathway_index: which pathway(s) to draw (None → all, an int, or a
            list of ints).
        connector_style: 'bezier' (default) or 'linear'.
        colors: per-pathway colors in pathway order.
        show_conformers: scatter individual conformer values around each
            level.
        thermo_lookup: deprecated and ignored (conformer values are read
            from the PESResult); accepted for backwards compatibility.
        title: figure title; defaults to the pathway names + temperature.
        label_points: annotate each point's value next to its bar.
        quantity: which relative quantity to draw, by registry id or alias
            (see goodvibes.quantities); the y-label follows the choice.

    Returns:
        The matplotlib Axes the profile(s) were drawn on.
    """
    if connector_style not in ("bezier", "linear"):
        raise ValueError(
            f"connector_style must be 'bezier' or 'linear', got {connector_style!r}"
        )
    if thermo_lookup is not None:
        import warnings
        warnings.warn(
            "plot_pes: thermo_lookup is no longer needed for show_conformers "
            "and is ignored; conformer values are read from the PESResult.",
            DeprecationWarning, stacklevel=2)
    if pathway_index is None:
        pathways = None
    elif isinstance(pathway_index, int):
        pathways = [pathway_index]
    else:
        pathways = list(pathway_index)
    try:
        profile = plot_profile(
            pes_result, quantity=quantity, temperatures=[pes_result.temperature],
            pathways=pathways, ax=ax, style={"connector": connector_style},
            colors=colors, show_conformers=show_conformers, label_points=label_points,
            title=title,
        )
    except ValueError as exc:
        raise ValueError(str(exc).replace("plot_profile:", "plot_pes:", 1)) from None
    return profile.ax


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

    Not yet implemented; planned for 5.1 over ``ConformerSet.populations``
    (ROADMAP milestone M3).
    """
    raise NotImplementedError(
        "plot_boltzmann_histogram is reserved for v5.1; "
        "see ROADMAP.md, milestone M3."
    )


def plot_temperature_scan(
    results_per_T: Sequence[tuple],
    *,
    ax=None,
):
    """Plot thermochemistry quantities (qh-G, S, H) vs temperature.

    Not yet implemented; planned for 5.1 over computed ``Series`` at
    several temperatures (ROADMAP milestone M3).
    """
    raise NotImplementedError(
        "plot_temperature_scan is reserved for v5.1; "
        "see ROADMAP.md, milestone M3."
    )
