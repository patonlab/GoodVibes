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
    STYLE_PRESETS / resolve_preset(name)             — figure presets for plot_profile
    plot_pes(pes_result, ax=None, **kw)              — 4.2-4.5 shim over plot_profile
    plot_selectivity_strip(selectivity,
                           thermo_lookup, ax=None)   — per-species scatter
    plot_boltzmann_histogram(conformers, ...)        — population bars
    plot_temperature_scan(conformers or profile, ...) — thermo or levels vs T
"""
from __future__ import annotations

import io
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

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


def _version() -> str:
    from .constants import __version__
    return __version__


#: id of the ``<metadata>`` element that carries a figure's document.
SVG_METADATA_ID = "goodvibes-reaction-profile"
_SVG_METADATA_RE = re.compile(
    r'<metadata id="' + SVG_METADATA_ID + r'">(.*?)</metadata>', re.DOTALL)


def embed_svg_metadata(svg: str, payload: Mapping[str, Any]) -> str:
    """Insert ``payload`` (JSON) as a ``<metadata id="goodvibes-reaction-profile">``
    element right after the opening ``<svg>`` tag."""
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    data = data.replace("]]>", "]]]]><![CDATA[>")          # a CDATA section cannot contain ']]>'
    m = re.search(r"<svg\b[^>]*>", svg)
    if m is None:
        raise ValueError("not an SVG document: no <svg> element")
    block = f'\n <metadata id="{SVG_METADATA_ID}"><![CDATA[{data}]]></metadata>'
    return svg[:m.end()] + block + svg[m.end():]


def read_svg_metadata(svg: str) -> Optional[dict]:
    """The payload ``ProfileAxes.save`` embedded in an SVG, or None."""
    m = _SVG_METADATA_RE.search(svg)
    if m is None:
        return None
    text = "".join(re.findall(r"<!\[CDATA\[(.*?)\]\]>", m.group(1), re.DOTALL))
    return json.loads(text)


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


@dataclass(frozen=True)
class StylePreset:
    """Figure size, font sizes (pt) and line widths (pt) for one target.

    Applied inside a ``matplotlib.rc_context`` for the figure only, never
    set globally. ``figsize`` is the overlay figure in inches; with
    ``layout='panels'`` each panel is ``panel_height`` tall.
    """
    name: str
    figsize: Tuple[float, float]
    panel_height: float
    font_size: float        # axis labels, tick labels, title
    label_size: float       # value labels, barrier labels, legend
    bar_width: float
    connector_width: float
    axes_width: float
    marker_size: float

    @property
    def rc(self) -> Dict[str, Any]:
        fs, aw = self.font_size, self.axes_width
        return {
            "font.size": fs, "axes.titlesize": fs, "axes.labelsize": fs,
            "xtick.labelsize": fs, "ytick.labelsize": fs, "legend.fontsize": self.label_size,
            "figure.titlesize": fs,
            "axes.linewidth": aw, "xtick.major.width": aw, "ytick.major.width": aw,
            "xtick.minor.width": 0.6 * aw, "ytick.minor.width": 0.6 * aw,
            "xtick.major.size": 4.4 * aw, "ytick.major.size": 4.4 * aw,
            "xtick.minor.size": 2.5 * aw, "ytick.minor.size": 2.5 * aw,
            "lines.linewidth": self.connector_width, "lines.markersize": self.marker_size,
        }


#: Named styles for ``style.preset`` (reaction-profile documents),
#: ``plot_profile(preset=...)`` and ``goodvibes-profile plot --preset``.
#: 'none' keeps matplotlib's defaults and a figure sized to the profile.
STYLE_PRESETS: Dict[str, Optional[StylePreset]] = {
    "none": None,
    # one journal column, about 85 mm wide
    "single-column": StylePreset("single-column", (3.35, 2.6), 1.9, 7, 6, 1.2, 0.7, 0.6, 3),
    # a full page width, about 178 mm
    "double-column": StylePreset("double-column", (7.0, 3.2), 2.4, 8, 7, 1.5, 0.9, 0.8, 4),
    # a 16:9 slide, readable from the back of the room
    "slide": StylePreset("slide", (10.0, 5.6), 3.2, 16, 14, 3.0, 1.8, 1.2, 7),
}


def resolve_preset(name: Optional[str]) -> Optional[StylePreset]:
    """The StylePreset called ``name`` (None or 'none' for the defaults)."""
    if name is None or isinstance(name, StylePreset):
        return name
    if name not in STYLE_PRESETS:
        raise ValueError(f"unknown style preset {name!r} (choose from {', '.join(STYLE_PRESETS)})")
    return STYLE_PRESETS[name]


class _ElementIds:
    """Stable, unique SVG ids (matplotlib ``gid``) for the drawn elements,
    and an index from each id back to what it shows."""

    def __init__(self):
        self.index: Dict[str, Dict[str, str]] = {}

    def __call__(self, artist, kind: str, **ref) -> str:
        ref = {k: str(v) for k, v in ref.items() if v is not None}
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", "-".join([kind] + list(ref.values())))
        gid, n = base, 2
        while gid in self.index:
            gid, n = f"{base}-{n}", n + 1
        self.index[gid] = {"kind": kind, **ref}
        artist.set_gid(gid)
        return gid


@dataclass
class ProfileAxes:
    """What ``plot_profile`` drew, with the numbers behind it.

    ``levels`` is {series id: {pathway name: {point label: value}}} in
    ``units``: the same evaluation the bars were drawn from, so a table
    written from it cannot disagree with the figure; ``uncertainty`` has
    the same shape for the series that carry one. ``order`` is the merged
    x order (point labels) and ``x`` maps a label to its position.
    ``element_ids`` maps each drawn element's SVG id to what it shows.
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
    uncertainty: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=dict)
    preset: Optional[str] = None
    rc: Dict[str, Any] = field(default_factory=dict, repr=False)
    label_size: Any = "x-small"
    pes_result: Any = field(default=None, repr=False)
    _ids: Any = field(default_factory=_ElementIds, repr=False)

    @property
    def element_ids(self) -> Dict[str, Dict[str, str]]:
        return dict(self._ids.index)

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
        with _import_matplotlib().rc_context(self.rc):
            arrow = axis.annotate("", xy=(xa, y1), xytext=(xa, y0),
                                  arrowprops=dict(arrowstyle="<->", color=col, shrinkA=0, shrinkB=0,
                                                  linewidth=0.9 * self.rc["lines.linewidth"] if self.rc
                                                  else 0.8))
            text = axis.annotate(fmt.format(y1 - y0), (xa, (y0 + y1) / 2), xytext=(-3 if last else 3, 0),
                                 textcoords="offset points", ha="right" if last else "left", va="center",
                                 fontsize=self.label_size, color=col)
        ref = {"series": sid, "pathway": name, "from": src, "to": dst}
        self._ids(arrow.arrow_patch, "barrier-arrow", **ref)   # the text of `arrow` is empty
        self._ids(text, "barrier-label", **ref)
        return text

    def to_document(self) -> dict:
        """The reaction-profile document of what was drawn: the drawn
        series with the drawn levels (and uncertainties), in ``units``.
        Needs the ``PESResult`` the figure came from (``plot_profile``
        keeps it)."""
        from .profile import Profile
        return Profile.from_figure(self).to_dict(include_conformers=False)

    def save(self, *paths: str, dpi: int = 200, bbox_inches: str = "tight", embed: bool = True, **kw) -> None:
        """Write the figure to each path (format by extension).

        An SVG keeps its text as text (``svg.fonttype = 'none'``), gives
        every bar, connector, label and error bar an ``id`` (see
        ``element_ids``) and, unless ``embed=False``, carries the
        reaction-profile document of what was drawn in a ``<metadata>``
        element, so ``goodvibes.load_profile("figure.svg")`` reads the
        numbers back. Output is deterministic (no date, fixed id salt).
        """
        plt = _import_matplotlib()
        for path in paths:
            path_kw = dict(kw)
            named = isinstance(path, (str, os.PathLike))
            fmt = (path_kw.get("format") or (os.fspath(path).rsplit(".", 1)[-1] if named else "")).lower()
            rc = dict(self.rc)
            if fmt == "svg":
                rc.update({"svg.fonttype": "none", "svg.hashsalt": "goodvibes"})
                path_kw.setdefault("metadata", {"Date": None})
            with plt.rc_context(rc):
                if fmt == "svg" and embed and self.pes_result is not None:
                    buf = io.StringIO()
                    path_kw["format"] = "svg"
                    self.figure.savefig(buf, dpi=dpi, bbox_inches=bbox_inches, **path_kw)
                    payload = {"generator": f"GoodVibes {_version()}", "document": self.to_document(),
                               "elements": self.element_ids}
                    text = embed_svg_metadata(buf.getvalue(), payload)
                    if named:
                        Path(path).write_text(text, encoding="utf-8")
                    elif isinstance(path, io.TextIOBase):
                        path.write(text)
                    else:
                        path.write(text.encode("utf-8"))
                else:
                    self.figure.savefig(path, dpi=dpi, bbox_inches=bbox_inches, **path_kw)

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
        return ax.add_patch(patch)
    if style == "step":
        xm = (x0 + x1) / 2
        return ax.plot([x0, xm, xm, x1], [y0, y0, y1, y1], color=color,
                       linewidth=linewidth, linestyle=linestyle, zorder=zorder)[0]
    return ax.plot([x0, x1], [y0, y1], color=color, linewidth=linewidth,
                   linestyle=linestyle, zorder=zorder)[0]


def _tick_label_rotation(fig, axis, texts, angle: float = 15.0) -> float:
    """The smallest of 15, 40 and 90 degrees at which slanted, right-aligned
    tick labels one data unit apart do not overlap.

    Parallel labels at angle a, d apart, run into each other when one is
    long enough to reach its neighbour (width * cos a > d) and the gap
    between their baselines (d * sin a) is less than the text height.
    """
    import math
    try:
        renderer = fig.canvas.get_renderer()
    except Exception:                                   # a canvas without a renderer
        return angle
    shown = [t for t in texts if t.get_text()]
    if len(shown) < 2:
        return angle
    x0, x1 = axis.transData.transform([(0.0, 0.0), (1.0, 0.0)])[:, 0]
    d = abs(x1 - x0)
    widths = []
    for t in shown:
        rot = t.get_rotation()
        t.set_rotation(0)
        widths.append(t.get_window_extent(renderer).width)
        t.set_rotation(rot)
    height = 1.2 * max(t.get_size() for t in shown) * fig.dpi / 72.0
    for a in (angle, 40.0):
        r = math.radians(a)
        if max(widths) * math.cos(r) <= d or d * math.sin(r) >= height:
            return a
    return 90.0


def _label_margin(fig, axis, stack_pts: float, minimum: float = 0.1) -> float:
    """The y margin (fraction of the data range) that keeps a stack of
    ``stack_pts`` points of value labels inside the axes."""
    height_pts = axis.get_position().height * fig.get_figheight() * 72.0
    f = stack_pts / height_pts if height_pts > 0 else 0.0
    return minimum if f >= 0.45 else max(minimum, f / (1.0 - 2.0 * f))


def _draw_error_bar(ax, x, y, u, *, cap, color, linewidth, zorder=3):
    """± u about y at x, with caps, as one artist (one SVG element)."""
    from matplotlib.collections import LineCollection
    segments = [[(x, y - u), (x, y + u)],
                [(x - cap, y - u), (x + cap, y - u)],
                [(x - cap, y + u), (x + cap, y + u)]]
    coll = LineCollection(segments, colors=color, linewidths=linewidth, zorder=zorder)
    ax.add_collection(coll, autolim=True)
    ax.update_datalim([(x, y - u), (x, y + u)])
    return coll


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
    preset: Optional[str] = None,
    uncertainty: bool = True,
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
            float, 'decimals': int, 'figsize': (w, h), 'linestyles': [...],
            'preset': name}.
        colors: per-pathway colours, a sequence in pathway order or a
            {name: colour} mapping. Default: black for one pathway, the
            matplotlib cycle otherwise.
        show_conformers: scatter each conformer at level + (conformer −
            species rollup) in the plotted quantity (computed series only).
        label_points: print each level's value next to its bar.
        title: figure title (default: pathway names, and the temperature
            when there is one).
        order: explicit x order of point labels (overrides the result's).
        preset: a ``STYLE_PRESETS`` name ('single-column', 'double-column',
            'slide'; default ``style['preset']`` or 'none'): figure size,
            fonts and line widths for that target, applied to this figure
            only. An explicit ``style['figsize']`` still wins.
        uncertainty: draw each series' ``uncertainty`` as an error bar
            (± the value) on its levels.

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
    pre = resolve_preset(preset if preset is not None else style.get("preset"))
    rc = pre.rc if pre else {}
    if pre:
        bar_lw, connector_lw, marker_size = pre.bar_width, pre.connector_width, pre.marker_size
        tick_size, label_size, legend_size = pre.font_size, pre.label_size, pre.label_size
        shift0, shift_step = 0.85 * pre.label_size, 1.15 * pre.label_size
    else:
        bar_lw, connector_lw, marker_size = 1.5, 1.0, 4
        tick_size, label_size, legend_size = "small", "x-small", "small"
        shift0, shift_step = 6, 8
    ids = _ElementIds()

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
    sigmas: Dict[str, Dict[str, Dict[str, float]]] = {}
    for s in series_list:
        if s.uncertainty:
            sigmas[s.id] = {p.name: s.evaluate_uncertainty(pes_result, p) for p in paths}

    colors_by_path = _resolve_colors(plt, paths, colors)
    # Series without an explicit linestyle take the next style from the cycle
    # that no other series asked for, so defaults never repeat a chosen one.
    # A list, not a set: a dash pattern such as (0, [3, 2]) is unhashable.
    _aliases = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}

    def _norm_ls(ls):
        return _aliases.get(ls, ls) if isinstance(ls, str) else ls
    explicit = [_norm_ls(s.style["linestyle"]) for s in series_list if "linestyle" in s.style]
    free = [ls for ls in linestyles if _norm_ls(ls) not in explicit] or linestyles
    ls_by_series = {}
    n_default = 0
    for s in series_list:
        if "linestyle" in s.style:
            ls_by_series[s.id] = s.style["linestyle"]
        else:
            ls_by_series[s.id] = free[n_default % len(free)]
            n_default += 1

    with plt.rc_context(rc):
        # Figure / axes
        figsize = style.get("figsize")
        if layout == "panels":
            if ax is not None:
                raise ValueError("plot_profile: ax= cannot be combined with layout='panels'")
            if figsize is None:
                figsize = ((pre.figsize[0], pre.panel_height * len(paths)) if pre
                           else (max(5, 0.9 * n_points + 1), 3.2 * len(paths)))
            fig, axes = plt.subplots(len(paths), 1, sharey=True, sharex=True, figsize=figsize, squeeze=False)
            axes = [a[0] for a in axes]
        else:
            if ax is None:
                if figsize is None:
                    figsize = pre.figsize if pre else (max(5, 0.9 * n_points + 1), 4)
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
                label_shift = shift0 + shift_step * si   # stack value labels when several series share a bar
                lv = levels[s.id][path.name]
                sig = sigmas.get(s.id, {}).get(path.name, {}) if uncertainty else {}
                # bars
                for point in path.points:
                    y = lv.get(point.label)
                    if y is None:
                        continue
                    x = xpos[point.label]
                    ref = dict(series=s.id, pathway=path.name, point=point.label)
                    if s.declared:
                        ids(axis.hlines(y, x - bar_half, x + bar_half, colors=color, linewidth=bar_lw,
                                        linestyle=ls, zorder=3), "bar", **ref)
                        ids(axis.plot([x], [y], marker="o", markersize=marker_size, markerfacecolor="white",
                                      markeredgecolor=color, linestyle="none", zorder=4)[0], "marker", **ref)
                    else:
                        ids(axis.hlines(y, x - bar_half, x + bar_half, colors=color, linewidth=bar_lw,
                                        zorder=3), "bar", **ref)
                    u = sig.get(point.label)
                    if u:
                        ids(_draw_error_bar(axis, x, y, u, cap=0.45 * bar_half, color=color,
                                            linewidth=connector_lw), "error", **ref)
                    if label_points:
                        above = point.is_ts
                        anchor = y + (u or 0.0) if above else y - (u or 0.0)
                        ids(axis.annotate(f"{y:.{decimals}f}", (x, anchor),
                                          xytext=(0, label_shift if above else -label_shift),
                                          textcoords="offset points", ha="center",
                                          va="bottom" if above else "top",
                                          fontsize=label_size, color=color), "label", **ref)
                # connectors along the edges
                for edge in path.edges:
                    if edge.kind == "none":
                        continue
                    y0, y1 = lv.get(edge.src), lv.get(edge.dst)
                    if y0 is None or y1 is None:
                        continue
                    edge_ls = ":" if edge.kind == "barrierless" else ls
                    ids(_draw_connector(axis, xpos[edge.src], y0, xpos[edge.dst], y1,
                                        style=connector, color=color, linestyle=edge_ls,
                                        linewidth=connector_lw),
                        "edge", series=s.id, pathway=path.name, **{"from": edge.src, "to": edge.dst})
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
                                ids(axis.scatter([x], [y], alpha=0.4, s=(1.06 * marker_size) ** 2,
                                                 color=color, zorder=4),
                                    "conformer", series=s.id, pathway=path.name, point=point.label)

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
        from matplotlib.font_manager import FontProperties
        label_pts = FontProperties(size=label_size).get_size_in_points()
        # value labels stacked above a TS bar / below a minimum, in points
        stack = (shift0 + shift_step * (len(series_list) - 1) + 1.2 * label_pts) if label_points else 0.0
        for i, axis in enumerate(axes):
            axis.set_xticks(list(range(n_points)))
            ticklabels = axis.set_xticklabels([display.get(lab, lab) for lab in order],
                                              rotation=15, ha="right", fontsize=tick_size)
            rotation = _tick_label_rotation(fig, axis, ticklabels)
            if rotation != 15:
                for t in ticklabels:
                    t.set_rotation(rotation)
            if pre:
                axis.tick_params(axis="y", labelsize=tick_size)
            axis.set_ylabel(ylabel)
            # room for the value labels above TS bars and below minima
            axis.margins(y=_label_margin(fig, axis, stack))
            axis.minorticks_on()
            axis.tick_params(axis='x', which='minor', bottom=False, top=False)
            axis.tick_params(axis='y', which='both', labelright=True, right=True)
            if layout == "panels":
                axis.set_title(paths[i].name, fontsize=tick_size, loc="left")

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
            ids(axes[0].legend(handles=handles, loc="best", fontsize=legend_size), "legend")

    return ProfileAxes(
        figure=fig, axes=axes, levels=levels, units=units, order=order, x=xpos,
        series=series_list, pathways=paths, colors=colors_by_path,
        linestyles=ls_by_series, layout=layout, uncertainty=sigmas,
        preset=pre.name if pre else None, rc=rc, label_size=label_size,
        pes_result=pes_result, _ids=ids,
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
# Conformer populations and temperature scans
# ---------------------------------------------------------------------------

def _as_conformer_set(source, name: str, quantity: Optional[str]):
    """A ConformerSet from a ConformerSet or a sequence of ThermoResults."""
    from .pes_model import ConformerSet
    if isinstance(source, ConformerSet):
        return source
    results = list(source)
    if not results:
        raise ValueError(f"{name}: no conformers")
    return ConformerSet.from_results(name, results, weight_by=quantity or "qh_gibbs")


def _conformer_names(cset) -> List[str]:
    from .utils import display_name
    return [display_name(f) if f else f"{cset.name} {i + 1}" for i, f in enumerate(cset.files)]


def plot_boltzmann_histogram(
    source: Any,
    *,
    temperature: float = 298.15,
    quantity: Optional[str] = None,
    top: Optional[int] = None,
    sort: bool = True,
    ax=None,
    title: Optional[str] = None,
):
    """Bar chart of conformer Boltzmann populations.

    Parameters:
        source: a ``ConformerSet``, a sequence of ``ThermoResult`` (one
            species' conformers), or a mapping ``{group: ConformerSet or
            results}``. With a mapping the populations are taken over all
            conformers together (for example the transition states of
            competing pathways, so the bars show where the selectivity
            comes from); bars are coloured by group and the legend gives
            each group's total population.
        temperature: K; each conformer is re-evaluated at it when the set
            carries its parsed data.
        quantity: the registry quantity weighted (default the set's
            ``weight_by``, ``qh_gibbs``; ``electronic`` for energy-only
            ensembles).
        top: draw only the ``top`` most populated conformers, plus one
            "other" bar for the rest.
        sort: most populated first (default); False keeps input order.
        ax: optional matplotlib Axes.
        title: figure title; generated when None.

    Returns:
        The matplotlib Axes. Each bar's ``gid`` is ``pop-<group>-<name>``.
    """
    import math

    from .constants import GAS_CONSTANT, J_TO_AU
    from .quantities import resolve_quantity
    plt = _import_matplotlib()
    groups = dict(source) if isinstance(source, Mapping) else {None: source}
    bars = []                                     # (group, name, value)
    qid = None
    for group, member in groups.items():
        cset = _as_conformer_set(member, str(group or "conformers"), quantity)
        qid = resolve_quantity(quantity or cset.weight_by).id
        values = [v.get(qid, temperature) for v in cset.vectors(temperature)]
        if any(v is None for v in values):
            raise ValueError(f"plot_boltzmann_histogram: {qid!r} is not available for every conformer "
                             "(an energy-only ensemble needs quantity='electronic')")
        bars.extend((group, name, v) for name, v in zip(_conformer_names(cset), values))
    if not bars:
        raise ValueError("plot_boltzmann_histogram: no conformers")
    rt = GAS_CONSTANT * temperature / J_TO_AU
    lowest = min(v for _g, _n, v in bars)
    weights = [math.exp(-(v - lowest) / rt) for _g, _n, v in bars]
    total = sum(weights)
    pops = [(g, n, w / total) for (g, n, _v), w in zip(bars, weights)]
    if sort:
        pops.sort(key=lambda b: -b[2])
    shown, rest = (pops[:top], pops[top:]) if top is not None and top < len(pops) else (pops, [])

    if ax is None:
        n = len(shown) + (1 if rest else 0)
        _, ax = plt.subplots(figsize=(max(3.5, 0.35 * n + 1.5), 3.5))
    group_names = list(groups)
    colour = {g: f"C{i}" for i, g in enumerate(group_names)}
    xs = list(range(len(shown)))
    for x, (g, name, p) in zip(xs, shown):
        ax.bar(x, p * 100.0, color=colour[g], edgecolor="black", linewidth=0.4,
               gid=f"pop-{g}-{name}" if g is not None else f"pop-{name}")
    names = [name for _g, name, _p in shown]
    if rest:
        ax.bar(len(shown), sum(p for _g, _n, p in rest) * 100.0, color="0.8", edgecolor="black",
               linewidth=0.4, gid="pop-other")
        names.append(f"{len(rest)} other")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60 if len(names) > 4 else 0, ha="right" if len(names) > 4 else "center")
    ax.set_ylabel("Population (%)")
    if len(group_names) > 1 or group_names[0] is not None:
        from matplotlib.patches import Patch
        share = {g: sum(p for gg, _n, p in pops if gg == g) for g in group_names}
        ax.legend(handles=[Patch(facecolor=colour[g], edgecolor="black", label=f"{g} ({share[g] * 100:.1f} %)")
                           for g in group_names], frameon=False)
    ax.set_title(title if title is not None else f"Boltzmann populations ({qid}, T = {temperature:g} K)")
    return ax


_SCAN_QUANTITIES = ("qh_gibbs", "qh_enthalpy", "qh_entropy")


def plot_temperature_scan(
    source: Any,
    temperatures: Optional[Sequence[float]] = None,
    *,
    quantities: Optional[Sequence[str]] = None,
    points: Optional[Sequence[str]] = None,
    pathway: Optional[str] = None,
    units: str = "kcal/mol",
    ax=None,
    title: Optional[str] = None,
):
    """Thermochemistry against temperature.

    Two kinds of ``source``:

    - a ``ConformerSet`` (or a sequence of ``ThermoResult``, one
      species' conformers) and ``temperatures``: one line per quantity
      (default Δqh-G, Δqh-H and T·Δqh-S) of the conformer ensemble (the
      gconf rollup), relative to its value at the first temperature;
    - a reaction-profile ``Profile``: one line per point (``points``,
      default every point of ``pathway`` but its zero; ``pathway``
      defaults to the first) giving its level against temperature over
      the document's series of one quantity (``quantities[0]``, default
      that of the first series with a temperature). A point is read on
      ``pathway`` when it is there, else on the first pathway that holds
      it (each relative to that pathway's zero). With ``temperatures``
      the document is evaluated at them first (it needs embedded
      conformers).

    Returns:
        The matplotlib Axes; each line's ``gid`` is ``scan-<quantity>``
        or ``scan-<point>``.
    """
    from .profile import Profile
    from .quantities import QUANTITIES, resolve_quantity
    plt = _import_matplotlib()
    units = canonical_units(units)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 3.5))

    if isinstance(source, Profile):
        prof = source.evaluate(temperatures=temperatures) if temperatures else source
        qid = resolve_quantity(quantities[0]).id if quantities else next(
            (s.quantity for s in prof.series if s.temperature is not None and s.levels), None)
        series = sorted((s for s in prof.series if s.quantity == qid and s.levels and s.temperature is not None),
                        key=lambda s: s.temperature)
        if not series:
            raise ValueError("plot_temperature_scan: the document has no evaluated series with a temperature"
                             + (f" of {qid!r}" if qid else ""))
        path = prof.pathways[pathway] if pathway else next(iter(prof.pathways.values()))
        pids = list(points) if points else [p for p in path.points if p != path.zero]
        scale = hartree_factor(units) / hartree_factor(prof.units)
        for pid in pids:
            # the named pathway when it holds the point, else the first that does
            names = [path.name] + [n for n, pw in prof.pathways.items()
                                   if n != path.name and (pid in pw.points or pid == pw.zero)]
            xs, ys = [], []
            for s in series:
                v = next(((s.levels.get(n) or {}).get(pid) for n in names
                          if (s.levels.get(n) or {}).get(pid) is not None), None)
                if v is not None:
                    xs.append(s.temperature)
                    ys.append(v * scale)
            if not xs:
                raise ValueError(f"plot_temperature_scan: no levels for point {pid!r}")
            display = prof.points[pid].display if pid in prof.points and prof.points[pid].display else pid
            ax.plot(xs, ys, marker="o", label=display, gid=f"scan-{pid}")
        ax.set_ylabel(f"{QUANTITIES[qid].label} ({units})")
        default_title = f"{prof.title or path.name}: {QUANTITIES[qid].label} vs T"
    else:
        if not temperatures:
            raise ValueError("plot_temperature_scan: a conformer set needs temperatures")
        cset = _as_conformer_set(source, "conformers", None)
        temps = [float(t) for t in temperatures]
        scale = hartree_factor(units)
        for q in (quantities or _SCAN_QUANTITIES):
            qid = resolve_quantity(q).id
            vals = [cset.gconf_corrected(T).get(qid, T) if len(cset.bbes) > 1 else cset.vectors(T)[0].get(qid, T)
                    for T in temps]
            if any(v is None for v in vals):
                raise ValueError(f"plot_temperature_scan: {qid!r} is not available for these conformers")
            ax.plot(temps, [(v - vals[0]) * scale for v in vals], marker="o",
                    label=QUANTITIES[qid].label, gid=f"scan-{qid}")
        ax.set_ylabel(f"change from {temps[0]:g} K ({units})")
        default_title = f"{cset.name}: thermochemistry vs T"
    ax.set_xlabel("T (K)")
    ax.legend(frameon=False)
    ax.set_title(title if title is not None else default_title)
    return ax
