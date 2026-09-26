"""Reaction-profile documents: the ``reaction-profile/1.0`` format.

A reaction-profile document describes an energy profile independently of
how its numbers were obtained: species (identity only), points (a
stoichiometric sum of species with a role and a display label), pathways
(ordered points, a zero and edges), methods (provenance), and series (a
quantity at a temperature, one line set on the axes). A series is either
*declared* (typed-in levels, e.g. literature values) or *computed*
(evaluated by GoodVibes from QC or MLIP data). Everything GoodVibes needs
to compute a series (which files belong to which species, the rollup and
thermochemistry options, optionally the parsed structures themselves)
lives under the ``goodvibes:`` namespace, which other readers ignore.

    from goodvibes import load_profile
    prof = load_profile("profile.yaml")          # or .json, .csv, a v2 PES YAML, legacy --- # PES text
    ev = prof.evaluate(thermo_data)               # {file: calc_bbe} or [ThermoResult]
    ev.dump("profile.json")                       # explicit form, levels filled, provenance
    ev.plot().save("profile.svg")

The normative description is ``docs/source/reaction_profile.md``; the JSON
Schema is ``goodvibes/schemas/reaction-profile-1.0.schema.json``. The Python
validator here is the reference implementation: it checks everything the
JSON Schema checks plus the referential rules a JSON Schema cannot express
(a pathway's points exist, a series' levels name points on that pathway,
...), and it accepts a few conveniences on input (quantity aliases, the
GoodVibes v2 PES YAML and the legacy ``--- # PES`` text) that it always
writes back in the explicit form.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import re
import warnings
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .constants import __version__, canonical_units, hartree_factor
from .pes_model import (
    EDGE_KINDS, ConformerSet, Edge, PESOptions, PESResult, Pathway, Point, Series,
    _normalise_role, merge_point_order, parse_point_label,
)
from .quantities import QUANTITIES, resolve_quantity

__all__ = [
    "Profile", "ProfilePoint", "ProfilePathway", "ProfileError", "ProfileWarning",
    "load_profile", "validate_document", "schema_errors", "load_json_schema",
    "SCHEMA_TAG", "SCHEMA_VERSION", "SCHEMA_ID",
]

SCHEMA_NAME = "reaction-profile"
SCHEMA_VERSION = "1.0"
SCHEMA_TAG = f"{SCHEMA_NAME}/{SCHEMA_VERSION}"
SCHEMA_FILE = "reaction-profile-1.0.schema.json"
SCHEMA_ID = ("https://raw.githubusercontent.com/patonlab/GoodVibes/master/goodvibes/schemas/"
             + SCHEMA_FILE)
NAMESPACE = "goodvibes"
DEFAULT_METHOD = "default"

ENSEMBLES = ("ideal-gas",)
ROLLUP_MODES = ("gconf", "boltzmann", "lowest")
LAYOUTS = ("overlay", "panels")
CONNECTORS = ("bezier", "linear", "step")
ANNOTATION_TYPES = ("barrier", "span")
SERIES_SOURCES = ("computed", "declared")

_CORE_KEYS = ("schema", "title", "description", "units", "ensemble", "default_temperature",
              "species", "points", "pathways", "order", "methods", "series", "annotations",
              "style", "provenance", NAMESPACE)
#: Keys a later 1.x minor will define; rejected, never silently ignored.
_RESERVED = {
    "selectivity": "the `selectivity` block is reserved for reaction-profile 1.1 (GoodVibes 5.1)",
}
_SPECIES_KEYS = {"name", "smiles", "inchi", "formula", "charge", "multiplicity"}
_SOURCE_KEYS = {"files", "dir", "dirs"}
_POINT_KEYS = {"species", "role", "display"}
_PATHWAY_KEYS = {"points", "zero", "edges"}
_EDGE_KEYS = {"from", "to", "kind"}
_SERIES_KEYS = {"id", "label", "method", "quantity", "temperature", "source", "levels",
                "uncertainty", "style", "standard_state"}
_ANNOTATION_KEYS = {"type", "pathway", "from", "to", "series", "label"}
_STYLE_KEYS = {"preset", "layout", "connector", "label_points", "decimals", "figsize"}
_METHOD_KEYS = {"program", "model", "level_of_theory", "solvent", "standard_state",
                "frequency_scaling", "quasi_harmonic", "hessian", "reference", "description"}
_NAMESPACE_KEYS = {"sources", "rollup", "thermo", "dedup", "conformers", "format"}
_ROLLUP_KEYS = {"mode", "weight_by"}
_DEDUP_KEYS = {"e_cutoff", "ro_cutoff", "rmsd_cutoff"}
_THERMO_KEYS = {"QS", "QH", "s_freq_cutoff", "h_freq_cutoff", "concentration", "freq_scale_factor",
                "zpe_scale_factor", "solv", "spc", "invert", "symm", "inertia", "strict_spc"}
_DIR_PREFIX = "@dir:"          # pes_loader's encoding of a directory pattern
_SCHEMA_RE = re.compile(r"^reaction-profile/(\d+)\.(\d+)$")


class ProfileError(ValueError):
    """A reaction-profile document is invalid. ``errors`` lists every
    problem found (``path: message``); ``warnings`` the non-fatal ones."""

    def __init__(self, errors: Sequence[str], warnings_: Sequence[str] = ()):
        self.errors = list(errors)
        self.warnings = list(warnings_)
        head = f"invalid reaction-profile document ({len(self.errors)} error{'s' if len(self.errors) != 1 else ''})"
        super().__init__(head + ":\n  " + "\n  ".join(self.errors))


class ProfileWarning(UserWarning):
    """A reaction-profile document was accepted with a caveat (an unknown
    key, a quantity alias, a v2 shorthand, ...)."""


# ---------------------------------------------------------------------------
# Document parts
# ---------------------------------------------------------------------------

@dataclass
class ProfilePoint:
    """A point: a stoichiometric sum of species (empty for a point that only
    carries declared values), a role and a display label."""
    id: str
    species: List[Tuple[int, str]] = field(default_factory=list)
    role: str = "minimum"
    display: Optional[str] = None
    extensions: Dict[str, Any] = field(default_factory=dict)

    def species_mapping(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for coeff, name in self.species:
            out[name] = out.get(name, 0) + coeff
        return out


@dataclass
class ProfilePathway:
    """A pathway: ordered point ids, the zero and the full edge list
    (``explicit_edges`` are the ones the document lists; the rest are the
    default consecutive ``step`` edges)."""
    name: str
    points: List[str]
    zero: str
    edges: List[Edge]
    explicit_edges: List[Edge] = field(default_factory=list)
    extensions: Dict[str, Any] = field(default_factory=dict)


class _Checker:
    """Collects errors and warnings with JSON-path-like locations."""

    def __init__(self, strict: bool = False):
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def error(self, where: str, msg: str) -> None:
        self.errors.append(f"{where}: {msg}" if where else msg)

    def warn(self, where: str, msg: str) -> None:
        text = f"{where}: {msg}" if where else msg
        if self.strict:
            self.errors.append(text + " (error under --strict)")
        else:
            self.warnings.append(text)

    def keys(self, where: str, mapping: Mapping, allowed: set) -> None:
        for key in mapping:
            if isinstance(key, str) and key.startswith("x-"):
                continue
            if key not in allowed:
                self.warn(where, f"unknown key {key!r}")

    def mapping(self, where: str, value, *, allow_none: bool = True) -> Optional[dict]:
        if value is None and allow_none:
            return {}
        if not isinstance(value, Mapping):
            self.error(where, f"must be a mapping, got {type(value).__name__}")
            return None
        return dict(value)

    def number(self, where: str, value, *, positive: bool = False, minimum: Optional[float] = None) -> Optional[float]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self.error(where, f"must be a number, got {value!r}")
            return None
        if positive and not value > 0:
            self.error(where, f"must be positive, got {value!r}")
            return None
        if minimum is not None and value < minimum:
            self.error(where, f"must be ≥ {minimum:g}, got {value!r}")
            return None
        return float(value)

    def string(self, where: str, value) -> Optional[str]:
        if not isinstance(value, str) or not value.strip():
            self.error(where, f"must be a non-empty string, got {value!r}")
            return None
        return value


def _extensions(mapping: Mapping) -> Dict[str, Any]:
    return {k: v for k, v in mapping.items() if isinstance(k, str) and k.startswith("x-")}


# ---------------------------------------------------------------------------
# Upgrading the GoodVibes v2 PES YAML / legacy text (PESSpec) to the explicit form
# ---------------------------------------------------------------------------

def _patterns_to_source(pattern) -> Dict[str, List[str]]:
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    files = [p for p in patterns if not p.startswith(_DIR_PREFIX)]
    dirs = [p[len(_DIR_PREFIX):] for p in patterns if p.startswith(_DIR_PREFIX)]
    out: Dict[str, List[str]] = {}
    if files:
        out["files"] = files
    if dirs:
        out["dirs"] = dirs
    return out


def _source_to_patterns(source) -> List[str]:
    if isinstance(source, str):
        return [source]
    if isinstance(source, (list, tuple)):
        return [str(s) for s in source]
    out: List[str] = []
    files = source.get("files")
    if isinstance(files, str):
        out.append(files)
    elif isinstance(files, (list, tuple)):
        out.extend(str(f) for f in files)
    from .pes_yaml import _normalize_dir
    if isinstance(source.get("dir"), str):
        out.append(_DIR_PREFIX + _normalize_dir(source["dir"]))
    for d in source.get("dirs", []) or []:
        out.append(_DIR_PREFIX + _normalize_dir(str(d)))
    return out


def _explicit_from_spec(spec, extra: Optional[Mapping] = None) -> dict:
    """The explicit reaction-profile mapping for a PESSpec (v2 YAML or
    legacy text). Point ids are the original point labels."""
    doc: Dict[str, Any] = {"schema": SCHEMA_TAG, "units": spec.options.units}
    doc["species"] = {name: {} for name in spec.species}
    doc["pathways"] = {}
    for name, labels in spec.pathways.items():
        entry: Dict[str, Any] = {"points": list(labels)}
        if name in spec.zero:
            entry["zero"] = spec.zero[name]
        doc["pathways"][name] = entry
    doc["style"] = {"decimals": spec.options.decimals}
    ns: Dict[str, Any] = {"sources": {DEFAULT_METHOD: {name: _patterns_to_source(pat)
                                                      for name, pat in spec.species.items()}}}
    if getattr(spec, "format_extras", None):
        ns["format"] = dict(spec.format_extras)
    doc[NAMESPACE] = ns
    for key, value in (extra or {}).items():
        doc.setdefault(key, value)
    return doc


def _looks_explicit(data: Mapping) -> bool:
    if "schema" in data or "points" in data or "series" in data:
        return True
    paths = data.get("pathways")
    return isinstance(paths, Mapping) and any(isinstance(v, Mapping) for v in paths.values())


# ---------------------------------------------------------------------------
# Parsing + validation of the explicit form
# ---------------------------------------------------------------------------

def _parse_species_sum(where: str, value, chk: _Checker) -> Optional[List[Tuple[int, str]]]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            return parse_point_label(value)
        except ValueError as exc:
            chk.error(where, str(exc))
            return None
    if isinstance(value, Mapping):
        if not value:
            chk.error(where, "an empty species mapping; omit `species` for a point without species")
            return None
        out = []
        for name, coeff in value.items():
            if isinstance(coeff, bool) or not isinstance(coeff, int) or coeff < 1:
                chk.error(f"{where}.{name}", f"coefficient must be a positive integer, got {coeff!r}")
                continue
            out.append((coeff, str(name)))
        return out
    chk.error(where, f"must be a species sum string, a {{species: coefficient}} mapping or null, got {value!r}")
    return None


def _parse(data: Mapping, strict: bool = False) -> Tuple["Profile", List[str]]:
    chk = _Checker(strict)
    if not isinstance(data, Mapping):
        raise ProfileError([f"document root must be a mapping, got {type(data).__name__}"])
    data = dict(data)
    upgraded_from = None

    # -- the v2 PES YAML shorthand ------------------------------------------
    if "schema" not in data:
        if _looks_explicit(data):
            chk.warn("schema", f"missing; assuming {SCHEMA_TAG}")
            data["schema"] = SCHEMA_TAG
        else:
            from .pes_yaml import parse_yaml_data
            try:
                spec = parse_yaml_data(data)
            except ValueError as exc:
                raise ProfileError([str(exc)]) from None
            extra = {k: v for k, v in data.items() if k not in ("pathways", "species", "zero", "format")}
            data = _explicit_from_spec(spec, extra)
            upgraded_from = "v2"

    # -- schema --------------------------------------------------------------
    tag = data.get("schema")
    m = _SCHEMA_RE.match(tag) if isinstance(tag, str) else None
    if m is None:
        chk.error("schema", f"must be 'reaction-profile/<major>.<minor>', got {tag!r}")
    elif int(m.group(1)) != 1:
        chk.error("schema", f"{tag!r} is a major version this GoodVibes does not read (it reads 1.x)")
    elif int(m.group(2)) > int(SCHEMA_VERSION.split(".")[1]):
        chk.warn("schema", f"{tag!r} is newer than {SCHEMA_TAG}; keys this version does not know are ignored")

    for key, why in _RESERVED.items():
        if key in data:
            chk.error(key, why)
    for key in data:
        if key in _CORE_KEYS or key in _RESERVED or (isinstance(key, str) and key.startswith("x-")):
            continue
        chk.warn("", f"unknown top-level key {key!r}")

    title = data.get("title")
    if title is not None and not isinstance(title, str):
        chk.error("title", "must be a string")
    description = data.get("description")
    if description is not None and not isinstance(description, str):
        chk.error("description", "must be a string")

    units = data.get("units", "kcal/mol")
    try:
        units = canonical_units(units)
    except (ValueError, TypeError) as exc:
        chk.error("units", str(exc))
        units = "kcal/mol"
    ensemble = data.get("ensemble", "ideal-gas")
    if ensemble not in ENSEMBLES:
        chk.error("ensemble", f"must be one of {', '.join(ENSEMBLES)}, got {ensemble!r}")
    default_T = data.get("default_temperature", 298.15)
    default_T = chk.number("default_temperature", default_T, positive=True) or 298.15

    namespace = chk.mapping(NAMESPACE, data.get(NAMESPACE)) or {}
    namespace = copy.deepcopy(namespace)
    sources = chk.mapping(f"{NAMESPACE}.sources", namespace.get("sources")) or {}

    # -- species -------------------------------------------------------------
    species: Dict[str, dict] = {}
    raw_species = chk.mapping("species", data.get("species")) or {}
    for name, entry in raw_species.items():
        where = f"species.{name}"
        entry = chk.mapping(where, entry)
        if entry is None:
            continue
        moved = {k: entry.pop(k) for k in list(entry) if k in _SOURCE_KEYS}
        if moved:
            chk.warn(where, f"{', '.join(sorted(moved))} belongs under {NAMESPACE}.sources; moved there")
            sources.setdefault(DEFAULT_METHOD, {})[str(name)] = moved
        chk.keys(where, entry, _SPECIES_KEYS)
        for key in ("name", "smiles", "inchi", "formula"):
            if key in entry and not isinstance(entry[key], str):
                chk.error(f"{where}.{key}", "must be a string")
        if "charge" in entry and (isinstance(entry["charge"], bool) or not isinstance(entry["charge"], int)):
            chk.error(f"{where}.charge", "must be an integer")
        if "multiplicity" in entry:
            mult = entry["multiplicity"]
            if isinstance(mult, bool) or not isinstance(mult, int) or mult < 1:
                chk.error(f"{where}.multiplicity", "must be a positive integer")
        species[str(name)] = entry

    # -- points --------------------------------------------------------------
    points: Dict[str, ProfilePoint] = {}
    raw_points = chk.mapping("points", data.get("points")) or {}
    for pid, entry in raw_points.items():
        where = f"points.{pid}"
        entry = chk.mapping(where, entry)
        if entry is None:
            continue
        chk.keys(where, entry, _POINT_KEYS)
        terms = _parse_species_sum(f"{where}.species", entry.get("species"), chk)
        try:
            role = _normalise_role(entry.get("role"))
        except ValueError as exc:
            chk.error(f"{where}.role", str(exc))
            role = "minimum"
        display = entry.get("display")
        if display is not None and not isinstance(display, str):
            chk.error(f"{where}.display", "must be a string")
            display = None
        points[str(pid)] = ProfilePoint(str(pid), terms or [], role, display, _extensions(entry))

    def ensure_point(pid: str, where: str) -> bool:
        """A pathway may name a point by its species sum ("A + B") without
        listing it under `points`; such a point is created on the fly."""
        if pid in points:
            return True
        try:
            terms = parse_point_label(pid)
        except ValueError:
            terms = None
        if terms and all(name in species for _c, name in terms):
            points[pid] = ProfilePoint(pid, terms)
            return True
        chk.error(where, f"point {pid!r} is not defined under `points` (and is not a sum of declared species)")
        return False

    # -- pathways ------------------------------------------------------------
    pathways: Dict[str, ProfilePathway] = {}
    raw_paths = chk.mapping("pathways", data.get("pathways"), allow_none=False)
    if raw_paths is not None and not raw_paths:
        chk.error("pathways", "must define at least one pathway")
    for name, entry in (raw_paths or {}).items():
        where = f"pathways.{name}"
        if isinstance(entry, (list, tuple)):
            entry = {"points": list(entry)}
        entry = chk.mapping(where, entry, allow_none=False)
        if entry is None:
            continue
        chk.keys(where, entry, _PATHWAY_KEYS)
        ids = entry.get("points")
        if not isinstance(ids, (list, tuple)) or not ids:
            chk.error(f"{where}.points", "must be a non-empty list of point ids")
            continue
        ids = [str(i).strip() for i in ids]
        ok = all([ensure_point(pid, f"{where}.points") for pid in ids])
        zero = str(entry.get("zero", ids[0])).strip()
        ok = ensure_point(zero, f"{where}.zero") and ok
        explicit: List[Edge] = []
        for i, e in enumerate(entry.get("edges") or []):
            ewhere = f"{where}.edges[{i}]"
            e = chk.mapping(ewhere, e, allow_none=False)
            if e is None:
                continue
            chk.keys(ewhere, e, _EDGE_KEYS)
            src, dst, kind = e.get("from"), e.get("to"), e.get("kind", "step")
            if src not in ids or dst not in ids:
                chk.error(ewhere, f"edge {src!r} -> {dst!r} must join two points of the pathway")
                continue
            if kind not in EDGE_KINDS:
                chk.error(f"{ewhere}.kind", f"must be one of {', '.join(EDGE_KINDS)}, got {kind!r}")
                continue
            explicit.append(Edge(src, dst, kind))
        edges = [Edge(a, b) for a, b in zip(ids, ids[1:])]
        for e in explicit:
            for j, d in enumerate(edges):
                if (d.src, d.dst) == (e.src, e.dst):
                    edges[j] = e
                    break
            else:
                edges.append(e)
        if ok:
            pathways[str(name)] = ProfilePathway(str(name), ids, zero, edges, explicit, _extensions(entry))

    for pid, pt in points.items():
        for _c, sname in pt.species:
            if sname not in species:
                chk.error(f"points.{pid}.species", f"species {sname!r} is not declared under `species`")

    # -- order ---------------------------------------------------------------
    order = data.get("order")
    if order is not None:
        if not isinstance(order, (list, tuple)) or not all(isinstance(o, str) for o in order):
            chk.error("order", "must be a list of point ids")
            order = None
        else:
            order = list(order)
            for pid in order:
                if pid not in points:
                    chk.error("order", f"point {pid!r} is not defined")
            for path in pathways.values():
                for pid in path.points:
                    if pid not in order:
                        chk.error("order", f"omits point {pid!r} of pathway {path.name!r}")

    # -- methods -------------------------------------------------------------
    methods: Dict[str, dict] = {}
    for mid, entry in (chk.mapping("methods", data.get("methods")) or {}).items():
        where = f"methods.{mid}"
        entry = chk.mapping(where, entry)
        if entry is None:
            continue
        chk.keys(where, entry, _METHOD_KEYS)
        methods[str(mid)] = entry

    # -- goodvibes namespace -------------------------------------------------
    chk.keys(NAMESPACE, namespace, _NAMESPACE_KEYS)
    clean_sources: Dict[str, Dict[str, Any]] = {}
    for mid, per_species in sources.items():
        where = f"{NAMESPACE}.sources.{mid}"
        per_species = chk.mapping(where, per_species)
        if per_species is None:
            continue
        clean_sources[str(mid)] = {}
        for sname, src in per_species.items():
            swhere = f"{where}.{sname}"
            if str(sname) not in species:
                chk.error(swhere, f"species {sname!r} is not declared under `species`")
            if isinstance(src, Mapping):
                chk.keys(swhere, src, _SOURCE_KEYS)
                if not any(k in src for k in _SOURCE_KEYS):
                    chk.error(swhere, "needs `files`, `dir` or `dirs`")
            elif not isinstance(src, (str, list, tuple)):
                chk.error(swhere, "must be a glob string, a list of globs or {files, dir, dirs}")
            clean_sources[str(mid)][str(sname)] = src
    if clean_sources:
        namespace["sources"] = clean_sources
    rollup = chk.mapping(f"{NAMESPACE}.rollup", namespace.get("rollup")) or {}
    chk.keys(f"{NAMESPACE}.rollup", rollup, _ROLLUP_KEYS)
    if rollup.get("mode", "gconf") not in ROLLUP_MODES:
        chk.error(f"{NAMESPACE}.rollup.mode", f"must be one of {', '.join(ROLLUP_MODES)}")
    if "weight_by" in rollup:
        try:
            rollup["weight_by"] = resolve_quantity(rollup["weight_by"]).id
        except ValueError as exc:
            chk.error(f"{NAMESPACE}.rollup.weight_by", str(exc))
    if rollup:
        namespace["rollup"] = rollup
    thermo = chk.mapping(f"{NAMESPACE}.thermo", namespace.get("thermo")) or {}
    chk.keys(f"{NAMESPACE}.thermo", thermo, _THERMO_KEYS | {"temperature"})
    dedup = namespace.get("dedup")
    if dedup is not None:
        dedup = chk.mapping(f"{NAMESPACE}.dedup", dedup) or {}
        chk.keys(f"{NAMESPACE}.dedup", dedup, _DEDUP_KEYS | {"scope"})
    method_ids = set(methods) | set(clean_sources)
    conformers = chk.mapping(f"{NAMESPACE}.conformers", namespace.get("conformers")) or {}
    for mid, per_species in conformers.items():
        where = f"{NAMESPACE}.conformers.{mid}"
        if not isinstance(per_species, Mapping):
            chk.error(where, "must map species to a list of conformers")
            continue
        method_ids.add(str(mid))
        for sname, items in per_species.items():
            if str(sname) not in species:
                chk.error(f"{where}.{sname}", f"species {sname!r} is not declared under `species`")
            if not isinstance(items, list) or not all(isinstance(i, Mapping) and "qcdata" in i and "options" in i
                                                     for i in items):
                chk.error(f"{where}.{sname}", "must be a list of {file, qcdata, options} entries")

    # -- series --------------------------------------------------------------
    series: List[Series] = []
    raw_series = data.get("series") or []
    if not isinstance(raw_series, list):
        chk.error("series", "must be a list")
        raw_series = []
    seen_ids = set()
    for i, entry in enumerate(raw_series):
        where = f"series[{i}]"
        entry = chk.mapping(where, entry, allow_none=False)
        if entry is None:
            continue
        chk.keys(where, entry, _SERIES_KEYS)
        sid = chk.string(f"{where}.id", entry.get("id"))
        if sid is None:
            continue
        where = f"series.{sid}"
        if sid in seen_ids:
            chk.error(where, "duplicate series id")
        seen_ids.add(sid)
        raw_q = entry.get("quantity")
        try:
            qty = resolve_quantity(raw_q)
        except ValueError as exc:
            chk.error(f"{where}.quantity", str(exc))
            continue
        if raw_q != qty.id:
            chk.warn(f"{where}.quantity", f"{raw_q!r} is an alias; written as {qty.id!r}")
        source = entry.get("source", "computed")
        if source not in SERIES_SOURCES:
            chk.error(f"{where}.source", f"must be 'computed' or 'declared', got {source!r}")
            continue
        T = entry.get("temperature")
        if T is not None:
            T = chk.number(f"{where}.temperature", T, positive=True)
        method = entry.get("method")
        if method is not None and str(method) not in method_ids:
            chk.error(f"{where}.method", f"method {method!r} is not defined under `methods` or "
                                         f"{NAMESPACE}.sources / conformers")
        levels = _parse_levels(f"{where}.levels", entry.get("levels"), pathways, chk, minimum=None)
        uncertainty = _parse_levels(f"{where}.uncertainty", entry.get("uncertainty"), pathways, chk, minimum=0.0)
        if source == "declared" and levels is None:
            chk.error(where, "a declared series needs `levels`")
            continue
        style = chk.mapping(f"{where}.style", entry.get("style")) or {}
        state = entry.get("standard_state")
        if state is not None:
            state = chk.mapping(f"{where}.standard_state", state)
        label = entry.get("label")
        if label is not None and not isinstance(label, str):
            chk.error(f"{where}.label", "must be a string")
            label = None
        if label is None:
            label = qty.label if T is None else f"{qty.label} {T:g} K"
        series.append(Series(
            id=sid, label=label, quantity=qty.id, temperature=T,
            method=str(method) if method is not None else None, style=style, levels=levels,
            declared=(source == "declared"), uncertainty=uncertainty, standard_state=state,
            extensions=_extensions(entry),
        ))

    # -- annotations ---------------------------------------------------------
    annotations: List[dict] = []
    raw_ann = data.get("annotations") or []
    if not isinstance(raw_ann, list):
        chk.error("annotations", "must be a list")
        raw_ann = []
    for i, a in enumerate(raw_ann):
        where = f"annotations[{i}]"
        a = chk.mapping(where, a, allow_none=False)
        if a is None:
            continue
        chk.keys(where, a, _ANNOTATION_KEYS)
        if a.get("type", "barrier") not in ANNOTATION_TYPES:
            chk.error(f"{where}.type", f"must be one of {', '.join(ANNOTATION_TYPES)}")
        path = pathways.get(a.get("pathway"))
        if path is None:
            chk.error(f"{where}.pathway", f"unknown pathway {a.get('pathway')!r}")
        else:
            for end in ("from", "to"):
                if a.get(end) not in path.points and a.get(end) != path.zero:
                    chk.error(f"{where}.{end}", f"{a.get(end)!r} is not a point of pathway {path.name!r}")
        if "series" in a and a["series"] not in seen_ids:
            chk.error(f"{where}.series", f"unknown series {a['series']!r}")
        annotations.append(dict(a))

    # -- style ---------------------------------------------------------------
    style = chk.mapping("style", data.get("style")) or {}
    chk.keys("style", style, _STYLE_KEYS)
    if style.get("preset", "none") != "none":
        chk.error("style.preset", "style presets are reserved for reaction-profile 1.1 (GoodVibes 5.1); "
                                  "use 'none' or omit the key")
    if "layout" in style and style["layout"] not in LAYOUTS:
        chk.error("style.layout", f"must be one of {', '.join(LAYOUTS)}")
    if "connector" in style and style["connector"] not in CONNECTORS:
        chk.error("style.connector", f"must be one of {', '.join(CONNECTORS)}")
    if "label_points" in style and not isinstance(style["label_points"], bool):
        chk.error("style.label_points", "must be true or false")
    if "decimals" in style:
        dec = style["decimals"]
        if isinstance(dec, bool) or not isinstance(dec, int) or not 0 <= dec <= 6:
            chk.error("style.decimals", "must be an integer from 0 to 6")
    if "figsize" in style:
        fs = style["figsize"]
        if (not isinstance(fs, (list, tuple)) or len(fs) != 2
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in fs)):
            chk.error("style.figsize", "must be [width, height] in inches")

    provenance = chk.mapping("provenance", data.get("provenance")) or {}

    if chk.errors:
        raise ProfileError(chk.errors, chk.warnings)
    prof = Profile(
        title=title, description=description, units=units, ensemble=ensemble, default_temperature=default_T,
        species=species, points=points, pathways=pathways, order=order, methods=methods, series=series,
        annotations=annotations, style=style, provenance=provenance, namespace=namespace,
        extensions=_extensions(data),
    )
    prof.upgraded_from = upgraded_from
    return prof, chk.warnings


def _parse_levels(where, raw, pathways, chk: _Checker, minimum):
    if raw is None:
        return None
    raw = chk.mapping(where, raw, allow_none=False)
    if raw is None:
        return None
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for pname, per_point in raw.items():
        pwhere = f"{where}.{pname}"
        path = pathways.get(str(pname))
        if path is None:
            chk.error(pwhere, f"unknown pathway {pname!r}")
            continue
        per_point = chk.mapping(pwhere, per_point, allow_none=False)
        if per_point is None:
            continue
        out[str(pname)] = {}
        for pid, value in per_point.items():
            if str(pid) not in path.points and str(pid) != path.zero:
                chk.error(f"{pwhere}.{pid}", f"{pid!r} is not a point of pathway {pname!r}")
                continue
            if value is None:
                out[str(pname)][str(pid)] = None
                continue
            v = chk.number(f"{pwhere}.{pid}", value, minimum=minimum)
            if v is not None:
                out[str(pname)][str(pid)] = v
    return out


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

class Profile:
    """A reaction-profile document (see the module docstring).

    Build one with :func:`load_profile`, :meth:`from_dict`,
    :meth:`from_table` (CSV of relative energies) or
    :meth:`from_pes_result`; evaluate its computed series with
    :meth:`evaluate`; write it with :meth:`dump`; draw it with
    :meth:`plot`; tabulate it with :meth:`to_rows` / :meth:`to_dataframe`.
    """

    def __init__(self, *, title=None, description=None, units="kcal/mol", ensemble="ideal-gas",
                 default_temperature=298.15, species=None, points=None, pathways=None, order=None,
                 methods=None, series=None, annotations=None, style=None, provenance=None,
                 namespace=None, extensions=None):
        self.title: Optional[str] = title
        self.description: Optional[str] = description
        self.units: str = canonical_units(units)
        self.ensemble: str = ensemble
        self.default_temperature: float = float(default_temperature)
        self.species: Dict[str, dict] = dict(species or {})
        self.points: Dict[str, ProfilePoint] = dict(points or {})
        self.pathways: Dict[str, ProfilePathway] = dict(pathways or {})
        self.order: Optional[List[str]] = list(order) if order else None
        self.methods: Dict[str, dict] = dict(methods or {})
        self.series: List[Series] = list(series or [])
        self.annotations: List[dict] = list(annotations or [])
        self.style: Dict[str, Any] = dict(style or {})
        self.provenance: Dict[str, Any] = dict(provenance or {})
        self.namespace: Dict[str, Any] = dict(namespace or {})
        self.extensions: Dict[str, Any] = dict(extensions or {})
        self.upgraded_from: Optional[str] = None      # 'v2' | 'legacy' when read from a GoodVibes PES file
        self.warnings: List[str] = []

    # -- construction ------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping, *, strict: bool = False) -> "Profile":
        """Parse and validate a document mapping (explicit form or the
        GoodVibes v2 PES YAML shorthand). Raises ProfileError listing every
        problem; warnings are kept on ``.warnings`` and emitted as
        ``ProfileWarning``."""
        prof, warns = _parse(data, strict=strict)
        prof.warnings = warns
        for w in warns:
            warnings.warn(w, ProfileWarning, stacklevel=2)
        return prof

    @classmethod
    def from_spec(cls, spec, kind: str = "legacy") -> "Profile":
        """Upgrade a PESSpec (``kind`` 'legacy' for ``--- # PES`` text,
        'v2' for the GoodVibes v2 PES YAML)."""
        prof, warns = _parse(_explicit_from_spec(spec))
        prof.warnings = warns
        prof.upgraded_from = kind
        return prof

    @classmethod
    def from_table(cls, source, *, layout: str = "auto", units: str = "kcal/mol", quantity: str = "gibbs",
                   temperature: Optional[float] = 298.15, method: Optional[str] = None,
                   series_id: str = "table", label: Optional[str] = None, title: Optional[str] = None
                   ) -> "Profile":
        """A declared-only profile from a CSV of relative energies.

        ``source`` is a path, a file object, CSV text or a list of row
        mappings. ``layout='wide'``: a ``point`` column, optional ``role``
        and ``display`` columns, then one column per pathway (a column
        named ``pathway:series`` gives several series); a point belongs to
        a pathway when its cell is not empty, in row order, and the first
        such point is the pathway's zero. ``layout='long'``: columns
        ``pathway``, ``point``, ``value`` and optionally ``series``,
        ``label``, ``quantity``, ``temperature``, ``units``, ``method``,
        ``role``, ``display``, ``uncertainty``. ``'auto'`` picks long when
        the header has ``pathway`` and ``value``. Values are in ``units``
        (or a long row's own units) and are relative to the pathway zero;
        an empty cell means the point is not on that pathway, ``null`` /
        ``none`` / ``—`` an unknown value.
        """
        rows = _read_rows(source)
        if not rows:
            raise ProfileError(["the table has no rows"])
        header = list(rows[0].keys())
        units = canonical_units(units)
        if layout == "auto":
            layout = "long" if {"pathway", "value"} <= set(header) else "wide"
        if layout not in ("wide", "long"):
            raise ValueError(f"layout must be 'auto', 'wide' or 'long', got {layout!r}")
        qty = resolve_quantity(quantity)
        doc: Dict[str, Any] = {"schema": SCHEMA_TAG, "units": units, "points": {}, "pathways": {},
                               "series": []}
        if title:
            doc["title"] = title
        if temperature is not None:
            doc["default_temperature"] = float(temperature)
        errors: List[str] = []

        def add_point(row):
            pid = (row.get("point") or "").strip()
            if not pid:
                return None
            pt = doc["points"].setdefault(pid, {})
            if (row.get("role") or "").strip():
                pt["role"] = row["role"].strip()
            if (row.get("display") or "").strip():
                pt["display"] = row["display"].strip()
            return pid

        series_defs: Dict[str, dict] = {}

        def series_def(sid, row=None):
            if sid not in series_defs:
                q = resolve_quantity((row or {}).get("quantity") or qty.id)
                T = _cell_float((row or {}).get("temperature"))
                T = T if T is not None else temperature
                lab = (row or {}).get("label") or label
                if not lab:
                    lab = q.label if T is None else f"{q.label} {T:g} K"
                sdef = {"id": sid, "label": lab, "quantity": q.id, "source": "declared", "levels": {}}
                if T is not None:
                    sdef["temperature"] = float(T)
                m = (row or {}).get("method") or method
                if m:
                    sdef["method"] = m
                    doc.setdefault("methods", {}).setdefault(m, {})
                series_defs[sid] = sdef
                doc["series"].append(sdef)
            return series_defs[sid]

        if layout == "wide":
            if "point" not in header:
                raise ProfileError(["a wide table needs a 'point' column"])
            value_cols = [c for c in header if c not in ("point", "role", "display")]
            if not value_cols:
                raise ProfileError(["a wide table needs at least one pathway column"])
            for row in rows:
                pid = add_point(row)
                if pid is None:
                    continue
                for col in value_cols:
                    cell = (row.get(col) or "").strip()
                    if cell == "":
                        continue
                    pname, _, sid = col.partition(":")
                    pname, sid = pname.strip(), (sid.strip() or series_id)
                    doc["pathways"].setdefault(pname, {"points": []})["points"].append(pid)
                    try:
                        value = _cell_float(cell)
                    except ValueError as exc:
                        errors.append(f"row {pid!r}, column {col!r}: {exc}")
                        continue
                    series_def(sid)["levels"].setdefault(pname, {})[pid] = value
            # a pathway appears once per series column; keep each point once, in row order
            for entry in doc["pathways"].values():
                entry["points"] = list(dict.fromkeys(entry["points"]))
        else:
            missing = {"pathway", "point", "value"} - set(header)
            if missing:
                raise ProfileError([f"a long table needs the column(s) {', '.join(sorted(missing))}"])
            for n, row in enumerate(rows, start=2):
                pid = add_point(row)
                pname = (row.get("pathway") or "").strip()
                if pid is None or not pname:
                    errors.append(f"line {n}: needs a pathway and a point")
                    continue
                pts = doc["pathways"].setdefault(pname, {"points": []})["points"]
                if pid not in pts:
                    pts.append(pid)
                sdef = series_def((row.get("series") or "").strip() or series_id, row)
                try:
                    value = _cell_float(row.get("value"))
                    row_units = (row.get("units") or "").strip()
                    factor = hartree_factor(units) / hartree_factor(row_units) if row_units else 1.0
                    if value is not None:
                        value *= factor
                    sdef["levels"].setdefault(pname, {})[pid] = value
                    unc = _cell_float(row.get("uncertainty"))
                    if unc is not None:
                        sdef.setdefault("uncertainty", {}).setdefault(pname, {})[pid] = unc * factor
                except ValueError as exc:
                    errors.append(f"line {n}: {exc}")
        if errors:
            raise ProfileError(errors)
        prof = cls.from_dict(doc)
        for sdef in doc["series"]:
            for pname, lv in sdef["levels"].items():
                zero = prof.pathways[pname].zero
                if lv.get(zero) not in (None, 0.0):
                    msg = (f"series {sdef['id']!r}: pathway {pname!r} is {lv[zero]:g} {units} at its zero "
                           f"point {zero!r}; levels are drawn as given")
                    prof.warnings.append(msg)
                    warnings.warn(msg, ProfileWarning, stacklevel=2)
        return prof

    @classmethod
    def from_pes_result(cls, pes_result: PESResult, *, quantity: str = "qh_gibbs",
                        title: Optional[str] = None, with_conformers: bool = False,
                        invocation: Optional[str] = None) -> "Profile":
        """An evaluated document for a PESResult built in Python (or loaded
        with ``load_pes``): points, pathways, the files of every species
        (as ``goodvibes.sources``) and one computed series of ``quantity``
        per temperature of the result (or the result's own series)."""
        if isinstance(pes_result.source, Profile):
            skeleton = pes_result.source.copy()
        else:
            skeleton = cls._skeleton_from_pes(pes_result)
        if title:
            skeleton.title = title
        computed = [s for s in (pes_result.series or []) if not s.declared]
        if not computed:
            computed = pes_result.default_series(quantity)
        return skeleton._evaluated({None: pes_result}, computed, with_conformers=with_conformers,
                                   invocation=invocation)

    @classmethod
    def _skeleton_from_pes(cls, pes_result: PESResult) -> "Profile":
        species: Dict[str, dict] = {}
        sources: Dict[str, dict] = {}
        points: Dict[str, ProfilePoint] = {}
        pathways: Dict[str, ProfilePathway] = {}
        for path in pes_result.pathways:
            for pt in list(path.points) + [path.zero]:
                if pt.label in points:
                    continue
                for _c, cset in pt.species:
                    if cset.name not in species:
                        species[cset.name] = _species_identity(cset)
                        sources[cset.name] = {"files": [os.path.splitext(os.path.basename(f))[0]
                                                        for f in cset.files]}
                points[pt.label] = ProfilePoint(pt.label, [(c, cs.name) for c, cs in pt.species],
                                                pt.role, pt.display)
            ids = [p.label for p in path.points]
            default = [Edge(a, b) for a, b in zip(ids, ids[1:])]
            explicit = [e for e in path.edges if e not in default]
            pathways[path.name] = ProfilePathway(path.name, ids, path.zero.label, list(path.edges), explicit)
        opts = pes_result.options
        mode = "lowest" if opts.lowest_only else ("gconf" if opts.gconf else "boltzmann")
        ns = {"sources": {DEFAULT_METHOD: sources}, "rollup": {"mode": mode}}
        return cls(units=opts.units, default_temperature=pes_result.temperature, species=species,
                   points=points, pathways=pathways, order=pes_result.order,
                   style={"decimals": opts.decimals}, namespace=ns)

    def copy(self) -> "Profile":
        return copy.deepcopy(self)

    # -- serialisation -----------------------------------------------------

    def to_dict(self, *, include_conformers: bool = True) -> dict:
        """The explicit form, ready for JSON or YAML (plain Python types
        only: NumPy scalars and arrays from parsed data are converted)."""
        return _plain(self._to_dict(include_conformers))

    def _to_dict(self, include_conformers: bool) -> dict:
        d: Dict[str, Any] = {"schema": SCHEMA_TAG}
        if self.title:
            d["title"] = self.title
        if self.description:
            d["description"] = self.description
        d["units"] = self.units
        d["ensemble"] = self.ensemble
        d["default_temperature"] = self.default_temperature
        d["species"] = {name: dict(entry) for name, entry in self.species.items()}
        d["points"] = {}
        for pid, pt in self.points.items():
            entry: Dict[str, Any] = {}
            if pt.species:
                entry["species"] = pt.species_mapping()
            entry["role"] = pt.role
            if pt.display is not None:
                entry["display"] = pt.display
            entry.update(pt.extensions)
            d["points"][pid] = entry
        d["pathways"] = {}
        for name, path in self.pathways.items():
            entry = {"points": list(path.points), "zero": path.zero}
            if path.explicit_edges:
                entry["edges"] = [{"from": e.src, "to": e.dst, "kind": e.kind} for e in path.explicit_edges]
            entry.update(path.extensions)
            d["pathways"][name] = entry
        if self.order:
            d["order"] = list(self.order)
        if self.methods:
            d["methods"] = copy.deepcopy(self.methods)
        d["series"] = [self._series_dict(s) for s in self.series]
        if self.annotations:
            d["annotations"] = copy.deepcopy(self.annotations)
        if self.style:
            d["style"] = dict(self.style)
        d.update(copy.deepcopy(self.extensions))
        ns = copy.deepcopy(self.namespace)
        if not include_conformers:
            ns.pop("conformers", None)
        if ns:
            d[NAMESPACE] = ns
        if self.provenance:
            d["provenance"] = copy.deepcopy(self.provenance)
        return d

    def _series_dict(self, s: Series) -> dict:
        out: Dict[str, Any] = {"id": s.id, "label": s.label}
        if s.method is not None:
            out["method"] = s.method
        out["quantity"] = s.quantity
        if s.temperature is not None:
            out["temperature"] = s.temperature
        out["source"] = "declared" if s.declared else "computed"
        if s.levels is not None:
            factor = hartree_factor(self.units) / hartree_factor(s.units or self.units)
            out["levels"] = {p: {k: (v * factor if v is not None else None) for k, v in lv.items()}
                             for p, lv in s.levels.items()}
        if s.uncertainty:
            out["uncertainty"] = copy.deepcopy(s.uncertainty)
        if s.style:
            out["style"] = dict(s.style)
        if s.standard_state:
            out["standard_state"] = dict(s.standard_state)
        out.update(s.extensions)
        return out

    def to_json(self, *, include_conformers: bool = True, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(include_conformers=include_conformers), indent=indent,
                          ensure_ascii=False, default=str)

    def to_yaml(self, *, include_conformers: bool = True) -> str:
        import yaml
        return yaml.safe_dump(self.to_dict(include_conformers=include_conformers), sort_keys=False,
                              allow_unicode=True, default_flow_style=None, width=100)

    def dump(self, path, *, include_conformers: bool = True, indent: Optional[int] = 2) -> None:
        """Write the explicit form to ``path`` (.json, .yaml or .yml).
        ``indent=None`` writes compact JSON (a document with embedded
        conformers is several times smaller)."""
        ext = os.path.splitext(str(path))[1].lower()
        if ext == ".json":
            text = self.to_json(include_conformers=include_conformers, indent=indent)
        elif ext in (".yaml", ".yml"):
            text = self.to_yaml(include_conformers=include_conformers)
        else:
            raise ValueError(f"cannot write a reaction-profile document as {ext or 'no extension'!r}; "
                             "use .json, .yaml or .yml")
        Path(path).write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")

    def validate(self, strict: bool = False) -> List[str]:
        """Re-validate the explicit form; returns the warnings, raises
        ProfileError on an error."""
        errors, warns = validate_document(self.to_dict(), strict=strict)
        if errors:
            raise ProfileError(errors, warns)
        return warns

    # -- data access ---------------------------------------------------------

    def get_series(self, sid: str) -> Series:
        for s in self.series:
            if s.id == sid:
                return s
        raise KeyError(f"no series {sid!r} (available: {[s.id for s in self.series]})")

    def merged_order(self) -> List[str]:
        if self.order:
            return list(self.order)
        return merge_point_order([p.points for p in self.pathways.values()])

    def levels(self) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
        """{series id: {pathway: {point: value in the document units}}} for
        every series with levels (declared, or evaluated)."""
        out = {}
        for s in self.series:
            if s.levels is None:
                continue
            factor = hartree_factor(self.units) / hartree_factor(s.units or self.units)
            out[s.id] = {p: {k: (v * factor if v is not None else None) for k, v in lv.items()}
                         for p, lv in s.levels.items()}
        return out

    def to_rows(self, layout: str = "long") -> List[dict]:
        """The levels as table rows. ``long``: one row per pathway × point ×
        series (pathway, point, display, role, series, label, quantity,
        temperature, method, value, units, uncertainty). ``wide``: one row per
        point (in the merged x order) with a column per pathway, or per
        ``pathway:series`` when there are several series; ``from_table``
        reads either back."""
        levels = self.levels()
        series = [s for s in self.series if s.id in levels]
        if layout == "long":
            rows = []
            for pname, path in self.pathways.items():
                for pid in path.points:
                    pt = self.points[pid]
                    for s in series:
                        lv = levels[s.id].get(pname, {})
                        if pid not in lv:
                            continue
                        unc = (s.uncertainty or {}).get(pname, {}).get(pid)
                        rows.append({
                            "pathway": pname, "point": pid, "display": pt.display or pid, "role": pt.role,
                            "series": s.id, "label": s.label, "quantity": s.quantity,
                            "temperature": s.temperature, "method": s.method, "value": lv[pid],
                            "units": self.units, "uncertainty": unc,
                        })
            return rows
        if layout != "wide":
            raise ValueError(f"layout must be 'long' or 'wide', got {layout!r}")
        multi = len(series) > 1
        cols = [(pname, s) for pname in self.pathways for s in series
                if pname in levels[s.id] and levels[s.id][pname]]
        rows = []
        for pid in self.merged_order():
            pt = self.points[pid]
            row = {"point": pid, "role": pt.role, "display": pt.display or pid}
            for pname, s in cols:
                key = f"{pname}:{s.id}" if multi else pname
                # "" = the point is not on this pathway; None = on it, value unknown
                row[key] = levels[s.id][pname].get(pid, "")
            rows.append(row)
        return rows

    def to_dataframe(self, layout: str = "long"):
        """``to_rows`` as a pandas DataFrame (pandas is optional)."""
        try:
            import pandas as pd
        except ImportError as exc:                          # pragma: no cover
            raise ImportError("to_dataframe requires pandas; install with `pip install pandas`.") from exc
        return pd.DataFrame(self.to_rows(layout))

    def write_table(self, path, layout: str = "wide", decimals: Optional[int] = None) -> None:
        """Write ``to_rows`` to a .csv (full precision) or .md (rounded to
        ``decimals``, default ``style.decimals`` or 1) file."""
        rows = self.to_rows(layout)
        ext = os.path.splitext(str(path))[1].lower()
        if ext == ".csv":
            fixed = {"point", "role", "display"} if layout == "wide" else set()
            nulls = ({k for r in rows for k in r} - fixed) if layout == "wide" else {"value"}
            text = _rows_to_csv(rows, null_cols=nulls)
        elif ext in (".md", ".markdown"):
            text = _rows_to_markdown(rows, decimals if decimals is not None else self.style.get("decimals", 1))
        else:
            raise ValueError(f"cannot write a table as {ext or 'no extension'!r}; use .csv or .md")
        Path(path).write_text(text, encoding="utf-8")

    # -- evaluation ----------------------------------------------------------

    def source_methods(self) -> List[str]:
        """Methods that GoodVibes can evaluate: those with
        ``goodvibes.sources`` or embedded ``goodvibes.conformers``."""
        out = list(self.namespace.get("sources", {}))
        out += [m for m in self.namespace.get("conformers", {}) if m not in out]
        return out

    def default_method(self) -> Optional[str]:
        methods = self.source_methods()
        if not methods:
            return None
        return DEFAULT_METHOD if DEFAULT_METHOD in methods else methods[0]

    def pes_options(self) -> PESOptions:
        rollup = self.namespace.get("rollup", {})
        mode = rollup.get("mode", "gconf")
        thermo = self.namespace.get("thermo", {})
        return PESOptions(units=self.units, decimals=int(self.style.get("decimals", 2)),
                          gconf=(mode == "gconf"), QH=True, spc_used=bool(thermo.get("spc")),
                          lowest_only=(mode == "lowest"))

    def to_pes_result(self, thermo_data=None, *, method: Optional[str] = None,
                      temperatures: Optional[Sequence[float]] = None,
                      options: Optional[PESOptions] = None) -> PESResult:
        """The PESResult model of this document for one method.

        With ``thermo_data`` ({file: calc_bbe} or a list of ``ThermoResult``)
        each species' conformers are the files its ``goodvibes.sources``
        entry matches; without it, the embedded ``goodvibes.conformers``
        are used when present; otherwise points have no species and only
        stored levels can be drawn. Computed series keep their stored
        levels only when no data is used (the data would recompute them).
        """
        method = method or self.default_method()
        sets, used_data = self._conformer_sets(method, thermo_data)
        pts: Dict[str, Point] = {}
        for pid, spec in self.points.items():
            if spec.species and all(name in sets for _c, name in spec.species):
                terms = [(c, sets[name]) for c, name in spec.species]
            else:
                terms = []
            pts[pid] = Point(label=pid, species=terms, role=spec.role, display=spec.display)
        pathways = [Pathway(name=path.name, points=[pts[i] for i in path.points], zero=pts[path.zero],
                            edges=list(path.edges))
                    for path in self.pathways.values()]
        opts = copy.copy(options) if options is not None else self.pes_options()
        opts.units = self.units
        series = []
        for s in self.series:
            if s.declared:
                series.append(s)
            elif s.method in (None, method) or method is None:
                series.append(replace(s, levels=None) if used_data else s)
        temps = list(temperatures) if temperatures else [self.default_temperature]
        return PESResult(pathways=pathways, options=opts, temperatures=temps, series=series,
                         order=self.order, source=self)

    def _conformer_sets(self, method, thermo_data) -> Tuple[Dict[str, ConformerSet], bool]:
        needed = {name for pt in self.points.values() for _c, name in pt.species}
        rollup = self.namespace.get("rollup", {})
        weight_by = rollup.get("weight_by", "qh_gibbs")
        sets: Dict[str, ConformerSet] = {}
        used = False
        if thermo_data is not None:
            from .pes_loader import _resolve_species
            data = _as_thermo_data(thermo_data)
            srcs = self.namespace.get("sources", {}).get(method, {}) if method else {}
            for name in sorted(needed):
                if name in srcs:
                    cset = _resolve_species(name, _source_to_patterns(srcs[name]), data)
                    sets[name] = ConformerSet(cset.name, cset.files, cset.bbes, cset.entries, weight_by)
            used = bool(sets)
        elif method and method in self.namespace.get("conformers", {}):
            from .io import dict_to_qcdata
            from .thermo import ThermoOptions, calc_bbe
            for name, items in self.namespace["conformers"][method].items():
                if name not in needed:
                    continue
                files, bbes = [], []
                for item in items:
                    qc = dict_to_qcdata(item["qcdata"])
                    opts = ThermoOptions(**{k: v for k, v in item["options"].items()
                                            if k in ThermoOptions.__dataclass_fields__})
                    files.append(item.get("file") or qc.file)
                    bbes.append(calc_bbe.from_options(qc, opts))
                sets[name] = ConformerSet(name, files, bbes, weight_by=weight_by)
            used = bool(sets)
        dedup = self.namespace.get("dedup")
        if dedup and sets:
            kw = {k: dedup[k] for k in ("e_cutoff", "ro_cutoff", "rmsd_cutoff") if k in dedup}
            sets = {name: cs.dedup(**kw) for name, cs in sets.items()}
        return sets, used

    def evaluate(self, thermo_data=None, *, temperatures: Optional[Sequence[float]] = None,
                 options: Optional[PESOptions] = None, default_series: Optional[Sequence[Series]] = None,
                 with_conformers: bool = False, invocation: Optional[str] = None,
                 base_temperature: Optional[float] = None) -> "Profile":
        """A new document with every computed series evaluated.

        The data is ``thermo_data`` ({file: calc_bbe} or ``ThermoResult``
        list) resolved through ``goodvibes.sources``, or, when it is None,
        the embedded ``goodvibes.conformers``. ``temperatures`` turns each
        computed series into one series per temperature (ids suffixed
        ``@<T>K``). A document without computed series gets
        ``default_series`` (default: Δqh-G(T) at ``default_temperature``).
        ``options`` overrides the rollup (the CLI passes its flags this
        way). ``base_temperature`` replaces ``default_temperature`` for
        computed series that give no temperature (the CLI passes its run
        temperature, so the document matches the tables and the ``pes``
        block of the same run). ``with_conformers`` embeds every structure's parsed data and
        thermochemistry options so the document can be re-evaluated, at
        any temperature, with no output files. Declared series are copied
        unchanged. Adds a ``provenance`` block.
        """
        computed = [s for s in self.series if not s.declared]
        if not computed:
            computed = list(default_series) if default_series else [
                Series(id=f"qh_gibbs@{self.default_temperature:g}K", label=QUANTITIES["qh_gibbs"].label,
                       quantity="qh_gibbs", temperature=self.default_temperature)]
        if temperatures:
            expanded = []
            for s in computed:
                base = re.sub(r"@[0-9.]+K$", "", s.id)
                q = QUANTITIES[s.quantity]
                for T in temperatures:
                    expanded.append(replace(s, id=f"{base}@{float(T):g}K", temperature=float(T),
                                            label=f"{q.label} {float(T):g} K", levels=None))
            computed = expanded
        by_method: Dict[Optional[str], PESResult] = {}
        for s in computed:
            m = s.method or self.default_method()
            if m in by_method:
                continue
            pes = self.to_pes_result(thermo_data, method=m, options=options,
                                     temperatures=[base_temperature] if base_temperature else None)
            if not any(pt.species for path in pes.pathways for pt in path.points):
                where = "the thermo data given" if thermo_data is not None else "embedded conformers"
                raise ProfileError([f"series {s.id!r}: no species of method {m!r} could be built from {where}; "
                                    f"check {NAMESPACE}.sources.{m} (or pass thermo_data)"])
            by_method[m] = pes
        keyed = {m: pes for m, pes in by_method.items()}
        return self._evaluated(keyed, computed, with_conformers=with_conformers, invocation=invocation,
                               method_of=lambda s: s.method or self.default_method())

    def _evaluated(self, pes_by_method, computed, *, with_conformers=False, invocation=None,
                   method_of=None) -> "Profile":
        method_of = method_of or (lambda s: None)
        new = self.copy()
        evaluated: Dict[str, Series] = {}
        warns: List[str] = []
        inputs: List[dict] = []
        conformers: Dict[str, Dict[str, list]] = {}
        thermo_summary = None
        seen_sets = set()
        for s in computed:
            m = method_of(s)
            pes = pes_by_method[m]
            T = s.temperature if s.temperature is not None else pes.temperature
            stub = replace(s, levels=None, temperature=T)
            levels = {}
            for path in pes.pathways:
                lv = stub.evaluate(pes, path)
                if lv:
                    levels[path.name] = lv
            evaluated[s.id] = replace(stub, levels=levels, declared=False, units=None)
            for path in pes.pathways:
                for pt in list(path.points) + [path.zero]:
                    for _c, cset in pt.species:
                        key = (m, cset.name)
                        if key in seen_sets:
                            continue
                        seen_sets.add(key)
                        for f, bbe in zip(cset.files, cset.bbes):
                            inputs.append(_input_record(f, cset.name, m if m != DEFAULT_METHOD else None))
                            reason = getattr(bbe, "spc_reason", None)
                            if reason:
                                warns.append(f"{os.path.basename(f)}: {reason}")
                        if cset.entries and thermo_summary is None:
                            thermo_summary = _options_summary(cset.entries[0].options)
                        if with_conformers:
                            if not cset.entries:
                                warns.append(f"species {cset.name!r}: conformers not embedded (no parsed input kept)")
                                continue
                            conformers.setdefault(m or DEFAULT_METHOD, {})[cset.name] = [
                                _conformer_record(e) for e in cset.entries]
        # evaluated series take the place of the first computed series; declared
        # series keep their positions
        out, placed = [], False
        for s in self.series:
            if s.declared:
                out.append(s)
            elif not placed:
                out.extend(evaluated.values())
                placed = True
        new.series = out if placed else list(evaluated.values()) + out
        any_pes = next(iter(pes_by_method.values()))
        mode = ("lowest" if any_pes.options.lowest_only else ("gconf" if any_pes.options.gconf else "boltzmann"))
        rollup = dict(new.namespace.get("rollup", {}))
        rollup["mode"] = mode
        new.namespace["rollup"] = rollup
        if thermo_summary:
            new.namespace["thermo"] = thermo_summary
        if with_conformers:
            new.namespace["conformers"] = conformers
        new.provenance = {
            "tool": "goodvibes",
            "goodvibes_version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "invocation": invocation or "goodvibes.profile.Profile.evaluate",
            "inputs": inputs,
            "warnings": warns,
        }
        return new

    # -- plotting --------------------------------------------------------------

    def plot(self, *, series=None, pathways=None, layout: Optional[str] = None, ax=None,
             label_points: Optional[bool] = None, connector: Optional[str] = None,
             title: Optional[str] = None, annotations: bool = True, **kw):
        """Draw the document with ``goodvibes.plot.plot_profile``.

        ``series`` (ids) defaults to every series; the document's ``style``
        gives the layout, connector, decimals, figure size and whether to
        label points unless overridden here; ``annotations`` draws the
        document's barrier / span annotations. A computed series needs
        levels (an evaluated document) or embedded conformers.
        """
        from .plot import plot_profile
        pes = self.to_pes_result()
        chosen = self.series if series is None else [self.get_series(s) if isinstance(s, str) else s
                                                     for s in ([series] if isinstance(series, str) else series)]
        if not chosen:
            raise ProfileError(["the document has no series to draw"])
        drawable = {p.name for p in pes.pathways if p.zero.species}
        for s in chosen:
            if s.levels is None and not s.declared and not drawable:
                raise ProfileError([f"series {s.id!r} is computed but has no levels: evaluate the document "
                                    "first (goodvibes ... --pes FILE --profile OUT, or Profile.evaluate)"])
        style = {k: self.style[k] for k in ("decimals", "figsize", "connector") if k in self.style}
        if connector:
            style["connector"] = connector
        prof = plot_profile(
            pes, series=chosen, pathways=pathways, layout=layout or self.style.get("layout", "overlay"),
            ax=ax, style=style,
            label_points=self.style.get("label_points", False) if label_points is None else label_points,
            title=title if title is not None else self.title, **kw)
        if annotations:
            drawn_paths = {p.name for p in prof.pathways}
            drawn_series = {s.id for s in prof.series}
            for a in self.annotations:
                sid = a.get("series", prof.series[0].id)
                if a["pathway"] not in drawn_paths or sid not in drawn_series:
                    continue
                if prof.level(a["pathway"], a["from"], sid) is None or prof.level(a["pathway"], a["to"], sid) is None:
                    continue
                dec = self.style.get("decimals", 1)
                fmt = (a["label"] + " {:+." + str(dec) + "f}") if a.get("label") else ("{:+." + str(dec) + "f}")
                prof.annotate_barrier(a["pathway"], a["from"], a["to"], series=sid, fmt=fmt)
        return prof

    def __repr__(self) -> str:
        return (f"Profile(title={self.title!r}, pathways={list(self.pathways)}, "
                f"series={[s.id for s in self.series]}, units={self.units!r})")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _plain(obj):
    """Recursively convert NumPy scalars/arrays and tuples to plain Python
    types so the document serialises identically to JSON and YAML."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):      # numpy scalar or array
        return _plain(obj.tolist())
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return float(obj)
    return obj


def _species_identity(cset: ConformerSet) -> dict:
    out = {}
    qc = getattr(cset.bbes[0], "qcdata", None)
    if qc is not None:
        if isinstance(getattr(qc, "charge", None), int):
            out["charge"] = qc.charge
        if isinstance(getattr(qc, "multiplicity", None), int):
            out["multiplicity"] = qc.multiplicity
    return out


def _as_thermo_data(thermo_data) -> dict:
    if isinstance(thermo_data, Mapping):
        return dict(thermo_data)
    out = {}
    for r in thermo_data:
        out[r.file] = r.bbe
    return out


def _input_record(path: str, species: str, method: Optional[str]) -> dict:
    rec = {"file": os.path.basename(path), "species": species}
    if method:
        rec["method"] = method
    if os.path.isfile(path):
        h = hashlib.sha1()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        rec["sha1"] = h.hexdigest()
    return rec


def _options_summary(options) -> dict:
    d = asdict(options)
    d.pop("temperature", None)
    return d


def _conformer_record(entry) -> dict:
    from .io import qcdata_to_dict
    qc = qcdata_to_dict(entry.qcdata)
    qc.pop("_cache_version", None)
    qc["file"] = os.path.basename(qc.get("file", "") or entry.file)
    qc.pop("sp_file", None)
    return {"file": os.path.basename(entry.file or qc["file"]), "qcdata": qc, "options": asdict(entry.options)}


def _cell_float(cell) -> Optional[float]:
    if cell is None:
        return None
    text = str(cell).strip()
    if text.lower() in ("", "null", "none", "nan", "—", "-", "na", "n/a"):
        return None
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"not a number: {text!r}") from None


def _read_rows(source) -> List[dict]:
    if isinstance(source, list):
        return [{str(k).strip(): ("" if v is None else str(v)) for k, v in row.items()} for row in source]
    if hasattr(source, "read"):
        text = source.read()
    elif isinstance(source, (str, os.PathLike)) and ("\n" not in str(source)) and os.path.exists(source):
        text = Path(source).read_text(encoding="utf-8-sig")
    elif isinstance(source, str):
        text = source
    else:
        raise ValueError(f"cannot read a table from {source!r}")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        return []
    dialect = "excel-tab" if "\t" in lines[0] and "," not in lines[0] else "excel"
    reader = csv.DictReader(lines, dialect=dialect)
    return [{str(k).strip(): (v or "").strip() for k, v in row.items() if k is not None} for row in reader]


def _rows_to_csv(rows: List[dict], null_cols=()) -> str:
    """CSV text; a None in one of ``null_cols`` is written ``null`` (an
    unknown value, as opposed to an empty cell: not on the pathway)."""
    if not rows:
        return ""
    buf = io.StringIO()
    cols = list(rows[0].keys())
    for r in rows[1:]:
        cols += [k for k in r if k not in cols]
    w = csv.DictWriter(buf, fieldnames=cols, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: (("null" if k in null_cols else "") if r.get(k) is None else r.get(k)) for k in cols})
    return buf.getvalue()


def _rows_to_markdown(rows: List[dict], decimals: int) -> str:
    if not rows:
        return ""
    cols = list(rows[0].keys())

    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.{decimals}f}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# JSON Schema + validation entry points
# ---------------------------------------------------------------------------

def load_json_schema() -> dict:
    """The reaction-profile JSON Schema shipped with GoodVibes."""
    path = Path(__file__).resolve().parent / "schemas" / SCHEMA_FILE
    return json.loads(path.read_text(encoding="utf-8"))


def schema_errors(data) -> List[str]:
    """Errors from the JSON Schema alone (requires ``jsonschema``); the
    explicit form only. Raises ImportError when jsonschema is missing."""
    import jsonschema
    validator = jsonschema.Draft202012Validator(load_json_schema())
    out = []
    for err in sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path)):
        where = "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in err.absolute_path).lstrip(".")
        out.append(f"{where or '(root)'}: {err.message}")
    return out


def validate_document(data, *, strict: bool = False, use_jsonschema: bool = True) -> Tuple[List[str], List[str]]:
    """(errors, warnings) for a document mapping.

    Runs the reference Python validator and, for a document in the
    explicit form (one with a ``schema`` key) when ``jsonschema`` is
    installed, the JSON Schema too; errors from the latter are prefixed
    ``schema:``. Never raises for an invalid document.
    """
    errors: List[str] = []
    warns: List[str] = []
    try:
        _prof, warns = _parse(data, strict=strict)
    except ProfileError as exc:
        errors.extend(exc.errors)
        warns = exc.warnings
    if use_jsonschema and isinstance(data, Mapping) and "schema" in data:
        try:
            extra = schema_errors(json.loads(json.dumps(data, default=str)))
        except ImportError:
            extra = []
        errors.extend(f"schema: {e}" for e in extra)
    return errors, warns


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_profile(source, *, strict: bool = False, **table_options) -> Profile:
    """Read a reaction-profile document.

    ``source`` is a Profile (returned as is), a mapping, or a path to:
    a reaction-profile document (.yaml / .yml / .json), a GoodVibes
    ``--json`` / ``--export`` payload with a ``profile`` block, a CSV/TSV
    table of relative energies (``table_options`` go to
    :meth:`Profile.from_table`), a GoodVibes v2 PES YAML or a legacy
    ``--- # PES`` text file (both upgraded to the explicit form).
    """
    if isinstance(source, Profile):
        return source
    if isinstance(source, Mapping):
        return Profile.from_dict(source, strict=strict)
    path = str(source)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        return Profile.from_table(path, **table_options)
    text = Path(path).read_text(encoding="utf-8")
    from .pes_loader import is_legacy_format
    if is_legacy_format(text):
        from .pes_legacy import parse_legacy
        return Profile.from_spec(parse_legacy(text))
    if ext == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:                          # pragma: no cover
            raise ImportError("reading YAML profiles requires PyYAML (`pip install pyyaml`)") from exc
        data = yaml.safe_load(text)
    if isinstance(data, Mapping) and "schema_version" in data and "results" in data:
        if "profile" not in data:
            raise ProfileError([f"{path} is a GoodVibes payload without a `profile` block "
                                "(write one with goodvibes ... --pes FILE --json OUT)"])
        data = data["profile"]
    return Profile.from_dict(data, strict=strict)
