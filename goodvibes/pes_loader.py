"""Format-agnostic PES loading pipeline.

Three stages, each independently testable:

    text  ──parse_legacy()/parse_yaml()──▶  PESSpec
    PESSpec + thermo_data ──build_pes_result()──▶  PESResult

`PESSpec` is the intermediate that both parsers produce: a description of
pathways, species (as file patterns), zero overrides, and options — but
*no* references to actual `calc_bbe` instances. The builder resolves
patterns against `thermo_data`, constructs `ConformerSet`s, and emits a
fully-realized `PESResult`.

The format dispatcher `load_pes()` sniffs the file content (presence of
`--- # PES`/`# SPECIES`/`# FORMAT` markers) to choose the parser; the
legacy path emits a `DeprecationWarning`.
"""
from __future__ import annotations

import fnmatch
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

from .pes_model import (
    ConformerSet, PESOptions, PESResult, Pathway, Point, parse_point_label,
)


# ---------------------------------------------------------------------------
# Intermediate: PESSpec
# ---------------------------------------------------------------------------

@dataclass
class PESSpec:
    """Format-agnostic intermediate. Both parsers emit this.

    Attributes:
        pathways: pathway name -> ordered list of point labels.
        species:  species name -> file pattern (glob or literal) or list of patterns.
        zero:     pathway name -> point label to use as zero. Optional;
                  pathways missing from this dict default to their first point.
        options:  PESOptions parsed from the FORMAT block.
        format_extras: any FORMAT keys not consumed by PESOptions
                       (kept verbatim so graph_reaction_profile can read them).
    """
    pathways: Dict[str, List[str]]
    species: Dict[str, Union[str, List[str]]]
    zero: Dict[str, str] = field(default_factory=dict)
    options: PESOptions = field(default_factory=PESOptions)
    format_extras: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

_LEGACY_MARKERS = {"PES", "SPECIES", "FORMAT"}


def is_legacy_format(text: str) -> bool:
    """Sniff for the `--- # PES` / `# SPECIES` / `# FORMAT` markers.

    Presence of any one of these markers anywhere in the file means legacy.
    Otherwise the file is treated as true YAML.
    """
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("---") and "#" in line:
            comment = line.split("#", 1)[1].strip().upper()
            if comment in _LEGACY_MARKERS:
                return True
    return False


# ---------------------------------------------------------------------------
# Pattern resolution
# ---------------------------------------------------------------------------

def _stem(path: str) -> str:
    """Basename without final extension. 'foo/bar.log' -> 'bar'."""
    return os.path.splitext(os.path.basename(path))[0]


def _strip_ext(pattern: str) -> str:
    """Drop a trailing .log/.out so patterns can be written either way."""
    for ext in (".log", ".out"):
        if pattern.endswith(ext):
            return pattern[: -len(ext)]
    return pattern


def resolve_pattern(pattern: str, thermo_data: dict) -> List[str]:
    """Resolve a file pattern against the keys of `thermo_data`.

    Patterns are matched against the basename (sans extension) of each key.
    A pattern with no glob characters must equal the stem exactly. A glob
    pattern uses `fnmatch`. Returns the matched thermo_data keys in their
    original (insertion) order.
    """
    pat = _strip_ext(pattern.strip())
    has_glob = any(c in pat for c in "*?[")
    matches: List[str] = []
    for key in thermo_data:
        stem = _stem(key)
        if has_glob:
            if fnmatch.fnmatch(stem, pat):
                matches.append(key)
        elif stem == pat:
            matches.append(key)
    return matches


def _resolve_species(
    name: str,
    pattern: Union[str, List[str]],
    thermo_data: dict,
) -> ConformerSet:
    """Build a ConformerSet for one species by resolving its pattern(s)."""
    patterns = [pattern] if isinstance(pattern, str) else list(pattern)
    files: List[str] = []
    for p in patterns:
        files.extend(resolve_pattern(p, thermo_data))
    # De-duplicate while preserving order (a species can list overlapping globs).
    seen = set()
    deduped = [f for f in files if not (f in seen or seen.add(f))]
    if not deduped:
        raise ValueError(
            f"species {name!r}: no files matched pattern(s) {patterns!r}. "
            f"available stems: {sorted({_stem(k) for k in thermo_data})}"
        )
    bbes = [thermo_data[f] for f in deduped]
    return ConformerSet(name=name, files=deduped, bbes=bbes)


# ---------------------------------------------------------------------------
# Builder: PESSpec + thermo_data  ->  PESResult
# ---------------------------------------------------------------------------

def build_pes_result(
    spec: PESSpec,
    thermo_data: dict,
    temperatures: Optional[List[float]] = None,
) -> PESResult:
    """Resolve patterns, build ConformerSets, parse point labels, return PESResult.

    All species referenced (transitively) by the pathways' points must
    resolve to at least one file in `thermo_data`; unresolved names raise
    `KeyError`.
    """
    # 1. Build the species map by resolving every species mentioned anywhere.
    needed: set = set()
    for points in spec.pathways.values():
        for point_label in points:
            for _, name in parse_point_label(point_label):
                needed.add(name)
    for zero_label in spec.zero.values():
        for _, name in parse_point_label(zero_label):
            needed.add(name)

    species_map: Dict[str, ConformerSet] = {}
    for name in needed:
        if name not in spec.species:
            raise KeyError(
                f"species {name!r} referenced by a pathway but not defined "
                f"in the SPECIES block (defined: {sorted(spec.species)})"
            )
        species_map[name] = _resolve_species(name, spec.species[name], thermo_data)

    # 2. Build pathways. For each pathway, parse every point, resolve zero.
    pathways: List[Pathway] = []
    for path_name, point_labels in spec.pathways.items():
        points = [Point.from_label(label, species_map) for label in point_labels]
        if path_name in spec.zero:
            zero_point = Point.from_label(spec.zero[path_name], species_map)
        else:
            zero_point = points[0]
        pathways.append(Pathway(name=path_name, points=points, zero=zero_point))

    return PESResult(
        pathways=pathways,
        options=spec.options,
        temperatures=temperatures or [298.15],
    )


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

def load_pes(
    path: str,
    thermo_data: dict,
    temperatures: Optional[List[float]] = None,
) -> PESResult:
    """Read a PES definition file, parse it, and return a `PESResult`.

    Sniffs the file format (legacy `--- # PES` markers vs proper YAML);
    legacy emits a `DeprecationWarning`.
    """
    text = Path(path).read_text()
    if is_legacy_format(text):
        warnings.warn(
            f"PES file {path!r} uses the legacy line-based format. "
            "This format is deprecated and will be removed in v5.1; "
            "see ROADMAP.md Sub-plan B for the new YAML schema.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .pes_legacy import parse_legacy
        spec = parse_legacy(text)
    else:
        from .pes_yaml import parse_yaml
        spec = parse_yaml(text)
    return build_pes_result(spec, thermo_data, temperatures=temperatures)
