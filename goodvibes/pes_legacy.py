"""Parser for the legacy line-based PES format.

The format predates v4.2; despite the `.yaml` extension it isn't valid YAML
(parens in species names like `Cu(III)-S` and the `[a, b, c]` PES point
lists with bare identifiers would either fail or parse strangely).

Sections are introduced by `--- # NAME` comment markers:

    --- # PES
        Reaction: [Int-I + TolS, Int-II, Int-III]
    --- # SPECIES
        Int-I: Int-I_*
    --- # FORMAT
        units: kcal/mol
        dec: 1
        zero: Int-I + TolS

The parser produces a `PESSpec`; format keys it doesn't consume (graph
styling — dpi, color, title, etc.) are preserved in `format_extras` so
`graph_reaction_profile` can keep reading them.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .pes_model import PESOptions
from .pes_loader import PESSpec


# Section terminators: a `---` line OR end of file.
def _split_sections(text: str) -> Dict[str, List[str]]:
    """Split the input into named sections {PES, SPECIES, FORMAT}.

    Lines under each section header are returned with leading/trailing
    whitespace and pure-comment lines stripped. Blank lines are kept so
    a downstream parser can tell that a list ended.
    """
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("---"):
            current = None
            if "#" in stripped:
                marker = stripped.split("#", 1)[1].strip().upper()
                if marker in {"PES", "SPECIES", "FORMAT"}:
                    current = marker
                    sections.setdefault(current, [])
            continue
        if current is None:
            continue
        if stripped.startswith("#"):
            continue
        sections[current].append(raw)
    return sections


def _split_kv(line: str) -> Optional[Tuple[str, str]]:
    """Split a `key: value` or `key = value` line. Returns None if not a kv line."""
    s = line.strip()
    if not s:
        return None
    # Allow either separator. We replace ':' with '=' so a single split works,
    # but only for the first occurrence — values can contain ':' in colors etc.
    if ":" in s and ("=" not in s or s.index(":") < s.index("=")):
        key, _, value = s.partition(":")
    elif "=" in s:
        key, _, value = s.partition("=")
    else:
        return None
    return key.strip(), value.strip()


def _parse_pathways(lines: List[str]) -> Dict[str, List[str]]:
    """Parse the PES section into {pathway_name: [point_label, ...]}.

    Each non-blank line is `name: [p1, p2, ...]`. Blank lines and comment
    lines are skipped.
    """
    pathways: Dict[str, List[str]] = {}
    for line in lines:
        kv = _split_kv(line)
        if kv is None:
            continue
        name, value = kv
        if not (value.startswith("[") and value.endswith("]")):
            raise ValueError(
                f"PES section: pathway {name!r} value must be a [bracketed list], "
                f"got {value!r}"
            )
        inner = value[1:-1]
        # Split on commas but preserve `+` — point labels are e.g. "A + B".
        points = [p.strip() for p in inner.split(",")]
        points = [p for p in points if p]
        if not points:
            raise ValueError(f"PES section: pathway {name!r} has no points")
        pathways[name] = points
    if not pathways:
        raise ValueError("PES section is empty or missing")
    return pathways


def _parse_species(lines: List[str]) -> Dict[str, str]:
    """Parse the SPECIES section into {species_name: pattern}.

    `folder: PATH` lines are ignored (legacy used these to prefix paths;
    the modern orchestrator already provides full paths in thermo_data).
    """
    species: Dict[str, str] = {}
    for line in lines:
        kv = _split_kv(line)
        if kv is None:
            continue
        key, value = kv
        if key.lower() == "folder":
            continue
        if not value:
            raise ValueError(f"SPECIES section: empty pattern for {key!r}")
        species[key] = value
    if not species:
        raise ValueError("SPECIES section is empty or missing")
    return species


def _parse_format(lines: List[str]) -> Tuple[PESOptions, Dict[str, str], Dict[str, str]]:
    """Parse the FORMAT section. Returns (PESOptions, zero overrides, extras).

    Recognised keys:
        units    -> options.units
        dec      -> options.decimals
        zero     -> applies to *all* pathways (legacy never supported per-pathway)
    Any other key is preserved verbatim in `extras` for downstream consumers
    (graph_reaction_profile reads dpi, color, title, etc.).
    """
    options = PESOptions()
    zero_default: Optional[str] = None
    extras: Dict[str, str] = {}
    for line in lines:
        kv = _split_kv(line)
        if kv is None:
            continue
        key, value = kv
        key_lower = key.lower()
        if key_lower == "units":
            if value not in ("kcal/mol", "kJ/mol"):
                raise ValueError(
                    f"FORMAT.units: expected 'kcal/mol' or 'kJ/mol', got {value!r}"
                )
            options = _replace(options, units=value)
        elif key_lower == "dec":
            try:
                options = _replace(options, decimals=int(value))
            except ValueError:
                raise ValueError(f"FORMAT.dec: expected integer, got {value!r}")
        elif key_lower == "zero":
            zero_default = value
        else:
            extras[key] = value
    zeros: Dict[str, str] = {}
    if zero_default is not None:
        zeros["__default__"] = zero_default      # applied to all pathways at build time
    return options, zeros, extras


def _replace(options: PESOptions, **kwargs) -> PESOptions:
    return PESOptions(
        units=kwargs.get("units", options.units),
        decimals=kwargs.get("decimals", options.decimals),
        gconf=kwargs.get("gconf", options.gconf),
        QH=kwargs.get("QH", options.QH),
        spc_used=kwargs.get("spc_used", options.spc_used),
        lowest_only=kwargs.get("lowest_only", options.lowest_only),
    )


def parse_legacy(text: str) -> PESSpec:
    """Parse the legacy line-based PES format into a `PESSpec`."""
    sections = _split_sections(text)
    if "PES" not in sections:
        raise ValueError("legacy PES file is missing the `--- # PES` section")
    if "SPECIES" not in sections:
        raise ValueError("legacy PES file is missing the `--- # SPECIES` section")

    pathways = _parse_pathways(sections["PES"])
    species = _parse_species(sections["SPECIES"])

    options = PESOptions()
    zero_default: Dict[str, str] = {}
    extras: Dict[str, str] = {}
    if "FORMAT" in sections:
        options, raw_zeros, extras = _parse_format(sections["FORMAT"])
        if "__default__" in raw_zeros:
            zero_default = {p: raw_zeros["__default__"] for p in pathways}

    return PESSpec(
        pathways=pathways,
        species=species,
        zero=zero_default,
        options=options,
        format_extras=extras,
    )
