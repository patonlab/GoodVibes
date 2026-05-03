"""Parser for the new (true YAML) PES format.

Schema:

    pathways:
      Reaction: ["Int-I + TolS + TolSH", "Int-II + TolSH", "Int-III"]

    species:
      Int-I:    {files: "Int-I_*.log"}
      TolS:     {files: ["TolS.log"]}
      Int-II:   {files: "Int-II_*.log"}

    zero:
      Reaction: "Int-I + TolS + TolSH"          # optional; defaults to points[0]

    format:
      units: kcal/mol
      decimals: 1

`pathways` is a dict of pathway-name → ordered list of point label strings.
`species` is a dict of species-name → {files: <glob string OR list>} (the
dict shape leaves room for per-species options later — scaling factors,
symmetry numbers — without breaking the schema). `zero` is optional and
applies per pathway. `format` parses into `PESOptions`.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Union

from .pes_legacy import _replace
from .pes_loader import PESSpec
from .pes_model import PESOptions


def parse_yaml(text: str) -> PESSpec:
    """Parse the YAML text into a `PESSpec`. PyYAML is required."""
    try:
        import yaml
    except ImportError:
        raise RuntimeError(
            "PyYAML is required to read modern PES YAML files. Install with "
            "`pip install pyyaml`, or use the legacy `--- # PES` format."
        )
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("PES YAML root must be a mapping")

    if "pathways" not in data:
        raise ValueError("PES YAML must define a top-level `pathways:` block")
    if "species" not in data:
        raise ValueError("PES YAML must define a top-level `species:` block")

    pathways = _parse_pathways(data["pathways"])
    species = _parse_species(data["species"])
    zero = _parse_zero(data.get("zero", {}), pathways)
    options, extras = _parse_format(data.get("format", {}))

    return PESSpec(
        pathways=pathways,
        species=species,
        zero=zero,
        options=options,
        format_extras=extras,
    )


def _parse_pathways(raw) -> Dict[str, List[str]]:
    if not isinstance(raw, dict):
        raise ValueError(f"`pathways` must be a mapping, got {type(raw).__name__}")
    pathways: Dict[str, List[str]] = {}
    for name, points in raw.items():
        if not isinstance(points, list):
            raise ValueError(
                f"pathway {name!r}: points must be a list of strings, "
                f"got {type(points).__name__}"
            )
        cleaned = [str(p).strip() for p in points if str(p).strip()]
        if not cleaned:
            raise ValueError(f"pathway {name!r} has no points")
        pathways[str(name)] = cleaned
    if not pathways:
        raise ValueError("`pathways` is empty")
    return pathways


def _parse_species(raw) -> Dict[str, Union[str, List[str]]]:
    if not isinstance(raw, dict):
        raise ValueError(f"`species` must be a mapping, got {type(raw).__name__}")
    species: Dict[str, Union[str, List[str]]] = {}
    for name, entry in raw.items():
        files = _extract_files(name, entry)
        species[str(name)] = files
    if not species:
        raise ValueError("`species` is empty")
    return species


#: Internal prefix used to encode a "match this directory" pattern in the
#: same `Union[str, List[str]]` value space as file globs. Users never see
#: it — they write `{dir: "X"}` in YAML and the loader translates.
_DIR_PREFIX = "@dir:"


def _extract_files(name: str, entry) -> Union[str, List[str]]:
    """A species entry can be:
        {files: "glob"}                      # match by filename stem (default)
        {files: ["a.log", "b.log", ...]}
        {dir: "subdir"}                      # match files whose parent dir basename matches
        {dirs: ["sub_a", "sub_b", ...]}      # multiple directories
        {files: [...], dir: "X"}             # combined: union of both rules
        "glob"                                # shorthand: bare string (file pattern)
        ["a.log", ...]                       # shorthand: bare list (file patterns)

    Directory entries can themselves be fnmatch globs (e.g. `dir: "TS_*"`).
    A trailing "/*" or "/**" on a directory name is dropped automatically
    so users can write either `dir: "X"` or `dir: "X/*"`.
    """
    if isinstance(entry, str):
        return entry
    if isinstance(entry, list):
        return [str(x) for x in entry]
    if isinstance(entry, dict):
        if "files" not in entry and "dir" not in entry and "dirs" not in entry:
            raise ValueError(
                f"species {name!r}: needs at least one of `files:`, `dir:`, "
                "or `dirs:`"
            )
        out: List[str] = []
        if "files" in entry:
            files = entry["files"]
            if isinstance(files, str):
                out.append(files)
            elif isinstance(files, list):
                out.extend(str(x) for x in files)
            else:
                raise ValueError(
                    f"species {name!r}: `files:` must be a string or list, "
                    f"got {type(files).__name__}"
                )
        if "dir" in entry:
            d = entry["dir"]
            if not isinstance(d, str):
                raise ValueError(
                    f"species {name!r}: `dir:` must be a string, "
                    f"got {type(d).__name__}"
                )
            out.append(_DIR_PREFIX + _normalize_dir(d))
        if "dirs" in entry:
            dirs = entry["dirs"]
            if not isinstance(dirs, list):
                raise ValueError(
                    f"species {name!r}: `dirs:` must be a list of strings, "
                    f"got {type(dirs).__name__}"
                )
            for d in dirs:
                if not isinstance(d, str):
                    raise ValueError(
                        f"species {name!r}: `dirs:` entries must be strings, "
                        f"got {type(d).__name__}"
                    )
                out.append(_DIR_PREFIX + _normalize_dir(d))
        # Single string vs list — preserve old single-string shape for
        # back-compat with external code that introspects PESSpec.species.
        return out[0] if len(out) == 1 else out
    raise ValueError(
        f"species {name!r}: entry must be a string, list, or mapping, "
        f"got {type(entry).__name__}"
    )


def _normalize_dir(d: str) -> str:
    """Strip trailing slashes / globs so `dir: "X/"` / `dir: "X/*"` /
    `dir: "X/**"` all normalize to `"X"`."""
    d = d.strip().rstrip("/")
    for tail in ("/**", "/*"):
        if d.endswith(tail):
            d = d[: -len(tail)]
    return d


def _parse_zero(raw, pathways: Dict[str, List[str]]) -> Dict[str, str]:
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"`zero` must be a mapping, got {type(raw).__name__}")
    zeros: Dict[str, str] = {}
    for path_name, label in raw.items():
        if path_name not in pathways:
            raise ValueError(
                f"`zero.{path_name}` references unknown pathway "
                f"(known: {sorted(pathways)})"
            )
        zeros[str(path_name)] = str(label).strip()
    return zeros


def _parse_format(raw) -> Tuple[PESOptions, Dict[str, str]]:
    if not raw:
        return PESOptions(), {}
    if not isinstance(raw, dict):
        raise ValueError(f"`format` must be a mapping, got {type(raw).__name__}")
    options = PESOptions()
    extras: Dict[str, str] = {}
    for key, value in raw.items():
        key_lower = str(key).lower()
        if key_lower == "units":
            if value not in ("kcal/mol", "kJ/mol"):
                raise ValueError(
                    f"format.units: expected 'kcal/mol' or 'kJ/mol', got {value!r}"
                )
            options = _replace(options, units=value)
        elif key_lower in ("decimals", "dec"):
            try:
                options = _replace(options, decimals=int(value))
            except (ValueError, TypeError):
                raise ValueError(f"format.decimals: expected integer, got {value!r}")
        else:
            extras[str(key)] = str(value)
    return options, extras
