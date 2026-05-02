"""Stable schema definitions for GoodVibes structured output.

The shape of the JSON written by `--json` / `goodvibes.api.write_json_results`
is now documented and stable. This module is the single source of truth
for that schema:

  - `SCHEMA_VERSION` — the current shipping version.
  - `SCHEMA_MIN_COMPATIBLE` — readers tolerate any version ≥ this.
  - `validate(payload)` — runtime sanity check; raises `ValueError`
     with a clear message on shape/version mismatches.
  - TypedDicts (`Payload`, `ResultEntry`, `ThermoBlock`, ...) — give
     static checkers and editors completion when working with parsed
     payloads.

The JSON file is also accepted by `--import` as a cache (skips
re-parsing the underlying QC outputs); the same schema therefore plays
two roles, which is item 10 of ROADMAP.md.

## Version policy (v1.0+)

  Major bump (X.0)        breaking change — fields removed or renamed,
                          or value types changed.
  Minor bump (1.X)        backwards-compatible addition — new fields,
                          new optional blocks, new enum values.
  Patch bump (1.0.X)      no-op (we don't bump for typo fixes etc.).

Readers SHOULD accept any minor version ≥ their min and ignore unknown
fields. Writers SHOULD emit the version they were built against.
"""
from __future__ import annotations

from typing import Any, List, Optional

try:
    from typing import TypedDict
except ImportError:                                # pragma: no cover  (py<3.8)
    from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------

#: The schema version this build of GoodVibes writes.
SCHEMA_VERSION = "1.0"

#: The oldest schema a reader in this build will accept. Bump only on
#: a true breaking change (e.g. v2.0).
SCHEMA_MIN_COMPATIBLE = "1.0"


# ---------------------------------------------------------------------------
# TypedDicts (for static analysis / editor support)
# ---------------------------------------------------------------------------

class ThermoBlock(TypedDict, total=False):
    """Per-file thermochemical numbers in atomic units (Hartree, Hartree/K).

    Entries are `Optional[float]` — `None` when the QC output didn't
    expose that quantity (e.g. `sp_energy` is `None` unless `--spc` was
    used, frequencies are absent for SP-only outputs).
    """
    scf_energy: Optional[float]
    sp_energy: Optional[float]
    zpe: Optional[float]
    zero_point_corr: Optional[float]
    enthalpy: Optional[float]
    qh_enthalpy: Optional[float]
    entropy: Optional[float]
    qh_entropy: Optional[float]
    gibbs_free_energy: Optional[float]
    qh_gibbs_free_energy: Optional[float]
    frequency_wn: Optional[List[float]]
    im_frequency_wn: Optional[List[float]]
    inverted_freqs: Optional[List[float]]
    point_group: Optional[str]
    symmno: Optional[int]
    linear_mol: Optional[bool]
    multiplicity: Optional[int]
    job_type: Optional[str]
    applied_freq_scale_factor: Optional[float]


class ResultEntry(TypedDict, total=False):
    """One per input QC file."""
    file: str                        # absolute path
    name: str                        # display name (basename sans ext)
    qcdata: Optional[dict]           # parsed QCData; reconstructable via io.dict_to_qcdata
    thermo: ThermoBlock              # per-file thermochemistry
    media_conc: Optional[float]      # neat-solvent concentration (mol/L) when --media matched
    boltzmann_factor: Optional[float]


class SelectivityResult(TypedDict, total=False):
    """One per temperature inside `selectivity`/`selectivity_lowest`."""
    temperature: float
    populations: dict                # {label: probability}, sums to 1.0
    raw_boltzmann: dict              # {label: un-normalised exp(-G/RT) sum}
    preferred: str                   # max-population label
    ee: Optional[float]              # 2-label only; in %
    ddG: Optional[float]             # 2-label only; in Hartree
    files_per_label: dict            # {label: list[file path]}


class SelectivityBlock(TypedDict):
    labels: List[str]                # ordered species names
    key: str                         # 'gibbs' | 'energy'
    results: List[SelectivityResult]


class PESPoint(TypedDict, total=False):
    label: str                       # original label, e.g. "2*A + B"
    species: List[dict]              # [{coefficient, name, files}]
    relative: dict                   # {scf, zpe, h, qh_h, ts, qh_ts, g, qh_g, spc}


class PESPathway(TypedDict, total=False):
    name: str
    temperature: float
    units: str                       # 'kcal/mol' or 'kJ/mol'
    zero_label: str
    points: List[PESPoint]


class PESBlock(TypedDict):
    pathways: List[PESPathway]


class Payload(TypedDict, total=False):
    """The top-level JSON document GoodVibes writes."""
    schema_version: str
    goodvibes_version: str
    generated_at: str                # ISO 8601 UTC
    options: dict
    results: List[ResultEntry]
    selectivity: SelectivityBlock          # optional
    selectivity_lowest: SelectivityBlock   # optional
    pes: PESBlock                          # optional


# ---------------------------------------------------------------------------
# Runtime validation
# ---------------------------------------------------------------------------

def _parse_version(s: str) -> tuple:
    """Parse '1.0' → (1, 0); '1.2.3' → (1, 2, 3). Tolerant of trailing junk."""
    parts = []
    for chunk in s.split('.'):
        try:
            parts.append(int(chunk))
        except ValueError:
            break
    return tuple(parts) if parts else (0,)


def validate(payload: Any) -> None:
    """Light runtime sanity check on a loaded payload.

    Verifies the document has a schema_version that this build can read
    and that the required top-level fields are present. Does NOT
    deeply type-check every field — that's the consumer's responsibility
    via the TypedDicts above.

    Raises:
        ValueError: with a one-line, actionable message on any failure.
    """
    if not isinstance(payload, dict):
        raise ValueError(
            f"GoodVibes payload must be a JSON object, got {type(payload).__name__}"
        )
    if 'schema_version' not in payload:
        raise ValueError(
            "GoodVibes payload missing required field 'schema_version'. "
            "Pre-v1.0 files (schema_version 0.x) are not accepted as input — "
            "regenerate with a v1.0+ build."
        )
    pv = _parse_version(payload['schema_version'])
    minv = _parse_version(SCHEMA_MIN_COMPATIBLE)
    cur = _parse_version(SCHEMA_VERSION)
    if pv < minv:
        raise ValueError(
            f"GoodVibes payload has schema_version={payload['schema_version']!r}; "
            f"this build accepts {SCHEMA_MIN_COMPATIBLE} or newer. "
            f"Regenerate the file with a recent GoodVibes."
        )
    if pv[0] > cur[0]:
        raise ValueError(
            f"GoodVibes payload has schema_version={payload['schema_version']!r}, "
            f"newer than this build understands ({SCHEMA_VERSION}). Upgrade GoodVibes."
        )
    for required in ('goodvibes_version', 'results'):
        if required not in payload:
            raise ValueError(
                f"GoodVibes payload missing required field {required!r}"
            )
    if not isinstance(payload['results'], list):
        raise ValueError(
            f"GoodVibes payload field 'results' must be a list, "
            f"got {type(payload['results']).__name__}"
        )
