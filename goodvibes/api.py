"""Programmatic API façade for GoodVibes (v4.2 item 5).

A small import-friendly entry point for notebooks and scripts. Replaces
the 15-positional-arg `calc_bbe` constructor with kwargs that have the
same defaults as the CLI, and returns a structured `ThermoResult` rather
than the raw `calc_bbe` instance.

    from goodvibes import compute_thermo, compute_batch
    r = compute_thermo("file.log", QH=True, spc="TZ")
    print(r.qh_gibbs_free_energy)

    rs = compute_batch(glob.glob("*.log"))

Internally just calls `calc_bbe`; no behavior change relative to the CLI.
The underlying `calc_bbe` and `QCData` instances stay accessible via
`result.bbe` / `result.qcdata` for advanced use (e.g. PES analysis,
direct attribute reads not yet promoted to the result dataclass).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from .constants import ATMOS, GAS_CONSTANT
from .io import parse_qcdata, read_initial
from .thermo import ThermoOptions, calc_bbe
from .utils import display_name
from .vib_scale_factors import canonicalize_level, scaling_data_dict


@dataclass(frozen=True)
class ThermoResult:
    """Bundle of thermochemistry for one structure.

    All numeric fields are in atomic units (Hartree for energies,
    Hartree/K for entropies); frequencies are cm⁻¹. Fields that aren't
    available for the given file (e.g. `qh_enthalpy` when `QH=False`,
    `sp_energy` when `spc=None`) are reported as `None` rather than
    a sentinel.
    """
    file: str                                   # absolute or as-passed path
    name: str                                   # display name (basename sans ext)

    # Energies (Hartree)
    scf_energy: Optional[float]
    sp_energy: Optional[float]
    zpe: Optional[float]
    enthalpy: Optional[float]
    qh_enthalpy: Optional[float]

    # Entropies (Hartree/K)
    entropy: Optional[float]
    qh_entropy: Optional[float]

    # Gibbs energies (Hartree)
    gibbs_free_energy: Optional[float]
    qh_gibbs_free_energy: Optional[float]

    # Frequencies (cm⁻¹)
    frequency_wn: Optional[List[float]]
    im_frequency_wn: Optional[List[float]]
    inverted_freqs: Optional[List[float]]

    # Metadata pulled from the source QCData
    point_group: Optional[str]
    symmno: Optional[int]
    linear_mol: bool
    multiplicity: Optional[int]
    job_type: Optional[str]
    level_of_theory: Optional[str]
    program: Optional[str]

    # Original objects for advanced use (PES analysis, attribute reads
    # that haven't been promoted to the result surface yet).
    bbe: Any
    qcdata: Any

    @property
    def has_thermo(self) -> bool:
        """True when calc_bbe found enough information to compute G(T).
        False for SP-only outputs (no frequency block)."""
        return self.gibbs_free_energy is not None


def compute_thermo(
    path: Optional[str] = None,
    *,
    qcdata: Any = None,
    QS: str = "grimme",
    QH: bool = False,
    s_freq_cutoff: float = 100.0,
    h_freq_cutoff: float = 100.0,
    temperature: float = 298.15,
    concentration: Optional[float] = None,
    freq_scale_factor: Optional[float] = None,
    zpe_scale_factor: Optional[float] = None,
    solv: Optional[str] = None,
    spc: Optional[str] = None,
    invert: Optional[float] = None,
    symm: bool = False,
    mm_freq_scale_factor: Optional[float] = None,
    inertia: str = "global",
) -> ThermoResult:
    """Compute thermochemistry for one QC output file.

    Pass *either* `path` (a filesystem location) or a pre-parsed
    `qcdata` (skips re-parsing). When `concentration` is None the
    gas-phase reference at 1 atm is used (`P / RT`).

    Parameters mirror the CLI:
        QS                 'grimme' (default) or 'truhlar' quasi-harmonic entropy
        QH                 apply Head-Gordon quasi-harmonic enthalpy correction
        s_freq_cutoff      entropy cutoff (cm⁻¹)
        h_freq_cutoff      enthalpy cutoff (cm⁻¹) — only used when QH=True
        temperature        K
        concentration      mol/L. None → gas phase 1 atm.
        freq_scale_factor  None → auto-lookup harm_fac from level of theory.
                           Applied to the partition-function frequencies
                           (used in H_vib and S_vib).
        zpe_scale_factor   None → auto-lookup zpe_fac from level of theory.
                           Applied to ZPE only. If `freq_scale_factor` is
                           explicitly set but `zpe_scale_factor` is None,
                           ZPE inherits `freq_scale_factor` (back-compat).
        solv               'none', or solvent name for free-space correction
        spc                None, 'link', or filename suffix for SPC files
        invert             None, or threshold for converting small imag → real
        symm               apply pymsym symmetry-number correction
    """
    if path is None and qcdata is None:
        raise ValueError("compute_thermo requires either `path` or `qcdata`")
    if concentration is None:
        concentration = ATMOS / (GAS_CONSTANT * temperature)

    options = ThermoOptions(
        QS=QS, QH=QH,
        s_freq_cutoff=s_freq_cutoff, h_freq_cutoff=h_freq_cutoff,
        temperature=temperature, concentration=concentration,
        freq_scale_factor=freq_scale_factor,
        zpe_scale_factor=zpe_scale_factor,
        solv=solv, spc=spc, invert=invert,
        symm=symm, mm_freq_scale_factor=mm_freq_scale_factor,
        inertia=inertia,
    )
    # from_options does the Truhlar-DB auto-lookup when freq/zpe scale
    # factors are None; we just need the level-of-theory string for the
    # result's `level_of_theory` field.
    lot = None
    if path is not None:
        try:
            scanned = read_initial(path)[0]
            if scanned and scanned != "none":
                lot = scanned
        except (IOError, OSError):
            pass
    bbe = calc_bbe.from_options(qcdata if qcdata is not None else path, options)
    return bbe_to_result(bbe, path, level_of_theory=lot)


def compute_batch(
    paths: Sequence[str],
    *,
    jobs: int = 1,
    **kwargs: Any,
) -> List[ThermoResult]:
    """Compute thermochemistry for a list of files.

    Parameters:
        paths: list of QC output file paths.
        jobs: parallelism level. ``1`` (default) is sequential — no
            process-pool overhead. ``> 1`` spawns that many worker
            processes via ``concurrent.futures.ProcessPoolExecutor``.
            ``0`` or negative uses ``os.cpu_count()`` (or 1 if unknown).
        **kwargs: forwarded unchanged to every ``compute_thermo`` call.

    Returns:
        Results in input order (matches ``executor.map``'s contract).
        Workers can't write to the orchestrator's logger, so per-file
        ``log.info`` from inside ``calc_bbe`` is silenced under
        ``jobs > 1``; warnings that surface through ``ThermoResult``
        (e.g. missing-frequency files) still come through unchanged.
    """
    if not paths:
        return []
    if jobs <= 0:
        import os
        jobs = os.cpu_count() or 1
    if jobs == 1 or len(paths) == 1:
        return [compute_thermo(p, **kwargs) for p in paths]

    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    fn = partial(compute_thermo, **kwargs)
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(fn, paths))


def to_dataframe(results: Sequence[ThermoResult]):
    """Convert a list of ThermoResults into a pandas DataFrame.

    Pandas is an optional dependency; raises ImportError with an install
    hint if it isn't available. The DataFrame has one row per result
    and columns for every public scalar field on `ThermoResult`
    (omits the `bbe` and `qcdata` references).
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "to_dataframe requires pandas; install with `pip install pandas`."
        ) from exc
    skip = {"bbe", "qcdata", "frequency_wn", "im_frequency_wn", "inverted_freqs"}
    rows = []
    for r in results:
        rows.append({
            f.name: getattr(r, f.name)
            for f in r.__dataclass_fields__.values()
            if f.name not in skip
        })
    return pd.DataFrame(rows)


def to_parquet(results: Sequence[ThermoResult], path: str) -> None:
    """Write a list of ThermoResults to a Parquet file at `path`.

    Same column set as `to_dataframe`. Requires pandas + a Parquet
    engine (pyarrow or fastparquet); install with
    `pip install goodvibes[full]` or `pip install pyarrow`.
    """
    df = to_dataframe(results)
    try:
        df.to_parquet(path, index=False)
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "to_parquet requires a Parquet engine. Install pyarrow "
            "(`pip install pyarrow`) or fastparquet, or use "
            "`pip install goodvibes[full]`."
        ) from exc


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def bbe_to_result(
    bbe: Any,
    path: Optional[str] = None,
    *,
    level_of_theory: Optional[str] = None,
) -> ThermoResult:
    """Project a `calc_bbe` instance into a `ThermoResult`.

    Useful for adapting CLI internals (which keep `thermo_data` as a
    `{path: calc_bbe}` dict) to the structured API without re-parsing.
    `level_of_theory` is read from the file via `read_initial()` if not
    supplied; pass it explicitly to avoid the extra file scan.
    """
    qc = getattr(bbe, "xyz", None)
    file = path if path is not None else getattr(qc, "file", "<unknown>")
    # Read level_of_theory from the file if it wasn't supplied (lookup
    # was already done at compute_thermo entry when freq_scale_factor
    # was None; for the explicit-factor path we still want the field
    # populated on the result for display).
    if level_of_theory is None and path is not None:
        try:
            lot = read_initial(path)[0]
            if lot and lot != "none":
                level_of_theory = lot
        except (IOError, OSError):
            pass

    sp = getattr(bbe, "sp_energy", None)
    if sp == "!":
        sp = None

    # calc_bbe leaves qh_enthalpy at 0.0 when CLI --QH is False; surface
    # that as None (the user can still read .bbe.qh_enthalpy directly).
    qh_h = getattr(bbe, "qh_enthalpy", None)
    if qh_h == 0.0:
        qh_h = None

    return ThermoResult(
        file=file,
        name=display_name(file),
        scf_energy=getattr(bbe, "scf_energy", None),
        sp_energy=sp,
        zpe=getattr(bbe, "zpe", None),
        enthalpy=getattr(bbe, "enthalpy", None),
        qh_enthalpy=qh_h,
        entropy=getattr(bbe, "entropy", None),
        qh_entropy=getattr(bbe, "qh_entropy", None),
        gibbs_free_energy=getattr(bbe, "gibbs_free_energy", None),
        qh_gibbs_free_energy=getattr(bbe, "qh_gibbs_free_energy", None),
        frequency_wn=getattr(bbe, "frequency_wn", None) or None,
        im_frequency_wn=getattr(bbe, "im_frequency_wn", None) or None,
        inverted_freqs=getattr(bbe, "inverted_freqs", None) or None,
        point_group=getattr(qc, "point_group", None) if qc else None,
        symmno=getattr(qc, "symmno", None) if qc else None,
        linear_mol=bool(getattr(qc, "linear_mol", False)) if qc else False,
        multiplicity=getattr(qc, "multiplicity", None) if qc else None,
        job_type=getattr(qc, "job_type", None) if qc else None,
        level_of_theory=level_of_theory,
        program=getattr(qc, "program", None) if qc else None,
        bbe=bbe,
        qcdata=qc,
    )
