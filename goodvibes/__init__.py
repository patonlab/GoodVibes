"""GoodVibes: quasi-harmonic thermochemistry from QC outputs.

Programmatic API (v4.2):

    from goodvibes import compute_thermo, compute_batch, ThermoResult
    r = compute_thermo("file.log", QH=True, spc="TZ")
    print(r.qh_gibbs_free_energy)

The `calc_bbe` class remains the canonical engine; `compute_thermo` is
just a kwargs façade that returns a structured `ThermoResult`.
"""
from .api import (
    ThermoResult,
    bbe_to_result,
    compute_batch,
    compute_thermo,
    to_dataframe,
)
from .constants import __version__
from .thermo import ThermoOptions

__all__ = [
    "ThermoResult",
    "ThermoOptions",
    "bbe_to_result",
    "compute_thermo",
    "compute_batch",
    "to_dataframe",
    "__version__",
]
