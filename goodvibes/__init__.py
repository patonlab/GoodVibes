"""GoodVibes: quasi-harmonic thermochemistry and reaction profiles from QC
and MLIP outputs.

Programmatic API:

    from goodvibes import compute_thermo, compute_batch, ThermoResult
    r = compute_thermo("file.log", QH=True, spc="TZ")
    print(r.qh_gibbs_free_energy)

    # file-free (ASE / MLIP)
    from goodvibes import QCData
    qc = QCData.from_vibrations(atoms, vib.get_vibrations(), atoms.get_potential_energy())
    r = compute_thermo(qcdata=qc)

    # reaction profiles
    from goodvibes import load_pes, plot_profile
    pes = load_pes("profile.yaml", {r.file: r.bbe for r in results})
    plot_profile(pes, temperatures=[298.15, 373.15]).save("profile.svg")

    # reaction-profile documents (reaction-profile/1.0)
    from goodvibes import load_profile
    doc = load_profile("profile.yaml").evaluate(results, with_conformers=True)
    doc.dump("profile.json"); doc.plot().save("profile.svg")

The `calc_bbe` class remains the canonical engine; `compute_thermo` is
just a kwargs façade that returns a structured `ThermoResult`.
"""
from .api import (
    ThermoResult,
    bbe_to_result,
    compute_batch,
    compute_thermo,
    to_dataframe,
    to_parquet,
)
from .constants import __version__
from .io import QCData
from .pes_loader import build_pes_result, load_pes
from .pes_model import (
    ComputedEntry,
    ConformerSet,
    Edge,
    PESOptions,
    PESResult,
    Pathway,
    Point,
    Series,
    ThermoVector,
    merge_point_order,
)
from .plot import ProfileAxes, plot_pes, plot_profile
from .profile import Profile, ProfileError, ProfileWarning, load_profile, validate_document
from .quantities import QUANTITIES, resolve_quantity
from .selectivity import SelectivityResult, compute_selectivity
from .thermo import MissingSinglePointError, ThermoOptions, calc_bbe

__all__ = [
    # thermochemistry
    "ThermoResult",
    "ThermoOptions",
    "MissingSinglePointError",
    "QCData",
    "calc_bbe",
    "bbe_to_result",
    "compute_thermo",
    "compute_batch",
    "to_dataframe",
    "to_parquet",
    # quantities
    "QUANTITIES",
    "resolve_quantity",
    # reaction profiles
    "ThermoVector",
    "ComputedEntry",
    "ConformerSet",
    "Point",
    "Edge",
    "Pathway",
    "Series",
    "PESOptions",
    "PESResult",
    "merge_point_order",
    "load_pes",
    "build_pes_result",
    "plot_profile",
    "plot_pes",
    "ProfileAxes",
    # reaction-profile documents
    "Profile",
    "ProfileError",
    "ProfileWarning",
    "load_profile",
    "validate_document",
    # selectivity
    "SelectivityResult",
    "compute_selectivity",
    "__version__",
]
