"""Registry of the thermochemical quantities GoodVibes can tabulate or plot.

One table drives the PES Rich table, the JSON ``pes`` block, ``plot_pes`` and
the CLI, so a quantity is spelled, labelled and converted the same way
everywhere. Values are always relative (point minus reference) in Hartree
until a consumer applies :func:`goodvibes.constants.hartree_factor`.

Public API:
    QUANTITIES        -- mapping id -> :class:`Quantity`, in display order
    resolve_quantity  -- accept an id or one of its aliases, return the Quantity
    quantity_ids      -- the canonical ids, in display order
"""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Quantity:
    id: str                 # canonical registry id
    label: str              # display label for the relative value, e.g. "ΔE"
    json_key: str           # key used in the JSON ``pes.relative`` block
    field: Optional[str]    # ThermoVector attribute, or None for a derived quantity
    scale_by_T: bool = False  # entropies: report T·ΔS rather than ΔS
    description: str = ""

    def spc_label(self, spc_used: bool) -> str:
        """Header used in the PES table: H and G are SPC-substituted when
        --spc is set, so their labels carry an _SPC suffix."""
        if spc_used and self.id in ("enthalpy", "qh_enthalpy", "gibbs", "qh_gibbs"):
            return self.label + "_SPC"
        return self.label


_ORDER: Tuple[Quantity, ...] = (
    Quantity("spc", "ΔE_SPC", "spc", "sp_energy", description="single-point electronic energy (None unless --spc)"),
    Quantity("electronic", "ΔE", "scf", "scf_energy", description="electronic energy at the frequency level"),
    Quantity("zpe", "ΔZPE", "zpe", "zpe", description="zero-point vibrational energy"),
    Quantity("e_zpe", "ΔE+ZPE", "e_zpe", None, description="electronic energy (SPC when applied) plus ZPE"),
    Quantity("enthalpy", "ΔH", "h", "enthalpy", description="enthalpy (SPC-substituted when --spc)"),
    Quantity("qh_enthalpy", "Δqh-H", "qh_h", "qh_enthalpy", description="quasi-harmonic enthalpy (Head-Gordon)"),
    Quantity("entropy", "T·ΔS", "ts", "entropy", scale_by_T=True, description="T times the RRHO entropy"),
    Quantity("qh_entropy", "T·Δqh-S", "qh_ts", "qh_entropy", scale_by_T=True, description="T times the quasi-harmonic entropy"),
    Quantity("gibbs", "ΔG(T)", "g", "gibbs", description="Gibbs energy (SPC-substituted when --spc)"),
    Quantity("qh_gibbs", "Δqh-G(T)", "qh_g", "qh_gibbs", description="quasi-harmonic Gibbs energy"),
)
QUANTITIES = {q.id: q for q in _ORDER}

_ALIASES = {
    # electronic energy
    "scf": "electronic", "e": "electronic", "energy": "electronic", "scf_energy": "electronic",
    # E + ZPE
    "e0": "e_zpe", "e+zpe": "e_zpe", "ezpe": "e_zpe",
    # enthalpy
    "h": "enthalpy", "qh_h": "qh_enthalpy", "qh-h": "qh_enthalpy", "qhh": "qh_enthalpy",
    # entropy
    "s": "entropy", "ts": "entropy", "t.s": "entropy", "qh_s": "qh_entropy", "qh-s": "qh_entropy",
    "qh_ts": "qh_entropy", "t.qh-s": "qh_entropy",
    # Gibbs
    "g": "gibbs", "g(t)": "gibbs", "gibbs_free_energy": "gibbs",
    "qh_g": "qh_gibbs", "qh-g": "qh_gibbs", "qhg": "qh_gibbs", "qh-g(t)": "qh_gibbs",
    "qh_gibbs_free_energy": "qh_gibbs",
    # single point
    "sp": "spc", "sp_energy": "spc", "e_spc": "spc",
}


def quantity_ids():
    """Canonical ids in display order."""
    return [q.id for q in _ORDER]


def resolve_quantity(name) -> Quantity:
    """Return the Quantity for an id or alias (case-insensitive), or raise ValueError."""
    if isinstance(name, Quantity):
        return name
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    try:
        return QUANTITIES[key]
    except KeyError:
        raise ValueError(
            f"unknown quantity {name!r}; expected one of {', '.join(quantity_ids())}"
        ) from None
