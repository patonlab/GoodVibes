"""PES data model.

Pure data + arithmetic; no I/O, no parsing, no global state. The model
is what the legacy parser and the new YAML parser produce, and what
the output layer (Rich tables, JSON) consumes.

Three layers:
    ConformerSet   one species, ≥1 conformers (calc_bbe instances)
    Point          stoichiometric sum of species at one node of a path
    Pathway        ordered list of points + a designated zero
    PESResult      container of pathways + options + temperatures

Arithmetic on ThermoVector collapses the legacy nine-parallel-list
pattern into one expression: rel = point.thermo(T) - zero.thermo(T).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

GAS_CONSTANT = 8.3144621            # J / K / mol
J_TO_AU = 4.184 * 627.509541 * 1000.0
KCAL_TO_AU = 627.509541


# ---------------------------------------------------------------------------
# ThermoVector
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThermoVector:
    """Bundle of thermo quantities for one species at one temperature.

    Energy fields are in Hartree; entropy fields are in Hartree/K (matching
    calc_bbe). sp_energy is None when --spc was not used; arithmetic
    propagates None (if either operand has sp_energy=None, the result does).
    """
    scf_energy: float
    zpe: float
    enthalpy: float
    qh_enthalpy: float
    entropy: float
    qh_entropy: float
    gibbs: float
    qh_gibbs: float
    sp_energy: Optional[float] = None

    def __add__(self, other: "ThermoVector") -> "ThermoVector":
        if not isinstance(other, ThermoVector):
            return NotImplemented
        return ThermoVector(
            scf_energy=self.scf_energy + other.scf_energy,
            zpe=self.zpe + other.zpe,
            enthalpy=self.enthalpy + other.enthalpy,
            qh_enthalpy=self.qh_enthalpy + other.qh_enthalpy,
            entropy=self.entropy + other.entropy,
            qh_entropy=self.qh_entropy + other.qh_entropy,
            gibbs=self.gibbs + other.gibbs,
            qh_gibbs=self.qh_gibbs + other.qh_gibbs,
            sp_energy=_add_opt(self.sp_energy, other.sp_energy),
        )

    def __sub__(self, other: "ThermoVector") -> "ThermoVector":
        if not isinstance(other, ThermoVector):
            return NotImplemented
        return ThermoVector(
            scf_energy=self.scf_energy - other.scf_energy,
            zpe=self.zpe - other.zpe,
            enthalpy=self.enthalpy - other.enthalpy,
            qh_enthalpy=self.qh_enthalpy - other.qh_enthalpy,
            entropy=self.entropy - other.entropy,
            qh_entropy=self.qh_entropy - other.qh_entropy,
            gibbs=self.gibbs - other.gibbs,
            qh_gibbs=self.qh_gibbs - other.qh_gibbs,
            sp_energy=_sub_opt(self.sp_energy, other.sp_energy),
        )

    def __mul__(self, k) -> "ThermoVector":
        if not isinstance(k, (int, float)):
            return NotImplemented
        return ThermoVector(
            scf_energy=self.scf_energy * k,
            zpe=self.zpe * k,
            enthalpy=self.enthalpy * k,
            qh_enthalpy=self.qh_enthalpy * k,
            entropy=self.entropy * k,
            qh_entropy=self.qh_entropy * k,
            gibbs=self.gibbs * k,
            qh_gibbs=self.qh_gibbs * k,
            sp_energy=self.sp_energy * k if self.sp_energy is not None else None,
        )

    __rmul__ = __mul__

    @classmethod
    def zero(cls, with_sp: bool = False) -> "ThermoVector":
        """Identity element for addition."""
        return cls(
            scf_energy=0.0, zpe=0.0, enthalpy=0.0, qh_enthalpy=0.0,
            entropy=0.0, qh_entropy=0.0, gibbs=0.0, qh_gibbs=0.0,
            sp_energy=0.0 if with_sp else None,
        )


def _add_opt(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a + b


def _sub_opt(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def _bbe_to_vector(bbe: Any) -> ThermoVector:
    """Project a calc_bbe instance into a ThermoVector.

    Treats sp_energy=='!' (the sentinel for "SPC requested but missing")
    as None — caller must decide whether that's an error.

    `calc_bbe` leaves `qh_enthalpy` at 0.0 unless the CLI `--QH` flag was
    set; in that case `qh_gibbs_free_energy` is computed from the plain
    `enthalpy`. Mirror that fallback here so arithmetic on the model
    produces correct relatives for both QH on and off runs.
    """
    sp = getattr(bbe, "sp_energy", None)
    if sp == "!":
        sp = None
    qh_h = bbe.qh_enthalpy if bbe.qh_enthalpy else bbe.enthalpy
    return ThermoVector(
        scf_energy=bbe.scf_energy,
        zpe=bbe.zpe,
        enthalpy=bbe.enthalpy,
        qh_enthalpy=qh_h,
        entropy=bbe.entropy,
        qh_entropy=bbe.qh_entropy,
        gibbs=bbe.gibbs_free_energy,
        qh_gibbs=bbe.qh_gibbs_free_energy,
        sp_energy=sp,
    )


# ---------------------------------------------------------------------------
# ConformerSet
# ---------------------------------------------------------------------------

@dataclass
class ConformerSet:
    """A named species + ≥1 calc_bbe entries.

    Encapsulates the rollup math: pure Boltzmann average, lowest-only,
    or gconf-corrected (lowest + Boltzmann adjustment + mixing entropy).
    """
    name: str
    files: List[str]
    bbes: List[Any]    # calc_bbe instances, parallel to `files`

    def __post_init__(self):
        if len(self.files) != len(self.bbes):
            raise ValueError(
                f"ConformerSet {self.name!r}: files and bbes length mismatch "
                f"({len(self.files)} vs {len(self.bbes)})"
            )
        if not self.bbes:
            raise ValueError(f"ConformerSet {self.name!r} has no conformers")

    @property
    def is_single(self) -> bool:
        return len(self.bbes) == 1

    def lowest_conformer(self) -> ThermoVector:
        """Vector for the conformer with the lowest qh_gibbs_free_energy."""
        idx = min(range(len(self.bbes)),
                  key=lambda i: self.bbes[i].qh_gibbs_free_energy)
        return _bbe_to_vector(self.bbes[idx])

    def boltzmann_weighted(self, T: float) -> ThermoVector:
        """Pure Boltzmann-weighted average over conformers, weights from qh-G."""
        if self.is_single:
            return _bbe_to_vector(self.bbes[0])
        probs = self._boltzmann_probs(T)
        result: Optional[ThermoVector] = None
        for p, bbe in zip(probs, self.bbes):
            term = _bbe_to_vector(bbe) * p
            result = term if result is None else result + term
        return result  # type: ignore[return-value]

    def gconf_corrected(self, T: float, QH: bool = True) -> ThermoVector:
        """Lowest conformer + Boltzmann adjustment + mixing entropy −R Σ pᵢ ln pᵢ.

        Mirrors the legacy `pes.get_pes` algorithm:
            H_tot   = H_min + (Σ pᵢ Hᵢ − H_min)
            S_tot   = S_min + (Σ pᵢ Sᵢ + Σ −R pᵢ ln pᵢ − S_min)
            G(T)    = H_tot − T·S_tot   (or qh-H/qh-S if QH)

        ZPE/SCF/SPC follow the plain Boltzmann sum (no mixing-entropy
        correction applies to those).
        """
        if self.is_single:
            return _bbe_to_vector(self.bbes[0])

        probs = self._boltzmann_probs(T)
        boltz = self.boltzmann_weighted(T)
        lowest = self.lowest_conformer()

        # Mixing entropy (Hartree / K). Skip terms where p == 0.
        mix_entropy = -GAS_CONSTANT / J_TO_AU * sum(
            p * math.log(p) for p in probs if p > 0.0
        )

        # Lowest + (Boltzmann avg − lowest) reproduces "lowest + adjustment"
        # in a way that is exact arithmetic on ThermoVectors.
        adjusted = lowest + (boltz - lowest)
        # adjusted == boltz exactly; the explicit form is here to mirror the
        # legacy code's intent. Now add mixing entropy and recompute G(T).
        s_total = adjusted.entropy + mix_entropy
        qs_total = adjusted.qh_entropy + mix_entropy
        h_for_g = adjusted.qh_enthalpy if QH else adjusted.enthalpy
        return ThermoVector(
            scf_energy=adjusted.scf_energy,
            zpe=adjusted.zpe,
            enthalpy=adjusted.enthalpy,
            qh_enthalpy=adjusted.qh_enthalpy,
            entropy=s_total,
            qh_entropy=qs_total,
            gibbs=adjusted.enthalpy - T * s_total,
            qh_gibbs=h_for_g - T * qs_total,
            sp_energy=adjusted.sp_energy,
        )

    def _boltzmann_probs(self, T: float) -> List[float]:
        g_min = min(b.qh_gibbs_free_energy for b in self.bbes)
        weights = [
            math.exp(-(b.qh_gibbs_free_energy - g_min) * J_TO_AU / GAS_CONSTANT / T)
            for b in self.bbes
        ]
        norm = sum(weights)
        return [w / norm for w in weights]


# ---------------------------------------------------------------------------
# Point — stoichiometric sum of species
# ---------------------------------------------------------------------------

# Matches "2*A", "2 * A", or "A". Coefficient is optional; default 1.
_TERM_RE = re.compile(r"""
    ^\s*
    (?:(\d+)\s*\*\s*)?       # optional integer coefficient
    (.+?)                    # species name (lazy, trims trailing ws)
    \s*$
""", re.VERBOSE)


def parse_point_label(label: str) -> List[Tuple[int, str]]:
    """Parse a point label like "2*A + B + C" into [(2,'A'), (1,'B'), (1,'C')].

    Whitespace-tolerant. Coefficients must be positive integers. Returns the
    terms in source order; duplicate species are allowed (they'll be summed).
    """
    if not label or not label.strip():
        raise ValueError("empty point label")
    terms: List[Tuple[int, str]] = []
    for chunk in label.split('+'):
        m = _TERM_RE.match(chunk)
        if m is None:
            raise ValueError(f"could not parse term {chunk!r} in point {label!r}")
        coeff_str, name = m.group(1), m.group(2).strip()
        if not name:
            raise ValueError(f"empty species name in point {label!r}")
        coeff = int(coeff_str) if coeff_str else 1
        if coeff < 1:
            raise ValueError(
                f"non-positive coefficient {coeff} in point {label!r}"
            )
        terms.append((coeff, name))
    return terms


@dataclass
class Point:
    """One node of a pathway: a stoichiometric sum of ConformerSets."""
    label: str                                       # original string, e.g. "2*A + B"
    species: List[Tuple[int, ConformerSet]]          # parsed: [(coeff, set), ...]

    def thermo(
        self,
        T: float,
        gconf: bool = True,
        QH: bool = True,
        lowest_only: bool = False,
    ) -> ThermoVector:
        """Total thermo at this point: Σ coeff_i × ConformerSet_i.thermo(T).

        Per-species rollup precedence:
            lowest_only=True  → the species' lowest qh-G conformer only
            else gconf=True   → lowest + Boltzmann adjustment + mixing entropy
            else              → pure Boltzmann-weighted average
        """
        total: Optional[ThermoVector] = None
        for coeff, cset in self.species:
            if lowest_only:
                term = cset.lowest_conformer() * coeff
            elif gconf:
                term = cset.gconf_corrected(T, QH=QH) * coeff
            else:
                term = cset.boltzmann_weighted(T) * coeff
            total = term if total is None else total + term
        if total is None:
            raise ValueError(f"Point {self.label!r} has no species")
        return total

    @classmethod
    def from_label(cls, label: str, species_map: dict) -> "Point":
        """Build a Point from a label string and {name: ConformerSet} map."""
        terms = parse_point_label(label)
        resolved: List[Tuple[int, ConformerSet]] = []
        missing = [name for _, name in terms if name not in species_map]
        if missing:
            raise KeyError(
                f"unknown species in point {label!r}: {missing} "
                f"(available: {sorted(species_map)})"
            )
        for coeff, name in terms:
            resolved.append((coeff, species_map[name]))
        return cls(label=label, species=resolved)


# ---------------------------------------------------------------------------
# Pathway / PESResult
# ---------------------------------------------------------------------------

@dataclass
class Pathway:
    """Ordered points + a designated zero (defaults to points[0])."""
    name: str
    points: List[Point]
    zero: Point = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        if not self.points:
            raise ValueError(f"Pathway {self.name!r} has no points")
        if self.zero is None:
            self.zero = self.points[0]

    def relative(
        self, T: float, gconf: bool = True, QH: bool = True, lowest_only: bool = False,
    ) -> List[ThermoVector]:
        """Per-point ΔThermo relative to self.zero, in Hartree."""
        kw = dict(gconf=gconf, QH=QH, lowest_only=lowest_only)
        zero = self.zero.thermo(T, **kw)
        return [p.thermo(T, **kw) - zero for p in self.points]


@dataclass
class PESOptions:
    units: str = "kcal/mol"          # 'kcal/mol' or 'kJ/mol'
    decimals: int = 2
    gconf: bool = True
    QH: bool = True
    spc_used: bool = False           # True when --spc was set
    lowest_only: bool = False        # True when --lowest-only overrides gconf/Boltzmann

    def to_user_units(self, hartree: Optional[float]) -> Optional[float]:
        if hartree is None:
            return None
        if self.units == "kJ/mol":
            return J_TO_AU / 1000.0 * hartree
        return KCAL_TO_AU * hartree   # default kcal/mol


@dataclass
class PESResult:
    """Top-level container: pathways + options + temperatures."""
    pathways: List[Pathway]
    options: PESOptions
    temperatures: List[float] = field(default_factory=lambda: [298.15])
