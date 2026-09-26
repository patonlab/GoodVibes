"""PES / reaction-profile data model.

Pure data + arithmetic; no I/O, no parsing, no global state. The model
is what the legacy parser and the YAML parser produce, and what the
output layer (Rich tables, JSON), ``plot_profile`` and the selectivity
code consume.

Layers:
    ThermoVector   the thermo quantities of one structure or sum at one T
    ComputedEntry  one parsed structure (QCData + ThermoOptions), evaluable
                   at any temperature (memoised)
    ConformerSet   one species, ≥1 conformers, with the ensemble rollups
    Point          stoichiometric sum of species at one node of a pathway,
                   with a role (reactant | minimum | ts | product) and a
                   display label
    Edge           a connection between two points (step | barrierless | none)
    Pathway        ordered points + edges + a designated zero
    Series         one line set on the axes / one column set in a table:
                   a quantity at a temperature, computed from the model or
                   declared (typed-in levels that are never re-evaluated)
    PESResult      pathways + options + temperatures + series

``Pathway.relative`` / ``Pathway.levels`` are the only places where
relative values are formed: every table, JSON block and figure reads
them, so they cannot disagree.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .constants import GAS_CONSTANT, J_TO_AU, KCAL_TO_AU, hartree_factor  # noqa: F401  (re-exported for callers)


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

    def get(self, quantity, T: Optional[float] = None) -> Optional[float]:
        """Value of a registry quantity in Hartree (see goodvibes.quantities).

        Entropy quantities are returned as T·S and therefore need ``T``.
        ``e_zpe`` is (sp_energy if present else scf_energy) + zpe, matching
        the table convention that H and G are SPC-substituted when --spc
        is used. Returns None where the underlying value is None (no SPC).
        """
        from .quantities import resolve_quantity
        q = resolve_quantity(quantity)
        if q.id == "e_zpe":
            base = self.sp_energy if self.sp_energy is not None else self.scf_energy
            return base + self.zpe
        value = getattr(self, q.field)
        if value is None:
            return None
        if q.scale_by_T:
            if T is None:
                raise ValueError(f"quantity {q.id!r} is T·S; a temperature is required")
            return T * value
        return value

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
    if sp == "!" or not isinstance(sp, (int, float)):
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
# ComputedEntry — one structure, evaluable at any temperature
# ---------------------------------------------------------------------------

@dataclass
class ComputedEntry:
    """One parsed structure plus the options it is evaluated with.

    ``thermo(T)`` returns the ThermoVector at temperature ``T`` and is
    memoised on the (temperature, options) pair, so a multi-temperature
    profile never re-parses a file and never evaluates the same
    temperature twice. When the options carry no explicit concentration
    the gas-phase standard state follows the temperature (P/RT), exactly
    as the CLI's ``--ti`` scan does.
    """
    qcdata: Any
    options: Any                       # goodvibes.thermo.ThermoOptions
    file: str = ""
    _cache: Dict[Any, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_bbe(cls, bbe: Any, file: Optional[str] = None) -> Optional["ComputedEntry"]:
        """Wrap a ``calc_bbe`` built by ``calc_bbe.from_options`` (or
        ``compute_thermo``). Returns None when the object does not carry
        the parsed input and its options (stubs, or the legacy
        15-argument constructor), in which case it can only be used at
        the temperature it was computed at.
        """
        qc = getattr(bbe, "qcdata", None)
        opts = getattr(bbe, "options", None)
        if qc is None or opts is None:
            return None
        entry = cls(qcdata=qc, options=opts, file=file or getattr(qc, "file", "") or "")
        entry._cache[opts] = bbe        # the base temperature is never recomputed
        return entry

    @classmethod
    def from_result(cls, result: Any) -> "ComputedEntry":
        """Wrap a ``ThermoResult`` from ``compute_thermo``."""
        entry = cls.from_bbe(result.bbe, file=result.file)
        if entry is None:
            raise ValueError(
                f"{result.name}: the result does not carry its QCData and ThermoOptions; "
                "build it with goodvibes.compute_thermo or calc_bbe.from_options")
        return entry

    @property
    def base_temperature(self) -> float:
        return self.options.temperature

    def bbe(self, T: Optional[float] = None, options: Any = None) -> Any:
        """The ``calc_bbe`` at temperature ``T`` (default: the base T)."""
        opts = options if options is not None else self.options
        if T is not None and T != opts.temperature:
            opts = replace(opts, temperature=T)
        hit = self._cache.get(opts)
        if hit is None:
            from .thermo import calc_bbe
            hit = calc_bbe.from_options(self.qcdata, opts)
            self._cache[opts] = hit
        return hit

    def thermo(self, T: Optional[float] = None, options: Any = None) -> ThermoVector:
        """ThermoVector at temperature ``T`` (default: the base T)."""
        return _bbe_to_vector(self.bbe(T, options))


# ---------------------------------------------------------------------------
# ConformerSet
# ---------------------------------------------------------------------------

@dataclass
class ConformerSet:
    """A named species + ≥1 conformers.

    ``bbes`` are the ``calc_bbe`` (or calc_bbe-shaped) objects at the base
    temperature, parallel to ``files``. When every one of them carries its
    parsed input and options (anything built through
    ``calc_bbe.from_options`` / ``compute_thermo``), ``entries`` is filled
    automatically and the set can be evaluated at any temperature;
    otherwise the rollups use the base-temperature values whatever ``T``
    is passed (the pre-4.6 behaviour).

    Encapsulates the rollup math: pure Boltzmann average, lowest-only,
    or gconf-corrected (lowest + Boltzmann adjustment + mixing entropy),
    weighting by ``weight_by`` (a registry quantity id; qh_gibbs by default).
    """
    name: str
    files: List[str]
    bbes: List[Any]    # calc_bbe instances, parallel to `files`
    entries: Optional[List[ComputedEntry]] = None
    weight_by: str = "qh_gibbs"

    def __post_init__(self):
        if len(self.files) != len(self.bbes):
            raise ValueError(
                f"ConformerSet {self.name!r}: files and bbes length mismatch "
                f"({len(self.files)} vs {len(self.bbes)})"
            )
        if not self.bbes:
            raise ValueError(f"ConformerSet {self.name!r} has no conformers")
        for f, b in zip(self.files, self.bbes):
            if not hasattr(b, "qh_gibbs_free_energy"):
                raise ValueError(
                    f"ConformerSet {self.name!r}: {f} has no thermochemistry (no frequencies in the "
                    "output; a single-point file cannot be a conformer)")
        if self.entries is None:
            built = [ComputedEntry.from_bbe(b, f) for f, b in zip(self.files, self.bbes)]
            self.entries = built if all(e is not None for e in built) else None  # type: ignore[assignment]
        elif len(self.entries) != len(self.bbes):
            raise ValueError(
                f"ConformerSet {self.name!r}: entries and bbes length mismatch "
                f"({len(self.entries)} vs {len(self.bbes)})"
            )
        from .quantities import resolve_quantity
        self.weight_by = resolve_quantity(self.weight_by).id

    # -- construction -------------------------------------------------------

    @classmethod
    def from_results(cls, name: str, results: Sequence[Any], weight_by: str = "qh_gibbs") -> "ConformerSet":
        """Build a species from ``ThermoResult`` objects (``compute_thermo``)."""
        results = list(results)
        if not results:
            raise ValueError(f"ConformerSet {name!r}: no results given")
        return cls(
            name=name,
            files=[r.file for r in results],
            bbes=[r.bbe for r in results],
            entries=[ComputedEntry.from_result(r) for r in results],
            weight_by=weight_by,
        )

    # -- basic queries ------------------------------------------------------

    @property
    def is_single(self) -> bool:
        return len(self.bbes) == 1

    @property
    def recomputable(self) -> bool:
        """True when the set can be evaluated at temperatures other than
        the one its conformers were computed at."""
        return self.entries is not None

    @property
    def base_temperature(self) -> Optional[float]:
        if self.entries:
            return self.entries[0].base_temperature
        return None

    def vectors(self, T: Optional[float] = None) -> List[ThermoVector]:
        """Per-conformer ThermoVectors at ``T`` (base temperature when
        None or when the set is not recomputable)."""
        if T is None or self.entries is None:
            return [_bbe_to_vector(b) for b in self.bbes]
        return [e.thermo(T) for e in self.entries]

    def _weights(self, vectors: List[ThermoVector], T: float, quantity: Optional[str] = None) -> List[float]:
        qid = quantity or self.weight_by
        values = [v.get(qid, T) for v in vectors]
        if any(x is None for x in values):
            raise ValueError(
                f"ConformerSet {self.name!r}: quantity {qid!r} is not available for every conformer")
        return list(values)  # type: ignore[arg-type]

    # -- public rollups -----------------------------------------------------

    def populations(self, T: float, quantity: Optional[str] = None) -> List[float]:
        """Boltzmann populations of the conformers at ``T`` (sum to 1),
        weighted by ``quantity`` (default: ``weight_by``)."""
        vecs = self.vectors(T)
        return self._boltzmann_probs_from(self._weights(vecs, T, quantity), T)

    def ensemble_free_energy(self, T: float, quantity: Optional[str] = None) -> float:
        """−RT ln Σᵢ exp(−Gᵢ/RT) in Hartree over the conformers, i.e. the
        conformationally averaged value of ``quantity`` (default:
        ``weight_by``) including the mixing entropy.

        For ``quantity='qh_gibbs'`` this equals
        ``gconf_corrected(T).qh_gibbs`` when ``weight_by`` is qh_gibbs.
        """
        vecs = self.vectors(T)
        values = self._weights(vecs, T, quantity)
        g_min = min(values)
        rt = GAS_CONSTANT * T / J_TO_AU      # Hartree
        z = sum(math.exp(-(g - g_min) / rt) for g in values)
        return g_min - rt * math.log(z)

    def s_conf(self, T: float, quantity: Optional[str] = None) -> float:
        """Conformational (mixing) entropy −R Σ pᵢ ln pᵢ in Hartree/K."""
        probs = self.populations(T, quantity)
        return -GAS_CONSTANT / J_TO_AU * sum(p * math.log(p) for p in probs if p > 0.0)

    def lowest_index(self, T: Optional[float] = None, quantity: Optional[str] = None) -> int:
        """Index of the conformer with the lowest ``quantity`` at ``T``."""
        vecs = self.vectors(T)
        values = self._weights(vecs, T if T is not None else 298.15, quantity)
        return min(range(len(values)), key=values.__getitem__)

    def lowest_conformer(self, T: Optional[float] = None) -> ThermoVector:
        """Vector for the conformer with the lowest ``weight_by`` value
        (qh-G by default) at ``T``."""
        vecs = self.vectors(T)
        return vecs[self.lowest_index(T)]

    def boltzmann_weighted(self, T: float) -> ThermoVector:
        """Pure Boltzmann-weighted average over conformers, weights from
        ``weight_by`` (qh-G by default)."""
        vecs = self.vectors(T)
        if self.is_single:
            return vecs[0]
        probs = self._boltzmann_probs_from(self._weights(vecs, T), T)
        result: Optional[ThermoVector] = None
        for p, vec in zip(probs, vecs):
            term = vec * p
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
        vecs = self.vectors(T)
        if self.is_single:
            return vecs[0]

        probs = self._boltzmann_probs_from(self._weights(vecs, T), T)
        boltz = self.boltzmann_weighted(T)
        lowest = self.lowest_conformer(T)

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

    def rollup(self, T: float, gconf: bool = True, QH: bool = True, lowest_only: bool = False) -> ThermoVector:
        """The species-level vector under the given rollup mode:
        lowest_only → lowest conformer; else gconf → gconf_corrected;
        else the pure Boltzmann average."""
        if lowest_only:
            return self.lowest_conformer(T)
        if gconf:
            return self.gconf_corrected(T, QH=QH)
        return self.boltzmann_weighted(T)

    def _boltzmann_probs(self, T: float) -> List[float]:
        return self._boltzmann_probs_from(self._weights(self.vectors(T), T), T)

    @staticmethod
    def _boltzmann_probs_from(values: Sequence[float], T: float) -> List[float]:
        g_min = min(values)
        weights = [math.exp(-(g - g_min) * J_TO_AU / GAS_CONSTANT / T) for g in values]
        norm = sum(weights)
        return [w / norm for w in weights]

    # -- duplicate removal --------------------------------------------------

    def dedup(self, e_cutoff: float = 0.05, ro_cutoff: float = 0.01,
              rmsd_cutoff: Optional[float] = None) -> "ConformerSet":
        """A new set without duplicate / enantiomeric conformers, using the
        same gates as ``goodvibes.sort.deduplicate`` (energy in kcal/mol,
        rotational constants as a fraction, optional Cartesian RMSD) and
        the same convention as the CLI's ``--dedup``: of each flagged pair
        the later file is the redundant copy and is dropped. A
        single-conformer set is returned unchanged."""
        if self.is_single:
            return self
        from .sort import deduplicate
        pairs = deduplicate(dict(zip(self.files, self.bbes)), e_cutoff=e_cutoff,
                            ro_cutoff=ro_cutoff, rmsd_cutoff=rmsd_cutoff)
        dropped = {a for a, _b in pairs}      # [duplicate, canonical], as selectivity._excluded_files
        keep = [i for i, f in enumerate(self.files) if f not in dropped]
        if len(keep) == len(self.files):
            return self
        return ConformerSet(
            name=self.name,
            files=[self.files[i] for i in keep],
            bbes=[self.bbes[i] for i in keep],
            entries=[self.entries[i] for i in keep] if self.entries is not None else None,
            weight_by=self.weight_by,
        )


# ---------------------------------------------------------------------------
# Point — stoichiometric sum of species
# ---------------------------------------------------------------------------

#: Point roles. A transition state is drawn with its label above the bar
#: and takes part in barrier annotations; minima/reactants/products are
#: labelled below.
POINT_ROLES = ("reactant", "minimum", "ts", "product")

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


def _normalise_role(role: Optional[str]) -> str:
    key = (role or "minimum").strip().lower()
    aliases = {"int": "minimum", "intermediate": "minimum", "min": "minimum",
               "transition_state": "ts", "transition-state": "ts", "tst": "ts",
               "r": "reactant", "p": "product", "reactants": "reactant", "products": "product"}
    key = aliases.get(key, key)
    if key not in POINT_ROLES:
        raise ValueError(f"unknown point role {role!r}; expected one of {', '.join(POINT_ROLES)}")
    return key


@dataclass
class Point:
    """One node of a pathway: a stoichiometric sum of ConformerSets.

    ``label`` is the point's identity (the string written in the PES file,
    e.g. "2*A + B"); ``display`` is what a figure prints for it (defaults
    to the label); ``role`` is one of ``POINT_ROLES``.
    """
    label: str                                       # original string, e.g. "2*A + B"
    species: List[Tuple[int, ConformerSet]]          # parsed: [(coeff, set), ...]
    role: str = "minimum"
    display: Optional[str] = None

    def __post_init__(self):
        self.role = _normalise_role(self.role)

    @property
    def id(self) -> str:
        return self.label

    @property
    def display_label(self) -> str:
        return self.display if self.display is not None else self.label

    @property
    def is_ts(self) -> bool:
        return self.role == "ts"

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
            term = cset.rollup(T, gconf=gconf, QH=QH, lowest_only=lowest_only) * coeff
            total = term if total is None else total + term
        if total is None:
            raise ValueError(f"Point {self.label!r} has no species")
        return total

    @classmethod
    def from_label(cls, label: str, species_map: dict, role: str = "minimum",
                   display: Optional[str] = None) -> "Point":
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
        return cls(label=label, species=resolved, role=role, display=display)


# ---------------------------------------------------------------------------
# Edge / Pathway / Series / PESResult
# ---------------------------------------------------------------------------

#: Edge kinds. ``step`` is an ordinary connector, ``barrierless`` a dotted
#: one (association / dissociation without a located TS), ``none`` no line.
EDGE_KINDS = ("step", "barrierless", "none")


@dataclass(frozen=True)
class Edge:
    """A connection between two points of a pathway (by point label)."""
    src: str
    dst: str
    kind: str = "step"

    def __post_init__(self):
        kind = (self.kind or "step").strip().lower()
        if kind not in EDGE_KINDS:
            raise ValueError(f"unknown edge kind {self.kind!r}; expected one of {', '.join(EDGE_KINDS)}")
        object.__setattr__(self, "kind", kind)


@dataclass
class Pathway:
    """Ordered points + edges + a designated zero (defaults to points[0]).

    ``edges`` default to one ``step`` edge between each pair of consecutive
    points; give an explicit list to mark barrierless steps or to omit a
    connector. Edges name points by label.
    """
    name: str
    points: List[Point]
    zero: Point = field(default=None)  # type: ignore[assignment]
    edges: List[Edge] = field(default=None)  # type: ignore[assignment]

    def __post_init__(self):
        if not self.points:
            raise ValueError(f"Pathway {self.name!r} has no points")
        if self.zero is None:
            self.zero = self.points[0]
        labels = {p.label for p in self.points}
        if self.edges is None:
            self.edges = [Edge(a.label, b.label) for a, b in zip(self.points, self.points[1:])]
        else:
            self.edges = [e if isinstance(e, Edge) else Edge(*e) for e in self.edges]
            for e in self.edges:
                for end in (e.src, e.dst):
                    if end not in labels:
                        raise ValueError(
                            f"Pathway {self.name!r}: edge {e.src!r} -> {e.dst!r} names a point "
                            f"that is not on the pathway (points: {[p.label for p in self.points]})")

    @property
    def labels(self) -> List[str]:
        return [p.label for p in self.points]

    def point(self, label: str) -> Point:
        for p in self.points:
            if p.label == label:
                return p
        if self.zero.label == label:
            return self.zero
        raise KeyError(f"Pathway {self.name!r} has no point {label!r}")

    def edge_kind(self, src: str, dst: str) -> Optional[str]:
        """Kind of the edge src -> dst, or None when there is none."""
        for e in self.edges:
            if e.src == src and e.dst == dst:
                return e.kind
        return None

    def with_edges(self, edges: Iterable) -> "Pathway":
        """Copy with explicit edges (tuples ``(src, dst[, kind])`` or ``Edge``)."""
        return Pathway(name=self.name, points=self.points, zero=self.zero, edges=list(edges))

    def relative(
        self, T: float, gconf: bool = True, QH: bool = True, lowest_only: bool = False,
    ) -> List[ThermoVector]:
        """Per-point ΔThermo relative to self.zero, in Hartree.

        This and :meth:`levels` are the only places relative values are
        formed; tables, JSON and figures all read them.
        """
        kw = dict(gconf=gconf, QH=QH, lowest_only=lowest_only)
        zero = self.zero.thermo(T, **kw)
        return [p.thermo(T, **kw) - zero for p in self.points]

    def levels(
        self, T: float, quantity: str = "qh_gibbs", gconf: bool = True, QH: bool = True,
        lowest_only: bool = False,
    ) -> Dict[str, Optional[float]]:
        """{point label: Δquantity relative to the zero, in Hartree} at ``T``.

        ``quantity`` is a registry id or alias (goodvibes.quantities);
        entropies are returned as T·ΔS. A value is None when the quantity
        is unavailable (single-point energy without --spc). A point
        without species (one that only carries declared values in a
        reaction-profile document) is absent from the result, and a pathway
        whose zero has no species gives an empty mapping.
        """
        from .quantities import resolve_quantity
        qid = resolve_quantity(quantity).id
        if not self.zero.species:
            return {}
        kw = dict(gconf=gconf, QH=QH, lowest_only=lowest_only)
        zero = self.zero.thermo(T, **kw)
        return {p.label: (p.thermo(T, **kw) - zero).get(qid, T) for p in self.points if p.species}

    @property
    def computable(self) -> bool:
        """True when every point (and the zero) has species, so the full
        thermochemistry table can be formed for this pathway."""
        return bool(self.zero.species) and all(p.species for p in self.points)


@dataclass
class PESOptions:
    units: str = "kcal/mol"          # any of constants.SUPPORTED_UNITS ('kcal/mol', 'kJ/mol', 'eV', 'hartree')
    decimals: int = 2
    gconf: bool = True
    QH: bool = True
    spc_used: bool = False           # True when --spc was set
    lowest_only: bool = False        # True when --lowest-only overrides gconf/Boltzmann

    def to_user_units(self, hartree: Optional[float]) -> Optional[float]:
        if hartree is None:
            return None
        return hartree_factor(self.units) * hartree

    @property
    def rollup_kw(self) -> Dict[str, bool]:
        """Keyword arguments for ``Point.thermo`` / ``Pathway.relative``."""
        return dict(gconf=self.gconf, QH=self.QH, lowest_only=self.lowest_only)


@dataclass
class Series:
    """One quantity at one temperature: a line set on the axes, a column
    set in a table.

    A *computed* series (``declared=False``) is evaluated from the model
    through ``Pathway.levels(temperature, quantity)``; ``temperature``
    None means the result's first temperature. Its ``levels``, when set,
    are a stored evaluation (a reaction-profile document written by
    ``Profile.evaluate``) and are used as they are. A *declared* series
    (``declared=True``) carries typed-in ``levels`` — literature values, a
    hand-entered table — as {pathway name: {point label: value}} in
    ``units`` (default: the result's units) and is never re-evaluated;
    a point missing from its levels is simply absent from the figure.
    ``style`` holds matplotlib overrides (``linestyle``, ``color``, ...).
    """
    id: str
    label: str
    quantity: str = "qh_gibbs"
    temperature: Optional[float] = None
    method: Optional[str] = None
    style: Dict[str, Any] = field(default_factory=dict)
    levels: Optional[Dict[str, Dict[str, Optional[float]]]] = None
    declared: bool = False
    units: Optional[str] = None
    uncertainty: Optional[Dict[str, Dict[str, float]]] = None
    standard_state: Optional[Dict[str, Any]] = None
    extensions: Dict[str, Any] = field(default_factory=dict)   # x-* keys of a reaction-profile document

    def __post_init__(self):
        from .quantities import resolve_quantity
        self.quantity = resolve_quantity(self.quantity).id
        if self.declared and self.levels is None:
            raise ValueError(f"Series {self.id!r}: a declared series needs its levels")
        if self.units is not None:
            from .constants import canonical_units
            self.units = canonical_units(self.units)

    @classmethod
    def declared_from(cls, id: str, label: str, levels: Mapping[str, Mapping[str, Optional[float]]],
                      *, quantity: str = "gibbs", temperature: Optional[float] = None,
                      method: Optional[str] = None, units: Optional[str] = None,
                      style: Optional[Dict[str, Any]] = None) -> "Series":
        """A declared series from {pathway: {point label: value}}."""
        return cls(id=id, label=label, quantity=quantity, temperature=temperature, method=method,
                   style=dict(style or {}), levels={k: dict(v) for k, v in levels.items()},
                   declared=True, units=units)

    def evaluate(self, result: "PESResult", pathway: Pathway) -> Dict[str, Optional[float]]:
        """{point label: value in the result's units} for one pathway.

        Stored levels (every declared series, and a computed series read
        from an evaluated document) are returned converted to the result's
        units; otherwise a computed series goes through ``Pathway.levels``.
        """
        if self.levels is not None:
            src_units = self.units or result.options.units
            factor = hartree_factor(result.options.units) / hartree_factor(src_units)
            stored = (self.levels or {}).get(pathway.name, {})
            return {p.label: (stored[p.label] * factor if stored.get(p.label) is not None else None)
                    for p in pathway.points if p.label in stored}
        T = self.temperature if self.temperature is not None else result.temperature
        factor = hartree_factor(result.options.units)
        raw = pathway.levels(T, self.quantity, **result.options.rollup_kw)
        return {k: (v * factor if v is not None else None) for k, v in raw.items()}


@dataclass
class PESResult:
    """Top-level container: pathways + options + temperatures + series."""
    pathways: List[Pathway]
    options: PESOptions
    temperatures: List[float] = field(default_factory=lambda: [298.15])
    series: List[Series] = field(default_factory=list)
    order: Optional[List[str]] = None     # optional override of the merged x order
    source: Any = field(default=None, repr=False, compare=False)   # the goodvibes.profile.Profile it was built from

    @property
    def temperature(self) -> float:
        """The first (base) temperature."""
        return self.temperatures[0] if self.temperatures else 298.15

    def pathway(self, name) -> Pathway:
        if isinstance(name, int):
            return self.pathways[name]
        for p in self.pathways:
            if p.name == name:
                return p
        raise KeyError(f"no pathway {name!r} (available: {[p.name for p in self.pathways]})")

    @property
    def recomputable(self) -> bool:
        """True when every species can be evaluated at other temperatures."""
        return all(cset.recomputable
                   for path in self.pathways
                   for point in path.points + [path.zero]
                   for _c, cset in point.species)

    def default_series(self, quantity: str = "qh_gibbs",
                       temperatures: Optional[Sequence[float]] = None) -> List[Series]:
        """One computed series per temperature (the implicit series of a
        result that declares none)."""
        from .quantities import resolve_quantity
        qty = resolve_quantity(quantity)
        temps = list(temperatures) if temperatures is not None else list(self.temperatures)
        if not temps:
            temps = [298.15]
        out = []
        for T in temps:
            label = qty.label if len(temps) == 1 else f"{qty.label} {T:g} K"
            out.append(Series(id=f"{qty.id}@{T:g}K", label=label, quantity=qty.id, temperature=T))
        return out

    def merged_order(self) -> List[str]:
        """Point labels in x order: ``order`` when set, else the union of
        the pathways' point sequences, each pathway's order preserved."""
        if self.order:
            return list(self.order)
        return merge_point_order([p.labels for p in self.pathways])

    def levels(self, series: Optional[Sequence[Series]] = None) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
        """{series id: {pathway name: {point label: value}}} in user units,
        for the given series (default: ``self.series`` or the implicit
        qh-G series per temperature)."""
        series = list(series) if series is not None else (self.series or self.default_series())
        return {s.id: {p.name: s.evaluate(self, p) for p in self.pathways} for s in series}


def merge_point_order(sequences: Sequence[Sequence[str]]) -> List[str]:
    """Merge several ordered label sequences into one order that keeps every
    sequence's own order. A label new to the merged list is inserted just
    before the earliest already-placed label that follows it in its own
    sequence, or appended when none does; so ``[R, Int1, TS1, P]`` and
    ``[R, TS1, P]`` merge to the first, and two branches ``[R, TS_R, P_R]``
    / ``[R, TS_S, P_S]`` keep each branch contiguous.
    """
    order: List[str] = []
    for seq in sequences:
        for i, label in enumerate(seq):
            if label in order:
                continue
            successors = [order.index(s) for s in seq[i + 1:] if s in order]
            if successors:
                order.insert(min(successors), label)
            else:
                order.append(label)
    return order
