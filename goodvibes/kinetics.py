"""Minimal kinetics from a reaction profile: Eyring rate constants, rate
ratios, a step table, the energy-span model and a mikimo export.

No microkinetics: for concentrations against time, export the profile to
mikimo (``write_mikimo_csv``) and model it there.

Energies are free energies of activation (or of reaction) in ``units``
(kcal/mol by default). An Eyring rate constant ``k = κ kB T / h ·
exp(-ΔG‡ / RT)`` is in s⁻¹ for a unimolecular step; for a bimolecular one it
is in M⁻¹ s⁻¹ when ΔG‡ refers to the 1 M standard state.

The energy span model (Kozuch and Shaik, Acc. Chem. Res. 2011, 44, 101)
gives a catalytic cycle's turnover frequency from its states in order:

    TOF = (kB T / h) (exp(-ΔG_r / RT) - 1) / Σ_ij exp((T_i - I_j - δG'_ij) / RT)

over transition states T_i and intermediates I_j, with δG'_ij = ΔG_r when
T_i comes after I_j and 0 otherwise. The TOF-determining TS and
intermediate (TDTS, TDI) give the largest term, the energy span is
δE = T_TDTS - I_TDI (+ ΔG_r when the TDTS comes first), and each state's
degree of TOF control is its share of the sum.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .constants import GAS_CONSTANT, J_TO_AU, hartree_factor

__all__ = [
    "eyring_rate", "barrier_for_rate", "rate_ratio", "half_life",
    "EnergySpan", "energy_span", "step_table", "mikimo_rows", "write_mikimo_csv",
]

PLANCK_CONSTANT = 6.62606957e-34      # J s (the value thermo.py uses)
BOLTZMANN_CONSTANT = 1.3806488e-23    # J / K


def _j_per_mol(value: float, units: str) -> float:
    """``value`` in ``units`` as J/mol."""
    return value / hartree_factor(units) * J_TO_AU


def _prefactor(T: float, kappa: float = 1.0) -> float:
    return kappa * BOLTZMANN_CONSTANT * T / PLANCK_CONSTANT


def eyring_rate(dG: float, temperature: float = 298.15, *, units: str = "kcal/mol",
                kappa: float = 1.0) -> float:
    """The Eyring rate constant ``κ kB T / h exp(-ΔG‡ / RT)`` for a free
    energy of activation ``dG`` in ``units`` (s⁻¹ for a unimolecular step)."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return _prefactor(temperature, kappa) * math.exp(-_j_per_mol(dG, units) / (GAS_CONSTANT * temperature))


def barrier_for_rate(k: float, temperature: float = 298.15, *, units: str = "kcal/mol",
                     kappa: float = 1.0) -> float:
    """The free energy of activation (``units``) that gives rate constant ``k``."""
    if k <= 0:
        raise ValueError("the rate constant must be positive")
    j = GAS_CONSTANT * temperature * math.log(_prefactor(temperature, kappa) / k)
    return j / J_TO_AU * hartree_factor(units)


def rate_ratio(dG_a: float, dG_b: float, temperature: float = 298.15, *, units: str = "kcal/mol") -> float:
    """``k_a / k_b = exp(-(ΔG‡_a - ΔG‡_b) / RT)``."""
    return math.exp(-_j_per_mol(dG_a - dG_b, units) / (GAS_CONSTANT * temperature))


def half_life(k: float) -> float:
    """``ln 2 / k``: the half-life (s) of a first-order step with rate constant ``k`` (s⁻¹)."""
    return math.log(2.0) / k


# ---------------------------------------------------------------------------
# Energy span
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EnergySpan:
    """The energy-span analysis of one catalytic cycle.

    ``span`` (δE) and ``reaction_energy`` (ΔG_r) are in ``units``; ``tof``
    is the exact energy-span TOF and ``tof_span`` its approximation
    ``kB T / h exp(-δE / RT)`` (both s⁻¹; zero or negative when the cycle is
    not exergonic). ``control`` maps every TS and intermediate to its
    degree of TOF control (the TSs sum to 1, and so do the intermediates).
    """
    tdts: str
    tdi: str
    span: float
    reaction_energy: float
    tof: float
    tof_span: float
    temperature: float
    units: str
    tdts_after_tdi: bool
    control: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return (f"energy span {self.span:.2f} {self.units} (TDTS {self.tdts}, TDI {self.tdi}"
                f"{'' if self.tdts_after_tdi else ', TDTS before TDI: + ΔG_r'}), "
                f"ΔG_r {self.reaction_energy:.2f} {self.units}, TOF {self.tof:.3g} s⁻¹ at {self.temperature:g} K")


def energy_span(levels: Mapping[str, float], transition_states: Iterable[str], *,
                reaction_energy: Optional[float] = None, temperature: float = 298.15,
                units: str = "kcal/mol") -> EnergySpan:
    """The energy span of a catalytic cycle.

    Parameters:
        levels: the states of one turnover in order, ``{point: level}``.
            Without ``reaction_energy``, the last state is the first one
            regenerated after the turnover (catalyst plus products): it sets
            ``ΔG_r = level(last) - level(first)`` and is not a state of the
            cycle.
        transition_states: the points of ``levels`` that are transition
            states; the others are intermediates.
        reaction_energy: ΔG_r of one turnover, when every point of
            ``levels`` is a state of the cycle.
        temperature: K.
        units: of the levels and of the results.
    """
    items = [(p, float(v)) for p, v in levels.items() if v is not None]
    ts = set(transition_states)
    if reaction_energy is None:
        if len(items) < 3:
            raise ValueError("energy_span needs at least an intermediate, a TS and the regenerated state")
        reaction_energy = items[-1][1] - items[0][1]
        items = items[:-1]
    tss = [(i, p, v) for i, (p, v) in enumerate(items) if p in ts]
    ints = [(j, p, v) for j, (p, v) in enumerate(items) if p not in ts]
    if not tss or not ints:
        raise ValueError("energy_span needs at least one transition state and one intermediate")
    RT = GAS_CONSTANT * temperature
    dGr = _j_per_mol(reaction_energy, units)

    def exponent(i, ti, j, ij):
        return (_j_per_mol(ti - ij, units) - (dGr if i > j else 0.0)) / RT

    terms = {(pi, pj): exponent(i, ti, j, ij) for i, pi, ti in tss for j, pj, ij in ints}
    top = max(terms.values())
    total = sum(math.exp(x - top) for x in terms.values())           # log-sum-exp
    log_sum = top + math.log(total)
    (tdts, tdi), _ = max(terms.items(), key=lambda kv: kv[1])
    pos = {p: i for i, (p, _v) in enumerate(items)}
    after = pos[tdts] > pos[tdi]
    span = dict(items)[tdts] - dict(items)[tdi] + (0.0 if after else reaction_energy)
    numerator = math.expm1(-dGr / RT)                                 # exp(-ΔG_r/RT) - 1
    tof = _prefactor(temperature) * numerator * math.exp(-log_sum) if math.isfinite(numerator) else math.inf
    tof_span = eyring_rate(span, temperature, units=units)
    control: Dict[str, float] = {}
    for (pi, pj), x in terms.items():
        share = math.exp(x - log_sum)
        control[pi] = control.get(pi, 0.0) + share
        control[pj] = control.get(pj, 0.0) + share
    return EnergySpan(tdts=tdts, tdi=tdi, span=span, reaction_energy=reaction_energy, tof=tof,
                      tof_span=tof_span, temperature=temperature, units=units, tdts_after_tdi=after,
                      control=control)


# ---------------------------------------------------------------------------
# Step table
# ---------------------------------------------------------------------------

def step_table(levels: Mapping[str, float], transition_states: Iterable[str], *,
               temperature: float = 298.15, units: str = "kcal/mol") -> List[dict]:
    """One row per transition state along a pathway (``levels`` in order):

    - ``ts``, ``from`` (the last intermediate before it), ``to`` (the first
      intermediate after it, or None);
    - ``barrier``: ``level(ts) - level(from)``, in ``units``;
    - ``barrier_from_lowest``: from the lowest point before the TS (the
      effective barrier when an earlier state is deeper);
    - ``step_energy``: ``level(to) - level(from)``;
    - ``k`` (s⁻¹) and ``half_life`` (s) of the Eyring rate over ``barrier``.
    """
    items = [(p, float(v)) for p, v in levels.items() if v is not None]
    ts = set(transition_states)
    rows = []
    for idx, (p, v) in enumerate(items):
        if p not in ts:
            continue
        before = [(q, w) for q, w in items[:idx] if q not in ts]
        after = [(q, w) for q, w in items[idx + 1:] if q not in ts]
        if not before:
            continue
        src, src_level = before[-1]
        dst = after[0] if after else None
        lowest = min(w for _q, w in items[:idx])
        barrier = v - src_level
        k = eyring_rate(barrier, temperature, units=units)
        rows.append({
            "ts": p, "from": src, "to": dst[0] if dst else None,
            "barrier": barrier, "barrier_from_lowest": v - lowest,
            "step_energy": dst[1] - src_level if dst else None,
            "k": k, "half_life": half_life(k), "temperature": temperature, "units": units,
        })
    return rows


# ---------------------------------------------------------------------------
# mikimo export
# ---------------------------------------------------------------------------

def _mikimo_names(points: Sequence[str], transition_states: Iterable[str]) -> List[str]:
    """mikimo state names for ``points`` in order: ``INT0, TS1, INT1, ...``
    with the last point ``Prod`` (mikimo reads a name with 'TS' as a
    transition state and one starting with R or P as a reactant or
    product, so point ids cannot be used as they are)."""
    ts = set(transition_states)
    if points and points[-1] in ts:
        raise ValueError(f"the last point, {points[-1]!r}, is a transition state; mikimo reads the last "
                         "state as the product, so end the profile at a minimum")
    names, n_int, n_ts = [], 0, 0
    for i, p in enumerate(points):
        if i == len(points) - 1:
            names.append("Prod")
        elif p in ts:
            n_ts += 1
            names.append(f"TS{n_ts}")
        else:
            names.append(f"INT{n_int}")
            n_int += 1
    return names


def mikimo_rows(profiles: Mapping[str, Mapping[str, float]], transition_states: Iterable[str], *,
                units: str = "kcal/mol") -> Tuple[List[str], List[List], Dict[str, str]]:
    """(header, rows, names) for mikimo's ``reaction_data.csv``: one row per
    profile (``{name: {point: level}}``, all with the same sequence of TS
    and intermediate states), levels converted to kcal/mol, and the map
    from each point id of the first profile to its mikimo name."""
    ts = set(transition_states)
    profiles = {name: [(p, v) for p, v in lv.items()] for name, lv in profiles.items()}
    if not profiles:
        raise ValueError("no profile to export")
    first = next(iter(profiles.values()))
    pattern = [p in ts for p, _v in first]
    for name, items in profiles.items():
        if [p in ts for p, _v in items] != pattern:
            raise ValueError(f"profile {name!r} does not have the same sequence of transition states and "
                             "intermediates as the first; export it to its own file")
        if any(v is None for _p, v in items):
            raise ValueError(f"profile {name!r} has a point without a level")
    header = _mikimo_names([p for p, _v in first], ts)
    to_kcal = hartree_factor("kcal/mol") / hartree_factor(units)
    rows = [[name] + [v * to_kcal for _p, v in items] for name, items in profiles.items()]
    return header, rows, dict(zip((p for p, _v in first), header))


def write_mikimo_csv(path, profiles: Mapping[str, Mapping[str, float]], transition_states: Iterable[str], *,
                     units: str = "kcal/mol") -> Dict[str, str]:
    """Write mikimo's ``reaction_data.csv`` (see ``mikimo_rows``) and return
    the point-to-state-name map. mikimo also needs ``rxn_network.csv``,
    which says which species each step consumes and forms; write that by
    hand with these state names."""
    header, rows, names = mikimo_rows(profiles, transition_states, units=units)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([""] + header)
        for row in rows:
            writer.writerow([row[0]] + [f"{v:.4f}" for v in row[1:]])
    return names
