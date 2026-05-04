"""Benchmark: msRRHO entropies vs experimental NIST S°(298.15 K, 1 bar).

Anchored on Pracht & Grimme, Chem. Sci. 2021, 12, 6551-6568 (DOI:
10.1039/D1SC00621E) — the msRRHO paper that GoodVibes already cites in
constants.py and implements as the default low-frequency entropy treatment.
The paper reports MAD < 1 cal mol⁻¹ K⁻¹ for medium-sized molecules. This
suite exercises the same machinery (compute_thermo, QS="grimme") on a
small set of well-characterized molecules and compares to NIST WebBook
reference values at 298.15 K and 1 bar.

Tests skip when the corresponding .out fixture is missing — the input decks
live alongside as benchmark_*_b3lyp.inp. To regenerate, run each .inp
through ORCA 6 and drop the resulting .out file in tests/orca6/.

Note on scope: at this molecule size (≤ n-butane), the lowest vibrational
modes are typically > 200 cm⁻¹, well above GoodVibes' default 100 cm⁻¹
msRRHO cutoff — so for these fixtures msRRHO and naive harmonic RRHO
produce nearly identical entropies. Demonstrating the msRRHO advantage
in isolation would require larger flexible systems (e.g. n-octane,
glycols, peptide backbones) where torsional modes drop below the cutoff.
"""
import os

import pytest

from goodvibes import compute_thermo
from goodvibes.constants import GAS_CONSTANT, KCAL_TO_AU


ORCA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "orca6")

# Hartree/K → cal mol⁻¹ K⁻¹: 1 Hartree = KCAL_TO_AU kcal/mol = 1000·KCAL_TO_AU cal/mol.
HARTREE_K_TO_CAL_MOL_K = KCAL_TO_AU * 1000.0   # 627509.541


def _conc_one_bar(temperature: float) -> float:
    """Ideal-gas concentration (mol L⁻¹) at 1 bar = 100 kPa.

    NIST tabulates S° at 1 bar; the GoodVibes default uses 1 atm
    (constants.ATMOS = 101.325 kPa), so we override the concentration
    to match the reference standard state."""
    return 100.0 / (GAS_CONSTANT * temperature)


# (filename, exp_S°(298.15 K, 1 bar) in J mol⁻¹ K⁻¹, tolerance in cal mol⁻¹ K⁻¹)
# Reference values: NIST Chemistry WebBook (https://webbook.nist.gov/chemistry/).
# Three tolerance tiers, each justified by the entropy contributions that
# single-conformer msRRHO can vs. cannot recover:
#   1.0 cal/(mol·K) — strictly rigid, high symmetry. Paper's MAD for this
#                     class. msRRHO ≈ exact partition functions.
#   2.0 cal/(mol·K) — low-barrier inversion / soft mode contributing
#                     ~R·ln 2 ≈ 1.4 cal/(mol·K) of state-doubling not
#                     fully captured by harmonic treatment of the mode.
#   3.0 cal/(mol·K) — multiple distinguishable rotamers. Carries a
#                     mixing-entropy term R·ln(N_conf) ≈ 2.2 cal/(mol·K)
#                     for N=3 that single-conformer msRRHO cannot recover.
#                     Closing this gap motivates roadmap items #11
#                     (Ensemble) and #13 (S_conf on Ensemble).
BENCHMARK = [
    # rigid, high symmetry
    ("benchmark_water_b3lyp.out",       188.835, 1.0),   # C2v
    ("benchmark_methane_b3lyp.out",     186.25,  1.0),   # Td
    ("benchmark_benzene_b3lyp.out",     269.20,  1.0),   # D6h
    ("benchmark_ethane_b3lyp.out",      229.16,  1.0),   # D3d (one rotamer)
    # soft modes / inversion
    ("benchmark_methylamine_b3lyp.out", 242.85,  2.0),   # NH2 umbrella
    # multiple rotamers (R·ln 3 ≈ 2.2 cal/(mol·K) gap)
    ("benchmark_ethanol_b3lyp.out",     281.6,   3.0),   # anti + 2 gauche
    ("benchmark_butane_b3lyp.out",      310.094, 3.0),   # anti + 2 gauche
]


@pytest.mark.parametrize("filename,exp_S_J,tol_cal", BENCHMARK)
def test_msRRHO_entropy_vs_experiment(filename, exp_S_J, tol_cal):
    """msRRHO S°(298.15 K, 1 bar) within tolerance of NIST experimental value."""
    fixture = os.path.join(ORCA_DIR, filename)
    if not os.path.exists(fixture):
        pytest.skip(f"fixture {filename} not yet generated; "
                    f"run {filename.replace('.out', '.inp')} through ORCA 6")

    T = 298.15
    r = compute_thermo(fixture, QS="grimme", temperature=T,
                       concentration=_conc_one_bar(T))

    computed_cal = r.qh_entropy * HARTREE_K_TO_CAL_MOL_K
    exp_cal = exp_S_J / 4.184
    assert abs(computed_cal - exp_cal) < tol_cal, (
        f"{filename}: msRRHO S°={computed_cal:.2f}, "
        f"NIST S°={exp_cal:.2f} cal mol⁻¹ K⁻¹, "
        f"|Δ|={abs(computed_cal - exp_cal):.2f} > tol={tol_cal}"
    )


# Note: a previous version of this file included a harmonic-vs-msRRHO
# comparison test on ethane/butane/methylamine. It was removed because at
# B3LYP-D3BJ/def2-TZVP the lowest real frequencies of these molecules are
# > 200 cm⁻¹ (e.g. ethane CH3-CH3 torsion at 304 cm⁻¹), well above the
# 100 cm⁻¹ msRRHO cutoff — so msRRHO and harmonic RRHO produce identical
# entropies for them. The msRRHO advantage shows up at larger / floppier
# scale (long alkanes, glycols, amino acids) which is out of scope here.
