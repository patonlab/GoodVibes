"""SelectivityResult v2: the major and ee-sign conventions, the ratio and
the per-label ensemble energies."""
import math

import pytest

from goodvibes.constants import GAS_CONSTANT, J_TO_AU, KCAL_TO_AU
from goodvibes.selectivity import compute_selectivity, selectivity_from_energies

T = 298.15
RT = GAS_CONSTANT * T / J_TO_AU                     # Hartree
KCAL = 1.0 / KCAL_TO_AU                             # one kcal/mol in Hartree


def test_the_ee_sign_follows_the_label_order():
    ab = selectivity_from_energies({"A": [0.0], "B": [1.0 * KCAL]}, T)
    ba = selectivity_from_energies({"B": [1.0 * KCAL], "A": [0.0]}, T)
    assert ab.major == ab.preferred == ba.major == "A"
    assert ab.ee_signed > 0 and ba.ee_signed == pytest.approx(-ab.ee_signed)
    assert ab.ee == ba.ee == pytest.approx(abs(ab.ee_signed))
    assert ab.ddG == pytest.approx(1.0 * KCAL)                   # RT ln(p_major/p_minor)
    assert ab.ratio == pytest.approx(math.exp(KCAL / RT))


def test_a_tie_makes_the_first_label_major():
    r = selectivity_from_energies({"X": [0.0], "Y": [0.0]}, T)
    assert r.major == "X" and r.ee_signed == pytest.approx(0.0) and r.ratio == pytest.approx(1.0)


def test_ensemble_energies_are_minus_rt_ln_z_per_label():
    conformers = {"A": [0.0, 0.5 * KCAL, 2.0 * KCAL], "B": [0.8 * KCAL]}
    r = selectivity_from_energies(conformers, T, quantity="qh_gibbs")
    for label, es in conformers.items():
        z = sum(math.exp(-e / RT) for e in es)
        assert r.ensemble_energies[label] == pytest.approx(-RT * math.log(z))
    diff = r.ensemble_energies["B"] - r.ensemble_energies["A"]
    assert r.ddG == pytest.approx(diff)                          # ΔΔG‡ is the ensemble gap
    assert r.quantity == r.key == "qh_gibbs"


def test_more_than_two_labels_have_a_ratio_but_no_ee():
    r = selectivity_from_energies({"a": [0.0], "b": [0.5 * KCAL], "c": [3.0 * KCAL]}, T)
    assert r.ee is None and r.ee_signed is None and r.ddG is None
    assert r.ratio == pytest.approx(math.exp(0.5 * KCAL / RT))   # major over the runner-up


def test_a_label_without_energies_has_population_zero():
    r = selectivity_from_energies({"A": [0.0], "B": []}, T)
    assert r.populations == {"A": 1.0, "B": 0.0}
    assert r.ee_signed == 100.0 and r.ratio is None and r.ddG is None
    assert r.ensemble_energies["B"] is None
    with pytest.raises(ValueError, match="at least two"):
        selectivity_from_energies({"A": [0.0]}, T)
    with pytest.raises(ValueError, match="No label has an energy"):
        selectivity_from_energies({"A": [], "B": [None]}, T)


class _Bbe:
    def __init__(self, g):
        self.qh_gibbs_free_energy = g


def test_compute_selectivity_fills_the_v2_fields():
    td = {"r1": _Bbe(-100.0), "r2": _Bbe(-100.0 + 0.3 * KCAL), "s1": _Bbe(-100.0 + 1.1 * KCAL)}
    r = compute_selectivity(td, {"R": ["r1", "r2"], "S": ["s1"]}, T)
    assert r.quantity == "qh_gibbs" and r.key == "gibbs"
    assert r.major == "R" and r.ee_signed > 0
    assert r.ensemble_energies["S"] == pytest.approx(-100.0 + 1.1 * KCAL)
    assert r.files_per_label == {"R": ["r1", "r2"], "S": ["s1"]}
