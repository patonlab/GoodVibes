"""Tests for goodvibes.sort: kabsch_rmsd, deduplicate, sort_thermo."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from goodvibes.sort import kabsch_rmsd, deduplicate, sort_thermo, SORT_KEYS
from goodvibes.thermo import calc_bbe
from goodvibes.constants import KCAL_TO_AU
from conftest import g16path


GAS_CONSTANT = 8.3144621
ATMOS = 101.325
T_DEFAULT = 298.15
CONC_DEFAULT = ATMOS / (GAS_CONSTANT * T_DEFAULT)


def _bbe(filename, scale=1.0):
    return calc_bbe(g16path(filename), 'grimme', False, 100.0, 100.0,
                    T_DEFAULT, CONC_DEFAULT, scale, None, None, None, 0)


# ===========================================================================
# kabsch_rmsd
# ===========================================================================

def _square(scale=1.0, offset=(0, 0, 0)):
    """A unit square in the xy plane, optionally translated."""
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float) * scale
    return pts + np.array(offset)


def test_kabsch_rmsd_identical():
    a = _square()
    assert kabsch_rmsd(a, a) == pytest.approx(0.0, abs=1e-12)


def test_kabsch_rmsd_translation_invariant():
    """Centering inside the algorithm should erase pure translations."""
    a = _square()
    b = _square(offset=(10.0, -3.0, 7.5))
    assert kabsch_rmsd(a, b) == pytest.approx(0.0, abs=1e-12)


def test_kabsch_rmsd_rotation_invariant():
    """Optimal-rotation alignment should erase pure rigid rotations."""
    a = _square()
    # 30° rotation about z
    theta = math.radians(30)
    R = np.array([[math.cos(theta), -math.sin(theta), 0],
                  [math.sin(theta),  math.cos(theta), 0],
                  [0, 0, 1]])
    b = a @ R.T
    assert kabsch_rmsd(a, b) == pytest.approx(0.0, abs=1e-12)


def test_kabsch_rmsd_reflection_corrected():
    """The Kabsch algorithm should handle reflections (chiral images)."""
    a = _square()
    # Reflection through xy plane is trivial (z=0 already), so flip x
    b = a.copy()
    b[:, 0] = -b[:, 0]
    # The 'reflection-corrected' Kabsch returns 0 for an enantiomer of a planar
    # achiral structure.
    rmsd = kabsch_rmsd(a, b)
    assert rmsd == pytest.approx(0.0, abs=1e-10)


def test_kabsch_rmsd_genuinely_different():
    """Different geometries (not related by rigid motion) give non-zero RMSD."""
    a = _square()
    # Stretch one corner — not a rigid transform
    b = a.copy()
    b[2] += [0.5, 0.5, 0]
    assert kabsch_rmsd(a, b) > 0.1


def test_kabsch_rmsd_lists_accepted():
    """Lists (not numpy arrays) are coerced to float arrays."""
    coords = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    assert kabsch_rmsd(coords, coords) == pytest.approx(0.0, abs=1e-12)


# ===========================================================================
# deduplicate — minimal stub objects (don't need full calc_bbe)
# ===========================================================================
# We use SimpleNamespace stubs so the test data is fully under control.

def _stub(scf_energy, roconst, cartesians=None):
    return SimpleNamespace(
        scf_energy=scf_energy,
        roconst=list(roconst),
        cartesians=list(cartesians) if cartesians else [],
    )


def test_deduplicate_empty():
    assert deduplicate({}) == []


def test_deduplicate_single():
    data = {'a.log': _stub(-76.0, [10, 20, 30])}
    assert deduplicate(data) == []


def test_deduplicate_identical_pair():
    """Same energy, same ro_const → flagged."""
    bbe = _stub(-76.0, [10, 20, 30])
    data = {'a.log': bbe, 'b.log': _stub(-76.0, [10, 20, 30])}
    dups = deduplicate(data)
    assert len(dups) == 1
    assert sorted(dups[0]) == ['a.log', 'b.log']


def test_deduplicate_energy_too_far():
    """Energy gap above e_cutoff → not flagged regardless of ro_const."""
    data = {'a.log': _stub(-76.0, [10, 20, 30]),
            'b.log': _stub(-77.0, [10, 20, 30])}  # 1 Ha apart
    assert deduplicate(data, e_cutoff=0.05) == []


def test_deduplicate_ro_too_far():
    """Energy match but ro_const differ by > ro_cutoff → not flagged."""
    data = {'a.log': _stub(-76.0, [10, 20, 30]),
            'b.log': _stub(-76.0, [10, 22, 30])}  # ~10% diff in middle
    assert deduplicate(data, e_cutoff=0.05, ro_cutoff=0.01) == []


def test_deduplicate_ro_within_cutoff():
    """Energy match + ro within cutoff → flagged."""
    data = {'a.log': _stub(-76.0, [10.0, 20.0, 30.0]),
            'b.log': _stub(-76.0, [10.0001, 20.0001, 30.0001])}
    assert len(deduplicate(data, ro_cutoff=0.01)) == 1


def test_deduplicate_e_cutoff_boundary():
    """Energy diff just below e_cutoff (kcal/mol) flags; at-or-above doesn't."""
    e_cutoff = 0.05  # kcal/mol
    e_cutoff_au = e_cutoff / KCAL_TO_AU
    data = {'a.log': _stub(-76.0, [10, 20, 30]),
            'b.log': _stub(-76.0 + 0.5 * e_cutoff_au, [10, 20, 30])}  # half cutoff
    assert len(deduplicate(data, e_cutoff=e_cutoff)) == 1
    # Just over the cutoff: shouldn't flag
    data['b.log'] = _stub(-76.0 + 1.5 * e_cutoff_au, [10, 20, 30])
    assert deduplicate(data, e_cutoff=e_cutoff) == []


def test_deduplicate_mismatched_roconst_length_skipped():
    """Different roconst lengths can't be compared → not flagged."""
    data = {'a.log': _stub(-76.0, [10, 20, 30]),  # 3 elements (nonlinear)
            'b.log': _stub(-76.0, [10])}          # 1 element (linear)
    assert deduplicate(data) == []


def test_deduplicate_all_zero_roconst_match():
    """Atoms / non-rotors have all-zero ro_const — should still match on energy."""
    data = {'a.log': _stub(-37.7, [0, 0, 0]),
            'b.log': _stub(-37.7, [0, 0, 0])}
    assert len(deduplicate(data)) == 1


def test_deduplicate_rmsd_cutoff_disabled_by_default():
    """rmsd_cutoff=None should skip the RMSD gate even if cartesians match."""
    # Stubs without cartesians: dedup still works because RMSD is skipped.
    data = {'a.log': _stub(-76.0, [10, 20, 30]),
            'b.log': _stub(-76.0, [10, 20, 30])}
    assert len(deduplicate(data, rmsd_cutoff=None)) == 1


def test_deduplicate_rmsd_cutoff_filters():
    """With rmsd_cutoff set, geometrically distinct structures are kept apart."""
    coords_a = _square().tolist()
    coords_b = (_square() + [10, 0, 0]).tolist()  # translated → kabsch_rmsd=0
    coords_c = _square()
    coords_c[2] += [2, 2, 0]                       # genuinely different
    data_close = {
        'a.log': _stub(-76.0, [10, 20, 30], coords_a),
        'b.log': _stub(-76.0, [10, 20, 30], coords_b),
    }
    assert len(deduplicate(data_close, rmsd_cutoff=0.1)) == 1
    data_far = {
        'a.log': _stub(-76.0, [10, 20, 30], coords_a),
        'c.log': _stub(-76.0, [10, 20, 30], coords_c.tolist()),
    }
    assert deduplicate(data_far, rmsd_cutoff=0.1) == []


def test_deduplicate_rmsd_cutoff_skipped_when_no_cartesians():
    """Files without cartesians under RMSD mode are quietly skipped — they
    can't be compared geometrically, so the pair is not flagged."""
    data = {'a.log': _stub(-76.0, [10, 20, 30], None),
            'b.log': _stub(-76.0, [10, 20, 30], None)}
    assert deduplicate(data, rmsd_cutoff=0.1) == []


def test_deduplicate_pair_count_n_choose_2():
    """All-identical structures → C(n, 2) duplicate pairs."""
    n = 4
    data = {f'{i}.log': _stub(-76.0, [10, 20, 30]) for i in range(n)}
    assert len(deduplicate(data)) == n * (n - 1) // 2


# ===========================================================================
# deduplicate — integration with real calc_bbe instances
# ===========================================================================

def test_deduplicate_real_water_self():
    """A real water calc_bbe vs. itself should flag as duplicate."""
    bbe = _bbe('01a_water_hf_freq.log')
    data = {'a.log': bbe, 'b.log': bbe}
    assert len(deduplicate(data)) == 1


def test_deduplicate_real_distinct_molecules():
    """Different molecules — water vs. ethane — should NOT flag."""
    data = {
        'water.log': _bbe('01a_water_hf_freq.log'),
        'ethane.log': _bbe('02_ethane_opt_freq_T398_P2.log'),
    }
    assert deduplicate(data) == []


# ===========================================================================
# sort_thermo
# ===========================================================================

def _energy_stub(scf_energy=None, qh_gibbs=None, linear_warning=False):
    ns = SimpleNamespace(linear_warning=linear_warning)
    if scf_energy is not None:
        ns.scf_energy = scf_energy
    if qh_gibbs is not None:
        ns.qh_gibbs_free_energy = qh_gibbs
    return ns


def test_sort_thermo_by_energy():
    data = {
        'b.log': _energy_stub(scf_energy=-2.0),
        'a.log': _energy_stub(scf_energy=-3.0),
        'c.log': _energy_stub(scf_energy=-1.0),
    }
    out = sort_thermo(data, 'energy')
    assert list(out) == ['a.log', 'b.log', 'c.log']


def test_sort_thermo_by_gibbs():
    data = {
        'a.log': _energy_stub(qh_gibbs=-1.5),
        'b.log': _energy_stub(qh_gibbs=-2.5),
        'c.log': _energy_stub(qh_gibbs=-2.0),
    }
    out = sort_thermo(data, 'gibbs')
    assert list(out) == ['b.log', 'c.log', 'a.log']


def test_sort_thermo_missing_attribute_goes_last():
    """Files without the requested attribute sort to the bottom."""
    data = {
        'no_gibbs.log': _energy_stub(scf_energy=-5.0),
        'has_gibbs.log': _energy_stub(qh_gibbs=-2.0),
    }
    out = sort_thermo(data, 'gibbs')
    assert list(out) == ['has_gibbs.log', 'no_gibbs.log']


def test_sort_thermo_none_value_goes_last():
    bbe1 = SimpleNamespace(scf_energy=None, linear_warning=False)
    bbe2 = SimpleNamespace(scf_energy=-3.0, linear_warning=False)
    out = sort_thermo({'a.log': bbe1, 'b.log': bbe2}, 'energy')
    assert list(out) == ['b.log', 'a.log']


def test_sort_thermo_linear_warning_goes_last():
    """linear_warning structures sort to the end even with a low energy."""
    data = {
        'flagged.log': _energy_stub(scf_energy=-100.0, linear_warning=True),
        'normal.log': _energy_stub(scf_energy=-50.0),
    }
    out = sort_thermo(data, 'energy')
    assert list(out) == ['normal.log', 'flagged.log']


def test_sort_thermo_preserves_data():
    """Sort returns a new dict with the same items (no values mutated)."""
    bbe = _energy_stub(scf_energy=-5.0)
    data = {'a.log': bbe}
    out = sort_thermo(data, 'energy')
    assert out['a.log'] is bbe


def test_sort_thermo_keys_constant():
    """SORT_KEYS exposes the canonical attribute names."""
    assert SORT_KEYS['energy'] == 'scf_energy'
    assert SORT_KEYS['gibbs'] == 'qh_gibbs_free_energy'


def test_sort_thermo_unknown_key_raises():
    """Unknown sort mode raises KeyError — caught upstream by the CLI."""
    with pytest.raises(KeyError):
        sort_thermo({}, 'not_a_key')


def test_sort_thermo_real_files():
    """End-to-end with real calc_bbe: water + ethane ordered by SCF energy
    (ethane is much more negative)."""
    data = {
        'water.log': _bbe('01a_water_hf_freq.log'),     # -76.01
        'ethane.log': _bbe('02_ethane_opt_freq_T398_P2.log'),   # -79.86
    }
    out = sort_thermo(data, 'energy')
    assert list(out) == ['ethane.log', 'water.log']
