"""Tests for the new --label / --selectivity API and the legacy --ee shim.

Three layers:
  1. Direct API: compute_selectivity, compute_selectivity_scan, the
     fnmatch-based bucket assignment, and label-spec parsing.
  2. Legacy --ee deprecation shim: still works, emits DeprecationWarning,
     and reproduces the old 6-tuple shape.
  3. CLI integration: subprocess runs of --label / --selectivity / --ee,
     plus the v0.2 JSON output's selectivity block.
"""

import json
import math
import os
import subprocess
import sys
import warnings
from types import SimpleNamespace

import pytest

from goodvibes.selectivity import (
    SelectivityResult, parse_label_args, load_label_yaml,
    assign_files_to_labels, compute_selectivity, compute_selectivity_scan,
    compute_selectivity_lowest_only, compute_selectivity_lowest_only_scan,
    get_selectivity, get_boltz,
)


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Diels–Alder exo/endo × 1,2- vs 1,4- TS set; one (DA_exo_14_ii) errored
# during the original calculation and is filtered by the orchestrator.
SEL_DIR = os.path.join(ROOT_DIR, 'goodvibes', 'examples', 'selectivity')


def _stub(g_value, scf=None):
    """Minimal calc_bbe stand-in: gives a Gibbs and (optionally) SCF energy."""
    return SimpleNamespace(
        qh_gibbs_free_energy=g_value,
        scf_energy=scf if scf is not None else g_value,
    )


# ---------------------------------------------------------------------------
# parse_label_args
# ---------------------------------------------------------------------------

def test_parse_label_args_basic():
    spec = parse_label_args(['R=*_R_*', 'S=*_S_*'])
    assert spec == {'R': '*_R_*', 'S': '*_S_*'}


def test_parse_label_args_strips_whitespace():
    spec = parse_label_args(['R = *_R_* ', '  S = *_S_*'])
    assert spec == {'R': '*_R_*', 'S': '*_S_*'}


def test_parse_label_args_missing_equals_raises():
    with pytest.raises(ValueError, match="NAME=PATTERN"):
        parse_label_args(['justaname'])


def test_parse_label_args_empty_side_raises():
    with pytest.raises(ValueError, match="non-empty"):
        parse_label_args(['=patternonly'])
    with pytest.raises(ValueError, match="non-empty"):
        parse_label_args(['nameonly='])


def test_parse_label_args_duplicate_raises():
    with pytest.raises(ValueError, match="Duplicate"):
        parse_label_args(['R=a', 'R=b'])


def test_parse_label_args_none_returns_empty():
    assert parse_label_args(None) == {}
    assert parse_label_args([]) == {}


def test_parse_label_args_preserves_order():
    """Insertion order is preserved — important for ratio printing."""
    spec = parse_label_args(['SS=*_SS_*', 'RR=*_RR_*', 'RS=*_RS_*'])
    assert list(spec) == ['SS', 'RR', 'RS']


# ---------------------------------------------------------------------------
# assign_files_to_labels
# ---------------------------------------------------------------------------

def test_assign_files_basic():
    files = ['/x/P_R_a.log', '/x/P_R_b.log', '/x/P_S_a.log']
    out = assign_files_to_labels(files, {'R': '*P_R_*', 'S': '*P_S_*'})
    assert out == {
        'R': ['/x/P_R_a.log', '/x/P_R_b.log'],
        'S': ['/x/P_S_a.log'],
    }


def test_assign_files_first_match_wins():
    """Files that match multiple patterns go to the first label in spec order."""
    files = ['/x/P_R_S_a.log']  # matches both '*P_R_*' and '*P_S_*'
    out = assign_files_to_labels(files, {'R': '*P_R_*', 'S': '*P_S_*'})
    assert out == {'R': ['/x/P_R_S_a.log'], 'S': []}


def test_assign_files_unmatched_dropped():
    """Files that match no pattern are silently excluded — they're not
    selectivity inputs."""
    files = ['/x/P_R_a.log', '/x/random.log']
    out = assign_files_to_labels(files, {'R': '*P_R_*', 'S': '*P_S_*'})
    assert out == {'R': ['/x/P_R_a.log'], 'S': []}


def test_assign_files_uses_basename_only():
    """A pattern matches the basename, not the full path. So a pattern of
    'tests/*.log' would NOT match '/abs/tests/foo.log' — only 'foo.log' is
    compared."""
    files = ['/abs/tests/foo.log']
    out = assign_files_to_labels(files, {'A': 'foo.log'})
    assert out == {'A': ['/abs/tests/foo.log']}


# ---------------------------------------------------------------------------
# compute_selectivity (synthetic data)
# ---------------------------------------------------------------------------

def test_compute_selectivity_two_label_basic():
    """3 R-conformers + 2 S-conformers; R is more stable on average."""
    thermo = {
        'R_1.log': _stub(-100.000),
        'R_2.log': _stub(-99.999),
        'R_3.log': _stub(-99.998),
        'S_1.log': _stub(-99.997),
        'S_2.log': _stub(-99.996),
    }
    files_per_label = {
        'R': ['R_1.log', 'R_2.log', 'R_3.log'],
        'S': ['S_1.log', 'S_2.log'],
    }
    result = compute_selectivity(thermo, files_per_label, 298.15)

    assert isinstance(result, SelectivityResult)
    assert result.temperature == 298.15
    assert result.labels == ['R', 'S']
    assert result.preferred == 'R'
    # Populations sum to 1.0 (within float tolerance)
    assert abs(sum(result.populations.values()) - 1.0) < 1e-12
    # ee = 100 * |p_a - p_b|
    assert math.isclose(
        result.ee, abs(result.populations['R'] - result.populations['S']) * 100,
        abs_tol=1e-9,
    )
    # ΔΔG = RT ln(p_major / p_minor), positive (favorable for major)
    assert result.ddG > 0


def test_compute_selectivity_n_way():
    thermo = {
        'P_RR_1.log': _stub(-100.000),
        'P_RS_1.log': _stub(-99.998),
        'P_SR_1.log': _stub(-99.997),
        'P_SS_1.log': _stub(-99.995),
    }
    spec = parse_label_args(['RR=*RR_*', 'RS=*RS_*', 'SR=*SR_*', 'SS=*SS_*'])
    files_per_label = assign_files_to_labels(list(thermo), spec)
    result = compute_selectivity(thermo, files_per_label, 298.15)

    assert result.labels == ['RR', 'RS', 'SR', 'SS']
    # N>2: ee and ddG are None
    assert result.ee is None
    assert result.ddG is None
    # Most stable RR is preferred, populations descending
    assert result.preferred == 'RR'
    assert (result.populations['RR'] > result.populations['RS']
            > result.populations['SR'] > result.populations['SS'])


def test_compute_selectivity_populations_normalized():
    thermo = {f'a_{i}.log': _stub(-100.0 - 0.001 * i) for i in range(5)}
    files_per_label = {'A': list(thermo)[:3], 'B': list(thermo)[3:]}
    result = compute_selectivity(thermo, files_per_label, 298.15)
    assert abs(sum(result.populations.values()) - 1.0) < 1e-12


def test_compute_selectivity_two_label_ddG_matches_RT_ln_ratio():
    """ΔΔG = RT ln(p_major / p_minor) — verify against direct math."""
    thermo = {'A.log': _stub(-100.000), 'B.log': _stub(-99.998)}
    R = compute_selectivity(thermo, {'A': ['A.log'], 'B': ['B.log']}, 298.15)
    pa, pb = R.populations['A'], R.populations['B']
    GAS_CONSTANT = 8.3144621
    J_TO_AU = 4.184 * 627.509541 * 1000.0
    expected_ddG = GAS_CONSTANT * 298.15 * math.log(max(pa, pb) / min(pa, pb)) / J_TO_AU
    assert math.isclose(R.ddG, expected_ddG, rel_tol=1e-12)


def test_compute_selectivity_files_per_label_isolated():
    """The returned dict is a copy, not a reference; mutating it shouldn't
    affect a re-run."""
    thermo = {'A.log': _stub(-100.0), 'B.log': _stub(-99.999)}
    spec = {'A': ['A.log'], 'B': ['B.log']}
    R = compute_selectivity(thermo, spec, 298.15)
    R.files_per_label['A'].append('hacked.log')
    assert spec['A'] == ['A.log']  # unmodified


def test_compute_selectivity_dup_list_semantics():
    """dup_list pairs are [duplicate, canonical]: only the FIRST element
    is excluded. The canonical (kept) structure stays in the sum.
    """
    thermo = {
        'A1.log': _stub(-100.0),
        'A2.log': _stub(-100.0),  # canonical form of A
        'B1.log': _stub(-99.99),
    }
    files_per_label = {'A': ['A1.log', 'A2.log'], 'B': ['B1.log']}
    # Pair [A1, A2] means A1 is the duplicate; A2 is kept.
    with_dups = compute_selectivity(thermo, files_per_label, 298.15,
                                    dup_list=[['A1.log', 'A2.log']])
    # Equivalent to dropping A1 from the bucket entirely:
    without_a1 = compute_selectivity(
        thermo, {'A': ['A2.log'], 'B': ['B1.log']}, 298.15)
    assert with_dups.raw_boltzmann['A'] == pytest.approx(
        without_a1.raw_boltzmann['A'])
    assert with_dups.populations['A'] == pytest.approx(
        without_a1.populations['A'])


def test_compute_selectivity_empty_label_raises():
    thermo = {'A.log': _stub(-100.0)}
    files_per_label = {'A': ['A.log'], 'B': []}
    with pytest.raises(ValueError, match="No files matched"):
        compute_selectivity(thermo, files_per_label, 298.15)


def test_compute_selectivity_too_few_labels_raises():
    with pytest.raises(ValueError, match="at least two"):
        compute_selectivity({}, {'only': []}, 298.15)


def test_compute_selectivity_no_usable_energy_raises():
    """All files in a label have None for the selected energy attr → error."""
    bad = SimpleNamespace(qh_gibbs_free_energy=None)
    thermo = {'a.log': bad, 'b.log': bad}
    with pytest.raises(ValueError, match="usable energy attribute"):
        compute_selectivity(thermo, {'A': ['a.log'], 'B': ['b.log']}, 298.15)


def test_compute_selectivity_key_energy_uses_scf():
    """key='energy' selects scf_energy instead of qh_gibbs_free_energy."""
    thermo = {
        'A.log': SimpleNamespace(qh_gibbs_free_energy=-1, scf_energy=-100.0),
        'B.log': SimpleNamespace(qh_gibbs_free_energy=-1, scf_energy=-99.9),
    }
    R = compute_selectivity(thermo, {'A': ['A.log'], 'B': ['B.log']}, 298.15,
                            key='energy')
    assert R.preferred == 'A'  # A has lower scf_energy


# ---------------------------------------------------------------------------
# compute_selectivity_scan
# ---------------------------------------------------------------------------

def test_compute_selectivity_scan_one_per_temperature():
    thermo = {'A.log': _stub(-100.0), 'B.log': _stub(-99.998)}
    spec = {'A': ['A.log'], 'B': ['B.log']}
    results = compute_selectivity_scan(thermo, spec, [200, 298.15, 400])
    assert [r.temperature for r in results] == [200, 298.15, 400]
    # Selectivity should fall as T rises (entropy-equalizing)
    ees = [r.ee for r in results]
    assert ees[0] > ees[1] > ees[2]


def test_compute_selectivity_scan_preserves_label_order():
    thermo = {'X.log': _stub(-100.0), 'Y.log': _stub(-99.999)}
    spec = {'Y': ['Y.log'], 'X': ['X.log']}  # deliberately reversed
    results = compute_selectivity_scan(thermo, spec, [298.15])
    assert results[0].labels == ['Y', 'X']


# ---------------------------------------------------------------------------
# compute_selectivity_lowest_only
# ---------------------------------------------------------------------------

def test_lowest_only_picks_min_energy_per_species():
    """Each species reduces to its single lowest-energy file before the
    Boltzmann calc."""
    thermo = {
        'R_1.log': _stub(-100.000),  # lowest in R
        'R_2.log': _stub(-99.998),
        'R_3.log': _stub(-99.996),
        'S_1.log': _stub(-99.999),  # lowest in S
        'S_2.log': _stub(-99.995),
    }
    spec = {'R': ['R_1.log', 'R_2.log', 'R_3.log'],
            'S': ['S_1.log', 'S_2.log']}
    result = compute_selectivity_lowest_only(thermo, spec, 298.15)
    # Only the lowest-energy file from each species is retained:
    assert result.files_per_label == {'R': ['R_1.log'], 'S': ['S_1.log']}
    # Standard Boltzmann math on those two files:
    direct = compute_selectivity(
        thermo, {'R': ['R_1.log'], 'S': ['S_1.log']}, 298.15)
    assert result.populations == direct.populations
    assert result.ee == direct.ee
    assert result.ddG == direct.ddG


def test_lowest_only_diverges_from_boltzmann_when_extras_help():
    """Boltzmann-averaged ee exceeds lowest-only ee when the dominant
    species has more conformers contributing — the conformational
    'mixing entropy' boosts its population."""
    thermo = {
        'R_1.log': _stub(-100.000),
        'R_2.log': _stub(-99.9990),    # ~0.6 kcal/mol higher
        'R_3.log': _stub(-99.9985),
        'R_4.log': _stub(-99.9980),
        'S_1.log': _stub(-99.999),    # only one S conformer near the floor
    }
    spec = {'R': ['R_1.log', 'R_2.log', 'R_3.log', 'R_4.log'],
            'S': ['S_1.log']}
    bw = compute_selectivity(thermo, spec, 298.15)
    lo = compute_selectivity_lowest_only(thermo, spec, 298.15)
    # Both prefer R because R_1 is the most stable individual conformer
    assert bw.preferred == 'R' and lo.preferred == 'R'
    # Boltzmann-averaged ee > lowest-only ee — the extra R conformers
    # tilt the BW selection further toward R.
    assert bw.ee > lo.ee


def test_lowest_only_matches_boltzmann_when_one_conformer_per_species():
    """Lowest-only and Boltzmann are identical when each species already
    has only one conformer."""
    thermo = {'R.log': _stub(-100.0), 'S.log': _stub(-99.999)}
    spec = {'R': ['R.log'], 'S': ['S.log']}
    bw = compute_selectivity(thermo, spec, 298.15)
    lo = compute_selectivity_lowest_only(thermo, spec, 298.15)
    assert bw.populations == lo.populations
    assert bw.ee == lo.ee
    assert bw.ddG == pytest.approx(lo.ddG)


def test_lowest_only_files_per_label_is_single_file():
    thermo = {f'a_{i}.log': _stub(-100.0 - 0.001 * i) for i in range(5)}
    spec = {'A': list(thermo)[:3], 'B': list(thermo)[3:]}
    result = compute_selectivity_lowest_only(thermo, spec, 298.15)
    for label in result.labels:
        assert len(result.files_per_label[label]) == 1


def test_lowest_only_excludes_dups():
    """If the lowest-energy file in a species is in dup_list[i][0], it's
    skipped and the next-lowest is selected instead."""
    thermo = {
        'A_dup.log': _stub(-100.0),    # would be picked, but excluded
        'A_canonical.log': _stub(-99.9999),
        'B.log': _stub(-99.99),
    }
    spec = {'A': ['A_dup.log', 'A_canonical.log'], 'B': ['B.log']}
    dups = [['A_dup.log', 'A_canonical.log']]
    result = compute_selectivity_lowest_only(thermo, spec, 298.15,
                                              dup_list=dups)
    assert result.files_per_label == {'A': ['A_canonical.log'], 'B': ['B.log']}


def test_lowest_only_empty_species_after_exclusion_raises():
    """If every conformer in a species is excluded as a duplicate, that
    species ends up empty and compute_selectivity raises ValueError."""
    thermo = {
        'A_only.log': _stub(-100.0),  # the only A; will be excluded
        'B.log': _stub(-99.99),
    }
    spec = {'A': ['A_only.log'], 'B': ['B.log']}
    dups = [['A_only.log', 'B.log']]
    with pytest.raises(ValueError, match="No files matched"):
        compute_selectivity_lowest_only(thermo, spec, 298.15, dup_list=dups)


def test_lowest_only_scan_one_per_temperature():
    thermo = {'R_1.log': _stub(-100.0), 'R_2.log': _stub(-99.998),
              'S_1.log': _stub(-99.999)}
    spec = {'R': ['R_1.log', 'R_2.log'], 'S': ['S_1.log']}
    results = compute_selectivity_lowest_only_scan(
        thermo, spec, [200, 298.15, 400])
    assert [r.temperature for r in results] == [200, 298.15, 400]
    # Each temperature uses the same single lowest conformer per species
    for r in results:
        assert r.files_per_label == {'R': ['R_1.log'], 'S': ['S_1.log']}


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def test_load_label_yaml_patterns_mode(tmp_path):
    yaml_path = tmp_path / 'spec.yaml'
    yaml_path.write_text("labels:\n  R: '*P_R_*'\n  S: '*P_S_*'\n")
    mode, data = load_label_yaml(str(yaml_path))
    assert mode == 'patterns'
    assert data == {'R': '*P_R_*', 'S': '*P_S_*'}


def test_load_label_yaml_files_mode(tmp_path):
    yaml_path = tmp_path / 'spec.yaml'
    yaml_path.write_text(
        "files:\n  R:\n    - a.log\n    - b.log\n  S:\n    - c.log\n"
    )
    mode, data = load_label_yaml(str(yaml_path))
    assert mode == 'files'
    assert data == {'R': ['a.log', 'b.log'], 'S': ['c.log']}


def test_load_label_yaml_both_keys_raises(tmp_path):
    yaml_path = tmp_path / 'spec.yaml'
    yaml_path.write_text("labels:\n  R: a\nfiles:\n  R: [a.log]\n")
    with pytest.raises(ValueError, match="not both"):
        load_label_yaml(str(yaml_path))


def test_load_label_yaml_empty_raises(tmp_path):
    yaml_path = tmp_path / 'spec.yaml'
    yaml_path.write_text("temperature: 298\n")
    with pytest.raises(ValueError, match="'labels' or 'files'"):
        load_label_yaml(str(yaml_path))


# ---------------------------------------------------------------------------
# Legacy --ee shim deprecation
# ---------------------------------------------------------------------------

def test_get_selectivity_emits_deprecation_warning():
    """get_selectivity (the --ee back-end) warns but still returns the
    legacy 6-tuple shape so the existing CLI print path keeps working."""
    files = [
        os.path.join(SEL_DIR, 'DA_exo_12_i.out'),
        os.path.join(SEL_DIR, 'DA_endo_12_i.out'),
    ]
    boltz_facs = {files[0]: 0.4, files[1]: 0.6}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = get_selectivity('*exo*:*endo*', files, boltz_facs,
                                 298.15, [])
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert len(result) == 6  # (ee, er, ratio, ddG, failed, pref)


def test_get_selectivity_invalid_pattern_raises():
    with pytest.raises(ValueError, match="exactly one colon"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            get_selectivity('only_one_pattern', [], {}, 298.15, [])


# ---------------------------------------------------------------------------
# get_boltz (still used by --boltz)
# ---------------------------------------------------------------------------

def test_get_boltz_normalizes():
    thermo = {'a.log': _stub(-100.0), 'b.log': _stub(-99.999)}
    facs = get_boltz(thermo, 298.15, dup_list=[])
    assert abs(sum(facs.values()) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def _run_cli(args, cwd):
    cmd = [sys.executable, '-m', 'goodvibes'] + list(args)
    env = {**os.environ, 'PYTHONPATH': ROOT_DIR}
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)


def _all_files():
    """Diels–Alder exo/endo × 1,2- vs 1,4- fixture set. The orchestrator
    drops DA_exo_14_ii.out (Error termination) automatically, so the
    contributing set is 7 files: 3 exo + 4 endo."""
    import glob
    return sorted(glob.glob(os.path.join(SEL_DIR, 'DA_*.out')))


def test_cli_label_two_species(tmp_path):
    """Two --label flags produce TWO Selectivity tables in the output:
    one Boltzmann-averaged, one lowest-conformer-only."""
    res = _run_cli(_all_files() + [
        '--label', 'exo=*_exo_*',
        '--label', 'endo=*_endo_*',
        '--vscal', '1.0', '--output', 'cli2',
    ], cwd=tmp_path)
    assert res.returncode == 0, f"stderr:\n{res.stderr}"
    assert 'Boltzmann-averaged' in res.stdout
    assert 'Lowest conformer only' in res.stdout
    assert res.stdout.count('Species') >= 2          # one Species header per table
    assert res.stdout.count('Ratio exo:endo') >= 2
    assert res.stdout.count('excess =') >= 2
    assert res.stdout.count('ΔΔG =') >= 2


def test_cli_label_n_way_no_ee_line(tmp_path):
    """N=4 selectivity prints ratio but no excess or ΔΔG summary."""
    res = _run_cli(_all_files() + [
        '--label', 'exo_12=*_exo_12*',
        '--label', 'endo_12=*_endo_12*',
        '--label', 'exo_14=*_exo_14*',
        '--label', 'endo_14=*_endo_14*',
        '--vscal', '1.0', '--output', 'clin',
    ], cwd=tmp_path)
    assert res.returncode == 0, f"stderr:\n{res.stderr}"
    sel_idx = res.stdout.rfind('Ratio exo_12:endo_12:exo_14:endo_14')
    assert sel_idx > 0
    summary_line = res.stdout[sel_idx: sel_idx + 200]
    assert 'excess =' not in summary_line


def test_cli_label_unknown_arg_format_fails(tmp_path):
    res = _run_cli(_all_files()[:2] + [
        '--label', 'no_equals_sign',
        '--output', 'clibad',
    ], cwd=tmp_path)
    assert res.returncode != 0
    assert 'FATAL ERROR' in (res.stdout + res.stderr)


def test_cli_label_and_selectivity_mutually_exclusive(tmp_path):
    yaml_path = tmp_path / 'spec.yaml'
    yaml_path.write_text("labels:\n  exo: '*_exo_*'\n  endo: '*_endo_*'\n")
    res = _run_cli(_all_files()[:2] + [
        '--label', 'exo=*_exo_*',
        '--selectivity', str(yaml_path),
        '--output', 'cliexcl',
    ], cwd=tmp_path)
    assert res.returncode != 0
    assert 'mutually exclusive' in (res.stdout + res.stderr)


def test_cli_label_and_ee_mutually_exclusive(tmp_path):
    res = _run_cli(_all_files()[:2] + [
        '--label', 'exo=*_exo_*', '--ee', '*_exo_*:*_endo_*',
        '--output', 'cliexclee',
    ], cwd=tmp_path)
    assert res.returncode != 0
    assert 'incompatible' in (res.stdout + res.stderr)


def test_cli_selectivity_yaml(tmp_path):
    """--selectivity FILE.yaml works as an alternative to --label."""
    yaml_path = tmp_path / 'spec.yaml'
    yaml_path.write_text("labels:\n  exo: '*_exo_*'\n  endo: '*_endo_*'\n")
    res = _run_cli(_all_files() + [
        '--selectivity', str(yaml_path),
        '--vscal', '1.0', '--output', 'cliyaml',
    ], cwd=tmp_path)
    assert res.returncode == 0, f"stderr:\n{res.stderr}"
    assert 'Boltzmann-averaged' in res.stdout
    assert 'Lowest conformer only' in res.stdout
    assert 'Ratio exo:endo' in res.stdout


def test_cli_temperature_scan(tmp_path):
    """--label combined with --ti emits one row per temperature."""
    res = _run_cli(_all_files() + [
        '--label', 'exo=*_exo_*',
        '--label', 'endo=*_endo_*',
        '--ti', '200,400,100',
        '--vscal', '1.0', '--output', 'cliscan',
    ], cwd=tmp_path)
    assert res.returncode == 0, f"stderr:\n{res.stderr}"
    assert 'Selectivity scan' in res.stdout
    for T in ['200', '300', '400']:
        assert T in res.stdout


def test_cli_json_includes_selectivity(tmp_path):
    out = tmp_path / 'results.json'
    res = _run_cli(_all_files() + [
        '--label', 'exo=*_exo_*',
        '--label', 'endo=*_endo_*',
        '--json', str(out),
        '--vscal', '1.0', '--output', 'clijson',
    ], cwd=tmp_path)
    assert res.returncode == 0, f"stderr:\n{res.stderr}"
    payload = json.loads(out.read_text())
    assert payload['schema_version'] == '1.0'
    assert 'selectivity' in payload
    assert 'selectivity_lowest' in payload
    sel = payload['selectivity']
    sel_lo = payload['selectivity_lowest']
    assert sel['labels'] == ['exo', 'endo']
    assert sel_lo['labels'] == ['exo', 'endo']
    r = sel['results'][0]
    r_lo = sel_lo['results'][0]
    assert math.isclose(r['temperature'], 298.15)
    assert math.isclose(r_lo['temperature'], 298.15)
    # Lowest-only: each species reduces to a single file
    for label in sel_lo['labels']:
        assert len(r_lo['files_per_label'][label]) == 1
    # Both methods should agree on the major species for this fixture
    assert r['preferred'] == r_lo['preferred']
    # Populations still sum to 1 in both methods
    assert abs(sum(r['populations'].values()) - 1.0) < 1e-9
    assert abs(sum(r_lo['populations'].values()) - 1.0) < 1e-9


def test_cli_ee_still_runs(tmp_path):
    """Legacy --ee still parses and runs without error."""
    res = _run_cli(_all_files() + [
        '--ee', '*_exo_*:*_endo_*',
        '--vscal', '1.0', '--output', 'cliee',
    ], cwd=tmp_path)
    assert res.returncode == 0, f"stderr:\n{res.stderr}"
