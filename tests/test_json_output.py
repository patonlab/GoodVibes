"""Tests for the --json structured output flag.

Covers two layers:
  1. Direct API: write_json_results against an in-memory thermo_data dict.
  2. CLI integration: subprocess runs that emit JSON and round-trip it.
"""

import json
import math
import os
import subprocess
import sys

import pytest

from goodvibes.thermo import calc_bbe
from goodvibes.output import (
    write_json_results,
    JSON_SCHEMA_VERSION,
    _bbe_to_json,
    _options_to_json,
)
from conftest import g16path


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GAS_CONSTANT = 8.3144621
ATMOS = 101.325
T_DEFAULT = 298.15
CONC_DEFAULT = ATMOS / (GAS_CONSTANT * T_DEFAULT)


def _bbe(filename):
    """Run calc_bbe with sensible defaults."""
    return calc_bbe(g16path(filename), 'grimme', False, 100.0, 100.0,
                    T_DEFAULT, CONC_DEFAULT, 1.0, None, None, None, 0)


# ---------------------------------------------------------------------------
# Direct API: shape checks
# ---------------------------------------------------------------------------

class _Opts:
    """Tiny stand-in for an argparse Namespace."""
    def __init__(self, **kw):
        # Defaults match what parse_arguments would set for a vanilla run.
        defaults = {
            'temperature': 298.15, 'temperature_interval': None, 'conc': None,
            'QS': 'grimme', 'QH': False, 'freq_cutoff': 100.0,
            'S_freq_cutoff': 100.0, 'H_freq_cutoff': 100.0,
            'freq_scale_factor': 1.0, 'media': None, 'freespace': None,
            'spc': None, 'invert': None, 'symm': False, 'duplicate': False,
            'boltz': None, 'inertia': 'global', 'mm_freq_scale_factor': None,
        }
        defaults.update(kw)
        for k, v in defaults.items():
            setattr(self, k, v)


def _calc_water():
    """A standalone calc_bbe for the H2O fixture."""
    return _bbe('01a_water_hf_freq.log')


def test_options_to_json_subset():
    """The serializer projects options down to the documented JSON keys
    only — random extra attributes (like --output) don't leak through."""
    opts = _Opts(media='h2o', boltz='gibbs')
    opts.output = 'should_not_appear'
    opts.custom_ext = 'should_not_appear'
    out = _options_to_json(opts)
    assert out['media'] == 'h2o'
    assert out['boltz'] == 'gibbs'
    assert out['QS'] == 'grimme'
    assert 'output' not in out
    assert 'custom_ext' not in out


def test_bbe_to_json_includes_qcdata_and_thermo():
    bbe = _calc_water()
    entry = _bbe_to_json(bbe, g16path('01a_water_hf_freq.log'))
    assert entry['name'] == '01a_water_hf_freq'
    assert os.path.isabs(entry['file'])
    assert 'qcdata' in entry
    # qcdata round-trips without the cache_version marker
    assert '_cache_version' not in entry['qcdata']
    assert entry['qcdata']['program'] == 'Gaussian'
    # Thermo numbers match calc_bbe's attributes
    assert entry['thermo']['scf_energy'] == bbe.scf_energy
    assert entry['thermo']['zpe'] == bbe.zpe
    assert entry['thermo']['gibbs_free_energy'] == bbe.gibbs_free_energy
    # Optional auxiliaries default to None
    assert entry['media_conc'] is None
    assert entry['boltzmann_factor'] is None


def test_bbe_to_json_passes_through_aux_fields():
    bbe = _calc_water()
    entry = _bbe_to_json(bbe, g16path('01a_water_hf_freq.log'),
                         media_conc=55.34, boltz_factor=0.5)
    assert entry['media_conc'] == 55.34
    assert entry['boltzmann_factor'] == 0.5


def test_write_json_results_payload(tmp_path):
    bbe = _calc_water()
    thermo_data = {g16path('01a_water_hf_freq.log'): bbe}
    out = tmp_path / "out.json"
    write_json_results(thermo_data, _Opts(), str(out))

    payload = json.loads(out.read_text())
    assert payload['schema_version'] == JSON_SCHEMA_VERSION
    assert payload['goodvibes_version']
    assert 'generated_at' in payload
    assert payload['options']['QS'] == 'grimme'
    assert len(payload['results']) == 1
    r = payload['results'][0]
    assert r['name'] == '01a_water_hf_freq'
    assert math.isclose(r['thermo']['scf_energy'], -76.010511, abs_tol=1e-5)


def test_write_json_results_preserves_options(tmp_path):
    bbe = _calc_water()
    thermo_data = {g16path('01a_water_hf_freq.log'): bbe}
    out = tmp_path / "out.json"
    write_json_results(thermo_data, _Opts(temperature=400.0, QH=True,
                                          media='h2o'), str(out))
    payload = json.loads(out.read_text())
    assert payload['options']['temperature'] == 400.0
    assert payload['options']['QH'] is True
    assert payload['options']['media'] == 'h2o'


def test_write_json_results_handles_sp_only(tmp_path):
    """Single-point-only files have no Gibbs free energy. The thermo block
    should still serialize, with None for fields calc_bbe didn't compute."""
    bbe = _bbe('06_carbon_atom_single_point.log')
    thermo_data = {g16path('06_carbon_atom_single_point.log'): bbe}
    out = tmp_path / "out.json"
    write_json_results(thermo_data, _Opts(), str(out))
    payload = json.loads(out.read_text())
    r = payload['results'][0]
    assert r['thermo']['scf_energy'] is not None
    # SP-only: gibbs / qh_gibbs / enthalpy / zpe should all be None
    assert r['thermo']['gibbs_free_energy'] is None
    assert r['thermo']['qh_gibbs_free_energy'] is None
    assert r['thermo']['enthalpy'] is None
    assert r['thermo']['zpe'] is None


def test_write_json_results_per_file_aux_dicts(tmp_path):
    """media_conc_per_file and boltz_facs are routed by file path."""
    bbe = _calc_water()
    fpath = g16path('01a_water_hf_freq.log')
    thermo_data = {fpath: bbe}
    out = tmp_path / "out.json"
    write_json_results(thermo_data, _Opts(),
                       str(out),
                       media_conc_per_file={fpath: 55.4},
                       boltz_facs={fpath: 0.123})
    r = json.loads(out.read_text())['results'][0]
    assert r['media_conc'] == 55.4
    assert r['boltzmann_factor'] == 0.123


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def _run_cli(args, cwd):
    cmd = [sys.executable, '-m', 'goodvibes'] + list(args)
    env = {**os.environ, 'PYTHONPATH': ROOT_DIR}
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, env=env)


def test_cli_json_flag_creates_file(tmp_path):
    """`goodvibes --json out.json file.log` writes a parseable JSON file."""
    out = tmp_path / "results.json"
    res = _run_cli([g16path('01a_water_hf_freq.log'), '--json', str(out),
                    '--output', 'jsontest'], cwd=tmp_path)
    assert res.returncode == 0, f"stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload['schema_version'] == JSON_SCHEMA_VERSION
    assert len(payload['results']) == 1
    assert math.isclose(payload['results'][0]['thermo']['scf_energy'],
                        -76.010511, abs_tol=1e-5)


def test_cli_json_flag_announces_output(tmp_path):
    """The CLI prints a confirmation line so users know the file landed."""
    out = tmp_path / "results.json"
    res = _run_cli([g16path('01a_water_hf_freq.log'), '--json', str(out),
                    '--output', 'jsontest'], cwd=tmp_path)
    assert 'Structured results written' in res.stdout
    assert f'schema v{JSON_SCHEMA_VERSION}' in res.stdout


def test_cli_json_multiple_files(tmp_path):
    out = tmp_path / "results.json"
    res = _run_cli([g16path('01a_water_hf_freq.log'),
                    g16path('05_methylene_triplet_carbene.log'),
                    g16path('22_hcn_linear_freq_noraman.log'),
                    '--json', str(out), '--output', 'jsontest'],
                   cwd=tmp_path)
    assert res.returncode == 0, f"stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    payload = json.loads(out.read_text())
    assert len(payload['results']) == 3
    names = {r['name'] for r in payload['results']}
    assert '01a_water_hf_freq' in names
    assert '05_methylene_triplet_carbene' in names
    assert '22_hcn_linear_freq_noraman' in names


def test_cli_json_with_media_records_concentration(tmp_path):
    """When --media matches the file, the per-file media_conc is recorded."""
    media_log = os.path.join(ROOT_DIR, 'goodvibes', 'examples',
                             'media_conc', 'H2O.log')
    out = tmp_path / "results.json"
    res = _run_cli([media_log, '--media', 'h2o', '--json', str(out),
                    '--output', 'jsontest'], cwd=tmp_path)
    assert res.returncode == 0, f"stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    r = json.loads(out.read_text())['results'][0]
    assert r['media_conc'] is not None
    assert math.isclose(r['media_conc'], 55.38, abs_tol=0.05)


@pytest.mark.parametrize("file", [
    '01a_water_hf_freq.log',
    '02_ethane_opt_freq_T398_P2.log',
    '08_alanine_C1_pcm_water.log',
    '22_hcn_linear_freq_noraman.log',
    '44_ts_sn2_identity_chloride.log',
])
def test_cli_json_thermo_matches_calc_bbe(tmp_path, file):
    """End-to-end: thermo numbers in the JSON match an independent calc_bbe
    invocation against the same file. Pass --vscal 1.0 so the CLI uses the
    same (unscaled) frequencies as our direct _bbe helper — otherwise the
    CLI auto-resolves a level-of-theory-specific scale factor."""
    out = tmp_path / "results.json"
    res = _run_cli([g16path(file), '--vscal', '1.0', '--json', str(out),
                    '--output', 'jsontest'], cwd=tmp_path)
    assert res.returncode == 0
    payload = json.loads(out.read_text())
    r = payload['results'][0]
    expected = _bbe(file)
    assert math.isclose(r['thermo']['scf_energy'], expected.scf_energy,
                        abs_tol=1e-7)
    if hasattr(expected, 'gibbs_free_energy'):
        assert math.isclose(r['thermo']['zpe'], expected.zpe, abs_tol=1e-7)
        assert math.isclose(r['thermo']['gibbs_free_energy'],
                            expected.gibbs_free_energy, abs_tol=1e-7)
