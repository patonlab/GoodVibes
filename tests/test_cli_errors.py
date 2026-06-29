"""
M0 safety-net tests (AUDIT.md tasks 0.2, 0.3, 0.4).

Three sections:

1. CLI error paths via subprocess — pin exit codes and messages for the
   fatal paths a user can hit from the command line.
2. ``sys.exit`` characterization — direct calls pinning the current
   exit-on-error behavior of library functions (validation, selectivity,
   output). These tests intentionally assert today's behavior; when task
   2.1 converts these sites to raise exceptions, update them to expect
   the new exception types.
3. Direct ``main()`` invocations — exercise flag interactions in-process
   so coverage instrumentation sees GoodVibes.py (subprocess runs don't
   count toward coverage).
"""

import json
import logging
import os
import subprocess
import sys

import pytest

from conftest import g16path
from goodvibes import GoodVibes as GV


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WATER = g16path('01a_water_hf_freq.log')
ETHANE = os.path.join(ROOT_DIR, 'goodvibes', 'examples', 'ethane.out')
ERR_FILE = g16path('51_err_scf_convergence_fe_complex.log')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cli(args, cwd):
    """Run ``python -m goodvibes <args>`` and return the CompletedProcess."""
    cmd = [sys.executable, '-m', 'goodvibes'] + args
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)


@pytest.fixture
def gv_logger_cleanup():
    """Reset the 'goodvibes' logger after a direct main() call.

    setup_logging() adds handlers without removing existing ones, so
    repeated in-process main() calls would duplicate output and leak the
    .dat file handle. Snapshot and restore around each test.
    """
    logger = logging.getLogger('goodvibes')
    before = list(logger.handlers)
    yield
    for handler in list(logger.handlers):
        if handler not in before:
            logger.removeHandler(handler)
            handler.close()


def run_main(monkeypatch, tmp_path, args, gv_logger_cleanup=None):
    """Invoke GV.main() in-process with argv set, cwd'd to tmp_path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['goodvibes'] + args)
    GV.main()
    return tmp_path


def build_options(monkeypatch, files=None, extra=None):
    """Build a real (options, files) pair via parse_arguments().

    Note parse_arguments() itself collects the file list from argv and
    sys.exit()s on a missing --spc partner (GoodVibes.py:250), so the
    files placed in argv must be consistent with the flags.
    """
    argv = ['goodvibes'] + (files if files is not None else [WATER]) + (extra or [])
    monkeypatch.setattr(sys, 'argv', argv)
    return GV.parse_arguments()


# ---------------------------------------------------------------------------
# 1. CLI error paths (task 0.2)
# ---------------------------------------------------------------------------

class TestCLIErrorPaths:

    def test_nonexistent_input_file_exits_1(self, tmp_path):
        result = run_cli(['no_such_file.log'], cwd=tmp_path)
        assert result.returncode == 1
        assert 'Specify at least one output file' in (result.stdout + result.stderr)

    def test_no_arguments_exits_1(self, tmp_path):
        result = run_cli([], cwd=tmp_path)
        assert result.returncode == 1
        assert 'Specify at least one output file' in (result.stdout + result.stderr)

    def test_unknown_flag_is_silently_ignored(self, tmp_path):
        # Characterization of current behavior: parse_arguments() uses
        # parse_known_args() (GoodVibes.py:204), so an unrecognized or
        # typo'd flag does NOT error — the run completes with defaults.
        # See AUDIT.md: a typo like --qss would silently use --qs default.
        # If this test starts failing because unknown flags are rejected,
        # that is an intentional improvement: update this test.
        result = run_cli(['--bad-flag-typo', WATER], cwd=tmp_path)
        assert result.returncode == 0

    def test_label_and_selectivity_are_mutually_exclusive(self, tmp_path):
        result = run_cli(
            [WATER, '--label', 'A=*water*', '--selectivity', 'sel.yaml'],
            cwd=tmp_path)
        assert result.returncode == 1
        assert 'mutually exclusive' in (result.stdout + result.stderr)

    def test_missing_spc_file_exits_1(self, tmp_path):
        result = run_cli([WATER, '--spc', 'TZ'], cwd=tmp_path)
        assert result.returncode == 1
        assert 'No SPC calculation file' in (result.stdout + result.stderr)

    def test_unknown_media_solvent_exits_1(self, tmp_path):
        result = run_cli([WATER, '--media', 'notasolvent'], cwd=tmp_path)
        assert result.returncode == 1
        assert 'Unknown solvent' in (result.stdout + result.stderr)

    def test_single_selectivity_label_exits_1(self, tmp_path):
        result = run_cli([WATER, '--label', 'R=*nomatch*'], cwd=tmp_path)
        assert result.returncode == 1
        assert 'at least two labels' in (result.stdout + result.stderr)

    def test_only_error_terminated_files_exits_1(self, tmp_path):
        result = run_cli([ERR_FILE], cwd=tmp_path)
        assert result.returncode == 1
        assert 'normally terminated' in (result.stdout + result.stderr)

    def test_help_exits_0(self, tmp_path):
        result = run_cli(['-h'], cwd=tmp_path)
        assert result.returncode == 0
        assert 'usage' in result.stdout.lower()

    def test_boltz_requires_uniform_level_of_theory(self, tmp_path):
        # WATER is HF/6-31G(d); the ethane example is B3LYP/6-31G(d).
        result = run_cli([WATER, ETHANE, '--boltz'], cwd=tmp_path)
        assert result.returncode == 1
        assert 'same level of theory' in (result.stdout + result.stderr)

    def test_missing_spc_exit_happens_at_argument_parsing(self, tmp_path):
        # Characterization: the missing-SPC check lives inside
        # parse_arguments() (GoodVibes.py:250), so it fires before any
        # file is parsed.
        result = run_cli([WATER, '--spc', 'nonexistent_suffix'], cwd=tmp_path)
        assert result.returncode == 1
        assert 'No SPC calculation file' in (result.stdout + result.stderr)


# ---------------------------------------------------------------------------
# 2. sys.exit characterization in library modules (task 0.4)
# ---------------------------------------------------------------------------

class TestSysExitCharacterization:
    """Pin the current exit-on-error behavior of library functions.

    AUDIT.md task 2.1 plans to convert these to a GoodVibesError
    hierarchy; when that lands, these tests should be updated to expect
    the new exceptions instead of SystemExit.
    """

    def test_validation_exits_when_no_files_survive(self, monkeypatch):
        options, _ = build_options(monkeypatch)
        with pytest.raises(SystemExit, match='normally terminated'):
            GV.collect_and_validate_files([ERR_FILE], options)

    def test_validation_exits_on_error_terminated_spc(
            self, monkeypatch, tmp_path):
        # Base file parses fine; its SPC partner has an error termination.
        base = tmp_path / 'mol.log'
        spc = tmp_path / 'mol_TZ.log'
        base.write_text(open(WATER, encoding='utf-8').read())
        spc.write_text(open(ERR_FILE, encoding='utf-8').read())
        options, files = build_options(
            monkeypatch, files=[str(base)], extra=['--spc', 'TZ'])
        with pytest.raises(SystemExit, match='Error termination found'):
            GV.collect_and_validate_files(files, options)

    @pytest.mark.filterwarnings('ignore::DeprecationWarning')
    def test_legacy_get_selectivity_exits_when_no_pattern_match(
            self, monkeypatch, tmp_path):
        from goodvibes.selectivity import get_selectivity
        monkeypatch.chdir(tmp_path)  # empty dir: globs match nothing
        with pytest.raises(SystemExit, match='filenames or selectivity pattern'):
            get_selectivity('*_R*:*_S*', ['water.log'], {}, 298.15, [])

    def test_print_pes_results_exits_on_missing_thermo_attrs(
            self, monkeypatch):
        from goodvibes.output import print_pes_results

        class _Empty:  # lacks qh_gibbs_free_energy
            pass

        options, _ = build_options(monkeypatch)
        with pytest.raises(SystemExit, match='Could not find thermodynamic data'):
            print_pes_results({'x.log': _Empty()}, options, [])


# ---------------------------------------------------------------------------
# 3. Direct main() flag-interaction tests (task 0.3)
# ---------------------------------------------------------------------------

class TestMainDirect:

    def _dat(self, tmp_path, name='GoodVibes_output.dat'):
        path = tmp_path / name
        assert path.exists(), f'{name} was not written'
        return path.read_text()

    def test_default_run_writes_dat_with_structure_row(
            self, monkeypatch, tmp_path, gv_logger_cleanup):
        run_main(monkeypatch, tmp_path, [WATER])
        text = self._dat(tmp_path)
        assert 'Structure' in text
        assert '01a_water_hf_freq' in text
        # Regression: the .dat console must not crop wide tables. Without
        # an explicit width, Rich defaults non-TTY files to 80 columns and
        # the rightmost columns -- including qh-G(T), the headline result
        # -- were silently missing from every .dat archive.
        assert 'qh-G(T)' in text

    def test_truhlar_qs_with_qh(self, monkeypatch, tmp_path, gv_logger_cleanup):
        run_main(monkeypatch, tmp_path, [WATER, '--qs', 'truhlar', '-q'])
        text = self._dat(tmp_path)
        assert 'Truhlar' in text
        assert 'qh-H' in text  # QH enthalpy column only present with -q

    def test_temperature_interval(self, monkeypatch, tmp_path, gv_logger_cleanup):
        run_main(monkeypatch, tmp_path, [WATER, '--ti', '250,350,50'])
        text = self._dat(tmp_path)
        assert '250.0' in text and '300.0' in text and '350.0' in text

    def test_boltz_populations(self, monkeypatch, tmp_path, gv_logger_cleanup):
        # Same level of theory (HF/6-31G(d)) required for Boltzmann factors.
        water_isotopes = g16path('01c_water_hf_freq_isotopes.log')
        run_main(monkeypatch, tmp_path, [WATER, water_isotopes, '--boltz'])
        text = self._dat(tmp_path)
        assert 'Boltz' in text

    def test_xyz_output(self, monkeypatch, tmp_path, gv_logger_cleanup):
        run_main(monkeypatch, tmp_path, [WATER, '--xyz'])
        xyz = tmp_path / 'GoodVibes_output.xyz'
        assert xyz.exists()
        lines = xyz.read_text().splitlines()
        assert lines[0].strip() == '3'  # water: 3 atoms

    def test_json_export_v1_schema(self, monkeypatch, tmp_path, gv_logger_cleanup):
        run_main(monkeypatch, tmp_path, [WATER, '--json', 'out.json'])
        payload = json.loads((tmp_path / 'out.json').read_text())
        assert payload['schema_version'] == '1.0'
        assert payload['results'], 'results list is empty'
        thermo = payload['results'][0]['thermo']
        assert 'qh_gibbs_free_energy' in thermo

    def test_csv_export(self, monkeypatch, tmp_path, gv_logger_cleanup):
        pytest.importorskip('pandas')
        run_main(monkeypatch, tmp_path, [WATER, '--csv', 'out.csv'])
        header = (tmp_path / 'out.csv').read_text().splitlines()[0]
        assert 'qh_gibbs_free_energy' in header

    def test_media_correction(self, monkeypatch, tmp_path, gv_logger_cleanup):
        run_main(monkeypatch, tmp_path, [WATER, '--media', 'h2o'])
        text = self._dat(tmp_path)
        assert '01a_water_hf_freq' in text

    def test_check_mode(self, monkeypatch, tmp_path, gv_logger_cleanup):
        run_main(monkeypatch, tmp_path, [WATER, '--check'])
        text = self._dat(tmp_path)
        assert 'check' in text.lower()

    def test_symm_point_group(self, monkeypatch, tmp_path, gv_logger_cleanup):
        pytest.importorskip('pymsym')
        run_main(monkeypatch, tmp_path, [WATER, '--symm'])
        text = self._dat(tmp_path)
        assert 'C2v' in text or 'Cs' in text or 'point group' in text.lower()

    def test_sort_by_gibbs(self, monkeypatch, tmp_path, gv_logger_cleanup):
        run_main(monkeypatch, tmp_path, [WATER, ETHANE, '--sort'])
        text = self._dat(tmp_path)
        assert '01a_water_hf_freq' in text and 'ethane' in text

    def test_custom_output_name(self, monkeypatch, tmp_path, gv_logger_cleanup):
        run_main(monkeypatch, tmp_path, [WATER, '--output', 'mytest'])
        assert (tmp_path / 'GoodVibes_mytest.dat').exists()
