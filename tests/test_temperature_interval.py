"""--ti temperature scans use a float grid and recompute G(T) at every step.

Regressions fixed here:
  * temperatures were built with range(int(start), int(end)+1, int(step)), so
    ``--ti 200,201,0.5`` died with 'range() arg 3 must not be zero' and
    ``--ti 298.15,398.15,50`` silently scanned 298, 348, 398 K;
  * the --label/--selectivity scan reused the base-temperature free energies
    and only changed RT in the exponent, so ee at 333 K from ``--ti`` did not
    match ee from ``--temp 333``;
  * the --ti thermochemistry table went through the legacy positional
    calc_bbe constructor, dropping --zpe-vscal and --symm, so its row at the
    base temperature disagreed with the main table.
"""
import json
import os
import re

import pytest

from conftest import datapath, g16path
from goodvibes.utils import parse_temperature_interval

from test_cli_errors import gv_logger_cleanup, run_main  # noqa: F401  (fixture re-export)

WATER = g16path('01a_water_hf_freq.log')
GCONF = datapath('gconf_ee_boltz')
TS_FILES = [os.path.join(GCONF, 'Aminoxylation_TS1_R.log'), os.path.join(GCONF, 'Aminoxylation_TS2_S.log')]
LABELS = ['--label', 'R=*_R*', '--label', 'S=*_S*']


@pytest.mark.parametrize('spec, expected', [
    ('250,350,50', [250.0, 300.0, 350.0]),
    ('298,318,10', [298.0, 308.0, 318.0]),
    ('200,201,0.5', [200.0, 200.5, 201.0]),
    ('298.15,398.15,50', [298.15, 348.15, 398.15]),
    ('100,200', [100.0 + 10.0 * i for i in range(11)]),
    ('300,300,10', [300.0]),
])
def test_parse_temperature_interval(spec, expected):
    assert parse_temperature_interval(spec) == pytest.approx(expected)


@pytest.mark.parametrize('spec', ['300', '300,200', '200,300,0', '200,300,-5', 'a,b', '1,2,3,4'])
def test_parse_temperature_interval_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        parse_temperature_interval(spec)


def _row_values(text, name, temp_str):
    """Numbers on the --ti table line for `name` at temperature `temp_str`."""
    for line in text.splitlines():
        if name in line and re.search(rf"\b{re.escape(temp_str)}\b", line):
            nums = re.findall(r"-?\d+\.\d{6}", line)
            if nums:
                return [float(x) for x in nums]
    raise AssertionError(f"no --ti row for {name} at {temp_str} K")


def test_ti_table_row_matches_single_temperature_table(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    # HF/6-31G(d): harm_fac 0.922 differs from zpe_fac 0.909, which is exactly
    # the case where the legacy constructor's ZPE scaling went wrong.
    run_main(monkeypatch, tmp_path, [WATER, '--temp', '300', '-q', '--output', 'single'])
    single = (tmp_path / 'GoodVibes_single.dat').read_text()
    main_row = re.search(r"^o\s+01a_water_hf_freq\s+(.*)$", single, re.M).group(1)
    main_vals = [float(x) for x in re.findall(r"-?\d+\.\d{6}", main_row)]
    assert len(main_vals) == 8          # E ZPE H qh-H T.S T.qh-S G qh-G
    run_main(monkeypatch, tmp_path, [WATER, '--ti', '250,350,50', '-q', '--output', 'scan'])
    scan = (tmp_path / 'GoodVibes_scan.dat').read_text()
    ti_vals = _row_values(scan, '01a_water_hf_freq', '300.0')   # H qh-H T.S T.qh-S G qh-G
    assert ti_vals == pytest.approx(main_vals[2:8], abs=1.5e-6)


def test_ti_selectivity_scan_recomputes_free_energies(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    run_main(monkeypatch, tmp_path, TS_FILES + LABELS + ['--ti', '273,333,30', '--json', 'scan.json'])
    scan = json.loads((tmp_path / 'scan.json').read_text())['selectivity']['results']
    at_333 = next(r for r in scan if abs(r['temperature'] - 333.0) < 1e-9)
    run_main(monkeypatch, tmp_path, TS_FILES + LABELS + ['--temp', '333', '--json', 'single.json'])
    single = json.loads((tmp_path / 'single.json').read_text())['selectivity']['results'][0]
    assert at_333['populations'] == pytest.approx(single['populations'], abs=1e-9)
    assert at_333['ee'] == pytest.approx(single['ee'], abs=1e-7)
    assert at_333['ddG'] == pytest.approx(single['ddG'], abs=1e-12)


def test_ti_accepts_fractional_temperatures(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    run_main(monkeypatch, tmp_path, [WATER, '--ti', '298.15,299.15,0.5'])
    text = (tmp_path / 'GoodVibes_output.dat').read_text()
    for t in ('298.1', '298.6', '299.1'):   # printed with one decimal
        assert re.search(rf"01a_water_hf_freq\s+{re.escape(t)}\b", text), t
