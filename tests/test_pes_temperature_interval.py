"""`--pes` together with `--ti` must produce the legacy per-temperature
PES tables instead of crashing.

Regression: main() discarded the return value of print_temperature_interval,
so print_pes_results received interval_bbe_data=None and died in zip(*None).
"""
from conftest import g16path

from test_cli_errors import gv_logger_cleanup, run_main  # noqa: F401  (fixture re-export)

WATER_A = g16path('01a_water_hf_freq.log')
WATER_A2 = g16path('01c_water_hf_freq_isotopes.log')
WATER_B = g16path('01b_water_hf_freq_scaled.log')


def _pes_yaml(tmp_path):
    pes = tmp_path / "pes.yaml"
    pes.write_text(
        "pathways:\n  rxn:\n    - A\n    - B\n"
        "species:\n"
        "  A: [01a_water_hf_freq, 01c_water_hf_freq_isotopes]\n"
        "  B: [01b_water_hf_freq_scaled]\n"
    )
    return str(pes)


def test_pes_with_temperature_interval_runs(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    run_main(monkeypatch, tmp_path,
             [WATER_A, WATER_A2, WATER_B, '--pes', _pes_yaml(tmp_path), '--ti', '250,350,50'])
    text = (tmp_path / 'GoodVibes_output.dat').read_text()
    # per-temperature thermo table ...
    assert '250.0' in text and '350.0' in text
    # ... and the PES section for each temperature, with both points.
    assert text.count('RXN') >= 3 or text.count('rxn') >= 3
    assert 'A' in text and 'B' in text


def test_check_and_temperature_interval_both_run(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    # --check used to short-circuit --ti (elif); both should be reported.
    run_main(monkeypatch, tmp_path, [WATER_A, WATER_B, '--check', '--ti', '250,350,50'])
    text = (tmp_path / 'GoodVibes_output.dat').read_text()
    assert '250.0' in text and '350.0' in text
