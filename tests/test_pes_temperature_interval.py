"""`--pes` together with `--ti` must produce the legacy per-temperature
PES tables instead of crashing.

Regression: main() discarded the return value of print_temperature_interval,
so print_pes_results received interval_bbe_data=None and died in zip(*None).
"""
import pytest

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


def test_pes_with_temperature_interval_builds_the_model_for_json_and_plot(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    """With --ti the PES model carries every scan temperature: the JSON
    ``pes`` block has one entry per pathway per temperature, its values
    agree with a direct evaluation at that temperature, and --pes-plot
    overlays the temperatures on one figure."""
    import json
    import math
    pytest.importorskip("matplotlib")
    from goodvibes import compute_thermo
    from goodvibes.constants import KCAL_TO_AU
    run_main(monkeypatch, tmp_path,
             [WATER_A, WATER_A2, WATER_B, '--pes', _pes_yaml(tmp_path), '--ti', '250,350,50',
              '--nogconf', '--json', 'out.json', '--pes-plot', 'profile.png'])
    payload = json.loads((tmp_path / 'out.json').read_text())
    entries = payload['pes']['pathways']
    assert [(e['name'], e['temperature']) for e in entries] == [('rxn', 250.0), ('rxn', 300.0), ('rxn', 350.0)]
    # Boltzmann-averaged A (two water conformers) vs B at 350 K, recomputed directly.
    at_350 = {f: compute_thermo(f, temperature=350.0) for f in (WATER_A, WATER_A2, WATER_B)}
    ga = [at_350[WATER_A].qh_gibbs_free_energy, at_350[WATER_A2].qh_gibbs_free_energy]
    from goodvibes.constants import GAS_CONSTANT, J_TO_AU
    w = [math.exp(-(g - min(ga)) * J_TO_AU / GAS_CONSTANT / 350.0) for g in ga]
    g_a = sum(p * g for p, g in zip(w, ga)) / sum(w)
    expected = (at_350[WATER_B].qh_gibbs_free_energy - g_a) * KCAL_TO_AU
    got = entries[2]['points'][1]['relative']['qh_g']
    assert got == pytest.approx(expected, abs=1e-6)
    assert (tmp_path / 'profile.png').stat().st_size > 0
    # the legacy per-temperature text is still printed
    text = (tmp_path / 'GoodVibes_output.dat').read_text()
    assert text.count('RXN: rxn') == 3
