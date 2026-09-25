"""--nogconf / --lowest-only must reach every PES consumer.

Regression: the CLI flags were only copied into the PES model inside
print_pes_tables, which runs *after* the JSON writer and --pes-plot, so
`--json` carried Boltzmann+gconf numbers whatever the user asked for.
"""
import json
from types import SimpleNamespace

import pytest

from conftest import g16path
from goodvibes.output import _pes_to_json, apply_cli_pes_options
from goodvibes.pes_loader import PESSpec, build_pes_result
from goodvibes.pes_model import PESOptions

from test_cli_errors import gv_logger_cleanup, run_main  # noqa: F401  (fixture re-export)


def _bbe(g):
    return SimpleNamespace(
        scf_energy=g - 0.001, zpe=0.005, enthalpy=g + 0.005, qh_enthalpy=0.0,
        entropy=1.6e-5, qh_entropy=1.6e-5,
        gibbs_free_energy=g, qh_gibbs_free_energy=g, sp_energy=None,
    )


def _two_conformer_result():
    td = {"a1.log": _bbe(-100.000), "a2.log": _bbe(-99.999), "b.log": _bbe(-99.990)}
    spec = PESSpec(pathways={"rxn": ["A", "B"]},
                   species={"A": ["a1", "a2"], "B": "b"},
                   options=PESOptions(units="kcal/mol", decimals=3))
    return build_pes_result(spec, td, temperatures=[298.15])


def _rel_b(result):
    return _pes_to_json(result, 298.15)["pathways"][0]["points"][1]["relative"]["qh_g"]


def test_apply_cli_pes_options_changes_json_rollup():
    ns = lambda **kw: SimpleNamespace(**{"gconf": True, "QH": False, "spc": None, "lowest_only": False, **kw})
    default = _rel_b(apply_cli_pes_options(_two_conformer_result(), ns()))
    nogconf = _rel_b(apply_cli_pes_options(_two_conformer_result(), ns(gconf=False)))
    lowest = _rel_b(apply_cli_pes_options(_two_conformer_result(), ns(lowest_only=True)))
    assert default != nogconf
    assert default != lowest
    assert lowest == pytest.approx((-99.990 - -100.000) * 627.509541, abs=1e-6)


def test_apply_cli_pes_options_is_idempotent_and_returns_result():
    r = _two_conformer_result()
    ns = SimpleNamespace(gconf=False, QH=True, spc="TZ", lowest_only=True)
    assert apply_cli_pes_options(r, ns) is r
    apply_cli_pes_options(r, ns)
    assert (r.options.gconf, r.options.QH, r.options.spc_used, r.options.lowest_only) == (False, True, True, True)


# --- end-to-end through main(): --json pes block must honour the flags ------

WATER_A = g16path('01a_water_hf_freq.log')
WATER_A2 = g16path('01c_water_hf_freq_isotopes.log')
WATER_B = g16path('01b_water_hf_freq_scaled.log')


def _run_json(monkeypatch, tmp_path, extra):
    pes = tmp_path / "pes.yaml"
    pes.write_text(
        "pathways:\n  rxn:\n    - A\n    - B\n"
        "species:\n"
        "  A: [01a_water_hf_freq, 01c_water_hf_freq_isotopes]\n"
        "  B: [01b_water_hf_freq_scaled]\n"
    )
    out = tmp_path / f"out{len(extra)}.json"
    run_main(monkeypatch, tmp_path,
             [WATER_A, WATER_A2, WATER_B, '--pes', str(pes), '--json', str(out)] + extra)
    payload = json.loads(out.read_text())
    return payload["pathways"] if "pathways" in payload else payload["pes"]["pathways"]


def test_json_pes_block_honours_nogconf_and_lowest_only(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    rel = lambda pw: pw[0]["points"][1]["relative"]["qh_g"]
    default = rel(_run_json(monkeypatch, tmp_path, []))
    nogconf = rel(_run_json(monkeypatch, tmp_path, ['--nogconf']))
    lowest = rel(_run_json(monkeypatch, tmp_path, ['--lowest-only']))
    assert default != nogconf
    assert default != lowest
