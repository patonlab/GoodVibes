"""Tests for the v4.2 PES output: Rich tables (`print_pes_tables`) and
JSON v1.0 (`_pes_to_json` + `write_json_results`)."""
import json
import os
from types import SimpleNamespace

import pytest

from goodvibes.output import (
    JSON_SCHEMA_VERSION, _pes_to_json, _build_pes_table,
    write_json_results,
)
from goodvibes.pes_loader import PESSpec, build_pes_result
from goodvibes.pes_model import PESOptions


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _bbe(g, scf=None, h=None, qh_h=None, sp=None):
    """calc_bbe-shaped stub. By default qh_enthalpy mimics the --QH=False
    path (left at 0.0); set qh_h explicitly to exercise the QH=True path."""
    if h is None:
        h = g + 0.005
    if scf is None:
        scf = g - 0.001
    return SimpleNamespace(
        scf_energy=scf, zpe=0.005,
        enthalpy=h, qh_enthalpy=qh_h if qh_h is not None else 0.0,
        entropy=1.6e-5, qh_entropy=1.6e-5,
        gibbs_free_energy=g, qh_gibbs_free_energy=g,
        sp_energy=sp,
    )


def _result_two_point(units="kcal/mol", spc=None):
    td = {
        "a.log": _bbe(-100.0, sp=(-100.5 if spc else None)),
        "b.log": _bbe(-50.0, sp=(-50.5 if spc else None)),
    }
    spec = PESSpec(
        pathways={"rxn": ["A", "B"]},
        species={"A": "a", "B": "b"},
        options=PESOptions(units=units, decimals=1, gconf=False, QH=False,
                           spc_used=bool(spc)),
    )
    return build_pes_result(spec, td, temperatures=[298.15])


# ---------------------------------------------------------------------------
# _pes_to_json
# ---------------------------------------------------------------------------

def test_pes_to_json_returns_none_for_empty():
    assert _pes_to_json(None, 298.15) is None


def test_pes_to_json_basic_structure():
    result = _result_two_point()
    block = _pes_to_json(result, 298.15)
    assert "pathways" in block
    assert len(block["pathways"]) == 1
    p = block["pathways"][0]
    assert p["name"] == "rxn"
    assert p["temperature"] == 298.15
    assert p["units"] == "kcal/mol"
    assert p["zero_label"] == "A"
    assert len(p["points"]) == 2


def test_pes_to_json_relative_g_matches_arithmetic():
    """Reactant -100 → product -50; ΔG = +50 Hartree → +50 × 627.51 ≈ 31375 kcal/mol."""
    result = _result_two_point()
    block = _pes_to_json(result, 298.15)
    rel = block["pathways"][0]["points"][1]["relative"]
    assert rel["g"] == pytest.approx(50.0 * 627.509541, rel=1e-9)
    assert rel["qh_g"] == pytest.approx(50.0 * 627.509541, rel=1e-9)


def test_pes_to_json_zero_point_relative_is_zero():
    result = _result_two_point()
    block = _pes_to_json(result, 298.15)
    rel = block["pathways"][0]["points"][0]["relative"]
    for k in ("scf", "zpe", "h", "qh_h", "ts", "qh_ts", "g", "qh_g"):
        assert rel[k] == pytest.approx(0.0, abs=1e-9)


def test_pes_to_json_spc_field_none_when_no_spc():
    result = _result_two_point(spc=False)
    block = _pes_to_json(result, 298.15)
    rel = block["pathways"][0]["points"][1]["relative"]
    assert rel["spc"] is None


def test_pes_to_json_spc_field_populated_when_spc_used():
    result = _result_two_point(spc=True)
    block = _pes_to_json(result, 298.15)
    rel = block["pathways"][0]["points"][1]["relative"]
    # ΔSPC = (-50.5) - (-100.5) = 50.0 Hartree → 50 × 627.51 ≈ 31375 kcal/mol
    assert rel["spc"] == pytest.approx(50.0 * 627.509541, rel=1e-9)


def test_pes_to_json_kj_mol_units():
    result = _result_two_point(units="kJ/mol")
    block = _pes_to_json(result, 298.15)
    rel = block["pathways"][0]["points"][1]["relative"]
    # 50 Hartree in kJ/mol = 50 * 4.184 * 627.509541 = 131290.5 kJ/mol
    expected = 50.0 * 4.184 * 627.509541
    assert rel["g"] == pytest.approx(expected, rel=1e-9)


def test_pes_to_json_species_breakdown():
    """Stoichiometric points emit per-species coefficients and resolved files."""
    td = {
        "a.log": _bbe(-100.0),
        "b.log": _bbe(-50.0),
        "c.log": _bbe(-200.0),
    }
    spec = PESSpec(
        pathways={"rxn": ["2*A + B", "C"]},
        species={"A": "a", "B": "b", "C": "c"},
        options=PESOptions(units="kcal/mol", decimals=1, gconf=False, QH=False),
    )
    result = build_pes_result(spec, td, temperatures=[298.15])
    block = _pes_to_json(result, 298.15)
    species = block["pathways"][0]["points"][0]["species"]
    assert species[0] == {"coefficient": 2, "name": "A", "files": ["a.log"]}
    assert species[1] == {"coefficient": 1, "name": "B", "files": ["b.log"]}


# ---------------------------------------------------------------------------
# write_json_results: schema integration
# ---------------------------------------------------------------------------

def test_schema_version_bumped_to_0_4():
    assert JSON_SCHEMA_VERSION == "1.0"


def test_write_json_includes_pes_block_when_pes_result_given(tmp_path):
    result = _result_two_point()
    options = SimpleNamespace(
        temperature=298.15, temperature_interval=None, conc=0.040874,
        QS='grimme', QH=False, freq_cutoff=100.0,
        S_freq_cutoff=100.0, H_freq_cutoff=100.0,
        freq_scale_factor=1.0, media=None, freespace='none', spc=None,
        invert=False, symm=False, duplicate=False, boltz=False,
        inertia='global', mm_freq_scale_factor=None,
    )
    out = tmp_path / "out.json"
    # Empty thermo_data; we only care about the pes block.
    write_json_results({}, options, str(out), pes_result=result)
    payload = json.loads(out.read_text())
    assert payload["schema_version"] == "1.0"
    assert "pes" in payload
    assert payload["pes"]["pathways"][0]["name"] == "rxn"


def test_write_json_omits_pes_block_when_no_pes_result(tmp_path):
    options = SimpleNamespace(
        temperature=298.15, temperature_interval=None, conc=0.040874,
        QS='grimme', QH=False, freq_cutoff=100.0,
        S_freq_cutoff=100.0, H_freq_cutoff=100.0,
        freq_scale_factor=1.0, media=None, freespace='none', spc=None,
        invert=False, symm=False, duplicate=False, boltz=False,
        inertia='global', mm_freq_scale_factor=None,
    )
    out = tmp_path / "out.json"
    write_json_results({}, options, str(out))
    payload = json.loads(out.read_text())
    assert "pes" not in payload


# ---------------------------------------------------------------------------
# _build_pes_table — Rich table structure
# ---------------------------------------------------------------------------

def test_build_pes_table_columns_no_spc_no_qh():
    """Without --spc and without --QH: 8 columns (marker + Species + 6 ΔX)."""
    result = _result_two_point()
    options = SimpleNamespace(spc=None, QH=False, gconf=False)
    table = _build_pes_table(
        result.pathways[0], options, 298.15, result.options
    )
    # Marker + Species + ΔE, ΔZPE, ΔH, T·ΔS, T·Δqh-S, ΔG(T), Δqh-G(T) = 9
    assert len(table.columns) == 9
    headers = [c.header for c in table.columns]
    assert "ΔE" in headers
    assert "ΔE_SPC" not in headers
    assert "Δqh-H" not in headers     # QH off → no qh-H column


def test_build_pes_table_columns_with_qh():
    result = _result_two_point()
    options = SimpleNamespace(spc=None, QH=True, gconf=False)
    table = _build_pes_table(
        result.pathways[0], options, 298.15, result.options
    )
    headers = [c.header for c in table.columns]
    assert "Δqh-H" in headers


def test_build_pes_table_columns_with_spc():
    result = _result_two_point(spc=True)
    options = SimpleNamespace(spc='tzpop', QH=False, gconf=False)
    table = _build_pes_table(
        result.pathways[0], options, 298.15, result.options
    )
    headers = [c.header for c in table.columns]
    assert "ΔE_SPC" in headers
    assert "ΔE" in headers           # plain ΔE still shown alongside


def test_build_pes_table_row_count_matches_points():
    """One row per pathway point."""
    result = _result_two_point()
    options = SimpleNamespace(spc=None, QH=False, gconf=False)
    table = _build_pes_table(
        result.pathways[0], options, 298.15, result.options
    )
    assert table.row_count == 2


def test_build_pes_table_renders_em_dash_for_none_sp():
    """SPC column shows '—' for points where sp_energy is None."""
    result = _result_two_point(spc=False)        # spc=None on bbes
    options = SimpleNamespace(spc='tzpop', QH=False, gconf=False)
    # spc_used signals "render the SPC column"; when sp_energy is None on
    # the bbes (e.g. user set --spc='link' but the file lacked SPC), the
    # builder should emit '—' rather than crash.
    result.options.spc_used = True
    table = _build_pes_table(
        result.pathways[0], options, 298.15, result.options
    )
    # Render to a string and look for em-dash in the body.
    from io import StringIO
    from rich.console import Console
    buf = StringIO()
    Console(file=buf, force_terminal=False, width=200).print(table)
    rendered = buf.getvalue()
    assert "—" in rendered


# ---------------------------------------------------------------------------
# Real fixture: examples/pes/azabor_PES_v2.yaml end-to-end
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AZABOR_V2 = os.path.join(REPO_ROOT, "goodvibes", "examples", "pes", "azabor_PES_v2.yaml")


@pytest.mark.skipif(not os.path.exists(AZABOR_V2), reason="azabor_PES_v2.yaml fixture missing")
def test_azabor_v2_loads_with_no_warnings():
    """The new-style YAML must NOT emit a DeprecationWarning."""
    import warnings
    from types import SimpleNamespace as NS
    from goodvibes.pes_loader import load_pes

    # Stub thermo_data: just enough conformer files to satisfy the globs.
    import glob
    geom_files = [
        f for f in sorted(glob.glob(os.path.join(os.path.dirname(AZABOR_V2), "*.log")))
        if "_sp_tzpop." not in f
    ]
    td = {f: _bbe(-1000.0 - i * 0.001) for i, f in enumerate(geom_files)}

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = load_pes(AZABOR_V2, td, temperatures=[298.15])
    assert len(result.pathways) == 1
    assert result.pathways[0].name == "Ph"
    assert len(result.pathways[0].points) == 6
