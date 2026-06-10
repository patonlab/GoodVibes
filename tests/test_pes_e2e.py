"""End-to-end PES regression test against the azabor_PES_v2.yaml fixture.

Closes the legacy `test_pes` skip slot. Parses 60 real Gaussian outputs,
threads them through `calc_bbe` → `load_pes` → `PESResult`, and asserts
golden ΔG values for three rollup modes (default gconf, --nogconf,
--lowest-only) plus the full JSON / Rich table render paths. Slow
(~3 s on parse) so wrapped in a session-scoped thermo_data fixture.
"""
import json
import math
import os

import pytest

from goodvibes.constants import ATMOS, GAS_CONSTANT, KCAL_TO_AU


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX_DIR = os.path.join(REPO_ROOT, "goodvibes", "examples", "pes")
YAML_V2 = os.path.join(EX_DIR, "azabor_PES_v2.yaml")
YAML_LEGACY = os.path.join(EX_DIR, "azabor_PES.yaml")
T_STD = 298.15

pytestmark = pytest.mark.skipif(
    not os.path.exists(YAML_V2),
    reason="azabor_PES_v2.yaml fixture missing",
)


# ---------------------------------------------------------------------------
# session-scoped fixture: parse the 60 .log files once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def azabor_thermo_data():
    """thermo_data dict for the azabor PES, computed at 298.15 K with --spc sp_tzpop."""
    import glob
    from goodvibes.thermo import calc_bbe

    geom_files = sorted(
        f for f in glob.glob(os.path.join(EX_DIR, "*.log"))
        if "_sp_tzpop." not in f
    )
    if not geom_files:
        pytest.skip("no .log fixtures present in goodvibes/examples/pes/")
    conc = ATMOS / (GAS_CONSTANT * T_STD)
    return {
        f: calc_bbe(f, "grimme", False, 100.0, 100.0, T_STD, conc, 1.0,
                    "none", "sp_tzpop", False)
        for f in geom_files
    }


@pytest.fixture(scope="module")
def azabor_pes_result(azabor_thermo_data):
    """PESResult built from azabor_PES_v2.yaml + thermo_data."""
    import warnings
    from goodvibes.pes_loader import load_pes
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        return load_pes(YAML_V2, azabor_thermo_data, temperatures=[T_STD])


# ---------------------------------------------------------------------------
# Golden ΔG values (kcal/mol, vs. zero point R1-An + Aza-Phos at 298.15 K)
# ---------------------------------------------------------------------------

# Mode 1: default — gconf-corrected, QH=False (CLI default).
GOLDEN_GCONF = [
    # (label,                      ΔG_qh,   ΔG,     ΔG_SPC)
    ("R1-An + Aza-Phos",            +0.000,  +0.000,  +0.000),
    ("R1-Comp + THF",               -3.787,  -5.392,  -5.695),
    ("AmTS + THF",                 +13.395, +12.349, +10.782),
    ("Azir-Comp + THF",            -59.041, -60.408, -63.800),
    ("OpenTS + THF",               -39.467, -40.650, -42.064),
    ("Syn-P + THF",                -91.402, -92.642, -97.939),
]

# Mode 2: --lowest-only — pick each species' lowest-qh-G conformer, no gconf.
GOLDEN_LOWEST = [
    ("R1-An + Aza-Phos",            +0.000,  +0.000,  +0.000),
    ("R1-Comp + THF",               -3.690,  -5.335,  -6.134),
    ("AmTS + THF",                 +13.210, +12.285, +10.876),
    ("Azir-Comp + THF",            -58.919, -60.020, -64.253),
    ("OpenTS + THF",               -39.037, -39.746, -42.544),
    ("Syn-P + THF",                -91.033, -91.692, -98.639),
]

# Mode 3: --nogconf — pure Boltzmann-averaged across conformers.
GOLDEN_NOGCONF_QH_GIBBS = [
    ("R1-An + Aza-Phos",            +0.000),
    ("R1-Comp + THF",               -3.669),
    ("AmTS + THF",                 +12.902),
    ("Azir-Comp + THF",            -59.061),
    ("OpenTS + THF",               -39.016),
    ("Syn-P + THF",                -91.188),
]

TOL = 0.001    # kcal/mol — assertions are tight to catch unit/arithmetic regressions


def _assert_relatives(pathway, mode_kw, golden):
    """Compare each Point's ΔG_qh, ΔG, ΔG_SPC against the golden table."""
    zero_th = pathway.zero.thermo(T_STD, **mode_kw)
    for (expected_label, *exp_kcal), point in zip(golden, pathway.points):
        assert point.label == expected_label
        rel = point.thermo(T_STD, **mode_kw) - zero_th
        actual = [
            rel.qh_gibbs * KCAL_TO_AU,
            rel.gibbs * KCAL_TO_AU,
            (rel.sp_energy or 0.0) * KCAL_TO_AU,
        ][:len(exp_kcal)]
        for got, want, name in zip(actual, exp_kcal, ("qh_g", "g", "spc")):
            assert math.isclose(got, want, abs_tol=TOL), (
                f"{point.label} {name}: got {got:+.3f}, expected {want:+.3f}"
            )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_load_pes_yaml_no_deprecation_warning(azabor_pes_result):
    """The new YAML schema must NOT emit a DeprecationWarning (the
    fixture is the v4.2-style azabor_PES_v2.yaml)."""
    result = azabor_pes_result
    assert len(result.pathways) == 1
    assert result.pathways[0].name == "Ph"
    assert len(result.pathways[0].points) == 6


def test_legacy_yaml_emits_deprecation_warning(azabor_thermo_data):
    """The line-based azabor_PES.yaml must trip a DeprecationWarning."""
    from goodvibes.pes_loader import load_pes
    with pytest.warns(DeprecationWarning, match="legacy line-based format"):
        load_pes(YAML_LEGACY, azabor_thermo_data, temperatures=[T_STD])


def test_default_gconf_mode_matches_golden(azabor_pes_result):
    """gconf=True, QH=False: the standard CLI default."""
    _assert_relatives(
        azabor_pes_result.pathways[0],
        dict(gconf=True, QH=False),
        GOLDEN_GCONF,
    )


def test_lowest_only_mode_matches_golden(azabor_pes_result):
    """--lowest-only: each species reduces to its lowest-qh-G conformer."""
    _assert_relatives(
        azabor_pes_result.pathways[0],
        dict(gconf=True, QH=False, lowest_only=True),
        GOLDEN_LOWEST,
    )


def test_nogconf_mode_matches_golden(azabor_pes_result):
    """--nogconf: pure Boltzmann average, no mixing-entropy correction."""
    pathway = azabor_pes_result.pathways[0]
    zero_th = pathway.zero.thermo(T_STD, gconf=False, QH=False)
    for (expected_label, expected_qh_g), point in zip(GOLDEN_NOGCONF_QH_GIBBS,
                                                       pathway.points):
        assert point.label == expected_label
        rel = point.thermo(T_STD, gconf=False, QH=False) - zero_th
        got = rel.qh_gibbs * KCAL_TO_AU
        assert math.isclose(got, expected_qh_g, abs_tol=TOL), (
            f"{point.label}: got {got:+.3f}, expected {expected_qh_g:+.3f}"
        )


def test_lowest_only_differs_from_gconf(azabor_pes_result):
    """Sanity check: gconf and lowest-only must disagree at conformer-rich
    points (otherwise we're not exercising the mode-switch logic).
    OpenTS+THF has the largest gap (~0.6 kcal/mol; 7 OpenTS conformers
    spread across ~0.04 Hartree)."""
    pathway = azabor_pes_result.pathways[0]
    zero_g = pathway.zero.thermo(T_STD, gconf=True, QH=False)
    zero_l = pathway.zero.thermo(T_STD, gconf=True, QH=False, lowest_only=True)
    point = pathway.points[4]    # OpenTS + THF
    rel_g = (point.thermo(T_STD, gconf=True, QH=False) - zero_g).qh_gibbs * KCAL_TO_AU
    rel_l = (point.thermo(T_STD, gconf=True, QH=False, lowest_only=True) - zero_l).qh_gibbs * KCAL_TO_AU
    assert abs(rel_g - rel_l) > 0.1, (
        f"gconf and lowest-only should differ by >0.1 kcal/mol "
        f"(got gconf={rel_g:+.3f}, lowest={rel_l:+.3f})"
    )


def test_spc_propagates_through_arithmetic(azabor_pes_result):
    """Every point should have a non-None sp_energy (--spc sp_tzpop was
    used during calc_bbe)."""
    pathway = azabor_pes_result.pathways[0]
    for point in pathway.points:
        rel = point.thermo(T_STD, gconf=True, QH=False) - pathway.zero.thermo(
            T_STD, gconf=True, QH=False
        )
        assert rel.sp_energy is not None, f"{point.label}: sp_energy is None"


# ---------------------------------------------------------------------------
# JSON v1.0 round-trip
# ---------------------------------------------------------------------------

def test_json_pes_block_matches_golden(azabor_pes_result, tmp_path):
    """write_json_results emits a `pes` block whose relative.qh_g values
    match the gconf golden."""
    from types import SimpleNamespace
    from goodvibes.output import write_json_results

    options = SimpleNamespace(
        temperature=T_STD, temperature_interval=None, conc=None,
        QS="grimme", QH=False, freq_cutoff=100.0,
        S_freq_cutoff=100.0, H_freq_cutoff=100.0,
        freq_scale_factor=1.0, media=None, freespace="none", spc="sp_tzpop",
        invert=False, symm=False, duplicate=False, boltz=False,
        inertia="global", mm_freq_scale_factor=None,
    )
    out = tmp_path / "azabor.json"
    write_json_results({}, options, str(out), pes_result=azabor_pes_result)
    payload = json.loads(out.read_text())

    assert payload["schema_version"] == "1.0"
    assert "pes" in payload
    pathway = payload["pes"]["pathways"][0]
    assert pathway["name"] == "Ph"
    assert len(pathway["points"]) == 6
    for (expected_label, exp_qh_g, _, _), pt in zip(GOLDEN_GCONF, pathway["points"]):
        assert pt["label"] == expected_label
        assert math.isclose(pt["relative"]["qh_g"], exp_qh_g, abs_tol=TOL)


# ---------------------------------------------------------------------------
# Rich table renderer doesn't crash on real data
# ---------------------------------------------------------------------------

def test_print_pes_tables_renders_without_crashing(azabor_pes_result):
    """Smoke: build the Rich table and confirm it has the expected shape."""
    from types import SimpleNamespace
    from goodvibes.output import _build_pes_table

    # Sync flags onto the model options (mirrors print_pes_tables).
    azabor_pes_result.options.gconf = True
    azabor_pes_result.options.QH = False
    azabor_pes_result.options.spc_used = True

    options = SimpleNamespace(spc="sp_tzpop", QH=False, gconf=True, conc=None)
    table = _build_pes_table(
        azabor_pes_result.pathways[0], options, T_STD, azabor_pes_result.options,
    )
    assert table.row_count == 6                   # one row per pathway point
    headers = [c.header for c in table.columns]
    assert "ΔE_SPC" in headers                    # SPC column appears
    assert "ΔG(T)_SPC" in headers
    assert "Δqh-H" not in headers                 # QH=False -> no qh-H column
    assert "RXN: Ph" in (table.title or "")
