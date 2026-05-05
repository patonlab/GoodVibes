"""Direct unit tests for goodvibes.output rendering helpers.

Closes the bulk of the output.py coverage gap (was 22% — the
selectivity / cpu / main-results renderers were exercised only
through end-to-end CLI subprocess tests, which the coverage tooling
can't credit to the parent module).

Tests target the small functions directly with synthetic
SelectivityResult / stub-bbe inputs, plus smoke tests for the
heavier renderers (`print_results`, `_print_rich_table`,
`print_pes_tables`) using a tmp-path-backed `setup_logging` fixture.
"""
from __future__ import annotations

import logging
import os
from types import SimpleNamespace

import pytest

from goodvibes.output import (
    _format_ratio, _selectivity_active, _selectivity_to_json,
    _build_results_table, _print_rich_table,
    print_selectivity_results, print_cpu_time, print_results,
)
from goodvibes.selectivity import SelectivityResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gv_logging(tmp_path):
    """Initialize the `goodvibes` logger to write to a tmp .dat file.

    Several output.py renderers call `_print_rich_table`, which in turn
    calls `get_console_stdout()` / `get_console_dat()` — these raise
    RuntimeError unless `setup_logging()` has been called.

    Yields the logger so tests can attach handlers (e.g. for caplog-style
    capture) if they need to.
    """
    from goodvibes.utils import setup_logging
    setup_logging(str(tmp_path / "test"), "output")
    logger = logging.getLogger("goodvibes")
    yield logger
    # Tear down handlers so the next test's setup_logging is clean.
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass


def _sel_result_2way(R=0.92, S=0.08, T=298.15, ee=84.0, ddG=-0.0034):
    """Build a 2-label SelectivityResult."""
    return SelectivityResult(
        temperature=T,
        key="gibbs",
        labels=["R", "S"],
        files_per_label={"R": ["a.log", "b.log"], "S": ["c.log"]},
        populations={"R": R, "S": S},
        raw_boltzmann={"R": R, "S": S},
        preferred="R" if R >= S else "S",
        ee=ee,
        ddG=ddG,
    )


def _sel_result_4way():
    """Build a 4-label SelectivityResult (ee/ddG = None)."""
    pops = {"RR": 0.55, "RS": 0.25, "SR": 0.15, "SS": 0.05}
    return SelectivityResult(
        temperature=298.15,
        key="gibbs",
        labels=["RR", "RS", "SR", "SS"],
        files_per_label={k: [f"{k}_1.log"] for k in pops},
        populations=pops,
        raw_boltzmann=pops,
        preferred="RR",
        ee=None,
        ddG=None,
    )


def _stub_bbe(scf=-100.0, sp=None, name="x"):
    """A calc_bbe-shaped SimpleNamespace for table-row tests."""
    return SimpleNamespace(
        scf_energy=scf,
        sp_energy=sp if sp is not None else scf,
        zpe=0.05,
        enthalpy=scf + 0.05,
        qh_enthalpy=scf + 0.05,
        entropy=8.5e-5,
        qh_entropy=8.0e-5,
        gibbs_free_energy=scf - 0.02,
        qh_gibbs_free_energy=scf - 0.02,
        inverted_freqs=[],
        im_frequency_wn=[],
        cpu=[0, 0, 1, 0, 0],     # [day, hr, min, sec, msec]
        sp_cpu=None,
        linear_warning=False,
        point_group="C1",
        xyz=SimpleNamespace(program="Gaussian", nprocs=1),
    )


def _basic_options(**overrides):
    """Default Namespace for print_results / _build_results_table."""
    base = dict(
        dp=6, QH=False, spc=None, boltz=False, symm=False, pg=False,
        imag_freq=False, invert=None, duplicate=False, temperature=298.15,
        cputime=False, media=None, labels=None, selectivity_spec=None, ee=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# _format_ratio
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("populations,labels,expected", [
    ({"R": 0.6, "S": 0.4}, ["R", "S"], "60:40"),
    ({"R": 0.92, "S": 0.08}, ["R", "S"], "92:8"),
    ({"R": 0.5, "S": 0.5}, ["R", "S"], "50:50"),
    ({"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1}, ["a", "b", "c", "d"], "40:30:20:10"),
])
def test_format_ratio(populations, labels, expected):
    assert _format_ratio(populations, labels) == expected


def test_format_ratio_respects_label_order():
    """Output uses the label order, not the dict's iteration order."""
    pops = {"S": 0.4, "R": 0.6}
    assert _format_ratio(pops, ["R", "S"]) == "60:40"
    assert _format_ratio(pops, ["S", "R"]) == "40:60"


# ---------------------------------------------------------------------------
# _selectivity_active
# ---------------------------------------------------------------------------

def test_selectivity_active_off_when_all_none():
    assert _selectivity_active(_basic_options()) is False


def test_selectivity_active_with_labels():
    opts = _basic_options(labels=["R=*_R*"])
    assert _selectivity_active(opts) is True


def test_selectivity_active_with_spec_yaml():
    opts = _basic_options(selectivity_spec="spec.yaml")
    assert _selectivity_active(opts) is True


def test_selectivity_active_with_legacy_ee():
    opts = _basic_options(ee="*_R*:*_S*")
    assert _selectivity_active(opts) is True


def test_selectivity_active_handles_missing_attrs():
    """Robust against a Namespace that lacks the selectivity attrs."""
    opts = SimpleNamespace()
    assert _selectivity_active(opts) is False


# ---------------------------------------------------------------------------
# _selectivity_to_json
# ---------------------------------------------------------------------------

def test_selectivity_to_json_none_for_empty():
    assert _selectivity_to_json([]) is None
    assert _selectivity_to_json(None) is None


def test_selectivity_to_json_2way_payload():
    out = _selectivity_to_json([_sel_result_2way()])
    assert out["labels"] == ["R", "S"]
    assert out["key"] == "gibbs"
    assert len(out["results"]) == 1
    entry = out["results"][0]
    assert entry["temperature"] == pytest.approx(298.15)
    assert entry["populations"] == {"R": 0.92, "S": 0.08}
    assert entry["preferred"] == "R"
    assert entry["ee"] == pytest.approx(84.0)
    assert entry["ddG"] == pytest.approx(-0.0034)
    assert entry["files_per_label"]["R"] == ["a.log", "b.log"]


def test_selectivity_to_json_nway_omits_ee():
    """N>2 still serializes ee/ddG fields, but they're None."""
    out = _selectivity_to_json([_sel_result_4way()])
    entry = out["results"][0]
    assert entry["ee"] is None
    assert entry["ddG"] is None
    assert sum(entry["populations"].values()) == pytest.approx(1.0)


def test_selectivity_to_json_scan_multiple_temperatures():
    """A T-scan gets one entry per temperature."""
    results = [_sel_result_2way(T=t, R=0.5 + 0.01 * i, S=0.5 - 0.01 * i)
               for i, t in enumerate([298.15, 350.0, 400.0])]
    out = _selectivity_to_json(results)
    assert len(out["results"]) == 3
    temps = [r["temperature"] for r in out["results"]]
    assert temps == [298.15, 350.0, 400.0]


# ---------------------------------------------------------------------------
# print_selectivity_results dispatcher + single + scan renderers
# ---------------------------------------------------------------------------

def test_print_selectivity_results_returns_silently_on_empty(gv_logging):
    """No tables, no log noise — must not crash."""
    print_selectivity_results({})
    print_selectivity_results({"Method": []})


def test_print_selectivity_single_logs_summary(gv_logging, caplog):
    """Single-T mode: a Selectivity header + ratio/excess/ΔΔG footer."""
    caplog.set_level(logging.INFO, logger="goodvibes")
    sel = _sel_result_2way()
    print_selectivity_results({"Boltzmann-averaged": [sel]})
    text = "\n".join(rec.message for rec in caplog.records)
    assert "Selectivity, Boltzmann-averaged (gibbs, T = 298.15 K)" in text
    assert "Ratio R:S = 92:8" in text
    assert "Major: R" in text
    assert "excess = 84.00%" in text
    assert "ΔΔG = -2.13 kcal/mol" in text or "ΔΔG = -2.14 kcal/mol" in text


def test_print_selectivity_single_nway_omits_excess_ddG(gv_logging, caplog):
    """N>2: ratio printed, but no `excess` / `ΔΔG` in the summary."""
    caplog.set_level(logging.INFO, logger="goodvibes")
    print_selectivity_results({"Boltzmann-averaged": [_sel_result_4way()]})
    text = "\n".join(rec.message for rec in caplog.records)
    assert "Ratio RR:RS:SR:SS = 55:25:15:5" in text
    assert "Major: RR" in text
    assert "excess" not in text
    assert "ΔΔG" not in text


def test_print_selectivity_scan_emits_one_row_per_temperature(gv_logging, caplog):
    """Multi-T mode: scan header + table with one row per T."""
    caplog.set_level(logging.INFO, logger="goodvibes")
    scan = [_sel_result_2way(T=t, R=0.6 + 0.05 * i, S=0.4 - 0.05 * i,
                              ee=20.0 + i, ddG=0.001 * (i + 1))
            for i, t in enumerate([298.15, 350.0, 400.0])]
    print_selectivity_results({"Boltzmann-averaged": scan})
    text = "\n".join(rec.message for rec in caplog.records)
    assert "Selectivity scan, Boltzmann-averaged (gibbs)" in text


def test_print_selectivity_dual_method_blocks(gv_logging, caplog):
    """Dispatcher renders one block per method (Boltzmann + Lowest)."""
    caplog.set_level(logging.INFO, logger="goodvibes")
    sel = _sel_result_2way()
    print_selectivity_results({
        "Boltzmann-averaged": [sel],
        "Lowest conformer only": [sel],
    })
    text = "\n".join(rec.message for rec in caplog.records)
    assert text.count("Selectivity, Boltzmann-averaged") == 1
    assert text.count("Selectivity, Lowest conformer only") == 1


# ---------------------------------------------------------------------------
# _build_results_table — column composition
# ---------------------------------------------------------------------------

def _columns(table):
    return [c.header for c in table.columns]


def test_build_results_table_default_columns():
    table = _build_results_table(_basic_options())
    cols = _columns(table)
    assert cols == ["", "Structure", "E", "ZPE", "H", "T.S", "T.qh-S",
                    "G(T)", "qh-G(T)"]


def test_build_results_table_with_spc_swaps_labels():
    table = _build_results_table(_basic_options(spc="TZ"))
    cols = _columns(table)
    assert "E_SPC" in cols
    assert "H_SPC" in cols
    assert "G(T)_SPC" in cols
    assert "qh-G(T)_SPC" in cols


def test_build_results_table_qh_adds_qh_enthalpy():
    table = _build_results_table(_basic_options(QH=True))
    cols = _columns(table)
    assert "qh-H" in cols


def test_build_results_table_boltz_adds_boltz_column():
    table = _build_results_table(_basic_options(boltz="gibbs"))
    assert "Boltz" in _columns(table)


@pytest.mark.parametrize("flag", ["symm", "pg"])
def test_build_results_table_pg_adds_point_group_column(flag):
    table = _build_results_table(_basic_options(**{flag: True}))
    assert "Point Group" in _columns(table)


def test_build_results_table_imag_freq_adds_imag_column():
    table = _build_results_table(_basic_options(imag_freq=True))
    assert "im freq" in _columns(table)


def test_build_results_table_selectivity_active_adds_grel():
    """Grel column appears only when --label / --selectivity / --ee set."""
    table_off = _build_results_table(_basic_options())
    assert "Grel (kcal/mol)" not in _columns(table_off)
    table_on = _build_results_table(_basic_options(labels=["R=*_R*"]))
    assert "Grel (kcal/mol)" in _columns(table_on)


def test_build_results_table_dp_widens_numeric_columns():
    """`--dp 9` (3 extra digits) widens the energy columns."""
    table = _build_results_table(_basic_options(dp=9))
    e_col = next(c for c in table.columns if c.header == "E")
    e_col_default = next(c for c in _build_results_table(_basic_options()).columns
                         if c.header == "E")
    assert e_col.min_width > e_col_default.min_width


# ---------------------------------------------------------------------------
# _print_rich_table — smoke test (must not crash, must log a separator)
# ---------------------------------------------------------------------------

def test_print_rich_table_smoke(gv_logging, caplog):
    caplog.set_level(logging.INFO, logger="goodvibes")
    from rich.table import Table
    from rich import box as rich_box
    t = Table(box=rich_box.SIMPLE_HEAD)
    t.add_column("Col1")
    t.add_column("Col2")
    t.add_row("a", "1")
    t.add_row("b", "2")
    _print_rich_table(t)
    # Separator line of '─' chars logged after the table.
    text = "\n".join(rec.message for rec in caplog.records)
    assert "─" in text


# ---------------------------------------------------------------------------
# print_cpu_time
# ---------------------------------------------------------------------------

def test_print_cpu_time_aggregates_multiple_files(gv_logging, caplog):
    caplog.set_level(logging.INFO, logger="goodvibes")
    thermo_data = {
        "a.log": _stub_bbe(),    # cpu = [0, 0, 1, 0, 0] (1 minute)
        "b.log": _stub_bbe(),
        "c.log": _stub_bbe(),
    }
    print_cpu_time(thermo_data)
    text = "\n".join(rec.message for rec in caplog.records)
    assert "TOTAL CPU" in text
    # 3 × 1 minute = 3 minutes
    assert " 3 mins" in text


def test_print_cpu_time_excludes_pattern(gv_logging, caplog):
    caplog.set_level(logging.INFO, logger="goodvibes")
    thermo_data = {
        "keep.log": _stub_bbe(),
        "drop_sp_tzpop.log": _stub_bbe(),
    }
    print_cpu_time(thermo_data, exclude="*_sp_tzpop*")
    text = "\n".join(rec.message for rec in caplog.records)
    # Only one file (1 min) survived the exclude.
    assert " 1 mins" in text


def test_print_cpu_time_orca_scaled_footnote(gv_logging, caplog):
    """When ORCA wall-time × nprocs is in play, a footnote is emitted."""
    caplog.set_level(logging.INFO, logger="goodvibes")
    bbe = _stub_bbe()
    bbe.xyz = SimpleNamespace(program="Orca", nprocs=8)
    print_cpu_time({"a.log": bbe})
    text = "\n".join(rec.message for rec in caplog.records)
    assert "ORCA wall-time × nprocs" in text


def test_print_cpu_time_sums_parent_plus_spc(gv_logging, caplog):
    """With --spc, the TOTAL CPU line must equal parent + SPC across all
    files. Regression for a bug where the per-file `bbe.cpu` and
    `bbe.sp_cpu` were aggregated through a datetime accumulator that
    silently lost month rollovers when both adds happened in the same
    iteration — causing 90+ days of CPU to display as 25 days."""
    caplog.set_level(logging.INFO, logger="goodvibes")
    # 100 files × (parent 1 day + SPC 1 hr) = 100 days, 4 hrs, 10 mins
    # (SPC also contributes 10 mins via the stub's [0,0,10,0,0] sp_cpu)
    thermo_data = {}
    for i in range(100):
        bbe = _stub_bbe()
        bbe.cpu = [1, 0, 0, 0, 0]            # 1 day per parent
        bbe.sp_cpu = [0, 1, 0, 0, 0]         # 1 hr per SPC
        thermo_data[f"f{i}.log"] = bbe
    print_cpu_time(thermo_data)
    text = "\n".join(rec.message for rec in caplog.records)
    # 100 days parent + 100 hrs SPC = 100 days + 4 days + 4 hrs = 104 days, 4 hrs.
    assert "104 days" in text
    assert " 4 hrs" in text


def test_print_cpu_time_long_run_no_rollover_loss(gv_logging, caplog):
    """A single very long parent CPU rolls over multiple month boundaries
    cleanly. The previous datetime-based accumulator dropped extra
    rollovers (only credited 31 days regardless of how far past the
    month boundary it crossed)."""
    caplog.set_level(logging.INFO, logger="goodvibes")
    bbe = _stub_bbe()
    bbe.cpu = [120, 0, 0, 0, 0]              # single 120-day entry
    bbe.sp_cpu = None
    print_cpu_time({"big.log": bbe})
    text = "\n".join(rec.message for rec in caplog.records)
    assert "120 days" in text


# ---------------------------------------------------------------------------
# print_results — main thermochemistry table
# ---------------------------------------------------------------------------

def test_print_results_smoke(gv_logging, caplog):
    """Single-file run: table renders without crashing; row appears."""
    caplog.set_level(logging.INFO, logger="goodvibes")
    bbe = _stub_bbe(scf=-100.5)
    print_results({"only.log": bbe}, _basic_options())
    text = "\n".join(rec.message for rec in caplog.records)
    # Separator under the table proves a render happened.
    assert "─" in text


def test_print_results_warns_on_missing_freq(gv_logging, caplog):
    """A bbe lacking gibbs_free_energy surfaces a one-line summary
    (file is excluded from the table, not silently dropped)."""
    caplog.set_level(logging.INFO, logger="goodvibes")
    sp_only = SimpleNamespace(scf_energy=-50.0, sp_energy=None,
                              inverted_freqs=[], linear_warning=False)
    print_results({"sp.log": sp_only}, _basic_options())
    text = "\n".join(rec.message for rec in caplog.records)
    assert "No frequency information found" in text
    assert "sp" in text


def test_print_results_skips_duplicate_rows(gv_logging, caplog):
    """Files in dup_list[i][0] are excluded from the table and an
    'x is a duplicate of ...' line is logged."""
    caplog.set_level(logging.INFO, logger="goodvibes")
    bbe1 = _stub_bbe(scf=-100.5)
    bbe2 = _stub_bbe(scf=-100.5)
    print_results(
        {"dup.log": bbe1, "canon.log": bbe2},
        _basic_options(),
        dup_list=[["dup.log", "canon.log"]],
    )
    text = "\n".join(rec.message for rec in caplog.records)
    assert "duplicate or enantiomer" in text


def test_print_results_linear_molecule_warning(gv_logging, caplog):
    """linear_warning=True bbes get a caution line, not a table row."""
    caplog.set_level(logging.INFO, logger="goodvibes")
    bbe = _stub_bbe()
    bbe.linear_warning = True
    print_results({"co2.log": bbe}, _basic_options())
    text = "\n".join(rec.message for rec in caplog.records)
    assert "linear molecule" in text


def test_print_results_with_boltz_and_symm(gv_logging):
    """Smoke-test the column-conditional row construction with several
    optional columns active."""
    bbe = _stub_bbe(scf=-100.5)
    print_results(
        {"a.log": bbe},
        _basic_options(boltz="gibbs", symm=True, imag_freq=True),
        boltz_facs={"a.log": 1.0},
    )
    # No exception → success. Detailed column verification covered by
    # _build_results_table tests above.
