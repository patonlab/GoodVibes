"""Tests for the goodvibes.api façade (v4.2 item 5).

Layered:
  1. Result-shape contract (`ThermoResult` exposes the expected fields).
  2. Parity with the underlying `calc_bbe` (the façade must NOT change
     numbers — same inputs ↔ identical floats).
  3. Default-argument behaviour: gas-phase concentration, frequency
     scale-factor auto-lookup, etc.
  4. compute_batch ordering and DataFrame round-trip.
"""
import math
import os

import pytest

from goodvibes import compute_batch, compute_thermo, ThermoResult
from goodvibes.api import to_dataframe
from goodvibes.constants import ATMOS, GAS_CONSTANT
from goodvibes.thermo import calc_bbe
from goodvibes.vib_scale_factors import canonicalize_level, scaling_data_dict


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G16 = os.path.join(REPO_ROOT, "tests", "g16")
WATER_HF = os.path.join(G16, "01a_water_hf_freq.log")
WATER_HF_HARMONIC = os.path.join(G16, "01b_water_hf_freq_scaled.log")


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------

def test_compute_thermo_returns_thermo_result():
    r = compute_thermo(WATER_HF)
    assert isinstance(r, ThermoResult)


def test_thermo_result_fields_populated_for_freq_file():
    """A normal opt+freq output populates every numeric thermo field."""
    r = compute_thermo(WATER_HF)
    assert r.scf_energy is not None
    assert r.zpe is not None
    assert r.enthalpy is not None
    assert r.entropy is not None
    assert r.qh_entropy is not None
    assert r.gibbs_free_energy is not None
    assert r.qh_gibbs_free_energy is not None
    assert r.has_thermo is True


def test_thermo_result_qh_enthalpy_none_when_qh_off():
    """qh_enthalpy is left at 0.0 by calc_bbe when QH=False; the API
    surfaces that as None (cleaner for downstream consumers)."""
    r = compute_thermo(WATER_HF, QH=False)
    assert r.qh_enthalpy is None


def test_thermo_result_qh_enthalpy_populated_when_qh_on():
    r = compute_thermo(WATER_HF, QH=True)
    assert r.qh_enthalpy is not None
    assert r.qh_enthalpy != 0.0


def test_thermo_result_metadata():
    r = compute_thermo(WATER_HF)
    assert r.name == "01a_water_hf_freq"
    assert r.program == "Gaussian"
    assert r.level_of_theory == "HF/6-31G(d)"
    assert r.linear_mol is False     # water is bent (C2v)


def test_thermo_result_exposes_underlying_bbe_and_qcdata():
    """Advanced users can read attributes not yet promoted to the result."""
    r = compute_thermo(WATER_HF)
    assert r.bbe is not None
    assert r.qcdata is not None
    # Sanity: original calc_bbe attribute reachable through .bbe
    assert r.bbe.scf_energy == r.scf_energy


def test_thermo_result_is_frozen():
    """Frozen dataclass — accidental mutation raises."""
    r = compute_thermo(WATER_HF)
    with pytest.raises(Exception):    # FrozenInstanceError
        r.scf_energy = 0.0


# ---------------------------------------------------------------------------
# Parity with calc_bbe
# ---------------------------------------------------------------------------

def test_parity_with_calc_bbe_default_args():
    """Default args (QH=False, gas-phase, auto scale factor) must yield
    bit-identical numbers to a direct calc_bbe call with the same
    resolved values."""
    T = 298.15
    conc = ATMOS / (GAS_CONSTANT * T)
    fac = scaling_data_dict[canonicalize_level("HF/6-31G(d)")].harm_fac

    api = compute_thermo(WATER_HF)
    entry = scaling_data_dict[canonicalize_level("HF/6-31G(d)")]
    direct = calc_bbe(WATER_HF, "grimme", False, 100.0, 100.0, T, conc,
                      entry.harm_fac, "none", None, None,
                      zpe_scale_fac=entry.zpe_fac)
    assert api.qh_gibbs_free_energy == direct.qh_gibbs_free_energy
    assert api.gibbs_free_energy == direct.gibbs_free_energy
    assert api.entropy == direct.entropy
    assert api.zpe == direct.zpe


def test_parity_with_calc_bbe_qh_on():
    T = 298.15
    conc = ATMOS / (GAS_CONSTANT * T)
    entry = scaling_data_dict[canonicalize_level("HF/6-31G(d)")]

    api = compute_thermo(WATER_HF, QH=True)
    direct = calc_bbe(WATER_HF, "grimme", True, 100.0, 100.0, T, conc,
                      entry.harm_fac, "none", None, None,
                      zpe_scale_fac=entry.zpe_fac)
    assert api.qh_enthalpy == direct.qh_enthalpy
    assert api.qh_gibbs_free_energy == direct.qh_gibbs_free_energy


def test_zpe_uses_zpe_fac_not_harm_fac():
    """The Truhlar database has separate harm_fac (for partition functions)
    and zpe_fac (for ZPE). Auto-lookup must use zpe_fac for ZPE, even
    when zpe_fac != harm_fac."""
    entry = scaling_data_dict[canonicalize_level("HF/6-31G(d)")]
    # Sanity: the two factors must actually differ for this LOT,
    # otherwise the test isn't proving anything.
    assert entry.zpe_fac != entry.harm_fac, "fixture LOT has identical zpe/harm factors"

    r_auto = compute_thermo(WATER_HF)
    # Build the comparison: ZPE computed with zpe_fac, H/S with harm_fac.
    T = 298.15
    conc = ATMOS / (GAS_CONSTANT * T)
    correct = calc_bbe(WATER_HF, "grimme", False, 100.0, 100.0, T, conc,
                       entry.harm_fac, "none", None, None,
                       zpe_scale_fac=entry.zpe_fac)
    # And the wrong version (harm_fac for both) for comparison.
    wrong = calc_bbe(WATER_HF, "grimme", False, 100.0, 100.0, T, conc,
                     entry.harm_fac, "none", None, None)

    assert r_auto.zpe == correct.zpe
    assert r_auto.zpe != wrong.zpe       # confirms zpe_fac != harm_fac path


def test_explicit_freq_scale_factor_overrides_lookup():
    """User-supplied freq_scale_factor must skip the DB lookup."""
    r_auto = compute_thermo(WATER_HF)                          # → 0.922
    r_explicit = compute_thermo(WATER_HF, freq_scale_factor=1.0)
    # Different scale factors → different ZPEs → different qh-G values.
    assert not math.isclose(r_auto.qh_gibbs_free_energy,
                            r_explicit.qh_gibbs_free_energy, abs_tol=1e-9)


def test_explicit_concentration_changes_entropy():
    """Doubling concentration must change translational entropy."""
    r_default = compute_thermo(WATER_HF)
    r_solution = compute_thermo(WATER_HF, concentration=1.0)
    assert not math.isclose(r_default.qh_entropy, r_solution.qh_entropy,
                            abs_tol=1e-12)


def test_explicit_temperature():
    r_298 = compute_thermo(WATER_HF, temperature=298.15)
    r_400 = compute_thermo(WATER_HF, temperature=400.0)
    assert r_298.qh_gibbs_free_energy != r_400.qh_gibbs_free_energy


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def test_default_freq_scale_factor_picks_db_value_for_known_lot():
    r = compute_thermo(WATER_HF)
    entry = scaling_data_dict[canonicalize_level("HF/6-31G(d)")]
    T, conc = 298.15, ATMOS / (GAS_CONSTANT * 298.15)
    direct = calc_bbe(WATER_HF, "grimme", False, 100.0, 100.0, T, conc,
                      entry.harm_fac, "none", None, None,
                      zpe_scale_fac=entry.zpe_fac)
    assert r.zpe == direct.zpe


def test_compute_thermo_requires_path_or_qcdata():
    with pytest.raises(ValueError, match="path.*qcdata"):
        compute_thermo()


def test_compute_thermo_accepts_pre_parsed_qcdata():
    """Passing qcdata avoids re-parsing — must yield the same result."""
    from goodvibes.io import parse_qcdata
    qc = parse_qcdata(WATER_HF)
    r_via_path = compute_thermo(WATER_HF)
    r_via_qcdata = compute_thermo(WATER_HF, qcdata=qc)
    assert r_via_path.qh_gibbs_free_energy == r_via_qcdata.qh_gibbs_free_energy


# ---------------------------------------------------------------------------
# compute_batch
# ---------------------------------------------------------------------------

def test_compute_batch_returns_list():
    rs = compute_batch([WATER_HF, WATER_HF_HARMONIC])
    assert isinstance(rs, list)
    assert len(rs) == 2
    assert all(isinstance(r, ThermoResult) for r in rs)


def test_compute_batch_preserves_order():
    rs = compute_batch([WATER_HF_HARMONIC, WATER_HF])
    assert rs[0].file == WATER_HF_HARMONIC
    assert rs[1].file == WATER_HF


def test_compute_batch_forwards_kwargs():
    """kwargs must be forwarded to every compute_thermo call."""
    rs_default = compute_batch([WATER_HF, WATER_HF])
    rs_qh = compute_batch([WATER_HF, WATER_HF], QH=True)
    assert rs_default[0].qh_enthalpy is None
    assert rs_qh[0].qh_enthalpy is not None


def test_compute_batch_empty_input():
    assert compute_batch([]) == []


# ---------------------------------------------------------------------------
# compute_batch parallelism (item 8)
# ---------------------------------------------------------------------------

def test_compute_batch_jobs_2_matches_sequential():
    """jobs=2 must produce bit-identical results to jobs=1 (input order)."""
    files = [WATER_HF, WATER_HF_HARMONIC, WATER_HF]
    seq = compute_batch(files, jobs=1)
    par = compute_batch(files, jobs=2)
    assert [r.file for r in seq] == [r.file for r in par]
    for s, p in zip(seq, par):
        assert s.qh_gibbs_free_energy == p.qh_gibbs_free_energy
        assert s.scf_energy == p.scf_energy


def test_compute_batch_jobs_0_uses_cpu_count():
    """jobs=0 should resolve to os.cpu_count() and still work."""
    rs = compute_batch([WATER_HF, WATER_HF_HARMONIC], jobs=0)
    assert len(rs) == 2
    assert rs[0].name == "01a_water_hf_freq"


def test_compute_batch_single_file_skips_executor(monkeypatch):
    """One-file batches should bypass ProcessPoolExecutor (pure overhead)."""
    import concurrent.futures as cf
    called = {"n": 0}
    real_init = cf.ProcessPoolExecutor.__init__

    def spy(self, *args, **kwargs):
        called["n"] += 1
        return real_init(self, *args, **kwargs)
    monkeypatch.setattr(cf.ProcessPoolExecutor, "__init__", spy)

    rs = compute_batch([WATER_HF], jobs=4)
    assert len(rs) == 1
    assert called["n"] == 0     # never instantiated


# ---------------------------------------------------------------------------
# --jobs CLI flag (end-to-end)
# ---------------------------------------------------------------------------

def test_cli_jobs_flag_matches_sequential(tmp_path):
    """`--jobs 2` must produce the same .dat output (modulo run-time
    metadata) as the default sequential run."""
    import subprocess
    import sys

    env = {**os.environ, "PYTHONPATH": REPO_ROOT}
    files = [WATER_HF, WATER_HF_HARMONIC]

    seq = subprocess.run(
        [sys.executable, "-m", "goodvibes", *files, "--output", "seq"],
        capture_output=True, text=True, cwd=tmp_path, env=env,
    )
    par = subprocess.run(
        [sys.executable, "-m", "goodvibes", *files, "--jobs", "2", "--output", "par"],
        capture_output=True, text=True, cwd=tmp_path, env=env,
    )
    assert seq.returncode == 0, f"seq stderr:\n{seq.stderr}"
    assert par.returncode == 0, f"par stderr:\n{par.stderr}"

    # Compare the per-file rows (lines starting with 'o ' followed by a name).
    def thermo_rows(text):
        return [ln for ln in text.splitlines()
                if ln.startswith("o      ") and "Found" not in ln]
    assert thermo_rows(seq.stdout) == thermo_rows(par.stdout)


# ---------------------------------------------------------------------------
# DataFrame export (optional pandas)
# ---------------------------------------------------------------------------

def test_to_dataframe_round_trip():
    pd = pytest.importorskip("pandas")
    rs = compute_batch([WATER_HF, WATER_HF_HARMONIC])
    df = to_dataframe(rs)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "qh_gibbs_free_energy" in df.columns
    assert "name" in df.columns
    # bbe / qcdata / freq lists are dropped (not DataFrame-friendly).
    assert "bbe" not in df.columns
    assert "qcdata" not in df.columns
    assert "frequency_wn" not in df.columns


def test_to_dataframe_values_match_results():
    pd = pytest.importorskip("pandas")
    rs = compute_batch([WATER_HF])
    df = to_dataframe(rs)
    assert df.loc[0, "qh_gibbs_free_energy"] == rs[0].qh_gibbs_free_energy
    assert df.loc[0, "name"] == rs[0].name


# ---------------------------------------------------------------------------
# Parquet export (v5.0 item 10)
# ---------------------------------------------------------------------------

def test_to_parquet_round_trip(tmp_path):
    """Parquet output round-trips the same numeric values as to_dataframe."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from goodvibes import to_parquet

    rs = compute_batch([WATER_HF, WATER_HF_HARMONIC])
    out = tmp_path / "results.parquet"
    to_parquet(rs, str(out))
    df = pd.read_parquet(out)
    assert len(df) == 2
    for r, row in zip(rs, df.itertuples(index=False)):
        assert row.qh_gibbs_free_energy == r.qh_gibbs_free_energy
        assert row.name == r.name


def test_cli_parquet_flag_writes_file(tmp_path):
    """--parquet PATH end-to-end via subprocess."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import subprocess
    import sys

    out = tmp_path / "results.parquet"
    res = subprocess.run(
        [sys.executable, "-m", "goodvibes", WATER_HF,
         "--parquet", str(out), "--output", "parquet_smoke"],
        capture_output=True, text=True, cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": REPO_ROOT},
    )
    assert res.returncode == 0, f"stderr:\n{res.stderr}"
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 1
    expected = compute_thermo(WATER_HF)
    assert math.isclose(df.iloc[0]["qh_gibbs_free_energy"],
                        expected.qh_gibbs_free_energy, abs_tol=1e-9)


def test_to_parquet_top_level_export():
    """`from goodvibes import to_parquet` must work."""
    import goodvibes
    assert hasattr(goodvibes, "to_parquet")
    assert "to_parquet" in goodvibes.__all__


# ---------------------------------------------------------------------------
# ThermoOptions + calc_bbe.from_options (v5.0 item 12)
# ---------------------------------------------------------------------------

def test_thermo_options_defaults_match_compute_thermo():
    """ThermoOptions defaults must match compute_thermo's defaults so
    calc_bbe.from_options(path, ThermoOptions()) == compute_thermo(path)."""
    from goodvibes import ThermoOptions
    opts = ThermoOptions()
    assert opts.QS == "grimme"
    assert opts.QH is False
    assert opts.s_freq_cutoff == 100.0
    assert opts.h_freq_cutoff == 100.0
    assert opts.temperature == 298.15
    assert opts.concentration is None
    assert opts.freq_scale_factor is None
    assert opts.zpe_scale_factor is None
    assert opts.symm is False
    assert opts.inertia == "global"


def test_thermo_options_is_frozen():
    """Frozen so it's safe across worker processes and shared instances."""
    from goodvibes import ThermoOptions
    opts = ThermoOptions()
    with pytest.raises(Exception):       # FrozenInstanceError
        opts.QS = "truhlar"


def test_from_options_with_path_matches_direct_calc_bbe():
    """calc_bbe.from_options(path, opts) must yield bit-identical numbers
    to a direct calc_bbe(...) call with the equivalent kwargs."""
    from goodvibes import ThermoOptions
    T = 298.15
    conc = ATMOS / (GAS_CONSTANT * T)
    entry = scaling_data_dict[canonicalize_level("HF/6-31G(d)")]

    opts = ThermoOptions(
        QH=True, temperature=T, concentration=conc,
        freq_scale_factor=entry.harm_fac,
        zpe_scale_factor=entry.zpe_fac,
    )
    via_opts = calc_bbe.from_options(WATER_HF, opts)
    direct = calc_bbe(WATER_HF, "grimme", True, 100.0, 100.0, T, conc,
                      entry.harm_fac, None, None, None,
                      zpe_scale_fac=entry.zpe_fac)
    assert via_opts.qh_gibbs_free_energy == direct.qh_gibbs_free_energy
    assert via_opts.qh_enthalpy == direct.qh_enthalpy
    assert via_opts.zpe == direct.zpe


def test_from_options_with_qcdata_skips_reparse():
    """Passing a pre-parsed QCData avoids the second parse_qcdata call."""
    from goodvibes import ThermoOptions
    from goodvibes.io import parse_qcdata

    qc = parse_qcdata(WATER_HF)
    opts = ThermoOptions()
    via_qcdata = calc_bbe.from_options(qc, opts)
    via_path = calc_bbe.from_options(WATER_HF, opts)
    # Same calc_bbe.scf_energy proves both went through the same parser.
    assert via_qcdata.scf_energy == via_path.scf_energy


def test_legacy_calc_bbe_constructor_emits_deprecation_warning():
    """Direct calc_bbe(file, QS, QH, ...) must trip the DeprecationWarning
    pointing users at from_options / compute_thermo."""
    with pytest.warns(DeprecationWarning, match="from_options"):
        calc_bbe(WATER_HF, "grimme", False, 100.0, 100.0, 298.15,
                 0.040874, 0.922, None, None, None)


def test_compute_thermo_does_not_emit_deprecation_warning():
    """The high-level façade routes through from_options internally —
    must not trip the legacy-constructor warning."""
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        compute_thermo(WATER_HF)
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, DeprecationWarning)]
    assert not any("calc_bbe" in m for m in msgs), (
        f"compute_thermo should not emit calc_bbe DeprecationWarning, "
        f"but got: {msgs}"
    )


def test_from_options_does_not_emit_deprecation_warning():
    """Same contract as compute_thermo — from_options is the v5.0 path."""
    import warnings
    from goodvibes import ThermoOptions
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        calc_bbe.from_options(WATER_HF, ThermoOptions())
    msgs = [str(w.message) for w in caught
            if issubclass(w.category, DeprecationWarning)]
    assert not any("calc_bbe" in m for m in msgs)


def test_thermo_options_re_exported_from_top_level():
    """`from goodvibes import ThermoOptions` must work alongside
    ThermoResult and compute_thermo."""
    import goodvibes
    assert hasattr(goodvibes, "ThermoOptions")
    assert "ThermoOptions" in goodvibes.__all__


def test_vscal_alone_inherits_for_zpe():
    """--vscal X (freq_scale_factor=X with zpe_scale_factor=None) must
    apply X to ZPE too, matching v3.x and v4.x.0 behaviour where
    --vscal was the single scale factor that scaled everything.
    Regression: an earlier draft of split-scaling auto-looked-up
    zpe_fac independently, surprising users with pinned scripts."""
    from goodvibes import ThermoOptions
    opts_vscal_only = ThermoOptions(freq_scale_factor=1.0)
    bbe = calc_bbe.from_options(WATER_HF, opts_vscal_only)

    # Compare against an unscaled reference (both factors = 1.0).
    opts_both = ThermoOptions(freq_scale_factor=1.0, zpe_scale_factor=1.0)
    bbe_ref = calc_bbe.from_options(WATER_HF, opts_both)
    assert bbe.zpe == bbe_ref.zpe


def test_zpe_vscal_alone_lets_freq_autolookup():
    """--zpe-vscal Y alone leaves freq_scale_factor to auto-lookup
    harm_fac. ZPE uses Y, partition functions use harm_fac."""
    from goodvibes import ThermoOptions
    entry = scaling_data_dict[canonicalize_level("HF/6-31G(d)")]
    opts = ThermoOptions(zpe_scale_factor=0.5)
    bbe = calc_bbe.from_options(WATER_HF, opts)

    # ZPE matches a separate calc with zpe=0.5
    ref = calc_bbe.from_options(
        WATER_HF,
        ThermoOptions(freq_scale_factor=entry.harm_fac, zpe_scale_factor=0.5),
    )
    assert bbe.zpe == ref.zpe
    assert bbe.qh_entropy == ref.qh_entropy


# ---------------------------------------------------------------------------
# bbe_to_result (CLI adapter)
# ---------------------------------------------------------------------------

def test_bbe_to_result_matches_compute_thermo():
    """bbe_to_result(bbe, path) must produce the same fields as
    compute_thermo(path), since both go through the same projection."""
    from goodvibes import bbe_to_result
    r_via_compute = compute_thermo(WATER_HF)
    r_via_adapter = bbe_to_result(r_via_compute.bbe, WATER_HF)
    assert r_via_adapter.scf_energy == r_via_compute.scf_energy
    assert r_via_adapter.qh_gibbs_free_energy == r_via_compute.qh_gibbs_free_energy
    assert r_via_adapter.level_of_theory == r_via_compute.level_of_theory
    assert r_via_adapter.name == r_via_compute.name


def test_bbe_to_result_skips_file_scan_when_lot_supplied():
    """Passing level_of_theory explicitly avoids the read_initial scan."""
    from goodvibes import bbe_to_result
    r = compute_thermo(WATER_HF)
    r2 = bbe_to_result(r.bbe, WATER_HF, level_of_theory="MP2/cc-pVTZ")
    assert r2.level_of_theory == "MP2/cc-pVTZ"


# ---------------------------------------------------------------------------
# --csv CLI flag (end-to-end)
# ---------------------------------------------------------------------------

def test_cli_csv_flag_writes_dataframe(tmp_path):
    """Subprocess: `goodvibes <file> --csv out.csv` writes a one-row CSV
    with columns matching ThermoResult and values matching compute_thermo."""
    pd = pytest.importorskip("pandas")
    import subprocess
    import sys

    out = tmp_path / "results.csv"
    res = subprocess.run(
        [sys.executable, "-m", "goodvibes", WATER_HF,
         "--csv", str(out), "--output", "csv_smoke"],
        capture_output=True, text=True, cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": REPO_ROOT},
    )
    assert res.returncode == 0, f"stderr:\n{res.stderr}"
    assert out.exists()
    df = pd.read_csv(out)
    assert len(df) == 1
    assert df.loc[0, "name"] == "01a_water_hf_freq"
    # Float compared as string in CSV — load and compare numerically.
    expected = compute_thermo(WATER_HF)
    assert math.isclose(df.loc[0, "qh_gibbs_free_energy"],
                        expected.qh_gibbs_free_energy, abs_tol=1e-9)


def test_cli_csv_flag_one_row_per_input(tmp_path):
    pd = pytest.importorskip("pandas")
    import subprocess
    import sys

    out = tmp_path / "multi.csv"
    res = subprocess.run(
        [sys.executable, "-m", "goodvibes", WATER_HF, WATER_HF_HARMONIC,
         "--csv", str(out), "--output", "csv_multi"],
        capture_output=True, text=True, cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": REPO_ROOT},
    )
    assert res.returncode == 0, f"stderr:\n{res.stderr}"
    df = pd.read_csv(out)
    assert len(df) == 2
    assert set(df["name"]) == {"01a_water_hf_freq", "01b_water_hf_freq_scaled"}
