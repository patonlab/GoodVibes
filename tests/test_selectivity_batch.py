"""compute_selectivity_batch: many jobs over temperatures, entropy cutoffs
and conformer windows, and the summary of the sweep."""
import os

import pytest

from conftest import g16path
from goodvibes import (ConformerSet, compute_selectivity, compute_selectivity_batch, compute_thermo,
                       read_xyz_frames, summarize_selectivity)
from goodvibes.selectivity import compute_selectivity_lowest_only

A = [g16path("01a_water_hf_freq.log"), g16path("01c_water_hf_freq_isotopes.log")]
B = [g16path("01b_water_hf_freq_scaled.log")]
JOBS = {"w": {"A": A, "B": B}}


def _reference(T, **options):
    """compute_selectivity on files computed directly at T with options."""
    td = {f: compute_thermo(f, temperature=T, **options).bbe for f in A + B}
    return td, {"A": A, "B": B}


def test_the_sweep_has_one_row_per_condition():
    rows = compute_selectivity_batch(JOBS, [298.15, 350.0], s_freq_cutoffs=[50, 150],
                                     conformer_windows=[0, 1.5], records=True)
    assert len(rows) == 2 * 3 * 3
    first = rows[0]
    assert (first["job"], first["temperature"], first["s_freq_cutoff"], first["conformer_window"]) == \
        ("w", 298.15, 50.0, None)
    assert [r["nominal"] for r in rows].count(True) == 2              # one per temperature
    assert first["labels"] == "A,B" and first["quantity"] == "qh_gibbs"
    assert first["population[A]"] + first["population[B]"] == pytest.approx(1.0)
    assert (first["n[A]"], first["n[B]"]) == (2, 1)


@pytest.mark.parametrize("T", [298.15, 350.0])
def test_the_nominal_row_matches_compute_selectivity(T):
    rows = compute_selectivity_batch(JOBS, [T], records=True)
    (row,) = rows
    ref = compute_selectivity(*_reference(T), T)
    assert row["nominal"] and row["major"] == ref.major
    assert row["ee"] == pytest.approx(ref.ee_signed)
    assert row["population[A]"] == pytest.approx(ref.populations["A"])


def test_a_cutoff_row_matches_a_direct_calculation():
    rows = compute_selectivity_batch(JOBS, [298.15], s_freq_cutoffs=[50], records=True)
    row = next(r for r in rows if r["s_freq_cutoff"] == 50.0)
    ref = compute_selectivity(*_reference(298.15, s_freq_cutoff=50.0), 298.15)
    assert row["population[A]"] == pytest.approx(ref.populations["A"], rel=1e-9)
    assert not row["nominal"]


def test_a_zero_window_is_lowest_only():
    rows = compute_selectivity_batch(JOBS, [298.15], conformer_windows=[0], records=True)
    row = next(r for r in rows if r["conformer_window"] == 0.0)
    ref = compute_selectivity_lowest_only(*_reference(298.15), 298.15)
    assert row["population[A]"] == pytest.approx(ref.populations["A"])
    assert row["n[A]"] == 1


def test_a_dataframe_by_default():
    pytest.importorskip("pandas")
    df = compute_selectivity_batch(JOBS, [298.15], s_freq_cutoffs=[50])
    assert list(df["s_freq_cutoff"]) == [50.0, 100.0] and df["nominal"].tolist() == [False, True]


def test_structures_can_be_globs_results_conformer_sets_and_frames(tmp_path):
    results = [compute_thermo(f) for f in A]
    pattern = os.path.join(os.path.dirname(B[0]), "01b_water*.log")
    rows = compute_selectivity_batch({
        "glob": {"A": results, "B": pattern},
        "set": {"A": ConformerSet.from_results("A", results), "B": B},
    }, records=True)
    assert rows[0]["ee"] == pytest.approx(rows[1]["ee"])
    xyz = tmp_path / "crest.xyz"
    water = "O 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n"
    xyz.write_text(f"3\n-5.0700\n{water}3\n-5.0690\n{water}3\n-5.0660\n{water}", encoding="utf-8")
    frames = read_xyz_frames(str(xyz))
    (row,) = compute_selectivity_batch({"crest": {"low": frames[:2], "high": frames[2:]}},
                                       quantity="electronic", records=True)
    assert row["major"] == "low" and row["quantity"] == "electronic"
    with pytest.raises(ValueError, match="needs quantity='electronic'"):
        compute_selectivity_batch({"crest": {"low": frames[:2], "high": frames[2:]}}, records=True)


@pytest.mark.parametrize("jobs, kwargs, error, match", [
    ({"j": {"A": A}}, {}, ValueError, "at least two labels"),
    (JOBS, {"conformer_windows": [-1]}, ValueError, "windows must be"),
    (JOBS, {"temperatures": [0]}, ValueError, "temperatures must be positive"),
    ({"j": {"A": A, "B": "nothing_here_*.log"}}, {}, ValueError, "no files match"),
    ({"j": {"A": A, "B": [42]}}, {}, TypeError, "cannot use a int"),
])
def test_bad_jobs(jobs, kwargs, error, match):
    with pytest.raises(error, match=match):
        compute_selectivity_batch(jobs, records=True, **kwargs)


def test_the_summary_states_the_nominal_value_and_its_range():
    rows = [
        {"job": "j", "temperature": 298.15, "s_freq_cutoff": c, "conformer_window": w,
         "nominal": c == 100.0 and w is None, "ee": ee}
        for c, w, ee in [(50.0, None, 88.2), (100.0, None, 92.1), (150.0, None, 93.8),
                         (100.0, 1.0, 90.4), (100.0, 3.0, 92.0)]
    ]
    (s,) = summarize_selectivity(rows)
    assert (s["nominal"], s["low"], s["high"]) == (92.1, 88.2, 93.8)
    assert s["text"] == ("ee +92 % (88 to 94 % over s_freq_cutoff 50–150 cm⁻¹, "
                         "conformer window 1–3 kcal/mol)")
    flat = [dict(r, ee=92.1) for r in rows]
    assert summarize_selectivity(flat)[0]["text"] == "ee +92 %"
    (d,) = summarize_selectivity([dict(rows[1], ddG=1.234)], "ddG", decimals=2)
    assert d["text"] == "ddG 1.23 kcal/mol"
