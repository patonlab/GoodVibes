"""Tests for the --media / --freespace CLI flags and the goodvibes.media module.

Covers three layers:
  1. Direct API: lookup_solvent + compute_media_conc semantics.
  2. get_free_space behavior and the FREESPACE_SOLVENTS list.
  3. CLI integration: subprocess-based runs that exercise --media / --freespace
     happy paths and error paths end-to-end.
"""

import math
import os
import subprocess
import sys
import warnings

import pytest

from goodvibes.media import compute_media_conc, lookup_solvent
from goodvibes.thermo import FREESPACE_SOLVENTS, get_free_space


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_DIR = os.path.join(ROOT_DIR, 'goodvibes', 'examples', 'media_conc')


# ---------------------------------------------------------------------------
# Direct API: lookup_solvent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alias", [
    'h2o', 'water', 'H2O', 'WATER',  # canonical and aliases, case-insensitive
    'benzene', 'meoh', 'methanol',
    'dmso', 'dmf', 'thf',
    'acetonitrile', 'mecn',
    'chloroform', 'dcm',
])
def test_lookup_solvent_known(alias):
    """Known aliases (any case) return a (mw, density) tuple of positive floats."""
    mw, density = lookup_solvent(alias)
    assert mw > 0
    assert density > 0


def test_lookup_solvent_unknown_raises():
    """Unknown solvent names raise ValueError with a helpful message."""
    with pytest.raises(ValueError, match="Unknown solvent"):
        lookup_solvent('not_a_real_solvent')


def test_lookup_solvent_error_lists_aliases():
    """The error message hints at common aliases so the user can self-correct."""
    with pytest.raises(ValueError) as exc_info:
        lookup_solvent('definitely_made_up')
    msg = str(exc_info.value)
    assert any(name in msg for name in ['water', 'dmso', 'acetone',
                                         'methanol', 'thf', 'dmf'])
    assert 'solvents.json' in msg


def test_lookup_solvent_typo_suggestions():
    """A near-miss typo gets a 'Did you mean' suggestion via difflib."""
    with pytest.raises(ValueError) as exc_info:
        lookup_solvent('wate')      # one char off from 'water'
    assert 'water' in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        lookup_solvent('benzeen')   # transposed from 'benzene'
    assert 'benzene' in str(exc_info.value)


# ---------------------------------------------------------------------------
# Direct API: compute_media_conc
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("media, fname, expected_M", [
    ('h2o', 'H2O.log', 1000 * 0.998 / 18.02),       # ≈ 55.38 M
    ('benzene', 'Benzene.log', 1000 * 0.8765 / 78.11),  # ≈ 11.22 M
    ('meoh', 'MeOH.log', 1000 * 0.7913 / 32.04),    # ≈ 24.70 M
])
def test_compute_media_conc_matched_file(media, fname, expected_M):
    """When the file basename matches the solvent alias, the neat
    concentration n*ρ/M is returned in mol/L."""
    fpath = os.path.join(MEDIA_DIR, fname)
    conc = compute_media_conc(media, fpath)
    assert math.isclose(conc, expected_M, rel_tol=1e-3)


def test_compute_media_conc_mismatched_file_returns_none():
    """When the solvent name doesn't match the file's basename, the
    correction is silently skipped (returns None) — running --media h2o on
    a benzene calc shouldn't apply a water correction."""
    fpath = os.path.join(MEDIA_DIR, 'Benzene.log')
    assert compute_media_conc('h2o', fpath) is None


# ---------------------------------------------------------------------------
# get_free_space and FREESPACE_SOLVENTS
# ---------------------------------------------------------------------------

def test_freespace_solvents_constant_matches_thermo():
    """The CLI-level FREESPACE_SOLVENTS list matches what get_free_space accepts."""
    assert FREESPACE_SOLVENTS == ("H2O", "toluene", "DMF", "AcOH", "chloroform")


@pytest.mark.parametrize("solv", FREESPACE_SOLVENTS)
def test_get_free_space_supported(solv):
    """Each supported solvent returns a free volume strictly less than the
    gas-phase fallback (1000 mL/L)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        v = get_free_space(solv)
    assert 0 < v < 1000.0


def test_get_free_space_unknown_returns_fallback_with_warning():
    """Unknown solvents trigger a UserWarning and return 1000 mL/L (gas)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        v = get_free_space("not_a_real_solvent")
    assert v == 1000.0
    assert any(issubclass(w.category, UserWarning) and 'not recognized' in str(w.message)
               for w in caught)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def _run_cli(args, cwd=None):
    """Run `python -m goodvibes <args>` and return CompletedProcess.

    PYTHONPATH is forced to the repo root so the subprocess imports the
    local source under test, not whatever `goodvibes` happens to be
    installed in site-packages. Tests can pass cwd=tmp_path to keep
    output .dat files out of the repo without losing the local-source
    guarantee.
    """
    cmd = [sys.executable, '-m', 'goodvibes'] + list(args)
    env = {**os.environ, 'PYTHONPATH': ROOT_DIR}
    return subprocess.run(cmd, capture_output=True, text=True,
                          cwd=cwd or ROOT_DIR, env=env)


def _h2o_path():
    return os.path.join(MEDIA_DIR, 'H2O.log')


def _benzene_path():
    return os.path.join(MEDIA_DIR, 'Benzene.log')


def test_cli_media_valid_solvent(tmp_path):
    """--media h2o on the H2O.log fixture completes successfully and prints
    the standard-state correction header."""
    res = _run_cli([_h2o_path(), '--media', 'h2o', '--output', 'mediatest'],
                   cwd=tmp_path)
    assert res.returncode == 0, f"stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    assert 'concentration correction' in res.stdout.lower() \
        or 'concentration correction' in res.stderr.lower()


def test_cli_media_mismatched_file_still_succeeds(tmp_path):
    """--media h2o on a non-water file is a silent no-op for the per-file
    correction; the run still completes."""
    res = _run_cli([_benzene_path(), '--media', 'h2o', '--output', 'mediatest'],
                   cwd=tmp_path)
    assert res.returncode == 0


def test_cli_media_unknown_solvent_fatal(tmp_path):
    """--media not_a_real_solvent fails fast with a helpful FATAL ERROR."""
    res = _run_cli([_h2o_path(), '--media', 'not_a_real_solvent',
                    '--output', 'mediatest'], cwd=tmp_path)
    assert res.returncode != 0
    out = res.stdout + res.stderr
    assert 'FATAL ERROR' in out
    assert 'Unknown solvent' in out
    # Helpful hint at known aliases is in the message
    assert 'solvents.json' in out


def test_cli_freespace_valid_solvent(tmp_path):
    """--freespace H2O completes and mentions the free-volume correction."""
    res = _run_cli([_h2o_path(), '--freespace', 'H2O', '--output', 'fstest'],
                   cwd=tmp_path)
    assert res.returncode == 0, f"stderr:\n{res.stderr}\nstdout:\n{res.stdout}"
    assert 'free volume' in res.stdout.lower()


def test_cli_freespace_unknown_solvent_fatal(tmp_path):
    """--freespace not_real fails fast with a list of supported solvents."""
    res = _run_cli([_h2o_path(), '--freespace', 'not_real',
                    '--output', 'fstest'], cwd=tmp_path)
    assert res.returncode != 0
    out = res.stdout + res.stderr
    assert 'FATAL ERROR' in out
    assert 'freespace' in out.lower() or 'free' in out.lower()
    # Each supported solvent name appears in the hint
    for s in FREESPACE_SOLVENTS:
        assert s in out


def test_cli_freespace_case_sensitive(tmp_path):
    """--freespace is case-sensitive (H2O works, h2o does not). This is a
    regression guard: callers that lowercase the input would silently fall
    back to gas phase before the explicit-error change."""
    res = _run_cli([_h2o_path(), '--freespace', 'h2o', '--output', 'fstest'],
                   cwd=tmp_path)
    assert res.returncode != 0
    assert 'FATAL ERROR' in res.stdout + res.stderr
