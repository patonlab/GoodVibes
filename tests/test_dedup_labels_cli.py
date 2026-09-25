"""--dedup with --label must not collapse an R/S pair of transition states."""
import json
import shutil

import pytest

from conftest import g16path

from test_cli_errors import gv_logger_cleanup, run_main  # noqa: F401  (fixture re-export)

TS = g16path('44_ts_sn2_identity_chloride.log')


def _pair(tmp_path):
    # The same output under two names stands in for a pair of enantiomeric TSs:
    # identical energy and rotational constants, exactly what dedup keys on.
    shutil.copy(TS, tmp_path / 'sn2_TS_R.log')
    shutil.copy(TS, tmp_path / 'sn2_TS_S.log')
    return [str(tmp_path / 'sn2_TS_R.log'), str(tmp_path / 'sn2_TS_S.log')]


def _populations(tmp_path, extra, out):
    run_main(monkeypatch=extra[0], tmp_path=tmp_path,
             args=extra[1] + ['--label', 'R=*_R*', '--label', 'S=*_S*', '--dedup', '--json', out] + extra[2])
    payload = json.loads((tmp_path / out).read_text())
    return payload['selectivity']['results'][0]['populations']


def test_dedup_is_scoped_per_label_by_default(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    files = _pair(tmp_path)
    pops = _populations(tmp_path, (monkeypatch, files, []), 'scoped.json')
    assert pops['R'] == pytest.approx(0.5) and pops['S'] == pytest.approx(0.5)
    assert 'scoped within each labelled species' in (tmp_path / 'GoodVibes_output.dat').read_text()


def test_dedup_global_restores_cross_species_comparison(monkeypatch, tmp_path, gv_logger_cleanup):  # noqa: F811
    files = _pair(tmp_path)
    # Across species the pair is a duplicate, one member is dropped and the
    # other label is left empty: the run stops with a clear error instead of
    # silently reporting 100:0.
    with pytest.raises((SystemExit, ValueError)):
        run_main(monkeypatch, tmp_path, files + ['--label', 'R=*_R*', '--label', 'S=*_S*',
                                                 '--dedup', '--dedup-global', '--output', 'glob'])
    text = (tmp_path / 'GoodVibes_glob.dat').read_text()
    assert 'No files matched' in text or 'duplicate' in text.lower()
