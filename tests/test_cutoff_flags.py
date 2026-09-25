"""-f sets both cut-offs; --fs / --fh override it for their own quantity.

Regression: a non-default -f used to overwrite explicit --fs/--fh values,
so `-f 50 --fh 80` applied 50 cm-1 to the enthalpy as well.
"""
import pytest

from test_cli_errors import build_options


@pytest.mark.parametrize('extra, expected_s, expected_h', [
    ([], 100.0, 100.0),
    (['-f', '50'], 50.0, 50.0),
    (['--tau', '75'], 75.0, 75.0),
    (['--fs', '30'], 30.0, 100.0),
    (['--fh', '30'], 100.0, 30.0),
    (['-f', '50', '--fh', '80'], 50.0, 80.0),
    (['-f', '50', '--fs', '80'], 80.0, 50.0),
    (['-f', '100', '--fs', '60', '--fh', '70'], 60.0, 70.0),
])
def test_cutoff_resolution(monkeypatch, extra, expected_s, expected_h):
    options, _ = build_options(monkeypatch, extra=extra)
    assert options.S_freq_cutoff == expected_s
    assert options.H_freq_cutoff == expected_h
