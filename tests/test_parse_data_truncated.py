# -*- coding: utf-8 -*-
"""Regression test: parse_data on a Gaussian output with no route section."""

from goodvibes.io import parse_data


def test_parse_data_truncated_gaussian(tmp_path):
    # A truncated/minimal Gaussian file (no '#' route line) must parse
    # without UnboundLocalError on empirical_dispersion
    out = tmp_path / "trunc.log"
    out.write_text(" Gaussian 16\n Full point group                 D*H\n")
    spe, program, _version, _solv, _file, _chg, emp_disp, _mult = parse_data(str(out))
    assert program == 'Gaussian'
    assert emp_disp == ''
