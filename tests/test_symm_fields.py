"""--symm / symm=True: the detected point group and symmetry number must be reported, and
pymsym's symmetry number must replace (not stack on) one already present in the output file."""

import math
import re

import pytest

from goodvibes import compute_thermo
from goodvibes.constants import GAS_CONSTANT

from conftest import ase_path

pymsym = pytest.importorskip("pymsym")

R_KCAL = GAS_CONSTANT / 4184.0  # kcal/(mol K)
T = 298.15


@pytest.fixture
def water_no_symm(tmp_path):
    """The water fixture with its point_group / symmno keys stripped, i.e. what an ASE-driven
    calculation that does not know its symmetry writes."""
    text = open(ase_path("01_water.extxyz")).read()
    lines = text.splitlines()
    lines[1] = re.sub(r"\s(point_group|symmno)=\S+", "", lines[1])
    out = tmp_path / "water_nosym.extxyz"
    out.write_text("\n".join(lines) + "\n")
    return str(out)


def test_symm_reports_detected_group_and_sigma(water_no_symm):
    plain = compute_thermo(water_no_symm, symm=False)
    sym = compute_thermo(water_no_symm, symm=True)
    assert (plain.point_group or "") == "" and plain.symmno == 1
    assert sym.point_group == "C2v" and sym.symmno == 2
    # sigma = 2 removes R ln 2 of rotational entropy; G rises by RT ln 2 (~0.41 kcal/mol)
    dS = (sym.entropy - plain.entropy) * 627.509541          # kcal/(mol K)
    assert dS == pytest.approx(-R_KCAL * math.log(2), rel=1e-6)
    for a, b in [(sym.gibbs_free_energy, plain.gibbs_free_energy), (sym.qh_gibbs_free_energy, plain.qh_gibbs_free_energy)]:
        assert (a - b) * 627.509541 == pytest.approx(R_KCAL * T * math.log(2), rel=1e-6)


def test_symm_does_not_double_count_file_symmno():
    """01_water.extxyz already says symmno=2; --symm must leave the entropy unchanged."""
    f = ase_path("01_water.extxyz")
    plain = compute_thermo(f, symm=False)
    sym = compute_thermo(f, symm=True)
    assert plain.symmno == 2 and sym.symmno == 2
    assert sym.point_group == "C2v"
    assert sym.entropy == pytest.approx(plain.entropy, abs=1e-12)
    assert sym.gibbs_free_energy == pytest.approx(plain.gibbs_free_energy, abs=1e-12)


def test_symm_off_keeps_file_values():
    r = compute_thermo(ase_path("01_water.extxyz"), symm=False)
    assert r.point_group == "C2V" and r.symmno == 2  # verbatim from the file
