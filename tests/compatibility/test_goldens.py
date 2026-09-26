"""Backwards-compatibility goldens for the CLI.

Each case runs ``python -m goodvibes`` in a scratch directory and compares the
``.dat`` archive (and, where requested, the ``--json`` payload) against a
checked-in golden. The goldens pin the user-visible output of the flag
combinations people rely on, so a refactor that changes a number, a column or
a line of text fails here first and has to update the golden deliberately.

Regenerate after an intentional change with::

    GOODVIBES_UPDATE_GOLDENS=1 pytest tests/compatibility -q

and mention the change in CHANGELOG.md.

Normalisation: the repository root becomes ``<ROOT>`` (with ``/`` separators
on every platform), the version banner becomes ``v<VERSION>``, and in JSON the
``generated_at`` timestamp, ``goodvibes_version``, the per-file ``qcdata``
block (raw parser output, covered by the parser tests) and the input
checksums of the ``profile`` provenance are dropped. JSON numbers
are compared with a relative tolerance so a last-digit difference between
platforms does not count as a regression; ``.dat`` values are printed at six
decimals and are compared verbatim.
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from goodvibes.constants import __version__

ROOT = Path(__file__).resolve().parents[2]
GOLDENS = Path(__file__).resolve().parent / "goldens"
UPDATE = os.environ.get("GOODVIBES_UPDATE_GOLDENS") == "1"

G16 = "tests/g16"
EX = "goodvibes/examples"
GCONF = f"{EX}/gconf_ee_boltz"
PES = f"{EX}/pes"

try:
    import pymsym  # noqa: F401
    HAS_PYMSYM = True
except ImportError:
    HAS_PYMSYM = False

# name, args (paths relative to ROOT), json?
CASES = [
    ("basic", [f"{G16}/01a_water_hf_freq.log"], False),
    ("qh_default", [f"{G16}/01a_water_hf_freq.log", "-q"], True),
    ("truhlar_qh", [f"{G16}/01a_water_hf_freq.log", "--qs", "truhlar", "-q"], False),
    ("temp_conc", [f"{G16}/01a_water_hf_freq.log", "--temp", "350", "-c", "1.0"], False),
    ("cutoffs", [f"{G16}/01a_water_hf_freq.log", "-q", "-f", "50", "--fh", "80", "--fs", "120"], False),
    ("vscal_zpe", [f"{G16}/01a_water_hf_freq.log", "-v", "0.95", "--zpe-vscal", "0.97"], False),
    ("ti_scan", [f"{G16}/01a_water_hf_freq.log", "--ti", "250,350,50", "-q"], True),
    ("boltz", [f"{G16}/01a_water_hf_freq.log", f"{G16}/01c_water_hf_freq_isotopes.log", "--boltz"], True),
    ("boltz_energy", [f"{G16}/01a_water_hf_freq.log", f"{G16}/01c_water_hf_freq_isotopes.log", "--boltz", "energy"], False),
    ("labels", [f"{GCONF}/Aminoxylation_TS1_R.log", f"{GCONF}/Aminoxylation_TS2_S.log",
                "--label", "R=*_R*", "--label", "S=*_S*"], True),
    ("labels_ti", [f"{GCONF}/Aminoxylation_TS1_R.log", f"{GCONF}/Aminoxylation_TS2_S.log",
                   "--label", "R=*_R*", "--label", "S=*_S*", "--ti", "273,333,30"], True),
    ("ee_legacy", [f"{GCONF}/Aminoxylation_TS1_R.log", f"{GCONF}/Aminoxylation_TS2_S.log",
                   "--boltz", "--ee", "*_R*:*_S*"], False),
    ("spc_suffix", [f"{EX}/ethane.out", "--spc", "TZ"], True),
    ("spc_link", [f"{EX}/ethane_spc.out", "--spc", "link", "-q"], False),
    ("media", [f"{EX}/media_conc/H2O.log", f"{EX}/media_conc/MeOH.log", "--media", "MeOH"], False),
    ("sort_dedup", [f"{GCONF}/aminox_cat_conf212_S.log", f"{GCONF}/aminox_cat_conf280_R.log",
                    f"{GCONF}/aminox_cat_conf65_S.log", "--sort", "--dedup"], False),
    ("imag_invert", [f"{G16}/44_ts_sn2_identity_chloride.log", f"{G16}/45_ts_diels_alder_butadiene_ethylene.log",
                     "--imag", "--invert"], False),
    ("check", [f"{G16}/01a_water_hf_freq.log", f"{G16}/02_ethane_opt_freq_T398_P2.log",
               f"{G16}/30_phenol_smd_thf_pbe0_d3bj.log", "--check"], False),
    ("cpu_xyz", [f"{G16}/01a_water_hf_freq.log", "--cpu", "--xyz"], False),
    ("orca6", ["tests/orca6/30_phenol_smd_thf_pbe0_d3bj.out", "-q", "--bav", "conf"], True),
    ("qchem", ["tests/qchem6/01a_water_hf_freq.out", "-q"], False),
    ("xtb", ["tests/xtb/01_water.out", "-q"], False),
    ("ase", ["tests/ase/01_water.extxyz", "tests/ase/44_ts_sn2.extxyz", "-q"], True),
    ("pes_legacy", [f"{GCONF}/aminox_cat_conf65_S.log", f"{GCONF}/aminox_subs_conf713.log",
                    f"{GCONF}/Aminoxylation_TS1_R.log", f"{GCONF}/Aminoxylation_TS2_S.log",
                    "--pes", f"{GCONF}/gconf_TS.yaml"], True),
    ("pes_legacy_nogconf", [f"{GCONF}/aminox_cat_conf65_S.log", f"{GCONF}/aminox_subs_conf713.log",
                            f"{GCONF}/Aminoxylation_TS1_R.log", f"{GCONF}/Aminoxylation_TS2_S.log",
                            "--pes", f"{GCONF}/gconf_TS.yaml", "--nogconf"], True),
    ("pes_legacy_ti", [f"{GCONF}/aminox_cat_conf65_S.log", f"{GCONF}/aminox_subs_conf713.log",
                       f"{GCONF}/Aminoxylation_TS1_R.log", f"{GCONF}/Aminoxylation_TS2_S.log",
                       "--pes", f"{GCONF}/gconf_TS.yaml", "--ti", "298,318,10"], True),
    ("graph_legacy", [f"{GCONF}/aminox_cat_conf65_S.log", f"{GCONF}/aminox_subs_conf713.log",
                      f"{GCONF}/Aminoxylation_TS1_R.log", f"{GCONF}/Aminoxylation_TS2_S.log",
                      "--pes", f"{GCONF}/gconf_TS.yaml", "--graph", f"{GCONF}/gconf_TS.yaml"], False),
    ("pes_v2_azabor", [f"{PES}/*.log", "--spc", "sp_tzpop", "--pes", f"{PES}/azabor_PES_v2.yaml", "-q"], True),
]
if HAS_PYMSYM:
    CASES.append(("symm", [f"{G16}/01a_water_hf_freq.log", f"{G16}/22_hcn_linear_freq_noraman.log", "--symm"], False))


def _expand(args):
    out = []
    for a in args:
        if a.startswith(("tests/", "goodvibes/")):
            if "*" in a:
                out.extend(sorted(str(p) for p in ROOT.glob(a)))
            else:
                out.append(str(ROOT / a))
        else:
            out.append(a)
    return out


def _normalise_text(text):
    root = str(ROOT)
    text = text.replace(root, "<ROOT>")
    if os.sep != "/":
        text = text.replace(root.replace("/", os.sep), "<ROOT>").replace("<ROOT>" + os.sep, "<ROOT>/")
        text = text.replace("\\", "/")
    return text.replace(f"v{__version__}", "v<VERSION>")


def _normalise_json(obj):
    if isinstance(obj, dict):
        # qcdata is raw parser output (coordinates, every frequency, ...) and is
        # covered by the parser tests; keeping it would make the goldens large
        # without pinning anything the thermo/selectivity/pes blocks do not.
        # sha1: the input files' checksums in a profile's provenance depend on
        # the checkout's line endings (Windows), not on GoodVibes.
        return {k: _normalise_json(v) for k, v in obj.items()
                if k not in ("generated_at", "goodvibes_version", "qcdata", "sha1")}
    if isinstance(obj, list):
        return [_normalise_json(v) for v in obj]
    if isinstance(obj, str):
        return _normalise_text(obj)
    return obj


def _assert_json_equal(got, want, path="$"):
    if isinstance(want, dict):
        assert isinstance(got, dict), path
        assert set(got) == set(want), f"{path}: keys {sorted(set(got) ^ set(want))}"
        for k in want:
            _assert_json_equal(got[k], want[k], f"{path}.{k}")
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), f"{path}: length {len(got)} != {len(want)}"
        for i, (g, w) in enumerate(zip(got, want)):
            _assert_json_equal(g, w, f"{path}[{i}]")
    elif isinstance(want, float) or isinstance(got, float):
        assert isinstance(got, (int, float)) and isinstance(want, (int, float)), path
        assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-12), f"{path}: {got!r} != {want!r}"
    else:
        assert got == want, f"{path}: {got!r} != {want!r}"


def _run(tmp_path, name, args, want_json):
    cmd = [sys.executable, "-m", "goodvibes", *_expand(args), "--output", name]
    if want_json:
        cmd += ["--json", "out.json"]
    env = dict(os.environ, PYTHONUTF8="1", MPLBACKEND="Agg")
    proc = subprocess.run(cmd, cwd=tmp_path, capture_output=True, text=True, env=env, encoding="utf-8")
    assert proc.returncode == 0, f"{name}: exit {proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    dat = (tmp_path / f"GoodVibes_{name}.dat").read_text(encoding="utf-8")
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8")) if want_json else None
    return _normalise_text(dat), (_normalise_json(payload) if want_json else None)


@pytest.mark.parametrize("name, args, want_json", CASES, ids=[c[0] for c in CASES])
def test_cli_golden(tmp_path, name, args, want_json):
    dat, payload = _run(tmp_path, name, args, want_json)
    dat_golden = GOLDENS / f"{name}.dat"
    json_golden = GOLDENS / f"{name}.json"
    if UPDATE:
        dat_golden.write_text(dat, encoding="utf-8")
        if want_json:
            json_golden.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
        return
    assert dat_golden.exists(), f"missing golden {dat_golden.name}; run with GOODVIBES_UPDATE_GOLDENS=1"
    want = dat_golden.read_text(encoding="utf-8")
    assert dat == want, f"{name}: .dat output changed (diff against {dat_golden})"
    if want_json:
        _assert_json_equal(payload, json.loads(json_golden.read_text(encoding="utf-8")))
