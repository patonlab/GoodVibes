"""The reaction-profile gallery (goodvibes/examples/gallery) rebuilds from
its committed compact inputs, and its numbers are the ones GoodVibes prints
from the program outputs."""
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "goodvibes" / "examples" / "gallery" / "build_gallery.py"


def _module():
    spec = importlib.util.spec_from_file_location("build_gallery", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("gallery")
    mod = _module()
    return mod, out, mod.build(str(out), formats=("png", "svg"))


def test_every_figure_is_built(built):
    mod, out, drawn = built
    assert list(drawn) == [name for name, _ in mod.GALLERY]
    for name in drawn:
        assert (out / f"{name}.png").stat().st_size > 10_000
        assert "<svg" in (out / f"{name}.svg").read_text(encoding="utf-8")


def test_committed_figures_exist_for_every_entry(built):
    mod, _out, _drawn = built
    for name, _ in mod.GALLERY:
        assert (ROOT / "docs" / "source" / "gallery" / f"{name}.png").is_file(), name


def test_gallery_numbers_match_the_cli(built):
    """Levels drawn from the compact documents equal what `goodvibes` prints
    from the outputs: `goodvibes gconf_ee_boltz/*.log --pes gconf_aminox_cat.yaml`
    gives Δqh-G(T) 19.91 kcal/mol for the R transition state, and
    `goodvibes pes/*.log --spc sp_tzpop --pes azabor_PES_v2.yaml` gives 13.40
    kcal/mol for AmTS at 298.15 K (the pes_v2_azabor golden adds -q)."""
    _mod, _out, drawn = built
    assert drawn["aminox_branches"].level("R", "TS-R", "G") == pytest.approx(19.912, abs=0.001)
    assert drawn["azabor_dft"].level("Ph", "AmTS + THF", "qh_gibbs@298.15K") == pytest.approx(13.396, abs=0.001)
    scan = drawn["azabor_temperatures"]
    assert scan.level("Ph", "AmTS + THF", "qh_gibbs@298.15K") == pytest.approx(13.396, abs=0.001)
    assert scan.level("Ph", "AmTS + THF", "qh_gibbs@423.15K") > scan.level("Ph", "AmTS + THF", "qh_gibbs@273.15K")
    q = drawn["azabor_quantities"]
    assert q.linestyles == {"E": ":", "H": "--", "G": "-"}
    assert q.level("Ph", "AmTS + THF", "G") == pytest.approx(13.396, abs=0.001)
    assert drawn["dft_vs_declared"].level("Ph", "AmTS + THF", "illustrative") == 18.4


def test_the_gallery_inputs_are_compact():
    for path, limit_kb in ((ROOT / "goodvibes/examples/profiles/azabor_profile.json", 500),
                           (ROOT / "goodvibes/examples/gallery/aminox_profile.json", 50)):
        assert path.stat().st_size < limit_kb * 1024, path.name
