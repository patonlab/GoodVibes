"""M2b figure features: style presets, uncertainty error bars, and SVG
output that carries element ids and the drawn reaction-profile document."""
import re
from pathlib import Path

import pytest

import matplotlib
matplotlib.use("Agg", force=True)
plt = pytest.importorskip("matplotlib.pyplot")
yaml = pytest.importorskip("yaml")

from goodvibes.pes_model import Series
from goodvibes.plot import (STYLE_PRESETS, embed_svg_metadata, plot_profile, read_svg_metadata,
                            resolve_preset)
from goodvibes.profile import PRESETS, Profile, load_profile, validate_document
from goodvibes.profile_cli import main as gvp
from test_plot_profile import _branches

EXAMPLES = Path(__file__).resolve().parents[1] / "goodvibes" / "examples"

DOC = """
schema: reaction-profile/1.0
title: "Error bars ]]> and presets"
units: kcal/mol
points:
  R:   {role: reactant}
  TS1: {role: ts, display: "TS1‡"}
  Int: {role: minimum}
  P:   {role: product}
pathways:
  main: [R, TS1, Int, P]
series:
  - id: G
    label: "ΔG (298 K)"
    quantity: gibbs
    temperature: 298.15
    source: declared
    levels:
      main: {R: 0.0, TS1: 18.4, Int: 3.2, P: -12.1}
    uncertainty:
      main: {TS1: 1.5, Int: 0.8}
annotations:
  - {pathway: main, from: R, to: TS1, label: "ΔG‡"}
style: {preset: single-column, label_points: true}
"""


@pytest.fixture
def doc():
    return Profile.from_dict(yaml.safe_load(DOC))


def _error_segments(prof):
    return {gid: ref for gid, ref in prof.element_ids.items() if ref["kind"] == "error"}


# -- presets -------------------------------------------------------------------

def test_the_validator_and_the_plot_agree_on_the_preset_names():
    assert tuple(STYLE_PRESETS) == PRESETS


@pytest.mark.parametrize("name", ["single-column", "double-column", "slide"])
def test_a_preset_sizes_the_figure_and_its_text(name):
    before = dict(matplotlib.rcParams)
    prof = plot_profile(_branches(), preset=name, label_points=True)
    pre = STYLE_PRESETS[name]
    assert tuple(prof.figure.get_size_inches()) == pytest.approx(pre.figsize)
    assert prof.preset == name
    assert all(t.get_fontsize() == pre.font_size for t in prof.ax.get_xticklabels())
    labels = [c for c in prof.ax.texts]
    assert labels and all(t.get_fontsize() == pre.label_size for t in labels)
    assert matplotlib.rcParams == before          # applied to this figure only
    prof.close()


def test_panels_stack_preset_height_per_pathway():
    prof = plot_profile(_branches(), preset="double-column", layout="panels")
    w, h = prof.figure.get_size_inches()
    assert (w, h) == pytest.approx((7.0, 2 * STYLE_PRESETS["double-column"].panel_height))
    prof.close()


def test_style_preset_explicit_figsize_and_unknown_names():
    prof = plot_profile(_branches(), style={"preset": "slide", "figsize": (4, 3)})
    assert prof.preset == "slide" and tuple(prof.figure.get_size_inches()) == (4, 3)
    prof.close()
    assert resolve_preset("none") is None and resolve_preset(None) is None
    with pytest.raises(ValueError, match="unknown style preset"):
        plot_profile(_branches(), preset="poster")


def test_document_preset_is_used_and_can_be_overridden(doc):
    assert doc.plot().preset == "single-column"
    assert doc.plot(preset="slide").preset == "slide"
    errors, _ = validate_document({**yaml.safe_load(DOC), "style": {"preset": "poster"}})
    assert any("style.preset" in e for e in errors)
    plt.close("all")


def test_long_tick_labels_are_tilted_and_value_labels_stay_inside_the_axes():
    azabor = load_profile(EXAMPLES / "profiles" / "azabor_profile.json")
    prof = azabor.plot(preset="single-column", label_points=True)
    assert {t.get_rotation() for t in prof.ax.get_xticklabels()} == {40.0}
    renderer = prof.figure.canvas.get_renderer()
    box = prof.ax.get_window_extent(renderer)
    for t in prof.ax.texts:
        tb = t.get_window_extent(renderer)
        assert box.y0 - 1 <= tb.y0 and tb.y1 <= box.y1 + 1, t.get_text()
    prof.close()


# -- uncertainty ---------------------------------------------------------------

def test_uncertainty_is_drawn_as_error_bars(doc):
    prof = doc.plot()
    errors = _error_segments(prof)
    assert {ref["point"] for ref in errors.values()} == {"TS1", "Int"}
    coll = next(c for c in prof.ax.collections if c.get_gid() == "error-G-main-TS1")
    ys = sorted({round(y, 6) for seg in coll.get_segments() for _x, y in seg})
    assert ys == pytest.approx([18.4 - 1.5, 18.4 + 1.5])
    prof.close()
    prof = doc.plot(uncertainty=False)
    assert not _error_segments(prof)
    assert prof.uncertainty["G"]["main"] == {"TS1": 1.5, "Int": 0.8}   # still recorded
    prof.close()


def test_uncertainty_follows_the_unit_conversion():
    res = _branches()
    s = Series.declared_from("lit", "lit", {"R": {"R": 0.0, "TS_R": 41.84}}, units="kJ/mol")
    s.uncertainty = {"R": {"TS_R": 4.184}}
    prof = plot_profile(res, series=[s], pathways="R")
    assert prof.uncertainty["lit"]["R"]["TS_R"] == pytest.approx(1.0)
    assert prof.level("R", "TS_R") == pytest.approx(10.0)
    prof.close()


# -- SVG -------------------------------------------------------------------------

def test_svg_carries_ids_and_the_drawn_document(doc, tmp_path):
    prof = doc.plot()
    out = tmp_path / "fig.svg"
    prof.save(str(out))
    svg = out.read_text(encoding="utf-8")
    for gid in prof.element_ids:
        assert svg.count(f'id="{gid}"') == 1, gid
    assert "<text" in svg                        # text stays text (editable)
    assert "dc:date" not in svg                  # deterministic
    payload = read_svg_metadata(svg)
    assert payload["generator"].startswith("GoodVibes")
    assert payload["elements"] == prof.element_ids
    back = load_profile(out)
    assert back.title == doc.title               # ']]>' in the title survives the CDATA
    assert back.series[0].levels == {"main": {"R": 0.0, "TS1": 18.4, "Int": 3.2, "P": -12.1}}
    assert back.series[0].uncertainty == {"main": {"TS1": 1.5, "Int": 0.8}}
    assert back.style["preset"] == "single-column"
    prof.close()


def test_svg_output_is_reproducible(doc, tmp_path):
    a, b = tmp_path / "a.svg", tmp_path / "b.svg"
    doc.plot().save(str(a))
    doc.plot().save(str(b))
    assert a.read_text(encoding="utf-8") == b.read_text(encoding="utf-8")
    plt.close("all")


def test_embed_false_and_other_formats_carry_no_document(doc, tmp_path):
    prof = doc.plot()
    prof.save(str(tmp_path / "plain.svg"), str(tmp_path / "f.png"), embed=False)
    assert read_svg_metadata((tmp_path / "plain.svg").read_text(encoding="utf-8")) is None
    with pytest.raises(Exception, match="carries no reaction-profile document"):
        load_profile(tmp_path / "plain.svg")
    prof.close()


def test_a_computed_profile_embeds_levels_without_conformers(tmp_path):
    azabor = load_profile(EXAMPLES / "profiles" / "azabor_profile.json")
    prof = azabor.plot(series=[azabor.series[0].id])
    out = tmp_path / "azabor.svg"
    prof.save(str(out))
    assert out.stat().st_size < 120_000          # no embedded conformers
    back = load_profile(out)
    assert "conformers" not in back.namespace
    drawn = prof.levels[azabor.series[0].id]
    assert set(back.series[0].levels) == set(drawn)
    for p in drawn:
        assert back.series[0].levels[p] == pytest.approx(drawn[p])
    assert validate_document(back.to_dict())[0] == []
    prof.close()


def test_a_figure_from_a_hand_built_result_gives_a_valid_document(tmp_path):
    prof = plot_profile(_branches(), label_points=True)
    prof.annotate_barrier("R", "R", "TS_R")
    doc = prof.to_document()
    assert validate_document(doc)[0] == []
    assert doc["series"][0]["temperature"] == 298.15
    assert {r["kind"] for r in prof.element_ids.values()} >= {"bar", "edge", "label", "legend",
                                                              "barrier-arrow", "barrier-label"}
    assert all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", g) for g in prof.element_ids)
    prof.close()


def test_svg_to_a_file_object_embeds_too(doc):
    import io
    prof = doc.plot()
    for buf in (io.StringIO(), io.BytesIO()):
        prof.save(buf, format="svg")
        text = buf.getvalue()
        text = text.decode("utf-8") if isinstance(text, bytes) else text
        assert read_svg_metadata(text)["document"]["title"] == doc.title
    png = io.BytesIO()
    prof.save(png, format="png")
    assert png.getvalue()[:4] == b"\x89PNG"
    prof.close()


@pytest.mark.parametrize("cdata", ["{not json", "42", '{"elements": {}}'])
def test_malformed_svg_metadata_is_a_profile_error(tmp_path, capsys, cdata):
    from goodvibes.profile import ProfileError
    svg = tmp_path / "bad.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg">\n <metadata id="goodvibes-reaction-profile">'
                   f"<![CDATA[{cdata}]]></metadata>\n</svg>\n", encoding="utf-8")
    with pytest.raises(ProfileError):
        load_profile(svg)
    assert gvp(["validate", str(svg)]) == 1
    assert "cannot read" in capsys.readouterr().err
    assert gvp(["table", str(svg)]) == 1


def test_embed_helpers_round_trip_awkward_text():
    svg = '<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg">\n</svg>\n'
    payload = {"document": {"title": "a ]]> b ]]]]> c <&>"}}
    assert read_svg_metadata(embed_svg_metadata(svg, payload)) == payload
    with pytest.raises(ValueError):
        embed_svg_metadata("<html/>", payload)


# -- goodvibes-profile ---------------------------------------------------------------

def test_cli_preset_and_svg_round_trip(tmp_path, capsys):
    src = tmp_path / "doc.yaml"
    src.write_text(DOC, encoding="utf-8")
    out = tmp_path / "fig.svg"
    assert gvp(["plot", str(src), "-o", str(out), "--preset", "slide", "--no-uncertainty"]) == 0
    back = load_profile(out)
    assert back.style["preset"] == "slide"
    assert not [g for g in read_svg_metadata(out.read_text(encoding="utf-8"))["elements"]
                if g.startswith("error")]
    capsys.readouterr()
    assert gvp(["validate", str(out)]) == 0
    assert "valid" in capsys.readouterr().out
    assert gvp(["table", str(out), "-o", str(tmp_path / "t.csv")]) == 0
    assert "18.4" in (tmp_path / "t.csv").read_text(encoding="utf-8")
    plain = tmp_path / "plain.svg"
    assert gvp(["plot", str(src), "-o", str(plain), "--no-embed"]) == 0
    assert gvp(["validate", str(plain)]) == 1
