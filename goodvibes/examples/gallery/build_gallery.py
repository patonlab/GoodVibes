"""Build the reaction-profile gallery from compact, committed inputs.

    python goodvibes/examples/gallery/build_gallery.py [--out DIR]

Every figure is drawn from a reaction-profile document or a CSV table in the
repository; no program output is read. Documents that embed their conformers
(``--with-conformers``) are re-evaluated here at other temperatures and for
other quantities, which is what makes the figures reproducible without the
original Gaussian outputs. The default output directory is
``docs/source/gallery`` (the images the documentation and the gallery README
show).
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")

from goodvibes import Series, load_profile  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXAMPLES))
AZABOR = os.path.join(EXAMPLES, "profiles", "azabor_profile.json")
AMINOX = os.path.join(HERE, "aminox_profile.json")
LEVELS_CSV = os.path.join(EXAMPLES, "profiles", "levels.csv")
DEFAULT_OUT = os.path.join(ROOT, "docs", "source", "gallery")
SINGLE = (6.4, 4.0)
DPI = 150

#: Illustrative values, not from a publication: the "Ph-lit" column of
#: ../profiles/levels.csv placed on the azabor pathway to show how a declared
#: series is drawn next to a computed one.
ILLUSTRATIVE = {"Ph": {"R1-An + Aza-Phos": 0.0, "AmTS + THF": 18.4, "Syn-P + THF": -88.0}}


def azabor_dft():
    """One computed series: Δqh-G at 298.15 K with every conformer shown."""
    prof = load_profile(AZABOR)
    prof.style["figsize"] = list(SINGLE)
    return prof.plot(series=["qh_gibbs@298.15K"], label_points=True, show_conformers=True,
                     title="Aza-borocyclization: Δqh-G at 298.15 K")


def azabor_temperatures():
    """The same structures re-evaluated at four temperatures, file-free."""
    prof = load_profile(AZABOR).evaluate(temperatures=[273.15, 298.15, 373.15, 423.15],
                                         invocation="build_gallery.py")
    prof.style["figsize"] = list(SINGLE)
    return prof.plot(title="Aza-borocyclization: Δqh-G from 273 to 423 K", annotations=False)


def azabor_quantities():
    """ΔE (single point), ΔH and Δqh-G of the same profile on one axes."""
    prof = load_profile(AZABOR)
    prof.series = [
        Series(id="E", label="ΔE (single point)", quantity="spc", temperature=298.15,
               style={"linestyle": ":"}),
        Series(id="H", label="ΔH (298 K)", quantity="enthalpy", temperature=298.15,
               style={"linestyle": "--"}),
        Series(id="G", label="Δqh-G (298 K)", quantity="qh_gibbs", temperature=298.15),
    ]
    prof.annotations = []
    ev = prof.evaluate(invocation="build_gallery.py")
    ev.style["figsize"] = list(SINGLE)
    return ev.plot(title="Aza-borocyclization: ΔE, ΔH and Δqh-G")


def aminox_branches():
    """Two competing transition states from one reactant, one panel each,
    ΔE dotted and Δqh-G solid, with the barriers marked."""
    prof = load_profile(AMINOX)
    prof.style["figsize"] = [SINGLE[0], 6.0]
    return prof.plot()


def declared_csv():
    """A table of relative energies (illustrative values) drawn directly."""
    prof = load_profile(LEVELS_CSV, quantity="gibbs", temperature=298.15,
                        title="Illustrative values read from a CSV table")
    prof.style["figsize"] = list(SINGLE)
    return prof.plot(label_points=True)


def dft_vs_declared():
    """A computed series and a declared one (hollow markers) on one pathway."""
    prof = load_profile(AZABOR)
    prof.series = [prof.get_series("qh_gibbs@298.15K"),
                   Series.declared_from("illustrative", "illustrative values (not published)", ILLUSTRATIVE,
                                        quantity="gibbs", temperature=298.15, style={"linestyle": "--"})]
    prof.annotations = []
    prof.style["figsize"] = list(SINGLE)
    return prof.plot(title="Computed (DFT) and declared values on one pathway")


GALLERY = [
    ("azabor_dft", azabor_dft),
    ("azabor_temperatures", azabor_temperatures),
    ("azabor_quantities", azabor_quantities),
    ("aminox_branches", aminox_branches),
    ("declared_csv", declared_csv),
    ("dft_vs_declared", dft_vs_declared),
]


def build(out_dir=DEFAULT_OUT, formats=("png",)):
    """Draw every figure; returns {name: ProfileAxes} (figures closed after saving)."""
    os.makedirs(out_dir, exist_ok=True)
    drawn = {}
    for name, make in GALLERY:
        fig = make()
        fig.save(*[os.path.join(out_dir, f"{name}.{ext}") for ext in formats], dpi=DPI)
        fig.close()
        drawn[name] = fig
    return drawn


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT, help="output directory (default: docs/source/gallery)")
    parser.add_argument("--formats", default="png", help="comma-separated: png,svg,pdf (default png)")
    args = parser.parse_args(argv)
    drawn = build(args.out, tuple(f.strip() for f in args.formats.split(",") if f.strip()))
    for name in drawn:
        print(f"wrote {os.path.join(args.out, name)}.{args.formats.split(',')[0]}")


if __name__ == "__main__":
    main()
