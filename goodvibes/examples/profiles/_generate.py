"""Regenerate azabor_profile.json from the Gaussian outputs in ../pes.

The outputs themselves (about 100 files, several MB each) are the raw data;
the document written here carries everything GoodVibes needs to redraw,
retabulate and re-evaluate the profile at any temperature: every
structure's parsed data and thermochemistry options (``--with-conformers``).
Run from anywhere:

    python goodvibes/examples/profiles/_generate.py

which is the Python form of

    cd goodvibes/examples/pes
    goodvibes *.log --spc sp_tzpop --pes azabor_PES_v2.yaml \\
        --profile ../profiles/azabor_profile.json --with-conformers
"""
import glob
import os

from goodvibes import compute_batch, load_profile

HERE = os.path.dirname(os.path.abspath(__file__))
PES = os.path.join(HERE, "..", "pes")


def main():
    files = sorted(f for f in glob.glob(os.path.join(PES, "*.log")) if not f.endswith("_sp_tzpop.log"))
    results = compute_batch(files, spc="sp_tzpop", jobs=os.cpu_count() or 1)
    prof = load_profile(os.path.join(PES, "azabor_PES_v2.yaml"))
    prof.title = "Aza-borocyclization (azabor example)"
    for pid, point in prof.points.items():
        if "TS" in pid:
            point.role = "ts"
            point.display = pid.split(" +")[0] + "‡"
    first, last = prof.pathways["Ph"].points[0], prof.pathways["Ph"].points[-1]
    prof.points[first].role, prof.points[last].role = "reactant", "product"
    prof.annotations = [{"type": "barrier", "pathway": "Ph", "from": "R1-Comp + THF", "to": "AmTS + THF"}]
    ev = prof.evaluate(results, temperatures=[298.15, 373.15], with_conformers=True,
                       invocation="goodvibes/examples/profiles/_generate.py")
    out = os.path.join(HERE, "azabor_profile.json")
    ev.dump(out, indent=None)
    print(f"wrote {out} ({os.path.getsize(out) // 1024} KB)")


if __name__ == "__main__":
    main()
