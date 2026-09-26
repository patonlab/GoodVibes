"""Regenerate the compact gallery inputs from program outputs.

    python goodvibes/examples/gallery/_generate_inputs.py

writes ``aminox_profile.json`` from the Gaussian outputs in
``../gconf_ee_boltz``. The azabor input is ``../profiles/azabor_profile.json``
(see ``../profiles/_generate.py``). Only these compact documents are needed to
build the gallery (``build_gallery.py``); the outputs are not.
"""
import glob
import os

from goodvibes import Profile, compute_batch
from goodvibes.io import read_initial

HERE = os.path.dirname(os.path.abspath(__file__))
AMINOX = os.path.join(HERE, "..", "gconf_ee_boltz")


def aminox_document(results, level_of_theory):
    """Two competing transition states (R and S) from one reactant pair."""
    return {
        "schema": "reaction-profile/1.0",
        "title": "Aminoxylation: R vs S transition states",
        "units": "kcal/mol",
        "species": {"cat": {}, "subs": {}, "TS-R": {}, "TS-S": {}},
        "points": {
            "cat + subs": {"species": "cat + subs", "role": "reactant"},
            "TS-R": {"species": "TS-R", "role": "ts", "display": "TS (R)‡"},
            "TS-S": {"species": "TS-S", "role": "ts", "display": "TS (S)‡"},
        },
        "pathways": {"R": ["cat + subs", "TS-R"], "S": ["cat + subs", "TS-S"]},
        "order": ["cat + subs", "TS-R", "TS-S"],
        "methods": {"dft": {"program": "gaussian", "level_of_theory": level_of_theory}},
        "series": [
            {"id": "E", "label": "ΔE", "method": "dft", "quantity": "electronic", "temperature": 298.15,
             "style": {"linestyle": ":"}},
            {"id": "G", "label": "Δqh-G (298 K)", "method": "dft", "quantity": "qh_gibbs",
             "temperature": 298.15},
        ],
        "annotations": [
            {"type": "barrier", "pathway": "R", "from": "cat + subs", "to": "TS-R", "series": "G"},
            {"type": "barrier", "pathway": "S", "from": "cat + subs", "to": "TS-S", "series": "G"},
        ],
        "style": {"layout": "panels", "label_points": True, "decimals": 1},
        "goodvibes": {
            "sources": {"dft": {
                "cat": {"files": "aminox_cat_*"},
                "subs": {"files": "aminox_subs_*"},
                "TS-R": {"files": "Aminoxylation_TS1_R"},
                "TS-S": {"files": "Aminoxylation_TS2_S"},
            }},
            "rollup": {"mode": "gconf"},
        },
    }


def main():
    files = sorted(glob.glob(os.path.join(AMINOX, "*.log")))
    results = compute_batch(files)
    lot = read_initial(files[0])[0]
    prof = Profile.from_dict(aminox_document(results, lot))
    ev = prof.evaluate(results, with_conformers=True,
                       invocation="goodvibes/examples/gallery/_generate_inputs.py")
    out = os.path.join(HERE, "aminox_profile.json")
    ev.dump(out, indent=None)
    print(f"wrote {out} ({os.path.getsize(out) // 1024} KB)")


if __name__ == "__main__":
    main()
