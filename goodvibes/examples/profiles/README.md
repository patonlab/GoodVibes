# Reaction-profile examples

Documents in the `reaction-profile/1.0` format (see
`docs/source/reaction_profile.md`). None of them needs output files.

| File | What it shows |
| --- | --- |
| `minimal.yaml` | the 20-line minimal profile: typed-in values only |
| `levels.csv` | a table of relative energies (two pathways) read as a declared-only profile |
| `azabor_profile.json` | the `../pes` aza-borocyclization profile at 298.15 and 373.15 K, evaluated from about 100 Gaussian outputs and carrying every structure's parsed data (`--with-conformers`): 380 KB instead of the outputs themselves |
| `_generate.py` | regenerates `azabor_profile.json` from the outputs in `../pes` |

```bash
goodvibes-profile plot minimal.yaml -o minimal.svg
goodvibes-profile plot levels.csv -o levels.png --quantity gibbs --temperature 298.15
goodvibes-profile table azabor_profile.json
goodvibes-profile evaluate azabor_profile.json -o hot.json --temperatures 273.15,423.15
```

New example sets follow the same rule: commit the evaluated document (or a
GoodVibes `--export` payload, or `.extxyz` files), keep the script that made
it next to it, and archive the raw program outputs elsewhere (e.g. Zenodo)
rather than in the repository.
