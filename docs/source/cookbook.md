# Cookbook

Task-oriented recipes for the workflows that landed in v4.x. Each
section is self-contained — copy, paste, modify the inputs, run.

For background on the underlying CLI flags and Python API, see the
[programmatic API guide](api_guide.md) and the
[main README](README.md).

---

## 1. One file → one structured result

The lowest-friction path for notebooks and scripts. Replaces the older
15-positional-arg `calc_bbe()` constructor.

```python
from goodvibes import compute_thermo

r = compute_thermo(
    "ethane.log",
    QS="grimme",            # default — Grimme quasi-RRHO entropy
    s_freq_cutoff=50,       # cm⁻¹ — soften modes below this
    spc="TZ",
    temperature=313.15,
)

print(f"qh-G(T) = {r.qh_gibbs_free_energy:.6f} Hartree")
print(f"point group: {r.point_group}, σ = {r.symmno}")
print(f"level of theory (auto-detected): {r.level_of_theory}")
print(f"frequency scale factor (auto-applied): {r.bbe.scale_fac}")
```

`compute_thermo` returns a frozen `ThermoResult` dataclass with every
attribute `calc_bbe` produces, plus `r.bbe` and `r.qcdata` for advanced
reads. Defaults match the CLI: gas-phase concentration (`P/RT`),
auto-lookup of the frequency scaling factor from the level of theory
via the Truhlar database.

---

## 2. Batch a directory with parallel parsing → DataFrame

The most common notebook workflow: parse hundreds of conformers,
filter, sort, export.

```python
import glob
from goodvibes import compute_batch, to_dataframe
from goodvibes.constants import KCAL_TO_AU

paths = sorted(glob.glob("conformers/*.log"))
results = compute_batch(paths, jobs=8)        # 8 worker processes

df = to_dataframe(results)
df = df.sort_values("qh_gibbs_free_energy")
df["ΔG_kcal"] = (df.qh_gibbs_free_energy - df.qh_gibbs_free_energy.min()) * KCAL_TO_AU

# Drop conformers more than 3 kcal/mol above the lowest
keep = df[df["ΔG_kcal"] < 3.0]
print(f"{len(keep)} of {len(df)} conformers within 3 kcal/mol of the lowest")
keep[["name", "qh_gibbs_free_energy", "ΔG_kcal"]].to_csv("survivors.csv", index=False)
```

`jobs=0` uses all CPU cores. Output preserves input order. Pandas is
optional — install with `pip install goodvibes[full]`.

The same thing from the shell, no Python:

```bash
goodvibes conformers/*.log --jobs 8 --csv all_thermo.csv
```

---

## 3. N-way selectivity (replaces `--ee`)

The v4.1 redesign generalizes `--ee a:b` (2-bucket only) to N-way
selectivity. Each bucket is named explicitly with `--label NAME=PATTERN`
(repeatable). The patterns are `fnmatch` globs against the input
filenames — no filesystem walks.

The example fixture in `goodvibes/examples/selectivity/` is a
Diels–Alder TS set: 8 transition states across two regiochemistries
(1,2- vs 1,4-) and two diastereomers (exo / endo).

**2-way (exo vs endo)**

```bash
cd goodvibes/examples/selectivity
goodvibes DA_*.out --label exo='*_exo_*' --label endo='*_endo_*'
```

```text
Selectivity, Boltzmann-averaged (gibbs, T = 298.15 K)
       Species   Files   Population (%)   ΔΔG (kcal/mol)
       exo           4             2.56            2.156
★      endo          4            97.44            0.000

Ratio exo:endo = 3:97   Major: endo   excess = 94.88%   ΔΔG = 2.16 kcal/mol

Selectivity, Lowest conformer only (gibbs, T = 298.15 K)
       Species   Files   Population (%)   ΔΔG (kcal/mol)
       exo           1             1.84            2.355
★      endo          1            98.16            0.000

Ratio exo:endo = 2:98   Major: endo   excess = 96.31%   ΔΔG = 2.36 kcal/mol
```

The two tables answer different questions: the Boltzmann row shows the
selectivity once you average over conformers; the lowest-conformer row
shows what the selectivity would be if only the most stable TS in each
species mattered. The gap between them tells you how much of the
selectivity is driven by conformer mixing.

**4-way (regio × stereo)**

```bash
goodvibes DA_*.out \
  --label exo_12='*_exo_12*'   --label endo_12='*_endo_12*' \
  --label exo_14='*_exo_14*'   --label endo_14='*_endo_14*'
```

For N > 2 the summary line drops `excess` and `ΔΔG` (those are 2-bucket
concepts) and just reports the ratio — `Ratio exo_12:endo_12:exo_14:endo_14 = 2:97:0:0`.

**Per-species subdirectories**

If your conformers are organized into one directory per species,
`--label` patterns are matched against the immediate parent
directory's basename in addition to the file's basename. So a
layout like

```text
selectivity_separated/
  exo/
    DA_exo_12_i.out
    DA_exo_12_ii.out
    ...
  endo/
    DA_endo_12_i.out
    ...
```

works with the directory names as labels:

```bash
cd selectivity_separated
goodvibes */*out --label exo='exo*' --label endo='endo*'
```

The shell expands `*/*out` to relative paths like `exo/DA_exo_12_i.out`,
and the `'exo*'` pattern matches the parent dir `exo`. The same
patterns also keep working on flat layouts (where the species is
encoded in the filename), so you don't need to know in advance
which layout your data uses.

**JSON output**

Add `--json results.json` and the file gets two top-level blocks,
`selectivity` and `selectivity_lowest`, each with the per-species
populations, ΔΔG, ee (when N=2), and the source files for every
species. Schema is `0.4`.

**Strip plot**

To visualize where the selectivity comes from — lowest-TS gap vs
conformer mixing — write a per-species ΔG strip plot:

```bash
goodvibes DA_*.out \
  --label exo='*_exo_*' --label endo='*_endo_*' \
  --strip-plot selectivity.png
```

The image shows one column per species with scattered conformer
ΔG values (relative to the global lowest). A tight cluster near the
bottom of a column means that species is dominated by its lowest
conformer; a wide spread means conformer mixing is contributing.

In Python:

```python
import matplotlib.pyplot as plt
from goodvibes import compute_batch
from goodvibes.selectivity import (
    compute_selectivity, parse_label_args, assign_files_to_labels,
)
from goodvibes.plot import plot_selectivity_strip

results = compute_batch(glob.glob("DA_*.out"))
thermo = {r.file: r.qh_gibbs_free_energy for r in results}
labels = parse_label_args(["exo=*_exo_*", "endo=*_endo_*"])
files_per_label = assign_files_to_labels(list(thermo), labels)
sel = compute_selectivity(thermo, files_per_label, 298.15)

ax = plot_selectivity_strip(sel, thermo)
plt.savefig("selectivity.png", dpi=200, bbox_inches="tight")
```

matplotlib is in the optional `[plot]` extras (or `[full]`) — install
with `pip install goodvibes[plot]`.

**Migration from `--ee`**

```bash
# v3.x
goodvibes *.log --ee 'P_R_*:P_S_*'

# v4.x equivalent
goodvibes *.log --label R='P_R_*' --label S='P_S_*'
```

`--ee` still works in v4.x with a `DeprecationWarning`; it's slated
for removal in v5.0.

---

## 4. PES with the new YAML format

The legacy line-based PES file (`--- # PES` markers) is auto-detected
and still works, but it isn't real YAML and has no stoichiometry
support. v4.2 adds a proper YAML schema with `pathways:` / `species:`
/ `format:` top-level keys and a `coeff*name` syntax for stoichiometric
sums.

```yaml
# azabor_PES_v2.yaml
pathways:
  Ph:
    - "R1-An + Aza-Phos"
    - "R1-Comp + THF"
    - "AmTS + THF"
    - "Azir-Comp + THF"
    - "OpenTS + THF"
    - "Syn-P + THF"

species:
  R1-An:      {files: "r1-li-3thf-c1*"}
  Aza-Phos:   {files: "azaoxy-phosphine-full*"}
  THF:        {files: "thf*"}
  R1-Comp:    {files: "r1-phosphine-2thf-full*"}
  Azir-Comp:  {files: "aziridinium-phos-full*"}
  Syn-P:      {files: "syn-product-phos-full*"}
  OpenTS:     {files: "openTS-phos-full*"}
  AmTS:       {files: "aminationTS-full-unfrz-c1*"}

format:
  units: kcal/mol
  decimals: 1
```

Stoichiometric example: a bimolecular reaction would write a point
as `"2*A + B"`. Each species' `files:` is a glob (single string) or
explicit list (`[a.log, b.log]`).

**Assigning species by directory.** When each species lives in its
own subdirectory, use `dir:` (single) or `dirs:` (list) instead of
file globs:

```yaml
species:
  R1-An:      {dir: "R1-An"}
  Aza-Phos:   {dir: "Aza-Phos"}
  AmTS:      {dir: "AmTS"}
  # combine if a species has both subdir conformers and a separate
  # explicit file:
  THF:        {files: "thf_extra.log", dir: "THF"}
```

`dir:` matches files whose immediate parent directory's basename
equals the value (or matches it as an fnmatch glob — `dir: "TS_*"`
catches every `TS_R/`, `TS_S/`, ...). Trailing `/`, `/*` or `/**`
on the dir name is ignored.

Run it from the directory above the per-species subdirectories:

```bash
cd goodvibes/examples/pes_separated
goodvibes */*log --spc tzpop --pes azabor_PES.yaml
```

The shell `*/*log` glob hands GoodVibes relative paths like
`R1-An/r1-li-3thf-c1.log` — the `dir: "R1-An"` rule sees `R1-An`
as the parent dir basename and assigns the file there.

Run it:

```bash
cd goodvibes/examples/pes
goodvibes *.log --pes azabor_PES_v2.yaml --spc sp_tzpop
```

By default each species' contribution is **gconf-corrected**: lowest
qh-G conformer + Boltzmann adjustment + the −R Σ pᵢ ln pᵢ mixing
entropy. Two flags change that:

| Mode | Flag | What it does |
| --- | --- | --- |
| gconf (default) | — | lowest + adjustment + mixing entropy |
| pure Boltzmann | `--nogconf` | Boltzmann-weighted average, no mixing entropy |
| lowest only | `--lowest-only` | use each species' single lowest qh-G conformer |

The mode tag appears in the table title:

```text
RXN: Ph  (kcal/mol)  at T = 298.15 K, p = 1 atm — lowest conformer per species
```

**Reaction-profile diagram**

```bash
goodvibes *.log --pes azabor_PES_v2.yaml --spc sp_tzpop \
                --pes-plot pes.png
```

Saves a step-plot of the pathway's qh-G profile (one column per
point, horizontal bar at each level, smooth bezier connectors).
matplotlib via `pip install goodvibes[plot]`.

If your PES YAML defines multiple pathways (e.g. an R-side and an
S-side TS sharing reactants and products), `--pes-plot` overlays
them on the same axes by default — different colors from the
matplotlib cycle, with a legend and shared x-axis. Pathways must
have the same number of points to be comparable.

For full control over colors, single-pathway selection, point
annotations, or per-conformer scatter, drop down to the API:

```python
from goodvibes.pes_loader import load_pes
from goodvibes.plot import plot_pes
import matplotlib.pyplot as plt

pes = load_pes("R_vs_S.yaml", thermo_data)        # 2-pathway YAML
ax = plot_pes(
    pes,
    colors=["#26a6a4", "#e76f51"],     # custom per-pathway palette
    connector_style="bezier",          # or "linear"
    label_points=True,                 # annotate ΔqhG at each level
)
plt.savefig("R_vs_S.png", dpi=200, bbox_inches="tight")

# Or pick one pathway and overlay individual conformer dots:
ax = plot_pes(
    pes, pathway_index=0,
    show_conformers=True,
    thermo_lookup={f: bbe.qh_gibbs_free_energy
                   for f, bbe in thermo_data.items()},
)
```

The legacy `--graph FILE.yaml` flag is still supported and reads
styling (dpi, color, title, legend, gridlines, ylim, ...) from a
YAML's `--- # FORMAT` block. It will be deprecated in v5.1 once
`--pes-plot` covers the remaining gaps.

---

## 5. PES + JSON for downstream analysis

```bash
goodvibes *.log --pes azabor_PES_v2.yaml --spc sp_tzpop --json pes.json
```

The JSON gets a `pes` block (schema v1.0):

```python
import json

with open("pes.json") as f:
    payload = json.load(f)

for path in payload["pes"]["pathways"]:
    print(f"\n=== {path['name']} ({path['units']}) ===")
    for pt in path["points"]:
        print(f"  {pt['label']:25s}  ΔqhG = {pt['relative']['qh_g']:+7.2f}")
```

Each point carries `label`, `species` (name + coefficient + resolved
files), and `relative` (Δ-values for E, ZPE, H, qh-H, T·S, T·qh-S, G,
qh-G, plus SPC variants when `--spc` was set). Plug straight into
plotting libraries or downstream pipelines.

---

## 6. Parse once, re-analyze many times

QC outputs are slow to parse, especially for large conformer ensembles
or composite-method SPCs. The unified v1.0 JSON payload (`--export`)
captures every parsed field once; subsequent runs read it back via
`--import` and skip the QC files entirely. Useful for re-running at a
different temperature, concentration, frequency cutoff, or quasi-RRHO
scheme without touching the original `.log`/`.out` files.

```bash
# First pass — parse + apply SPC + export the structured payload.
goodvibes conformers/*.log --spc TZ --export thermo.json

# Re-run at 350 K with the same files but no QC parsing. Re-pass --spc
# to keep the cached SPC numbers driving G(T)_SPC; drop it for plain G.
goodvibes --import thermo.json --spc TZ -t 350

# Re-run with the Truhlar frequency-raising entropy scheme and a
# stricter low-frequency cutoff. Still no parsing.
goodvibes --import thermo.json --spc TZ --qs truhlar -f 150

# Combine cached --spc with selectivity at a new temperature.
goodvibes --import thermo.json --spc TZ -t 313.15 \
          --label R='cat_R*' --label S='cat_S*'
```

`--export` writes the same payload as `--json`, so a single file covers
both downstream pipelines and re-import. Once exported, the original
`.log`/`.out` files can be archived, moved, or deleted — `--import`
works with just the JSON. The `--spc` energies are cached on the QCData
record, so re-passing `--spc <suffix>` reuses them without ever
re-reading the SPC files.

---

## See also

- The full CLI flag table in the [main README](README.md).
- The [programmatic API reference](api_guide.md) for `compute_thermo`,
  `compute_batch`, `ThermoResult`, and `to_dataframe`.
- The full module reference covers `goodvibes.pes_loader`,
  `goodvibes.pes_model`, `goodvibes.selectivity`, etc., for users
  embedding GoodVibes in larger pipelines.
