# Cookbook

Task-oriented recipes for the workflows that landed in v4.x. Each
section is self-contained — copy, paste, modify the inputs, run.

For background on the underlying CLI flags and Python API, see the
[programmatic API guide](api_guide.md) and the
[main README](README.md).

---

## 0. A table of numbers → a publication figure

No output files are needed. Type the relative energies into a CSV, one
column per pathway (an empty cell means the point is not on that
pathway; `role: ts` puts the value label above the bar):

```text
point,role,display,main
R,reactant,R,0.0
TS1,ts,TS1‡,18.4
Int,minimum,Int,3.2
TS2,ts,TS2‡,12.9
P,product,P,-12.1
```

```bash
goodvibes-profile plot levels.csv -o fig.svg -o fig.pdf --label-points --preset single-column
goodvibes-profile table fig.svg          # the SVG carries its numbers
```

- `--preset` sizes the figure, fonts and line widths for one journal
  column (about 85 mm), a full page width (`double-column`, about
  178 mm) or a `slide`.
- The SVG keeps its text as text, gives every bar, connector and label
  an `id` for editing in Inkscape or Illustrator, and embeds the
  reaction-profile document of what was drawn.
- The values are taken as ΔG at 298.15 K in kcal/mol unless you pass
  `--quantity`, `--temperature` and `--units`.

For error bars use the long layout, with one row per point and an
`uncertainty` column:

```text
pathway,point,role,display,value,uncertainty
main,R,reactant,R,0.0,
main,TS1,ts,TS1‡,18.4,1.5
main,Int,minimum,Int,3.2,0.8
main,TS2,ts,TS2‡,12.9,1.2
main,P,product,P,-12.1,1.0
```

From Python:

```python
from goodvibes import load_profile

prof = load_profile("levels.csv", quantity="gibbs", temperature=298.15)
fig = prof.plot(preset="single-column", label_points=True)
fig.save("fig.svg", "fig.pdf")
```

`goodvibes-profile convert levels.csv -o levels.yaml` writes the same
data as a [reaction-profile document](reaction_profile.md). Edit it to
add methods, barrier annotations or a second series. The
[gallery](gallery.md) shows what else a document can draw.

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
print(f"frequency scale factor: {r.freq_scale_factor} ({r.scale_factor_source})")
```

`compute_thermo` returns a frozen `ThermoResult` dataclass with every
attribute `calc_bbe` produces, plus `r.bbe` and `r.qcdata` for advanced
reads. Defaults match the CLI: gas-phase concentration (`P/RT`),
auto-lookup of the frequency scaling factor from the level of theory
via the Truhlar database. A level that is not in the database is used
unscaled with a `ScaleFactorWarning` (`r.scale_factor_source ==
"none-found"`); pass `freq_scale_factor=` to set one.

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
species. Each result also records `major`, the signed `ee_signed`, `ratio`
(major over runner-up) and each species' `ensemble_energies`
(−RT ln Σ exp(−G/RT), in Hartree).

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
thermo = {r.file: r.bbe for r in results}          # compute_selectivity reads calc_bbe objects
labels = parse_label_args(["exo=*_exo_*", "endo=*_endo_*"])
files_per_label = assign_files_to_labels(list(thermo), labels)
sel = compute_selectivity(thermo, files_per_label, 298.15)

ax = plot_selectivity_strip(sel, {r.file: r.qh_gibbs_free_energy for r in results})
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

`--ee` still works with a deprecation notice; it will be removed in v6.0.

---

## 3b. Selectivity sweeps, populations and temperature scans

**How robust is the prediction?** `compute_selectivity_batch` evaluates
many selectivity jobs at once, over temperatures, quasi-harmonic entropy
cutoffs and conformer energy windows. Each file is parsed once and then
re-evaluated for every condition:

```python
from goodvibes import compute_selectivity_batch, summarize_selectivity

jobs = {"DA": {"endo": "DA_endo_*.out", "exo": "DA_exo_*.out"}}   # globs, paths, results or ConformerSets
df = compute_selectivity_batch(jobs, [298.15, 353.15],
                               s_freq_cutoffs=[50, 150],           # cm⁻¹; the nominal 100 is always included
                               conformer_windows=[0, 1.0, 3.0])    # kcal/mol above each label's lowest
for s in summarize_selectivity(df):
    print(s["temperature"], s["text"])
```

```text
298.15 ee +95 % (94 to 96 % over s_freq_cutoff 50–150 cm⁻¹, conformer window 0–3 kcal/mol)
353.15 ee +90 % (88 to 93 % over s_freq_cutoff 50–150 cm⁻¹, conformer window 0–3 kcal/mol)
```

The DataFrame has one row per job and condition, with these columns:
- `job`, `temperature`, `s_freq_cutoff`, `conformer_window`;
- `nominal`: the nominal cutoff with all conformers;
- `major` and `ee`;
- `ddG` in kcal/mol, and `ratio`;
- per label, `population[<label>]` and `n[<label>]`.

`records=True` returns plain dicts, and `quantity="electronic"` weights
energy-only ensembles (`read_xyz_frames`).

The signs follow the `SelectivityResult` conventions:
- `major` is the most populated label; on a tie, the first listed.
- `ee = (p₁ − p₂) × 100`, with the labels in the order given, so it is
  positive when the first label is the major one.
- `ddG` and `ratio` compare the major with the runner-up.

**Where does the selectivity come from?** `plot_boltzmann_histogram`
draws the conformer populations. Given a mapping, it pools the groups into
one distribution, and its legend gives each group's total:

```python
import glob
from goodvibes import ConformerSet, compute_batch
from goodvibes.plot import plot_boltzmann_histogram, plot_temperature_scan

endo = compute_batch(sorted(glob.glob("DA_endo_*.out")))
exo = compute_batch(sorted(glob.glob("DA_exo_*.out")))
ax = plot_boltzmann_histogram({"endo": endo, "exo": exo}, temperature=298.15)   # legend: endo (97.4 %), exo (2.6 %)
ax.figure.savefig("populations.png", dpi=200, bbox_inches="tight")

ax = plot_temperature_scan(ConformerSet.from_results("endo", endo), [273.15, 298.15, 323.15, 353.15])
```

`plot_temperature_scan` draws a conformer ensemble's Δqh-G, Δqh-H and
T·Δqh-S against temperature. For a reaction-profile document, it draws each
point's level across the document's series temperatures instead.

**Selectivity in a reaction-profile document.** A `selectivity` block (see
[the format](reaction_profile.md)) names the competing branch points and
the point they share. GoodVibes predicts the selectivity from each branch
barrier, for computed and declared series alike, and checks the
Curtin–Hammett preconditions:

```python
from goodvibes import load_profile

prof = load_profile("tests/profile_conformance/valid/05_selectivity.yaml")
for r in prof.evaluate_selectivity():
    print(r.name, r.major, f"{r.ee_signed:+.1f} %", r.curtin_hammett)   # er TS_R +76.7 % satisfied
```

```bash
goodvibes-profile selectivity profile.json               # summary line and branch table per block
goodvibes-profile diff before.json after.json --tolerance 0.05
```

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

Run it from the directory above the per-species subdirectories (the
shipped `goodvibes/examples/pes` set is flat; this layout is one you
arrange yourself, e.g. one directory per species):

```bash
cd my_pes_project        # contains R1-An/, Aza-Phos/, THF/, ... subdirectories
goodvibes */*log --spc sp_tzpop --pes azabor_PES.yaml
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
matplotlib via `pip install goodvibes[plot]`. `--pes-plot-quantity E`
(or `H`, `G`, `E+ZPE`, ...) draws another quantity; with `--ti` the
scan temperatures are overlaid on one axes, one linestyle each.

If your PES YAML defines multiple pathways (e.g. an R-side and an
S-side TS sharing reactants and products), `--pes-plot` overlays
them on the same axes by default — different colors from the
matplotlib cycle, with a legend. The x axis is the merge of the
pathways' point sequences, so pathways of different lengths, or
branches that share a reactant, line up by point label.

For full control drop down to `plot_profile`, which returns a
`ProfileAxes` holding the figure, the axes and the drawn levels (the
same evaluation the tables use):

```python
from goodvibes import load_pes, plot_profile, Series

pes = load_pes("R_vs_S.yaml", thermo_data)        # 2-pathway YAML

# Two temperatures on one axes (linestyle per temperature; the species
# are re-evaluated at each T from their parsed inputs).
fig = plot_profile(pes, temperatures=[298.15, 373.15],
                   colors={"R": "#26a6a4", "S": "#e76f51"},
                   label_points=True)
fig.annotate_barrier("R", "A", "TS_R")          # ΔΔ between two points
fig.save("R_vs_S.svg", "R_vs_S.pdf")
print(fig.levels["qh_gibbs@298.15K"]["R"])     # {point label: kcal/mol}

# ΔE and Δqh-G on one axes; declared (literature) values as a third
# series, drawn with hollow markers and converted to the figure's units.
lit = Series.declared_from("lit", "lit. B3LYP (298 K)",
                           {"R": {"A": 0.0, "TS_R": 18.4, "B": -12.1}},
                           quantity="gibbs", temperature=298.15, units="kcal/mol")
plot_profile(pes, series=[pes.default_series("E")[0],
                          pes.default_series("qh_gibbs")[0], lit],
             layout="panels", show_conformers=True).save("compare.png")
```

Points carry a `role` (`reactant`, `minimum`, `ts`, `product`) and a
`display` label; edges between points are `step` (default),
`barrierless` (dotted connector) or `none`:

```python
path = pes.pathway("R")
path.point("TS_R").role, path.point("TS_R").display = "ts", "TS_R‡"
pes.pathways[0] = path.with_edges([("A", "TS_R"), ("TS_R", "B", "barrierless")])
```

The older `plot_pes(pes, ...)` keeps working as a thin wrapper that
returns the matplotlib Axes.

The legacy `--graph FILE.yaml` flag is still supported and reads
styling (dpi, color, title, legend, gridlines, ylim, ...) from a
YAML's `--- # FORMAT` block. It will be removed in v6.0 once
`--pes-plot` covers the remaining gaps.

**Building a profile without a PES file**

`ConformerSet`, `Point`, `Pathway` and `PESResult` can be assembled
directly from `compute_thermo` results, for example from an MLIP
workflow that never writes an output file (next recipe):

```python
from goodvibes import ConformerSet, PESResult, PESOptions, Pathway, Point, plot_profile

species = {name: ConformerSet.from_results(name, results)
           for name, results in {"R": r_confs, "TS_R": ts_r_confs, "TS_S": ts_s_confs}.items()}
pathways = [Pathway("R-path", [Point.from_label("R", species), Point.from_label("TS_R", species, role="ts")]),
            Pathway("S-path", [Point.from_label("R", species), Point.from_label("TS_S", species, role="ts")])]
pes = PESResult(pathways, PESOptions(units="kcal/mol"), temperatures=[298.15, 373.15])
plot_profile(pes).save("profile.svg")

# Ensemble properties of one species
cs = species["TS_R"]
cs.populations(298.15), cs.s_conf(298.15), cs.ensemble_free_energy(373.15)
cs.dedup()                                    # same gates as --dedup
```

The Rich tables the CLI prints are available without its logging
set-up: `goodvibes.output.pes_tables(pes)` returns one
`rich.table.Table` per pathway.

---

## 4b. MLIP / ASE workflows without output files

`QCData.from_atoms` and `QCData.from_vibrations` build the parsed
record GoodVibes needs from an ASE `Atoms`, an energy and the
vibrational analysis, so a MACE / ANI / xTB-in-ASE pipeline never has
to write an `.extxyz` first:

```python
from ase.io import read
from ase.vibrations import Vibrations
from mace.calculators import mace_off
from goodvibes import QCData, compute_thermo, ConformerSet

calc = mace_off(model="medium")
results = {}
for label, pattern in {"R": "R_c*.xyz", "TS_R": "TS_R_c*.xyz", "TS_S": "TS_S_c*.xyz"}.items():
    results[label] = []
    for i, atoms in enumerate(read(pattern, index=":")):
        atoms.calc = calc
        vib = Vibrations(atoms, delta=0.01, name=f"vib_{label}_{i}"); vib.run()
        qc = QCData.from_vibrations(atoms, vib.get_vibrations(), atoms.get_potential_energy(),
                                    name=f"{label}_c{i}", method="MACE-OFF23",
                                    job_type="TS" if label.startswith("TS") else "Freq")
        results[label].append(compute_thermo(qcdata=qc, QS="grimme", temperature=298.15))

species = {label: ConformerSet.from_results(label, rs) for label, rs in results.items()}
```

What the constructors do:

- energies default to eV (`energy_units="hartree"`, `"kcal/mol"`,
  `"kJ/mol"` otherwise), frequencies to cm⁻¹ (`"eV"`, `"meV"`); a
  negative or complex frequency is an imaginary mode;
- `from_vibrations` drops the 6 (5 for a linear molecule)
  translational/rotational modes of a 3N finite-difference Hessian,
  takes an imaginary mode smaller than 15 cm⁻¹ (`imag_threshold_cm1`)
  as numerical noise (real at |ν|, with a `RuntimeWarning`) and warns
  when a `job_type="TS"` structure does not have exactly one imaginary
  mode or a minimum has any;
- masses are the most-abundant-isotope values QC programs use
  (`masses="atoms"` takes ASE's standard weights); the point group and
  symmetry number come from pymsym when it is installed (`symm="auto"`),
  or pass `symm=<int>`;
- `method=` is recorded as `level_of_theory`; when it matches an entry
  of the scaling-factor database the usual scale factors apply. An MLIP
  (a `method` naming no basis set, such as `MACE-OFF23`) is used unscaled
  by design (`scale_factor_source == "mlip-unscaled"`); a QM level the
  database lacks is used unscaled with a `ScaleFactorWarning`.

`compute_batch` accepts `QCData` objects alongside paths, and
`ThermoResult.name` / `program` are the `name=` given and `"ase"`.

**DFT//MLIP composites.** `QCData.with_single_point` attaches a
higher-level single-point energy to a structure, in place of a `--spc`
output file: enthalpies and free energies use it, the frequencies (and their
scale factor) stay those of the MLIP.

```python
composite = qc.with_single_point(-232.3301, "hartree", "wB97X-D/def2-TZVP")
r = compute_thermo(qcdata=composite)      # r.spc_applied is True
```

The units are required. The attached energy survives re-evaluation at
other temperatures, `--export` caches and embedded conformers.

**Ensembles in one file.** `read_xyz_frames` reads every frame of a
multi-frame `.xyz` / `.extxyz` (a CREST `crest_conformers.xyz`, an xtb
trajectory, or MLIP energies written with `ase.io.write`) as an energy-only
`QCData`. Weighted by the electronic energy, an ensemble gives Boltzmann
populations and its ensemble energy:

```python
from goodvibes import ConformerSet, compute_batch, read_xyz_frames

frames = read_xyz_frames("crest_conformers.xyz", method="GFN2-xTB")
ens = ConformerSet.from_results("crest", compute_batch(frames), weight_by="electronic")
print(ens.populations(298.15)[:5], ens.lowest_index())
```

A plain `.xyz` comment line gives the energy as a bare number (CREST) or
`energy: <value>` (xtb), in hartree. An extxyz one gives it as `energy=`,
`free_energy=` or `total_energy=` (eV, the ASE convention), or as
`scf_energy=` (hartree). `energy_key=` and `energy_units=` override the
key and the units. Free energies need frequencies, so the thermochemical
quantities of these frames are None.

---

## 4c. Reaction-profile documents and the file-free `goodvibes-profile`

`--profile PATH` writes the evaluated profile as a
[reaction-profile document](reaction_profile.md): points, pathways, one
series of the plotted quantity per temperature, and provenance. With
`--with-conformers` it also carries every structure's parsed data, so it
can be re-evaluated at another temperature without the outputs.

```bash
cd goodvibes/examples/pes
goodvibes *.log --spc sp_tzpop --pes azabor_PES_v2.yaml --profile azabor.json --with-conformers
goodvibes-profile plot azabor.json -o azabor.svg --label-points
goodvibes-profile evaluate azabor.json -o hot.json --temperatures 298.15,373.15
goodvibes-profile table hot.json -o azabor_si.md
```

A CSV of literature values is a profile too, and a document can mix
computed and declared series (hollow markers) with barrier annotations:

```bash
goodvibes-profile plot levels.csv -o lit.svg --quantity gibbs --temperature 298.15
goodvibes-profile convert levels.csv -o lit.yaml     # then add series, methods, annotations by hand
```

A document with a DFT and an MLIP method can give each its own
thermochemistry options. `goodvibes.thermo.by_method` re-evaluates the
method's structures with them:

```yaml
goodvibes:
  sources:
    dft:  {R: {files: "R_dft*"}, TS: {files: "TS_dft*"}}
    mace: {R: {files: "R_mace*"}, TS: {files: "TS_mace*"}}
  thermo:
    by_method:
      mace: {freq_scale_factor: 1.0, QS: truhlar, s_freq_cutoff: 50}
```

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
