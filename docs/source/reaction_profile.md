# The reaction-profile format

`reaction-profile/1.x` is a small, tool-independent document format for
reaction energy profiles: the species and points of a mechanism, the
pathways through them, and one or more *series* of relative energies
(computed by GoodVibes from quantum-chemistry or MLIP data, or typed in from
a paper). One document is the data behind a figure, its table and its
provenance. Version 1.1 adds `selectivity`: competing branches and the
selectivity their barriers predict.

- **Status.** 1.x is a draft until a second, independent tool reads and
  writes it; minor versions only ever add optional keys.
- **Licence.** The JSON Schema and this specification text are CC0-1.0
  (public domain); GoodVibes itself is MIT.
- **Schema.** [`reaction-profile-1.1.schema.json`](https://goodvibespy.readthedocs.io/en/latest/reaction-profile-1.1.schema.json)
  (JSON Schema draft 2020-12), installed with GoodVibes; it validates 1.0
  and 1.1 documents. The 1.0 schema,
  [`reaction-profile-1.0.schema.json`](https://goodvibespy.readthedocs.io/en/latest/reaction-profile-1.0.schema.json),
  stays published unchanged.
- **Reference implementation.** `goodvibes.profile` (Python) and the
  `goodvibes-profile` command.
- **Conformance kit.** `tests/profile_conformance/` in the repository:
  documents a reader must accept (`valid/`), reject for structural reasons
  (`invalid-structural/`, also rejected by the JSON Schema) and reject for
  referential reasons (`invalid-semantic/`, which a JSON Schema cannot
  express).

## A minimal profile

Typed-in values, no QC data, 20 lines:

```yaml
schema: reaction-profile/1.0
title: Minimal profile
units: kcal/mol
points:
  R:   {role: reactant}
  TS1: {role: ts, display: "TS1‡"}
  Int: {role: minimum}
  TS2: {role: ts, display: "TS2‡"}
  P:   {role: product}
pathways:
  main: [R, TS1, Int, TS2, P]
series:
  - id: G
    label: "ΔG (298 K)"
    quantity: gibbs
    temperature: 298.15
    source: declared
    levels:
      main: {R: 0.0, TS1: 18.4, Int: 3.2, TS2: 12.9, P: -12.1}
```

```bash
goodvibes-profile plot minimal.yaml -o minimal.svg
```

## Document structure

A document is a YAML or JSON mapping. Only `schema` and `pathways` are
required.

| Key | Type | Meaning |
| --- | --- | --- |
| `schema` | string | `reaction-profile/1.<minor>` |
| `title`, `description` | string | free text |
| `units` | `kcal/mol` \| `kJ/mol` \| `eV` \| `hartree` | units of every level in the document (default `kcal/mol`) |
| `ensemble` | `ideal-gas` | the statistical ensemble; the only one defined in 1.0 |
| `default_temperature` | number > 0 | K; used by computed series without a temperature (default 298.15; the `goodvibes` command evaluates them at its run temperature instead) |
| `species` | mapping | identity of each species (no files, no values) |
| `points` | mapping | the nodes of the profile |
| `pathways` | mapping | ordered points, a zero and edges |
| `order` | list | optional x order of point ids across all pathways |
| `methods` | mapping | provenance of the series (program, level of theory, ...) |
| `series` | list | the energies: one line set on the axes, one column set in a table |
| `annotations` | list | barriers and spans to mark on the figure |
| `selectivity` | list | (1.1) competing branches sharing a reference point |
| `style` | mapping | presentation hints |
| `provenance` | mapping | who made the document, from what |
| `goodvibes` | mapping | GoodVibes' recipe namespace; other readers ignore it |
| `x-…` | any | extension keys; preserved, never validated |

### species

`name: {smiles, inchi, formula, name, charge, multiplicity}`, every key
optional (`{}` or `null` is fine). Species carry identity only; where their
structures come from belongs to a tool's namespace.

### points

`id: {species, role, display}`.

- `species` is a stoichiometric sum, as a string (`"A + 2*B"`) or a mapping
  (`{A: 1, B: 2}`, required when a species name contains `+`). Omit it for a
  point that only appears in declared series.
- `role` is `reactant`, `minimum` (default), `ts` or `product`. A figure
  labels transition states above their bar and other points below.
- `display` is the label a figure prints (default: the id), e.g. `"TS1‡"`.

A pathway may name a point by its species sum without listing it under
`points` (`pathways: {p: ["A + B", "TS"]}` with species `A`, `B`, `TS`
declared); the point is created with that id.

### pathways

`name: {points: [ids], zero: id, edges: [...]}`, or just the list of ids.

- `zero` (default: the first point) is the reference every level of the
  pathway is relative to. It may be any defined point.
- Edges default to one `step` between each pair of consecutive points. A
  listed edge `{from, to, kind}` replaces the default edge between the same
  two points or adds a new one; `kind` is `step`, `barrierless` (drawn
  dotted) or `none` (no connector).
- Pathways may have different lengths and share points; a figure aligns
  them on one x axis by point id (`order` overrides the merged order).

### methods

`id: {program, model, level_of_theory, solvent, standard_state,
frequency_scaling, quasi_harmonic, hessian, reference, description}`, all
optional and free-form, e.g.
`standard_state: {concentration_M: 1.0}` or `{pressure_atm: 1}`,
`reference: {doi, note}`.

### series

`{id, label, method, quantity, temperature, source, levels, uncertainty,
style, standard_state}`; `id` and `quantity` are required, ids are unique.

- `source: declared` — typed-in values; `levels` is required and the series
  is never re-evaluated. `source: computed` (default) — evaluated by a tool
  from structures; its `levels` are absent until it is evaluated.
- `levels` is `{pathway: {point: value}}`: values in the document `units`,
  relative to that pathway's zero. A point missing from a pathway's levels
  is not drawn; `null` means the point is on the pathway but its value is
  unknown.
- `uncertainty` has the same shape, values ≥ 0.
- `temperature` in K (`null` for a temperature-independent value).
- `style` holds drawing hints such as `{linestyle: dashed}`.

`quantity` is one of:

| id | label | meaning |
| --- | --- | --- |
| `spc` | ΔE_SPC | single-point electronic energy |
| `electronic` | ΔE | electronic energy at the frequency level |
| `zpe` | ΔZPE | zero-point vibrational energy |
| `e_zpe` | ΔE+ZPE | electronic energy (single point when applied) plus ZPE |
| `enthalpy` | ΔH | enthalpy |
| `qh_enthalpy` | Δqh-H | quasi-harmonic enthalpy (Head-Gordon) |
| `entropy` | T·ΔS | T times the RRHO entropy |
| `qh_entropy` | T·Δqh-S | T times the quasi-harmonic entropy |
| `gibbs` | ΔG(T) | Gibbs energy |
| `qh_gibbs` | Δqh-G(T) | quasi-harmonic Gibbs energy |

GoodVibes also accepts aliases on input (`G`, `E`, `H`, `qh-G`, ...) and
writes the canonical id.

### annotations

`{type: barrier | span, pathway, from, to, series, label}` marks the
difference `to − from` on one pathway (in `series`, default the first
drawn one).

### selectivity (1.1)

Each entry names points that compete from one shared point: enantiomeric or
diastereomeric transition states, regioisomeric pathways, a chemoselective
choice.

```yaml
schema: reaction-profile/1.1
selectivity:
  - id: er                     # unique
    label: R vs S              # optional
    kind: enantio              # enantio | diastereo | regio | chemo | other (default)
    reference: Int             # the point the branches leave from
    branches: [TS_R, TS_S]     # at least two points, in the order that sets the ee sign
    series: G                  # optional: the series to use (default: every series with the levels)
    interconversion: TS_swap   # optional: the barrier between the states feeding the branches
```

**What it predicts.** For a series, each branch's barrier is
`ΔG‡ᵢ = level(branchᵢ) − level(reference)`, both read on one pathway that
holds the two points (so pathways with different zeros are fine). The
branch populations are the Curtin–Hammett distribution
`pᵢ = exp(−ΔG‡ᵢ/RT) / Σⱼ exp(−ΔG‡ⱼ/RT)` at the series' temperature
(`default_temperature` when it has none). A computed series rolled up over
conformers gives each branch's ensemble free energy, so the conformers of
every branch count. Declared series work too, so literature values predict
a selectivity the same way.

**Conventions.**
- The *major* branch has the largest population; on an exact tie, the
  first listed.
- With two branches, `ee = (p₁ − p₂) × 100` with the branches in the
  order listed: positive when the first branch is the major one. The
  excess without a sign is `|ee|`, and `ΔΔG‡ = RT ln(p_major/p_minor) ≥ 0`.
- With more branches, `ratio = p_major / p_runner-up`.

**Curtin–Hammett.** The distribution assumes the states feeding the
branches interconvert faster than they react, from the reference as their
common ground state. A reader reports the assumption as:
- *violated* when a branch lies at or below the reference, when a point
  between the reference and a branch on its pathway lies below the
  reference (a deeper resting state, so the energy span, not the barrier
  from the reference, applies), or when the `interconversion` barrier (its
  level minus the reference's) is not lower than the lowest branch
  barrier;
- *satisfied* when a lower `interconversion` barrier is given;
- *assumed* otherwise.

A branch that is not a transition state gets a note: its population is an
equilibrium ratio, not a kinetic selectivity.

### style

`{preset, layout: overlay | panels, connector: bezier | linear | step,
label_points: bool, decimals: 0-6, figsize: [w, h]}`, all hints a reader
may ignore. `preset` names a target for the figure:

| preset | figure | text |
| --- | --- | --- |
| `none` (default) | sized to the profile | the plotting library's defaults |
| `single-column` | about 85 mm wide (3.35 × 2.6 in) | 7 pt; value labels 6 pt |
| `double-column` | about 178 mm wide (7.0 × 3.2 in) | 8 pt; value labels 7 pt |
| `slide` | 16:9 (10 × 5.6 in) | 16 pt; value labels 14 pt |

An explicit `figsize` overrides the preset's.

### provenance

Free-form. GoodVibes writes `{tool, goodvibes_version, generated_at,
invocation, inputs: [{file, species, method, sha1}], warnings}`.

## Rules a reader enforces

The JSON Schema checks structure. A conforming reader additionally rejects:

1. a pathway point, zero, `order` entry or edge end that is not a defined
   point (or, for pathway points, a sum of declared species);
2. an edge whose ends are not both on the pathway, or an `order` that omits
   a pathway point;
3. a point that uses an undeclared species;
4. a series whose `levels` / `uncertainty` name an unknown pathway, or a
   point that is neither on that pathway nor its zero;
5. duplicate series ids, and a `method` that is defined nowhere;
6. an annotation naming an unknown pathway, point or series;
7. (1.1) a selectivity entry with a duplicate id, a reference, branch or
   `interconversion` point that is not defined, a branch that is the
   reference or listed twice, a branch or `interconversion` point on no
   pathway with the reference, or an unknown `series`.

Unknown keys that do not start with `x-` produce a warning (an error in
strict mode).

## Versioning

`reaction-profile/MAJOR.MINOR`. A minor adds optional keys only; a reader
of 1.0 reads any 1.x document, warning that keys it does not know are
ignored. A major version changes meaning and is refused by readers of the
previous major.

A document using a 1.1 key must declare `reaction-profile/1.1`: a
`selectivity` block in a 1.0 document is an error. A writer should declare
the oldest version that can express the document: GoodVibes writes 1.0
unless the document has a `selectivity` block, so a 1.0 reader reads
everything else it writes without a warning.

## The GoodVibes namespace

Everything GoodVibes needs to *compute* a series lives under `goodvibes:`.

```yaml
goodvibes:
  sources:                   # method -> species -> which output files
    default:
      R1-An:  {files: "r1-li-3thf-*"}         # glob on the file stem
      AmTS:   {dir: "AmTS"}                     # every file in that directory
      THF:    {files: [thf.log], dirs: ["THF_*"]}
  rollup: {mode: gconf, weight_by: qh_gibbs}   # gconf | boltzmann | lowest
  thermo: {QS: grimme, QH: false, s_freq_cutoff: 100.0, spc: sp_tzpop}
  dedup: {e_cutoff: 0.05, ro_cutoff: 0.01}
  conformers: {...}          # written by --with-conformers; see below
```

`goodvibes.thermo` records the options the structures were computed with.
Its `by_method` key maps a method id to the options that method sets for
itself: `QS`, `QH`, `s_freq_cutoff`, `h_freq_cutoff`, `concentration`,
`freq_scale_factor`, `zpe_scale_factor`, `solv`, `invert`, `symm` and
`inertia`.

```yaml
  thermo:
    QS: grimme
    by_method:
      mace: {freq_scale_factor: 1.0, QS: truhlar, s_freq_cutoff: 50}
```

When a document is evaluated, that method's structures are re-evaluated with
these options in place of the ones they were computed with, so a DFT and an
MLIP method can differ in scaling and qRRHO settings within one evaluation.
Setting either scale factor resolves both again: a `freq_scale_factor`
alone sets both, and a factor that is left out or null is looked up for
the level of theory. A method id that is not defined, an unknown option and
`spc` / `strict_spc` (the single point is read with the output files) are
errors. The re-evaluation needs the parsed structures: thermo data from
`compute_thermo` / `calc_bbe.from_options`, the `goodvibes` command, or
embedded conformers.

`goodvibes.conformers` (written with `--with-conformers` or
`Profile.evaluate(..., with_conformers=True)`) stores every structure's
parsed data and thermochemistry options. A document that carries them can be
re-evaluated at any temperature, redrawn and retabulated with no output
files: the azabor example (about 100 Gaussian outputs) becomes a 380 KB JSON
file (`goodvibes/examples/profiles/azabor_profile.json`).

## Making and using documents

**From output files** (the `goodvibes` command):

```bash
goodvibes *.log --spc sp_tzpop --pes azabor_PES_v2.yaml --profile azabor.json
goodvibes *.log --pes profile.yaml --profile evaluated.json --with-conformers
goodvibes *.log --pes profile.yaml --json out.json      # payload 1.1: the `profile` block
```

`--pes` accepts a reaction-profile document, the v2 PES YAML or the legacy
`--- # PES` text; the two older formats are upgraded (point ids are their
point labels). A v2 / legacy file gets one computed series of
`--pes-plot-quantity` per temperature (`--ti` gives several). A
reaction-profile document keeps its own series, and `--pes-plot` draws them
with its declared values and annotations. `--nogconf` and `--lowest-only`
override the document's `goodvibes.rollup`; without them the document's
rollup is used.

**Without output files** (`goodvibes-profile`):

```bash
goodvibes-profile validate profile.yaml [--strict]
goodvibes-profile plot azabor.json -o azabor.svg -o azabor.pdf [--series ID,...] [--layout panels] [--preset single-column]
goodvibes-profile table azabor.json [-o table.csv | table.md] [--long]
goodvibes-profile convert levels.csv -o profile.yaml --quantity gibbs --temperature 298.15
goodvibes-profile convert old_pes.yaml -o profile.yaml          # v2 / legacy -> explicit form
goodvibes-profile evaluate azabor.json -o hot.json --temperatures 298.15,373.15
goodvibes-profile plot azabor.json --temperatures 273,373 -o scan.png
goodvibes-profile selectivity profile.json [--id er] [--temperatures 273,298] [-o sel.csv | --json]
goodvibes-profile diff old.json new.json [--tolerance 0.05] [--series G] [--json]
```

`selectivity` prints each block's major branch, signed ee, ΔΔG‡,
Curtin–Hammett status and branch table. `diff` lists what changed between
two documents (points, pathways, series metadata, every level beyond
`--tolerance` after converting to one set of units, selectivity blocks and
annotations) and exits 1 when they differ. The `goodvibes` command prints a
document's selectivities after its PES tables, and `--json` adds them as a
`profile_selectivity` list.

`evaluate` and `--temperatures` need embedded conformers. A GoodVibes
`--json` payload with a `profile` block is accepted wherever a document is,
and so is an SVG figure GoodVibes saved.

**Figures carry their data.** An SVG written by `goodvibes-profile plot`,
`Profile.plot(...).save` or `ProfileAxes.save` has:
- text kept as text;
- an `id` on every bar, connector, value label, error bar and barrier
  marker (for example `bar-G-main-TS1`);
- a `<metadata id="goodvibes-reaction-profile">` element holding, as JSON,
  the reaction-profile document of what was drawn (the drawn series with
  their levels and uncertainties, no embedded conformers) and an index from
  each `id` to the series, pathway and point it shows.

`load_profile("fig.svg")` and `goodvibes-profile table fig.svg` read the
numbers back; `--no-embed` (`save(..., embed=False)`) leaves the document
out. A series' `uncertainty` is drawn as ± error bars (`--no-uncertainty`
to omit them).

**Tables of relative energies.** A CSV (or TSV) is read as a declared-only
profile. Wide layout: a `point` column, optional `role` and `display`, then
one column per pathway (`pathway:series` for several series); an empty cell
means the point is not on that pathway, `null` an unknown value. Long
layout: `pathway, point, value` plus optional `series, label, quantity,
temperature, units, method, role, display, uncertainty`.

```text
point,role,display,Ph,Ph-lit
R,reactant,R,0.0,0.0
TS1,ts,TS1‡,20.1,18.4
P,product,P,-12.6,-12.1
```

**From Python:**

```python
import glob
from goodvibes import compute_batch, load_profile

prof = load_profile("profile.yaml")                 # .yaml/.json/.csv, v2 or legacy PES
results = compute_batch(glob.glob("*.log"), spc="sp_tzpop")
ev = prof.evaluate(results, temperatures=[298.15, 373.15], with_conformers=True)
ev.dump("profile.json")                             # .json / .yaml
fig = ev.plot(label_points=True)                    # ProfileAxes
fig.save("profile.svg")
ev.to_dataframe("long")                             # pandas, one row per pathway × point × series
ev.write_table("si_table.md")
ev.evaluate_selectivity()                           # 1.1 selectivity blocks -> [SelectivityResult]
ev.diff(load_profile("other.json"), tolerance=0.05) # ProfileDiff; empty when they agree
```

`Profile.from_table(...)`, `Profile.from_pes_result(pes)` (for a model built
in Python) and `validate_document(mapping)` complete the API; see
`goodvibes.profile`.
