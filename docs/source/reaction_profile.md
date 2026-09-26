# The reaction-profile format

`reaction-profile/1.0` is a small, tool-independent document format for
reaction energy profiles: the species and points of a mechanism, the
pathways through them, and one or more *series* of relative energies
(computed by GoodVibes from quantum-chemistry or MLIP data, or typed in from
a paper). One document is the data behind a figure, its table and its
provenance.

- **Status.** 1.0 is a draft until a second, independent tool reads and
  writes it; minor versions only ever add optional keys.
- **Licence.** The JSON Schema and this specification text are CC0-1.0
  (public domain); GoodVibes itself is MIT.
- **Schema.** [`goodvibes/schemas/reaction-profile-1.0.schema.json`](https://raw.githubusercontent.com/patonlab/GoodVibes/master/goodvibes/schemas/reaction-profile-1.0.schema.json)
  (JSON Schema draft 2020-12), installed with GoodVibes.
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

### style

`{layout: overlay | panels, connector: bezier | linear | step,
label_points: bool, decimals: 0-6, figsize: [w, h]}`. `preset` is reserved
(only `none` is accepted in 1.0).

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
6. an annotation naming an unknown pathway, point or series.

Unknown keys that do not start with `x-` produce a warning (an error in
strict mode). Keys reserved for later minors (`selectivity`, a `style.preset`
other than `none`) are rejected, never silently ignored.

## Versioning

`reaction-profile/MAJOR.MINOR`. A minor adds optional keys only; a reader
of 1.0 reads any 1.x document, warning that keys it does not know are
ignored. A major version changes meaning and is refused by readers of the
previous major.

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
goodvibes-profile plot azabor.json -o azabor.svg -o azabor.pdf [--series ID,...] [--layout panels]
goodvibes-profile table azabor.json [-o table.csv | table.md] [--long]
goodvibes-profile convert levels.csv -o profile.yaml --quantity gibbs --temperature 298.15
goodvibes-profile convert old_pes.yaml -o profile.yaml          # v2 / legacy -> explicit form
goodvibes-profile evaluate azabor.json -o hot.json --temperatures 298.15,373.15
goodvibes-profile plot azabor.json --temperatures 273,373 -o scan.png
```

`evaluate` and `--temperatures` need embedded conformers. A GoodVibes
`--json` payload with a `profile` block is accepted wherever a document is.

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
```

`Profile.from_table(...)`, `Profile.from_pes_result(pes)` (for a model built
in Python) and `validate_document(mapping)` complete the API; see
`goodvibes.profile`.
