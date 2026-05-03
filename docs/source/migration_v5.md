# Migrating from v4.x to v5.0

GoodVibes v5.0 cleans up the programmatic surface that started landing
in v4.2. The CLI and `.dat` output are unchanged — if you only use
`goodvibes` from the shell, **no migration is needed**. This page is
for users embedding GoodVibes in scripts, notebooks, or libraries.

## TL;DR

| Old | New |
| --- | --- |
| `calc_bbe(file, QS, QH, cutoff, H_FREQ_CUTOFF, temp, conc, scale_fac, ...)` | `compute_thermo(file, **kwargs)` (returns `ThermoResult`) |
| Same, but you need the underlying `calc_bbe` instance | `calc_bbe.from_options(file_or_qcdata, ThermoOptions(...))` |
| 15 positional args | One `ThermoOptions` dataclass |

The legacy `calc_bbe(file, QS, QH, ...)` constructor still works in
v5.0 but emits a `DeprecationWarning`. It will be removed in v6.0.

## The high-level path: `compute_thermo`

For most uses, replace direct `calc_bbe` calls with `compute_thermo`:

```python
# Before (v4.x)
from goodvibes.thermo import calc_bbe

bbe = calc_bbe(
    "file.log", "grimme", True, 100.0, 100.0, 298.15, None, 0.99,
    None, "TZ", None, False, None, "global"
)
print(bbe.qh_gibbs_free_energy)

# After (v5.0)
from goodvibes import compute_thermo

r = compute_thermo("file.log", QH=True, freq_scale_factor=0.99, spc="TZ")
print(r.qh_gibbs_free_energy)
```

`compute_thermo` returns a frozen [`ThermoResult`](api_guide.md)
dataclass. Every attribute on the old `calc_bbe` instance is still
accessible via `r.bbe.<attr>` if you need it.

For batches:

```python
from goodvibes import compute_batch
results = compute_batch(["a.log", "b.log", "c.log"], jobs=8)
```

## The medium path: `from_options`

If you specifically need a `calc_bbe` instance (e.g. you're embedding
the legacy `pes.get_pes` workflow), use the new classmethod:

```python
# Before
from goodvibes.thermo import calc_bbe
bbe = calc_bbe(
    "file.log", "grimme", True, 100.0, 100.0, 298.15, None, 0.99,
    None, "TZ", None, False, None, "global"
)

# After
from goodvibes.thermo import calc_bbe, ThermoOptions

opts = ThermoOptions(QH=True, freq_scale_factor=0.99, spc="TZ")
bbe = calc_bbe.from_options("file.log", opts)
```

`from_options` accepts either a path (parses internally) or a
pre-parsed `QCData` (skips re-parsing):

```python
from goodvibes.io import parse_qcdata
qc = parse_qcdata("file.log")
bbe = calc_bbe.from_options(qc, opts)
```

Both forms emit no `DeprecationWarning`.

## `ThermoOptions` field reference

```python
@dataclass(frozen=True)
class ThermoOptions:
    QS: str = "grimme"               # 'grimme' or 'truhlar' entropy scheme
    QH: bool = False                 # Head-Gordon enthalpy correction
    s_freq_cutoff: float = 100.0     # entropy cutoff (cm⁻¹)
    h_freq_cutoff: float = 100.0     # enthalpy cutoff (cm⁻¹)
    temperature: float = 298.15      # K
    concentration: float | None = None        # mol/L; None = gas-phase 1 atm
    freq_scale_factor: float | None = None    # None = auto-lookup harm_fac
    zpe_scale_factor: float | None = None     # None = auto-lookup zpe_fac
    solv: str | None = None
    spc: str | None = None
    invert: float | None = None
    symm: bool = False
    mm_freq_scale_factor: float | None = None
    inertia: str = "global"
```

The frozen dataclass means it's safe to share across worker processes
(parallel parsing) and across multiple `from_options` calls without
accidental mutation.

## Frequency-scaling defaults

`freq_scale_factor` and `zpe_scale_factor` are auto-resolved from the
file's level of theory via the [Truhlar
database](https://t1.chem.umn.edu/freqscale/index.html) when `None`.
Three precedence cases (matches the CLI `--vscal` / `--zpe-vscal`
flags):

| `freq_scale_factor` | `zpe_scale_factor` | Result |
| --- | --- | --- |
| `None` | `None` | both auto-looked-up (`harm_fac` / `zpe_fac`) |
| `X` | `None` | both = `X` (back-compat: `--vscal` alone scales everything) |
| `None` | `Y` | `freq` auto-looked-up; ZPE uses `Y` |
| `X` | `Y` | both explicit |

## Level-of-theory metadata

`ThermoResult.level_of_theory` is now populated automatically (in v4.x
it was on a separate `read_initial()` scan). For advanced users:

```python
r = compute_thermo("file.log")
print(r.level_of_theory)   # 'B3LYP/6-31G(d)' or None if unrecognised
print(r.program)           # 'Gaussian', 'Orca', ...
```

## CLI changes in v5.0

Most flags are unchanged. Two cache flags are renamed:

| v4.x | v5.0+ | Notes |
| --- | --- | --- |
| `--cache-save FILE` | `--export FILE` | Now writes the v1.0 unified JSON (richer than the legacy cache envelope; same file works as a re-import source). Old flag still works with a `DeprecationWarning`. |
| `--cache-read FILE` | `--import FILE` | Auto-detects v1.0 unified JSON OR legacy cache envelope, so existing on-disk caches keep working. Old flag deprecated. |

`--export PATH` is a synonym for `--json PATH` — both write the same
v1.0 schema. Use whichever name reads better in your script.

Caveat: in cache-only mode (`--import FILE` with no positional
`.log` files), the level-of-theory info is unavailable, so the
frequency scale factor falls back to 1.0 unless you pass `--vscal` /
`--zpe-vscal` explicitly. Auto-restoring scale factors from the
cached `options` block is a planned follow-up.

### SPC results are cached too

When you `--export` a run that used `--spc SUFFIX`, the parsed
single-point energy (and the SPC's solvation, charge, dispersion,
multiplicity, version-program metadata) is now written into the
`qcdata` block alongside the parent's parse output. On a subsequent
`--import FILE --spc SUFFIX` run with the same suffix, the SPC file
isn't touched — the cached numbers drive the thermo. This means you
can move or delete the SPC outputs after exporting, and re-runs still
work:

```bash
# First pass: parse + persist.
goodvibes job.log --spc TZ --export job.json

# Later: SPC files can be archived; --import is enough.
goodvibes --import job.json --spc TZ
```

The cache is suffix-aware: if you re-run with `--spc QZ` (a different
suffix), GoodVibes re-resolves the SPC file from disk and refreshes
the cached numbers. Cache entries that don't match the current `--spc`
value are bypassed transparently.

## JSON output schema is now stable (v1.0)

The JSON written by `--json` / `--export` was a preview through v4.x
(`schema_version: "0.4"`) and is now stable from v1.0. Use
`goodvibes.schema` for runtime validation:

```python
import json
from goodvibes import schema

payload = json.load(open("results.json"))
schema.validate(payload)             # raises ValueError on shape/version mismatch
print(schema.SCHEMA_VERSION)         # '1.0'
```

Pre-v1.0 payloads are rejected by `validate()` — regenerate with a
v5.0+ build. The v1.0 contract is **additive**: v1.x minor versions
introduce new fields without breaking old readers, and v2.0 is the
next allowed-to-break point.

## Parquet export

```bash
goodvibes *.log --parquet thermo.parquet
```

Same column set as `--csv`; same `goodvibes[full]` extras (adds
pandas + pyarrow).

```python
from goodvibes import compute_batch, to_parquet
results = compute_batch(glob.glob("*.log"))
to_parquet(results, "thermo.parquet")
```

## What's NOT changing in v5.0

- Most CLI flags (`--vscal`, `--zpe-vscal`, `--csv`, `--jobs`,
  `--label`, `--selectivity`, etc.).
- The `.dat` output format — back-compat goldens in
  `tests/compatibility/` will guard the existing format.
- `pes.get_pes` and its parallel-list attributes (still works as a
  back-compat shim around the v4.2 PES model; will be removed in v5.1
  with a one-cycle deprecation window).

## When to actually migrate

| Situation | What to do |
| --- | --- |
| You only use `goodvibes` from the shell | Nothing — CLI is unchanged. |
| You import `compute_thermo` already | Nothing — that's the v4.2 façade and is the v5.0 path too. |
| You import `calc_bbe` and call it directly | Switch to `compute_thermo` for ergonomics, or `calc_bbe.from_options` if you specifically want the `calc_bbe` instance. |
| You depend on cited paper values produced with v4.x.0 (uniform `harm_fac`) | Pin to `goodvibes==4.2.0` until you're ready to update. v5.0 uses Truhlar's separate `zpe_fac` / `harm_fac` (the *correct* split per Alecu et al., JCTC 2010). |

See the [v4.x → v5.0 ROADMAP entry on
GitHub](https://github.com/patonlab/GoodVibes/blob/master/ROADMAP.md)
for the full list of v5.0 changes.
