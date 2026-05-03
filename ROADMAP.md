# GoodVibes Roadmap

Tracking the v4.1 → v4.2 → v5.0 progression of GoodVibes.

The library is mature (v4.0): six native QC parsers (Gaussian, ORCA,
NWChem, Q-Chem 6, xTB, ASE), a comprehensive test suite, and a clean
modular structure. This roadmap addresses three structural gaps that
day-to-day lab use has surfaced:

1. **No clean programmatic API.** `calc_bbe` is invoked with 15 positional
   arguments and writes to a `.dat` file; embedding GoodVibes in
   notebooks or pipelines (CREST → conformer search → thermo) is awkward.
2. **No structured output.** Downstream tools and dashboards want JSON /
   CSV / Parquet — today they parse the `.dat` text.
3. **Doesn't scale to large conformer ensembles.** `pes.py` /
   `selectivity.py` / `sort.deduplicate` work on lists; a 10³–10⁴
   conformer batch is awkward.

Plus several in-flight features that are partially detected
(CBS/Gn composites) or untested (`--media`, `--freespace`).

---

## v4.1 — Polish & complete (4–6 weeks, backwards-compatible)

| # | Item | Status | Commit |
| --- | --- | --- | --- |
| 1 | CBS/Gn composite method detection (CBS-QB3, CBS-4M, G3, G3B3) | Deferred — needs Gaussian fixtures | — |
| 2 | `--media` / `--freespace` integration tests + clear errors when solvent is unknown | ✅ Done | [`04e2b63`](../../commit/04e2b63) |
| 3 | Test coverage gaps: sort, validation modules | ✅ Done | [`f88abfa`](../../commit/f88abfa) |
| 3a | Test coverage gaps: selectivity, PES modules | ✅ Done — coverage shipped with the redesigns (selectivity ≈87%, pes_loader 100%, pes_model 96%, pes_yaml 90%, pes_legacy 95%) | — |
| 4 | `--json OUTPUT.json` preview (schema v0.1 → v0.3) | ✅ Done | [`6e6187f`](../../commit/6e6187f) |
| Sub-plan A | **Selectivity redesign**: N-way labels, structured `SelectivityResult`, dual Boltzmann + lowest-conformer output, JSON v0.3 | ✅ Done | [`5fb8176`](../../commit/5fb8176) |

---

## v4.2 — Mid features (1–2 months, no API breaks)

| # | Item | Status |
| --- | --- | --- |
| 5 | **Programmatic API façade** `goodvibes.api`: `compute_thermo(path) -> ThermoResult` and `compute_batch(...)`. `ThermoResult` mirrors `calc_bbe`'s public attributes + the source `QCData`. Internally just calls `calc_bbe` — no behavior change. Re-exported from `goodvibes/__init__.py`. | ✅ Done |
| 6 | **Pandas DataFrame export + CSV writer**: `goodvibes.api.to_dataframe(results)`, `--csv PATH` CLI flag, `goodvibes[full]` extras group (ase + pyyaml + pandas). | ✅ Done |
| 8 | **Parallel parsing** with `concurrent.futures.ProcessPoolExecutor`, `--jobs N` CLI flag, `compute_batch(paths, jobs=N)`. ~3× speedup at 8 cores on 46 azabor PES fixtures (18.6 s → 6.2 s). | ✅ Done |
| Sub-plan B | **PES rewrite**: 3-layer data model (`ConformerSet`/`Point`/`Pathway`), true-YAML input alongside legacy parser, stoichiometry support, Rich tables, JSON v0.4 `pes` block, `--lowest-only` flag. Plot deferred to v5.0. | ✅ Done — [`03549d1`](../../commit/03549d1) |
| Docs | **Refresh Read the Docs** for v4.1 + v4.2: bumped Sphinx config to v4.0, swapped deprecated `sphinxcontrib-napoleon` / `recommonmark` → `sphinx.ext.napoleon` / `myst-parser`, added a dedicated `api_guide.md` page (kwargs API, batch/parallel, DataFrame/CSV, what's-new) and a `cookbook.md` page with five v4.x recipes, expanded module reference to all 17 modules, added `.readthedocs.yaml`. Stays on Sphinx; the mkdocs migration is reserved for v5.0. | ✅ Done |
| Follow-up A | **Separate ZPE vs harmonic frequency scaling factors**. The Truhlar database (Alecu et al., JCTC 2010) stores `zpe_fac` and `harm_fac` as distinct per-LOT values. Auto-detection now reads both: `harm_fac` for the partition-function frequencies (`calc_vibrational_energy`, `calc_rrho_entropy`) and `zpe_fac` for ZPE (`calc_zeropoint_energy`). New `--zpe-vscal` CLI flag and `zpe_scale_factor` API kwarg for explicit override. When `--vscal` alone is set, ZPE inherits it (back-compat for users with pinned scripts). Pre-refactor v3.x used `zpe_fac` uniformly; v4.x.0 used `harm_fac` uniformly — the new behavior matches Truhlar's intent and is the right default going forward. | ✅ Done |

**Sequencing.** Item 5 was the keystone — unblocked the structured
DataFrame/CSV work in 6. Item 8 is independent. Docs refresh sits
on top of whichever subset has shipped at the time it's tackled.

---

## v5.0 — Major (2–3 months, breaking changes allowed with migration path)

| # | Item | Status |
| --- | --- | --- |
| 10 | **First-class structured outputs**: stable `schema_version: "1.0"` in `goodvibes.schema` (TypedDicts + version constants + runtime validator). Parquet export via `to_parquet(results, path)` + `--parquet PATH` CLI flag (pandas + pyarrow behind `[full]` extras). Cache subsumed: new `--export` / `--import` flags read/write the unified v1.0 payload; `--cache-save` / `--cache-read` deprecated as aliases (still accept the legacy envelope on read). JSON writer no longer gated on `--ti` mode. | ✅ Done |
| 11 | **Ensemble container** with lazy parsing, streaming Boltzmann/dedup. Refactor `selectivity`, `pes`, `sort.deduplicate` to consume `Ensemble`. Handles 10⁴ conformers without holding them all in memory. | Pending |
| 12 | **Clean programmatic API as the headline**: `from goodvibes import compute_thermo, ThermoOptions, ThermoResult`. New `ThermoOptions` frozen dataclass replaces the 15 positional args. New `calc_bbe.from_options(qcdata_or_path, ThermoOptions)` classmethod is the v5.0+ entry point — also handles auto-lookup of Truhlar `harm_fac`/`zpe_fac` so any caller (not just `compute_thermo`) gets sensible defaults. Direct `calc_bbe(file, QS, ...)` calls now emit a `DeprecationWarning` (legacy form still works through v5.x; removed in v6.0). Internal callers (`compute_thermo`, parallel worker) migrated to `from_options`. New `docs/source/migration_v5.md` page covers the path for users embedding GoodVibes. `Ensemble` re-export deferred until item 11 lands. | ✅ Done |
| 13 | **Conformational entropy correction**: S_conf = −R Σ pᵢ ln pᵢ on Ensemble; `boltzmann_averaged_G(T)` includes −T·S_conf. Wires into `pes.get_pes` so Gconf becomes first-class. | Pending |
| 14 | **Visualization**: new `goodvibes/plot.py` with `plot_selectivity_strip` (per-species ΔG scatter + Boltzmann-mean / lowest overlays) and `plot_pes` (PES profile from `PESResult` — multi-pathway overlay, bezier or linear connectors, configurable colors, optional point-value annotations, optional per-conformer scatter). New `--strip-plot PATH` and `--pes-plot PATH` CLI flags; `--graph FILE.yaml` retained for YAML-driven styling until v5.1. `[plot]` extras add matplotlib. Stubs for `plot_boltzmann_histogram` and `plot_temperature_scan` lock in the v5.1 API. | ✅ Done |
| 15 | **Hindered-rotor treatment** (Pitzer–Gwinn / Truhlar HO-QHO). Auto-detect torsional modes from the normal-mode analysis + redundant internals; replace the HO contribution to S/H/ZPE with the rotor partition function. Manual `--hindered-rotor MODE,V,I_red` override. Moved from v4.2 (was item 9) — auto-detection is the version users actually want; the manual flag alone has too much friction. | Pending |

**Breaking changes.** `pes.get_pes` and `selectivity.get_boltz`
signatures change (list[dict] → Ensemble) with a one-cycle shim
accepting both. CLI flags + `.dat` output unchanged across the entire
roadmap.

---

## Cross-cutting concerns

- **Backwards compat.** CLI + `.dat` output identical across v4.x. v5.0
  may rename internal kwargs; CLI stays. A `tests/compatibility/`
  directory will diff `.dat` against checked-in goldens for the 20
  most-used flag combinations.
- **Deprecation policy.** Anything deprecated in v4.2 is removed no
  earlier than v5.1. CI runs `pytest -W error::DeprecationWarning`.
- **Docs.** mkdocs + mkdocstrings auto-API docs at v5.0. Cookbook section
  with notebook examples (CREST → `Ensemble` → Boltzmann G).
- **Performance targets.** 1k conformers parsed in <30s on 8 cores;
  ensemble Boltzmann/dedup memory <200 MB at 10k conformers.
- **CI.** Add Python 3.13 at v5.0 cut.

---

## Sub-plan A — Selectivity redesign (shipped in v4.1)

A worked example of how items in this roadmap are designed before
implementation begins.

### Goals

1. **N-way selectivity**: generalize from 2-bucket ee to N-bucket dr
   (e.g. four diastereomers). The 2-bucket case is a special instance.
2. **Explicit labels** instead of fragile filename globs. Decouple
   algorithm from filename templating.
3. **Structured `SelectivityResult` dataclass** + JSON output.
4. **Temperature scan** that composes naturally with `--ti`.
5. **Dual reporting**: Boltzmann-averaged AND lowest-conformer-only,
   so the user can see how much selectivity comes from the gap between
   the lowest TSs vs. conformer mixing.

### Result type

```python
@dataclass(frozen=True)
class SelectivityResult:
    temperature: float                      # K
    key: str                                # 'gibbs' | 'energy'
    labels: List[str]                       # ordered species names
    files_per_label: Dict[str, List[str]]
    populations: Dict[str, float]           # normalized: Σ = 1.0
    raw_boltzmann: Dict[str, float]
    preferred: str                          # max-population label
    ee: Optional[float] = None              # 2-label only, in %
    ddG: Optional[float] = None             # 2-label only, in Hartree
```

Numeric data only. Ratio strings (`60:40`, `40:30:20:10`) are derived
in the print layer from `populations`. For N>2, `ee` and `ddG` are None;
consumers derive any ratios they want from `populations`.

### CLI

- `--label NAME=PATTERN` (repeatable; fnmatch against basenames of files
  already in `thermo_data` — no filesystem walks).
- `--selectivity FILE.yaml` (alternative for many species or shareable
  specs). Top-level key `labels:` for patterns, or `files:` for explicit
  per-species file lists.
- `--label`/`--selectivity` combine with `--ti` for temperature scans.
- `--ee 'a:b'` keeps working in v4.x with a `DeprecationWarning`;
  removed in v5.0.

### Output

- Two stacked Rich tables per result: Boltzmann-averaged + Lowest
  conformer only. Each has Species/Files/Population (%)/ΔΔG (kcal/mol)
  columns plus a summary line (ratio, major, ee + ΔΔG‡ for N=2).
- Temperature scan: one row per T per method.
- JSON schema v0.3: top-level `selectivity` and `selectivity_lowest`
  blocks, both with the same shape.

### Resolved decisions

- **Pattern matching**: `fnmatch` against basenames of files already in
  `thermo_data`. No filesystem walks. The candidate set is exactly what
  the user passed on the command line.
- **Ratio formatting**: numeric data only on the dataclass; ratio
  strings live only in the print layer.
- **N=2 vs N>2 reporting**: N=2 emits ratio + ee + ΔΔG‡; N>2 emits only
  the ratio. No pairwise data.
- **Empty species**: `compute_selectivity` raises `ValueError`; `main()`
  translates to `fatal()` for the CLI.
- **`dup_list` semantics**: each pair is `[duplicate, canonical]` —
  excluding only `dup[0]` is correct; the canonical structure stays in
  the sum.

---

## Sub-plan B — PES rewrite (v4.2)

The current `pes.get_pes` is line-based parsing (despite the `.yaml`
extension and `---` document markers), and the internal data model is
~30 parallel lists nested four levels deep
(`g_qhgvals[pathway][step][structure][conformer]`). New features
(stoichiometry, JSON output, conformational entropy in v5.0) are hard
to wire in without a redesign.

### Design goals

1. **Three-layer data model** that matches how chemists think about a
   PES: pathway → point → species → conformers.
2. **Two input formats**: keep the existing line-based "yaml" working
   (auto-detected, `DeprecationWarning`, removed in v5.1); add a true
   YAML schema as the documented format.
3. **Stoichiometry**: `2*A + B` syntax for multi-mol reactions, valid
   in both formats.
4. **Rich tables + JSON v0.4** output, mirroring the v4.1 selectivity
   redesign style.
5. **Plot deferred** — `graph_reaction_profile` is matplotlib-tangled
   and orthogonal; rewritten as part of v5.0 visualization.

### Data model

```python
@dataclass(frozen=True)
class ThermoVector:
    """One species' thermo bundle in Hartree. Supports +, -, * scalar."""
    sp_energy: Optional[float]
    scf_energy: float
    zpe: float
    enthalpy: float
    qh_enthalpy: float
    entropy: float                  # S, not T·S — temperature lives on Pathway
    qh_entropy: float
    gibbs: float
    qh_gibbs: float

@dataclass
class ConformerSet:
    """A named species + ≥1 calc_bbe entries; encapsulates the gconf math."""
    name: str
    files: List[str]
    def boltzmann_weighted(self, T: float) -> ThermoVector: ...
    def lowest_conformer(self) -> ThermoVector: ...
    def gconf_corrected(self, T: float, QH: bool = True) -> ThermoVector: ...

@dataclass
class Point:
    """Stoichiometric sum at one node: [(coeff, ConformerSet), ...]."""
    label: str                      # "Int-I + TolS + TolSH" or "2*A + B"
    species: List[Tuple[int, ConformerSet]]
    def thermo(self, T: float, gconf: bool = True, QH: bool = True) -> ThermoVector: ...

@dataclass
class Pathway:
    name: str
    points: List[Point]
    zero: Point                     # default = points[0]
    def relative(self, T: float, ...) -> List[ThermoVector]: ...

@dataclass
class PESResult:
    pathways: List[Pathway]
    options: PESOptions             # units, dp, gconf, QH
    temperatures: List[float]       # supports --ti
```

`ThermoVector` arithmetic collapses 9 simultaneous parallel-list
subtractions into one expression:
`relative = point.thermo(T) - zero.thermo(T)`.

### Stoichiometry

Coefficient syntax: `2*A`, `2 * A`, `A` (implicit 1). Integer
coefficients only. Parsing is regex-based on the point string and
applied identically in both input formats. The thermo math is
multiplication on `ThermoVector` (`2 * A.thermo(T) + B.thermo(T)`).

### Input formats

**Legacy (auto-detected by `--- # PES` / `# SPECIES` / `# FORMAT` comment markers):**
parser kept for v4.x; emits `DeprecationWarning` on use; removed v5.1.
The example at [goodvibes/examples/pes/azabor_PES.yaml](goodvibes/examples/pes/azabor_PES.yaml)
(6-step pathway with conformer ensembles, stoichiometric sums, and
`--spc tzpop` pairs) is the back-compat regression fixture.

**True YAML:**

```yaml
pathways:
  Reaction: ["Int-I + TolS + TolSH", "Int-II + TolSH", "Int-III"]

species:
  Int-I:    {files: "Int-I_*.log"}            # glob
  Int-II:   {files: "Int-II_*.log"}
  Int-III:  {files: "Int-III_*.log"}
  TolS:     {files: "TolS.log"}
  TolSH:    {files: "TolSH.log"}

zero:
  Reaction: "Int-I + TolS + TolSH"             # optional; default = points[0]

format:
  units: kcal/mol
  decimals: 1
```

Each species is a dict to leave room for future per-species options
(scaling factor overrides, symmetry numbers, ...) without breaking the
schema. `files:` accepts a string (glob or single file) or list.

### Output format

**Rich table** per pathway × per temperature, columns matching the
current `.dat` layout (`DE`, `DZPE`, `DH`, `qh-DH`, `T·DS`, `T·qh-DS`,
`DG(T)`, `qh-DG(T)`, plus SPC variants when `--spc` is set).

**JSON v0.4** adds a top-level `pes` block:

```json
"pes": {
  "pathways": [{
    "name": "Reaction",
    "temperature": 298.15,
    "units": "kcal/mol",
    "points": [{
      "label": "Int-I + TolS + TolSH",
      "species": [
        {"coefficient": 1, "name": "Int-I", "files": ["Int-I_Oax.log"]},
        {"coefficient": 1, "name": "TolS", "files": ["TolS.log"]},
        {"coefficient": 1, "name": "TolSH", "files": ["TolSH.log"]}
      ],
      "relative": {"scf": 0.0, "zpe": 0.0, "h": 0.0, "qh_h": 0.0,
                   "ts": 0.0, "qh_ts": 0.0, "g": 0.0, "qh_g": 0.0}
    }]
  }]
}
```

`.dat` text output unchanged across v4.x.

### Code structure

| File | Change |
| --- | --- |
| `goodvibes/pes_model.py` | New: `ThermoVector`, `ConformerSet`, `Point`, `Pathway`, `PESResult`. Pure data model + arithmetic; no I/O. |
| `goodvibes/pes_legacy.py` | New: extract current `pes.get_pes` parser, target the new model. Emits `DeprecationWarning`. |
| `goodvibes/pes_yaml.py` | New: true-YAML parser → new model. Stoichiometry parsing lives here (shared with legacy). |
| `goodvibes/pes.py` | Becomes a dispatcher: sniff format, parse, return `PESResult`. `get_pes` retained as back-compat shim until v5.1. |
| `goodvibes/output.py` | New `print_pes_tables(result)` Rich renderer. Existing text path kept for v4.x. JSON writer extended with `pes` block. |
| `goodvibes/GoodVibes.py` | No new flags; `--pes FILE` already handles both formats. |
| `tests/test_pes.py` | New: model tests (stub data), parser tests (both formats), integration test against [examples/pes/](goodvibes/examples/pes/). |

### Migration

| Version | Behavior |
| --- | --- |
| v4.2 | Both parsers work; legacy emits `DeprecationWarning`. New model is the runtime backbone. JSON schema bumps to 0.4. |
| v5.0 | Refactor `ConformerSet` onto `Ensemble` (item 11). `Pathway.relative()` becomes streaming. No user-visible change. |
| v5.1 | Legacy parser removed. CLI fails with a clear error pointing at the new schema. |

### Decisions

- **Stoichiometry**: integer coefficients via `n*Species` syntax in
  both formats; default 1.
- **Plot rewrite**: deferred to v5.0 visualization. `graph_reaction_profile`
  keeps reading the legacy `get_pes` attributes via a thin compat shim
  in v4.2 — no behavior change.
- **Legacy timing**: deprecated in v4.2, removed in v5.1.
- **Slot in roadmap**: v4.2, alongside item 5 (API façade) and item 6
  (DataFrame export) — both depend on a clean `PESResult` for JSON
  serialization.
