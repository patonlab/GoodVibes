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
|---|---|---|---|
| 1 | CBS/Gn composite method detection (CBS-QB3, CBS-4M, G3, G3B3) | Deferred — needs Gaussian fixtures | — |
| 2 | `--media` / `--freespace` integration tests + clear errors when solvent is unknown | ✅ Done | [`04e2b63`](../../commit/04e2b63) |
| 3 | Test coverage gaps: sort, validation modules | ✅ Done | [`f88abfa`](../../commit/f88abfa) |
| 3a | Test coverage gaps: selectivity, PES modules | Deferred — both modules slated for redesign first | — |
| 4 | `--json OUTPUT.json` preview (schema v0.1 → v0.3) | ✅ Done | [`6e6187f`](../../commit/6e6187f) |
| Sub-plan A | **Selectivity redesign**: N-way labels, structured `SelectivityResult`, dual Boltzmann + lowest-conformer output, JSON v0.3 | ✅ Done | [`5fb8176`](../../commit/5fb8176) |

---

## v4.2 — Mid features (1–2 months, no API breaks)

| # | Item | Status |
|---|---|---|
| 5 | **Programmatic API façade** `goodvibes.api`: `compute_thermo(path) -> ThermoResult` and `compute_batch(...)`. `ThermoResult` mirrors `calc_bbe`'s public attributes + the source `QCData`. Internally just calls `calc_bbe` — no behavior change. | Pending |
| 6 | **Pandas DataFrame export + CSV writer**: `goodvibes.api.to_dataframe(results)`. Pandas as an optional dep (`goodvibes[full]`). Add `--csv` CLI flag. | Pending |
| 7 | **Wigner tunneling correction** for TS rates: κ_W = 1 + (1/24)(hcν‡/kT)². Eckart deferred to v5.0. Auto-applied when `--tunneling wigner` and `len(im_frequency_wn) == 1`. | Pending |
| 8 | **Parallel parsing** with `concurrent.futures.ProcessPoolExecutor`, `--jobs N`. Target: 1,000 ORCA outputs in <30s on 8 cores. | Pending |
| 9 | **Hindered-rotor treatment** (Pitzer–Gwinn / Truhlar HO-QHO interpolation). Manual `--hindered-rotor MODE_INDEX,V,I_red` for v4.2; auto detection deferred to v5.0+. | Pending |
| Sub-plan B | **PES rewrite** | Not yet drafted |

**Sequencing.** Item 5 is the keystone — unblocks 6 and enables clean
Wigner integration in 7. Items 8 and 9 are independent and can run in
parallel with 5/6/7.

---

## v5.0 — Major (2–3 months, breaking changes allowed with migration path)

| # | Item | Status |
|---|---|---|
| 10 | **First-class structured outputs**: promote `--json` and `--csv` to documented stable schemas (`schema_version: "1.0"`). Parquet via pandas behind extras. Define `goodvibes.schema`. **Subsumes `--cache-save` / `--cache-read`** under one schema with `--export` / `--import` modes. | Pending |
| 11 | **Ensemble container** with lazy parsing, streaming Boltzmann/dedup. Refactor `selectivity`, `pes`, `sort.deduplicate` to consume `Ensemble`. Handles 10⁴ conformers without holding them all in memory. | Pending |
| 12 | **Clean programmatic API as the headline**: `from goodvibes import compute_thermo, Ensemble, ThermoResult`. Deprecate `calc_bbe`'s 15-arg constructor; add `calc_bbe.from_options(qcdata, ThermoOptions)`. Migration doc. | Pending |
| 13 | **Conformational entropy correction**: S_conf = −R Σ pᵢ ln pᵢ on Ensemble; `boltzmann_averaged_G(T)` includes −T·S_conf. Wires into `pes.get_pes` so Gconf becomes first-class. | Pending |
| 14 | **Visualization** (stretch): PES diagram, Boltzmann histogram, T-scan curves. Behind `goodvibes[plot]`. | Pending |

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

## Sub-plan B — PES rewrite

Not yet drafted. Will be added when planning starts.
