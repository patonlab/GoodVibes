# Programmatic API

Starting with **v4.2**, GoodVibes exposes a clean kwargs-based Python API
in addition to the CLI. The same parser and thermochemistry engine
power both — the API just wraps `calc_bbe` in a friendlier signature
and returns a structured result.

## Quick start

```python
from goodvibes import compute_thermo

r = compute_thermo("structure.log")
print(f"qh-G(T) = {r.qh_gibbs_free_energy:.6f} Hartree")
print(f"level   = {r.level_of_theory}")
```

`compute_thermo` returns a frozen `ThermoResult` dataclass with every
attribute `calc_bbe` produces (energies in Hartree, entropies in
Hartree/K, frequencies in cm⁻¹), plus references to the underlying
`bbe` and `qcdata` for advanced use.

```python
@dataclass(frozen=True)
class ThermoResult:
    file: str                       # absolute path
    name: str                       # basename without extension
    scf_energy: float
    sp_energy: float | None         # None unless --spc was used
    zpe: float
    enthalpy: float
    qh_enthalpy: float | None       # None when QH=False
    entropy: float
    qh_entropy: float
    gibbs_free_energy: float
    qh_gibbs_free_energy: float
    frequency_wn: list[float] | None
    im_frequency_wn: list[float] | None
    inverted_freqs: list[float] | None
    point_group: str | None
    symmno: int | None
    linear_mol: bool
    multiplicity: int | None
    job_type: str | None
    level_of_theory: str | None
    program: str | None             # 'Gaussian', 'Orca', ...
    bbe: Any                        # original calc_bbe instance
    qcdata: Any                     # parsed QCData
```

## Common options

```python
r = compute_thermo(
    "structure.log",
    QS="grimme",            # default — Grimme quasi-RRHO entropy
    s_freq_cutoff=50,       # cm⁻¹ — soften vibrational modes below this
    spc="TZ",               # use 'structure_TZ.log' for the single point
    temperature=313.15,     # K
    concentration=1.0,      # mol/L; defaults to gas-phase 1 atm
    freq_scale_factor=None, # None → auto-lookup from level of theory
)
```

All keyword names match the CLI flags. Defaults match what the CLI
does when those flags aren't passed.

## Batch and parallel processing

```python
from goodvibes import compute_batch
import glob

paths = glob.glob("conformers/*.log")

# Sequential (default).
results = compute_batch(paths)

# Parallel: spawn 8 worker processes.
results = compute_batch(paths, jobs=8)

# Use all available CPU cores.
results = compute_batch(paths, jobs=0)
```

`compute_batch` preserves input order. With `jobs > 1`, parsing and
thermochemistry run in a `ProcessPoolExecutor`; on a typical laptop
this gives 2–3× speedup at 8 cores once the file count is large enough
to amortise process startup (~50 files).

## Pandas DataFrame export

```python
from goodvibes import compute_batch, to_dataframe

results = compute_batch(glob.glob("*.log"))
df = to_dataframe(results)
df.to_csv("thermo.csv", index=False)
df.sort_values("qh_gibbs_free_energy").head()
```

`to_dataframe` requires pandas; install with `pip install goodvibes[full]`
(includes pandas, ase, and pyyaml).

The CLI flag `--csv PATH` does the same thing without leaving the shell:

```bash
goodvibes *.log --csv thermo.csv
```

## Skipping a re-parse

If you've already parsed an output file (e.g. via
:py:func:`goodvibes.io.parse_qcdata`), pass it in to avoid re-reading:

```python
from goodvibes.io import parse_qcdata
from goodvibes import compute_thermo

qc = parse_qcdata("structure.log")
r = compute_thermo(qcdata=qc)
```

## What's new in v4.x at a glance

- **v4.1 — Selectivity redesign.** N-way `--label NAME=PATTERN` (or
  `--selectivity FILE.yaml`) replaces the 2-only `--ee a:b`. Outputs
  Boltzmann-averaged AND lowest-conformer-only tables. Structured
  `SelectivityResult` exposed on the JSON output.
- **v4.1 — `--json PATH`.** Structured output (schema v1.0) with
  per-file thermochemistry, parsed metadata, options, plus optional
  `selectivity` and `pes` blocks.
- **v4.2 — PES rewrite.** New 3-layer model
  (`ConformerSet`/`Point`/`Pathway`); true-YAML input format alongside
  the legacy line-based format (auto-detected, deprecated); stoichiometric
  sums (`2*A + B`); `--lowest-only` mode for "lowest qh-G conformer per
  species" PES tables.
- **v4.2 — Programmatic API.** This page.
- **v4.2 — `--jobs N` parallel parsing.** ~3× speedup at 8 cores.
- **v4.2 — `--csv PATH`.** Per-structure DataFrame export.
- **v4.2 — ORCA CPU-time scaling.** ORCA prints wall time only;
  GoodVibes now multiplies by the parsed MPI process count to give
  an effective CPU time, matching the Gaussian/NWChem/xTB convention.
  Footnoted on the `TOTAL CPU` line.

See the [project ROADMAP](https://github.com/patonlab/GoodVibes/blob/master/ROADMAP.md)
for what's coming in v5.0 (`Ensemble` container, conformational entropy
correction, visualization, and more).
