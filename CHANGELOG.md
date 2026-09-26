# Changelog

All notable changes to GoodVibes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/). Output that the compatibility
goldens under `tests/compatibility/` pin is only changed deliberately, and
every such change is listed under **Output changes**.

## [Unreleased]

### Added
- A gallery of reproducible reaction-profile figures
  (`goodvibes/examples/gallery`, `docs/source/gallery.md`): Δqh-G with every
  conformer, one profile at four temperatures, ΔE / ΔH / Δqh-G on one axes,
  competing transition states in panels, a CSV table, and computed with
  declared values. `build_gallery.py` redraws every figure from compact
  committed inputs (reaction-profile documents with embedded conformers and a
  CSV table), without the program outputs; `tests/test_gallery.py` rebuilds
  them and checks their numbers against the CLI.
- The reaction-profile format, `reaction-profile/1.0` (M2a of the direction
  plan): a tool-independent document for reaction energy profiles (species,
  points with roles and display labels, pathways with a zero and edges,
  methods, computed and declared series, annotations, style, provenance),
  specified in `docs/source/reaction_profile.md` and published as a JSON
  Schema (`goodvibes/schemas/reaction-profile-1.0.schema.json`, CC0).
  - `goodvibes.profile`: `Profile`, `load_profile`, `validate_document`.
    Reads the explicit form (YAML/JSON), the v2 PES YAML, the legacy
    `--- # PES` text, CSV/TSV tables of relative energies (wide or long) and
    GoodVibes payloads with a `profile` block; `evaluate` fills the computed
    series from thermo data or from embedded conformers (at any
    temperature), `dump`, `to_rows` / `to_dataframe` / `write_table` (CSV,
    Markdown), `plot`, `from_pes_result`.
  - The reference validator checks the JSON Schema's structural rules plus
    the referential ones it cannot express; the conformance kit
    (`tests/profile_conformance/`) pins their agreement. Unknown keys warn
    (error with `strict`); `x-*` keys are preserved; keys reserved for a
    later minor are rejected.
  - `goodvibes --profile PATH` writes the evaluated document;
    `--with-conformers` embeds every structure's parsed data and options
    so the document can be re-evaluated without the output files.
  - `--pes` accepts a reaction-profile document. `--pes-plot` then draws the
    document's own series (declared values, annotations); pathways whose
    points carry only declared values are left out of the Rich tables and
    the `pes` block, with a note. With `--ti` such a document is tabulated
    by the model at every scan temperature (the legacy text path is kept
    for the v2 and legacy formats), and `--graph` refuses it. Its computed
    series that give no temperature are evaluated at the run temperature,
    like the tables and the `pes` block.
  - `goodvibes-profile` (new console script): `validate`, `plot`, `table`,
    `convert`, `evaluate`. It reads documents and tables only, never QC
    outputs.
  - `goodvibes/examples/profiles/`: a minimal document, a CSV table, and the
    azabor profile with embedded conformers (380 KB in place of about 100
    Gaussian outputs) plus the script that regenerates it.
- The profile model (M1 of the direction plan):
  - `ComputedEntry` (a parsed structure plus its `ThermoOptions`, evaluable
    at any temperature and memoised) and `calc_bbe.options`, the resolved
    options kept on every result built through `from_options` /
    `compute_thermo`.
  - `ConformerSet.from_results`, `entries`, `weight_by`, `vectors(T)` and
    the public rollups `populations`, `ensemble_free_energy`, `s_conf`,
    `rollup` and `dedup` (same gates and convention as `--dedup`). A set
    built from real results re-evaluates its conformers at other
    temperatures instead of reusing the base-temperature values.
  - `Point.role` (`reactant | minimum | ts | product`) and `Point.display`;
    `Edge` (`step | barrierless | none`) and `Pathway.edges` /
    `with_edges` / `point`; `Pathway.levels(T, quantity)`. The PES Rich
    table and JSON block now read `Pathway.relative` instead of forming
    the differences themselves.
  - `Series`: a quantity at a temperature, computed from the model or
    declared (typed-in levels that are never re-evaluated, with their own
    units); `PESResult.series`, `default_series`, `merged_order`,
    `levels`, `pathway`, `order`; `merge_point_order`.
  - `plot_profile`: every pathway on one merged x axis (pathways of
    different lengths and branches that share a point line up by label);
    several series on one axes (temperature overlay, ΔE with Δqh-G,
    literature values as hollow markers) with linestyle per series and
    colour per pathway; `layout="panels"`; TS labels above and minima
    below the bar; barrierless edges dotted; returns a `ProfileAxes` with
    the drawn levels, `annotate_barrier` and `save`. `plot_pes` is a thin
    wrapper over it.
  - `goodvibes.output.pes_tables`: the CLI's PES Rich tables as
    `rich.table.Table` objects, usable without `setup_logging`.
- `QCData.from_atoms` and `QCData.from_vibrations`: thermochemistry from an
  ASE `Atoms`, an energy and a vibrational analysis with no output file
  (MLIP workflows). Unit conversion (eV / Hartree / kcal/mol / kJ/mol;
  cm⁻¹ / eV / meV), removal of the translational and rotational modes of a
  3N Hessian, a noise threshold for small imaginary modes, imaginary-mode
  count checks against the declared job type, pymsym symmetry detection,
  isotopic or ASE masses. `compute_batch` accepts `QCData` objects.
- `QCData.level_of_theory` (filled by the `.extxyz` parser and
  `from_atoms`) drives the scale-factor lookup for file-free inputs and
  `ThermoResult.level_of_theory`.
- The most-abundant-isotope mass table now covers every element to
  uranium (was H–Xe).
- Everything above, plus `load_pes`, `QCData`, `ThermoOptions`,
  `calc_bbe`, the quantity registry and `compute_selectivity`, is
  importable from `goodvibes`.
- `CHANGELOG.md`, `CITATION.cff`, `CONTRIBUTING.md`.
- `--strict-spc`, `--dedup-global`, `--pes-plot-quantity` documented in the
  README option table.

### Changed
- PyYAML is a core dependency (reaction-profile documents and PES files
  are YAML); `jsonschema` is optional and in the `test` extra; new
  `profile` extra (matplotlib).
- `--nogconf` / `--lowest-only` override the rollup of the `--pes` file
  only when given; without them a reaction-profile document's
  `goodvibes.rollup` is used (the v2 and legacy formats have no rollup
  setting, so their behaviour is unchanged).
- A computed `Series` may carry stored `levels` (an evaluated document);
  `Pathway.levels` skips points without species.
- `plot_pes` no longer rejects pathways of different lengths (they share
  the merged x axis) or `show_conformers=True` with several pathways
  (dots are drawn per pathway in its colour).
- `compute_thermo` leaves an unset `concentration` unresolved in the
  stored `ThermoOptions` (it is still the gas-phase P/RT when evaluated),
  so a result re-evaluated at another temperature gets that temperature's
  standard state. The numbers of a single call are unchanged.
- One removal version for everything deprecated in 4.x: **6.0** (`--ee`,
  `--cache-save`/`--cache-read`, the legacy `--- # PES` format, `--graph`,
  the 15-argument `calc_bbe` constructor). Messages and docs previously
  said v5.0, v5.1, v6.0 or "a future release".
- `docs/source/migration_v5.md` describes the API generation that shipped
  in 4.2-4.4 rather than a future v5.0 release; the cookbook selectivity
  recipe now runs (`{r.file: r.bbe}`), and its `dir:` example no longer
  points at a directory that does not exist.
- pytest configuration in `pyproject.toml`: a `DeprecationWarning` for the
  legacy `calc_bbe` constructor raised from inside `goodvibes` is an error.
- `tests/compatibility/`: 29 CLI goldens (`.dat` and `--json`) that pin the
  user-visible output of the common flag combinations.
- `goodvibes.quantities`: one registry of the quantities GoodVibes tabulates
  or plots (ids, labels, JSON keys, aliases) and `ThermoVector.get()`.
- `plot_pes(quantity=...)` and `--pes-plot-quantity` (alias `--gtype`) to
  draw ΔE, ΔE+ZPE, ΔH, Δqh-H, T·ΔS, ΔG or Δqh-G profiles. Restores the
  `--gtype E` capability added for issue #57 in 2022 and lost in the 4.x
  PES rewrite.
- Energy units `eV` and `hartree` alongside `kcal/mol` and `kJ/mol`
  everywhere a display unit is chosen (`goodvibes.constants.hartree_factor`).
- `goodvibes.utils.parse_temperature_interval` (float `--ti` grids).
- `calc_bbe.qcdata`: the parsed input is kept on the result.

### Fixed
- `plot_profile` gave a series without an explicit linestyle the next style
  in the cycle even when another series had asked for it, so two series
  could both be dotted; default styles now skip the chosen ones.
- `goodvibes/examples/profiles/levels.csv`, referenced by the documentation,
  was never committed (`*.csv` is ignored); CSV files under
  `goodvibes/examples` are now tracked, and the file says its values are
  illustrative.
- `solvents.json` and `scaling_factors.json` were read with the platform's
  default encoding, so importing GoodVibes failed under a non-UTF-8 locale
  (e.g. `LANG=C`); both are now read as UTF-8.
- `plot_pes(show_conformers=True)` placed conformer dots for multi-species
  points tens of thousands of kcal/mol away from the bar; dots now sit at
  the point level plus each conformer's offset from its species rollup.
  `thermo_lookup` is no longer required (ignored with a DeprecationWarning).
- `--ti` truncated temperatures and steps to integers.
- The physical constants were defined in four places.
- A requested single-point correction that could not be applied (missing or
  unparseable `--spc` partner, `--spc link` without a link job, cache without
  the SPC) silently left H and G at the frequency-level energy. It now warns
  (`RuntimeWarning`, and a line in the `.dat`), `ThermoResult.spc_applied`
  records the outcome, and `strict_spc=True` / `--strict-spc` makes it an
  error (`MissingSinglePointError`).
- `--dedup` compared structures across `--label` species and could merge an
  R/S transition-state pair (its gates cannot tell enantiomers apart). It is
  now scoped within each species; `--dedup-global` restores the old scope.
- `invert='auto'` never kept the reaction coordinate of an ASE or Q-Chem
  transition state: those parsers wrote `job_type='TS'` while the rule
  tested for `'TSFreq'`. Both parsers now report `TSFreq` when frequencies
  are present and the rule accepts either spelling.
- `write_thermo_extxyz` wrote an imaginary mode from ASE's complex
  frequency array as `0.0 cm-1`; it now writes `-|ν|`.

### Output changes
- `--json` / `--export` payloads are schema **1.1**: a `profile` block (the
  evaluated reaction-profile document, without conformers) is added when
  `--pes` is used. 1.0 readers ignore it; `--import` reads 1.0 and 1.1.
  The `.dat` line reporting the JSON file says `schema v1.1`.
- The gconf notice above the PES tables follows the rollup actually used
  (it read the command-line flag, so a document's Boltzmann rollup was
  announced as gconf).
- `--pes --ti`: the PES model is built for the scan, so the `--json` `pes`
  block is now written (one entry per pathway per temperature; it was
  absent) and `--pes-plot` works, overlaying the temperatures on one axes.
  The printed per-temperature PES text is unchanged.
- `--json` / `--export`: the `qcdata` block has a `level_of_theory` key
  (empty for the Gaussian, ORCA, NWChem, Q-Chem and xTB parsers).
- A `--pes` file whose species match no file, or whose pathway names an
  undefined species, ends with a `✗ FATAL ERROR` line instead of a Python
  traceback.
- The `.dat` archive no longer contains terminal escape codes. PES table
  titles (italic) and column headers (bold) were written with ANSI styling
  on ordinary terminals since 4.2; on a dumb terminal (`TERM=dumb`, e.g.
  CircleCI) they were plain, which is why the compatibility goldens passed
  on GitHub Actions and failed on CircleCI. The `.dat` copy of every Rich
  table is now rendered as a plain file on every terminal.
- The CLI prints a `!` deprecation notice (stdout and `.dat`) when `--ee` or
  the legacy `--- # PES` text format is used. The Python
  `DeprecationWarning` for these was attributed to GoodVibes' own modules
  and hidden by the default warning filters, so CLI users never saw it.
- `--json` / `--export`: `thermo.job_type` for an ASE (`.extxyz`) or Q-Chem
  transition state with frequencies is `TSFreq` (was `TS`), matching the
  Gaussian and ORCA parsers.
- `--ti` thermochemistry table: rows are now computed through
  `calc_bbe.from_options` with the same options as the single-temperature
  table, so `--symm` is honoured in the scan (it used to be dropped). Runs
  without `--symm` print identical numbers.
- `--label`/`--selectivity` with `--ti`: populations, ee and ΔΔG at each
  scan temperature are now computed from free energies re-evaluated at
  that temperature; previously the base-temperature free energies were
  reused and only RT in the Boltzmann factor changed. Results at the base
  temperature are unchanged.

## [4.4.0] - 2026-09-19

See the GitHub release notes for this and earlier versions.
