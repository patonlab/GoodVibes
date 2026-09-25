# Changelog

All notable changes to GoodVibes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/). Output that the compatibility
goldens under `tests/compatibility/` pin is only changed deliberately, and
every such change is listed under **Output changes**.

## [Unreleased]

### Added
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
