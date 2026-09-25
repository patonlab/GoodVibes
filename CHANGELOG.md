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

### Output changes
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
