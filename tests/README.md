# Tests

Run the whole suite from the repository root (an editable install with the
test extras is required: `pip install -e ".[test]"`):

```bash
pytest -q
```

The suite is pure pytest; `conftest.py` holds the fixture-directory path
helpers (`g16path`, `orca_path`, `orca5_path`, `xtb_path`, `ase_path`,
`qchem_path`, `datapath`) and the categorised file lists (frequency jobs,
transition states, single points, linear molecules, error cases) each
parser's tests parametrise over.

## Fixture data

Every parser is exercised against real program output checked in under
`tests/<program>/`. Inputs (`.com`, `.inp`, `.qcin`) sit next to the outputs
so a fixture can be regenerated.

| Directory | Program | Output files | Index |
| --- | --- | --- | --- |
| `g16/` | Gaussian 16 | 54 |`README.md` inside for the file index |
| `orca5/` | ORCA 5 | 63 | |
| `orca6/` | ORCA 6 | 71 |`README.md` inside for the file index |
| `qchem6/` | Q-Chem 6 | 51 |`README.md` inside for the file index |
| `xtb/` | xTB | 42 |`README.md` inside for the file index |
| `ase/` | ASE extended XYZ | 6 |`README.md` inside for the file index |

The legacy `test_goodvibes.py` and the PES / selectivity end-to-end tests
use the worked examples under `goodvibes/examples/` instead (those files are
in the git repository but are not shipped in the wheel).

Across the fixture sets the files follow one numbering scheme: 01-43 are
standard calculations (HF, DFT, MP2, CCSD, semi-empirical, TD-DFT, ONIOM,
various solvation models), 44-50 are transition states, 51-60 are
deliberate error cases (SCF failure, non-converged optimisation, bad
charge/multiplicity, missing basis, memory, timeout, syntax) and 61 is an
empty file.

## Test modules

776 tests at the time of writing. The first line of each module's
docstring is reproduced here; regenerate this table rather than editing it
by hand when modules are added.

| Module | Tests | Covers |
| --- | --- | --- |
| `test_api.py` | 47 | Tests for the goodvibes.api façade (v4.2 item 5). |
| `test_benchmark_entropy.py` | 1 | Benchmark: msRRHO entropies vs experimental NIST S°(298.15 K, 1 bar). |
| `test_cache.py` | 21 | Tests for QCData JSON caching (serialization round-trip, precision, integration). |
| `test_cli_errors.py` | 28 | M0 safety-net tests (AUDIT.md tasks 0.2, 0.3, 0.4). |
| `test_cli_examples.py` | 35 | CLI integration tests derived from readme_cli_examples. |
| `test_cutoff_flags.py` | 1 | -f sets both cut-offs; --fs / --fh override it for their own quantity. |
| `test_file_resolution.py` | 4 | The path a caller passes must be the file that gets parsed. |
| `test_goodvibes.py` | 10 | Legacy end-to-end tests on goodvibes/examples/ (pre-v4 suite). |
| `test_hessian.py` | 10 | Tests for io.parse_hessian (Cartesian Hessian + per-atom mass extraction). |
| `test_io_ase.py` | 17 | Tests for parsing ASE-driven calculations encoded as extxyz. |
| `test_io_g16.py` | 22 | Tests for parsing Gaussian 16 output files using goodvibes.io. |
| `test_io_orca.py` | 25 | Tests for parsing ORCA 6 output files using goodvibes.io. |
| `test_io_qchem.py` | 22 | Tests for parsing Q-Chem 6 output files using goodvibes.io. |
| `test_io_xtb.py` | 20 | Tests for parsing xtb output files using goodvibes.io. |
| `test_issue_114_ase_zpe.py` | 6 | Regression tests for issue #114 (ASE extxyz ingest and the ZPE gate). |
| `test_json_output.py` | 12 | Tests for the --json structured output flag. |
| `test_media.py` | 15 | Tests for the --media / --freespace CLI flags and the goodvibes.media module. |
| `test_modules.py` | 27 | Unit tests for extracted modules: utils, validation. |
| `test_output_rendering.py` | 35 | Direct unit tests for goodvibes.output rendering helpers. |
| `test_parse_data_truncated.py` | 1 | Regression test: parse_data on a Gaussian output with no route section. |
| `test_pes_cli_options.py` | 3 | --nogconf / --lowest-only must reach every PES consumer. |
| `test_pes_e2e.py` | 9 | End-to-end PES regression test against the azabor_PES_v2.yaml fixture. |
| `test_pes_legacy.py` | 16 | Tests for goodvibes.pes_legacy — the line-based `--- # PES` format. |
| `test_pes_loader.py` | 29 | Tests for goodvibes.pes_loader — pattern resolution + builder + dispatcher. |
| `test_pes_model.py` | 36 | Tests for goodvibes.pes_model — pure data + arithmetic, no I/O. |
| `test_pes_output.py` | 17 | Tests for the v4.2 PES output: Rich tables (`print_pes_tables`) and JSON v1.0 (`_pes_to_json` + `write_json_results`). |
| `test_pes_temperature_interval.py` | 2 | `--pes` together with `--ti` must produce the legacy per-temperature PES tables instead of crashing. |
| `test_pes_yaml.py` | 24 | Tests for goodvibes.pes_yaml — the proper YAML PES format. |
| `test_plot.py` | 33 | Tests for goodvibes.plot — selectivity strip plots, PES profiles. |
| `test_schema.py` | 14 | Tests for goodvibes.schema — version constants + payload validator. |
| `test_selectivity.py` | 50 | Tests for the new --label / --selectivity API and the legacy --ee shim. |
| `test_sort.py` | 30 | Tests for goodvibes.sort: kabsch_rmsd, deduplicate, sort_thermo. |
| `test_supporting.py` | 13 | Tests for supporting modules: vib_scale_factors and media. |
| `test_symm_fields.py` | 3 | --symm / symm=True: the detected point group and symmetry number must be reported, and pymsym's symmetry number must replace (not stack on) one already present in the output file. |
| `test_thermo_ase.py` | 11 | Tests for thermochemistry calculations on ASE-driven extxyz fixtures. |
| `test_thermo_g16.py` | 66 | Tests for thermochemistry calculations on Gaussian 16 output files. |
| `test_thermo_orca.py` | 11 | Tests for thermochemistry calculations on ORCA 6 output files. |
| `test_thermo_orca5.py` | 4 | Lightweight regression coverage for ORCA 5 output parsing and thermo. |
| `test_thermo_qchem.py` | 19 | Tests for thermochemistry calculations on Q-Chem 6 output files. |
| `test_thermo_xtb.py` | 5 | End-to-end thermochemistry tests on xtb output files. |
| `test_validation.py` | 17 | Tests for goodvibes.validation: collect_and_validate_files, print_check_fails, and check_files (smoke). |
| `test_vmm_removed.py` | 5 | The ONIOM MM-region frequency scaling feature (``--vmm`` / ``mm_freq_scale_factor`` / ``QCData.fract_modelsys``) was removed in v4.5. |

## Tolerances

Gaussian comparisons are exact to the precision Gaussian prints. ORCA
thermochemistry comparisons use a 5e-6 Eh tolerance: ORCA prints enthalpy
and Gibbs values to 8 decimals from a higher-precision internal value, so
GoodVibes reproduces them to sub-microhartree but not always to 1e-6. ORCA
tests pass `inertia='conf'` so the quasi-RRHO average moment of inertia is
computed per conformer, matching ORCA, rather than Grimme's global value.

## Deprecation warnings

The legacy 15-argument `calc_bbe(...)` constructor emits a
`DeprecationWarning`; several older test modules still call it on purpose
to pin its behaviour, so those warnings in the pytest summary are expected.
