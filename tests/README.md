# Tests

## Directory Structure

```
tests/
├── conftest.py              # Shared fixtures, path helpers, categorized file lists
├── test_goodvibes.py        # Legacy test suite (see note below)
├── test_io_g16.py           # Gaussian 16 parsing tests (goodvibes.io)
├── test_io_orca.py          # ORCA 6 parsing tests (goodvibes.io)
├── test_thermo_g16.py       # Gaussian 16 thermochemistry tests (goodvibes.thermo)
├── test_thermo_orca.py      # ORCA 6 thermochemistry tests (goodvibes.thermo)
├── test_supporting.py       # vib_scale_factors and media module tests
├── g16/                     # Gaussian 16 test data (62 .com inputs, 63 .log outputs)
│   └── README.md            # File index with method, job type, and key features
└── orca6/                   # ORCA 6 test data (63 .inp inputs, 64 .out outputs)
    └── README.md            # File index with method, job type, and key features
```

## Test Modules

### `test_io_g16.py` — Gaussian 16 Parsing

Tests `getoutData`, `parse_data`, `level_of_theory`, `read_initial`, and
`gaussian_jobtype` from `goodvibes/io.py` against the G16 log files.

- **Atom extraction** — atom types and counts
- **Frequency extraction** — mode counts (3N-6 nonlinear, 3N-5 linear)
- **Cartesian coordinates** — shape and extraction
- **Single-point detection** — SP-only files have no frequencies
- **SCF energy** — parsed energy matches grep of log file
- **Program detection** — identified as Gaussian
- **Charge / multiplicity** — neutral, cation, anion, triplet, quintet
- **Job progress** — Normal termination vs Incomplete vs Error
- **Solvation model** — PCM, CPCM, SMD detection
- **Level of theory** — method/basis string
- **Job type classification** — SP, Freq, GSFreq, TSFreq

### `test_io_orca.py` — ORCA 6 Parsing

Tests the subset of `goodvibes/io.py` functions that work with ORCA 6 output.
`parse_data` and `read_initial` have independent ORCA parsing paths that work
correctly. `getoutData` (cclib) and `level_of_theory` (Gaussian archive format)
are marked `xfail` for ORCA 6.

- **Energy, program, charge/multiplicity** — via `parse_data`
- **Progress and solvation** — via `read_initial`
- **getoutData** — xfail (cclib 1.7.2 incompatible with ORCA 6)
- **level_of_theory** — xfail (relies on Gaussian archive section)

### `test_thermo_g16.py` — Gaussian 16 Thermochemistry

Tests `calc_bbe` and individual thermo functions from `goodvibes/thermo.py`.

**Ground-truth validation against Gaussian output:**

Many tests compare `calc_bbe` results directly against values printed by
Gaussian in the log file, providing end-to-end validation:

- **ZPE vs Gaussian** — `calc_bbe` ZPE compared against the Gaussian
  `Zero-point correction=` line across 49 files
- **Enthalpy vs Gaussian** — `calc_bbe` enthalpy compared against
  `Sum of electronic and thermal Enthalpies=` across 49 files
- **Gibbs free energy vs Gaussian** — `calc_bbe` Gibbs energy compared against
  `Sum of electronic and thermal Free Energies=` across 49 files
- **Non-standard T/P** — ground-truth validation for files computed at
  non-default temperature and pressure (e.g. T=398.15 K, P=2 atm)
- **Non-standard scaling** — ground-truth validation for files computed with
  `Freq=(Scale=0.95)`, including deuterium isotope substitution

**Quasi-harmonic method tests:**

- **Grimme at 298.15 K** — full thermodynamic quantities (E, ZPE, H, TS, TqhS, G, qhG)
- **Truhlar at 298.15 K** — same quantities with Truhlar quasi-harmonic entropy
- **Head-Gordon QH enthalpy** — quasi-harmonic enthalpy correction (QH=True)

**Parameterized feature tests:**

- **Temperature variations** — sweep over 100–500 K
- **Frequency scaling** — effect of scale factor on ZPE and G
- **Transition states** — imaginary frequency count and sign
- **Solvation files** — PCM/CPCM/SMD files run without error
- **Linear molecules** — correct mode count (3N-5)
- **Linked jobs** — frequency extraction from multi-step Gaussian jobs
- **Single-point only** — no thermochemistry for SP files
- **Error files** — graceful handling of malformed output

**Unit tests for individual thermo functions:**

- `calc_translational_energy` — E_trans = 3/2 RT
- `calc_rotational_energy` — nonlinear (3/2 RT), linear (RT), atom (0)
- `calc_electronic_entropy` — S_elec = R ln(multiplicity)
- `calc_damp` — Grimme damping function behavior above, at, and below cutoff

### `test_thermo_orca.py` — ORCA 6 Thermochemistry

All tests are marked `xfail`. `calc_bbe` relies on `getoutData` (cclib) which
cannot parse ORCA 6 output, and `thermo.py` lacks ORCA-specific frequency
parsing. These tests document the intended coverage for when ORCA 6 support
is added.

### `test_supporting.py` — Supporting Modules

- **`vib_scale_factors`** — reference index bounds, dict lookup, hyphen stripping
- **`media`** — common solvent entries exist, positive MW/density, water properties

### `test_goodvibes.py` — Legacy Tests

This file contains the original test suite and remains for direct comparison
with earlier test results. It uses example files from `goodvibes/examples/`
rather than the newer `tests/g16/` test data. Tests cover quasi-harmonic
corrections (Grimme/Truhlar), temperature corrections, single-point
corrections, scaling factor search, concentration corrections, media
corrections, and potential energy surface analysis.

## Known xfail Cases

Several expected failures are tracked in the test suite:

| Category | Reason |
|----------|--------|
| ORCA 6 (`getoutData`, `level_of_theory`, `calc_bbe`) | cclib 1.7.2 cannot parse ORCA 6 output; `thermo.py` lacks ORCA frequency parsing |
| Anharmonic VPT2 (files 12, 23) | cclib reads both harmonic and anharmonic frequency blocks, doubling vibrational contributions |
| CCSD Opt+Freq (file 40) | cclib extracts the MP2 energy instead of the converged CCSD energy |

## Test Data

The `g16/` and `orca6/` subdirectories contain synthetic Gaussian 16 and
ORCA 6 output files covering:

- **Files 01–43** — standard calculations (HF, DFT, MP2, CCSD, semi-empirical,
  TD-DFT, ONIOM) with various basis sets, solvation models, and features
- **Files 44–50** — transition states (SN2, Diels-Alder, H-abstraction, E2,
  umbrella inversion, ring opening)
- **Files 51–60** — deliberate error cases (SCF failure, opt non-convergence,
  bad charge/multiplicity, missing basis, memory, timeout, syntax, linear bend,
  basis linear dependency, missing blank line)
- **File 61** — empty log file for edge-case testing

See `g16/README.md` and `orca6/README.md` for the complete file index.
