![GoodVibes](https://github.com/patonlab/GoodVibes/blob/master/goodvibes.png)
===

[![Tests](https://github.com/patonlab/GoodVibes/actions/workflows/tests.yml/badge.svg)](https://github.com/patonlab/GoodVibes/actions/workflows/tests.yml)
[![PyPI version](https://badge.fury.io/py/goodvibes.svg)](https://badge.fury.io/py/goodvibes)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/goodvibes/badges/downloads.svg)](https://anaconda.org/conda-forge/goodvibes)
[![Documentation Status](https://readthedocs.org/projects/goodvibespy/badge/?version=stable)](https://goodvibespy.readthedocs.io/en/stable/?badge=stable)
[![DOI](https://zenodo.org/badge/54848929.svg)](https://zenodo.org/badge/latestdoi/54848929)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/goodvibes/badges/license.svg)](https://anaconda.org/conda-forge/goodvibes)

GoodVibes computes quasi-harmonic thermochemical corrections from electronic structure calculations (Gaussian, ORCA, NWChem, QChem, xTB, ASE). It corrects the poor description of low-frequency vibrations by the rigid-rotor harmonic oscillator (RRHO) treatment using the approaches of [Grimme](http://dx.doi.org/10.1002/chem.201200497) and [Truhlar](http://dx.doi.org/10.1021/jp205508z).

#### Features

- Grimme quasi-RRHO (mRRHO) and Truhlar quasi-harmonic entropy corrections
- Head-Gordon quasi-harmonic enthalpy correction
- Variable temperature and concentration thermochemistry
- Automated vibrational frequency scaling factor lookup (~200 levels of theory)
- Single-point energy corrections (link jobs or separate files)
- Boltzmann-weighted populations and N-way stereoselectivity (`--label`)
- Potential energy surface analysis with YAML-defined pathways, stoichiometric sums (`2*A + B`), and Gconf corrections
- Structured JSON output (`--json`) for downstream pipelines
- Symmetry-corrected entropy via pymsym point-group detection
- Solvent standard-state concentration and free-space corrections
- Skip re-parsing across runs with `--export` / `--import` (v1.0 unified JSON; legacy `--cache-save` / `--cache-read` retained as deprecated aliases)
- Duplicate structure detection (`--dedup`) and energy-sorted output (`--sort`)
- Supports Gaussian, ORCA, NWChem, QChem, xTB, and ASE output files


#### Installation

Requires Python >= 3.9.

```bash
pip install goodvibes
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install goodvibes
```

Or from conda-forge:

```bash
conda install -c conda-forge goodvibes
```

For development (editable install):

```bash
pip install -e .
```

#### Quick Start

```bash
goodvibes <output_file(s)> -q
```

The `-q` flag applies quasi-harmonic corrections (Grimme entropy + Head-Gordon enthalpy). Use `-h` for the full list of options.

#### Supported Programs

GoodVibes reads output files from:

- **Gaussian** (09, 16) `.log` / `.out` -- optimization, frequency, single-point, link jobs, ONIOM, VPT2 anharmonic
- **ORCA** (5, 6) `.out` -- optimization, frequency, single-point, DLPNO-CCSD(T)
- **NWChem** `.out` -- optimization, frequency, single-point
- **QChem** (6) `.out` / `.qcin` -- optimization, frequency, single-point, linked jobs
- **xTB** `.out` -- frequency calculations (ORCA/xTB integration compatible)
- **ASE** (Atomic Simulation Environment) `.extxyz` -- extended XYZ format with energy & frequency data

The program is auto-detected from the output file contents. Additional file extensions can be registered with `--custom_ext`.

#### Documentation

Full documentation is available on [Read the Docs](https://goodvibespy.readthedocs.io/en/latest/).

#### Examples

##### Example 1: Grimme-type quasi-harmonic correction with a cut-off of 150 cm<sup>-1</sup>
```bash
goodvibes examples/methylaniline.out -f 150

   Structure                    E        ZPE             H        T.S     T.qh-S          G(T)       qh-G(T)
   *********************************************************************************************************
o  methylaniline      -326.664901   0.142118   -326.514489   0.039668   0.039465   -326.554157   -326.553954
   *********************************************************************************************************

```

The output shows both standard harmonic and quasi-harmonic corrected thermochemical data (in Hartree). The corrected enthalpy and entropy values are always less than or equal to the harmonic value.

##### Example 2: Quasi-harmonic thermochemistry with a larger basis set single point energy correction link job
```bash
goodvibes examples/ethane_spc.out --spc link

   Structure                E_SPC             E        ZPE         H_SPC        T.S     T.qh-S      G(T)_SPC   qh-G(T)_SPC
   ***********************************************************************************************************************
o  ethane_spc          -79.858399    -79.830421   0.073508    -79.779414   0.027540   0.027542    -79.806954    -79.806956
   ***********************************************************************************************************************
```

This calculation contains a multi-step job: an optimization and frequency calculation with a small basis set followed by (--Link1--) a larger basis set single point energy. Note the use of the `--spc link` option. The standard harmonic and quasi-harmonic corrected thermochemical data are obtained from the small basis set partition function combined with the larger basis set single point electronic energy. In this example, GoodVibes automatically recognizes the level of theory used in the frequency calculation, B3LYP/6-31G(d), and applies the [Truhlar group scaling factors](https://t1.chem.umn.edu/freqscale/index.html) of 0.991 (harmonic, used for H<sub>vib</sub> and S<sub>vib</sub>) and 0.977 (ZPE). Use `-v 1.0` to suppress all scaling, or `--zpe-vscal Y` to override only the ZPE factor.

Alternatively, if a single point energy calculation has been performed separately, provided both file names share a common root e.g. `ethane.out` and `ethane_TZ.out` then use of the `--spc TZ` option is appropriate. This will give identical results as above.

```bash
goodvibes examples/ethane.out --spc TZ

   Structure                E_SPC             E        ZPE         H_SPC        T.S     T.qh-S      G(T)_SPC   qh-G(T)_SPC
   ***********************************************************************************************************************
o  ethane              -79.858399    -79.830421   0.073508    -79.779414   0.027540   0.027542    -79.806954    -79.806956
   ***********************************************************************************************************************
```


##### Example 3: Changing the temperature (from standard 298.15 K to 1000 K) and concentration (from standard state in gas phase, 1 atm, to standard state in solution, 1 mol/l)
```bash
goodvibes examples/methylaniline.out --temp 1000 --conc 1.0

   Structure                    E        ZPE             H        T.S     T.qh-S          G(T)       qh-G(T)
   *********************************************************************************************************
o  methylaniline      -326.664901   0.142118   -326.452307   0.218212   0.216559   -326.670519   -326.668866
   *********************************************************************************************************
```

This correction from 1 atm to 1 mol/l is responsible for the addition 1.89 kcal/mol to the Gibbs energy of each species (at 298K). It affects the translational entropy, which is the only component of the molecular partition function to show concentration dependence. In the example above the correction is larger due to the increase in temperature.

##### Example 4: Analyzing the Gibbs energy across an interval of temperatures 300-1000 K with a stepsize of 100 K, applying a (Truhlar type) cut-off of 100 cm<sup>-1</sup>
```bash
goodvibes examples/methylaniline.out --ti '300,1000,100' --qs truhlar -f 120

   Structure               Temp/K                        H        T.S     T.qh-S          G(T)       qh-G(T)
   ******************************************************************************************************
o  methylaniline            300.0              -326.514399   0.040005   0.039842   -326.554404   -326.554241
o  methylaniline            400.0              -326.508735   0.059816   0.059596   -326.568551   -326.568331
o  methylaniline            500.0              -326.501670   0.082625   0.082349   -326.584296   -326.584020
o  methylaniline            600.0              -326.493429   0.108148   0.107816   -326.601577   -326.601245
o  methylaniline            700.0              -326.484222   0.136095   0.135707   -326.620317   -326.619930
o  methylaniline            800.0              -326.474218   0.166216   0.165772   -326.640434   -326.639990
o  methylaniline            900.0              -326.463545   0.198300   0.197800   -326.661845   -326.661346
o  methylaniline           1000.0              -326.452307   0.232169   0.231614   -326.684476   -326.683921
   ******************************************************************************************************
```

Note that the energy and ZPE are not printed in this instance since they are temperature-independent. The Truhlar-type quasi-harmonic correction sets all frequencies below 120 cm<sup>-1</sup> to a value of 100. Constant pressure is assumed, so that the concentration is recomputed at each temperature.

##### Example 5: Analyzing the Gibbs Energy using scaled vibrational frequencies
```bash
goodvibes examples/methylaniline.out -v 0.95

   Structure                    E        ZPE             H        T.S     T.qh-S          G(T)       qh-G(T)
   *********************************************************************************************************
o  methylaniline      -326.664901   0.135012   -326.521265   0.040238   0.040091   -326.561503   -326.561356
   *********************************************************************************************************
```

The frequencies are scaled by a factor of 0.95 before they are used in the computation of the vibrational energies (including ZPE) and entropies.

##### Example 6: Writing Cartesian coordinates
```bash
goodvibes examples/HCN*.out --xyz
```

Optimized cartesian-coordinates found in files HCN_singlet.out and HCN_triplet.out are written to GoodVibes_output.xyz

##### Example 7: Analyzing multiple files at once
```bash
goodvibes examples/*.out --cpu

   Structure                    E        ZPE             H        T.S     T.qh-S          G(T)       qh-G(T)
   *********************************************************************************************************
o  Al_298K            -242.328708   0.000000   -242.326347   0.017670   0.017670   -242.344018   -242.344018
o  Al_400K            -242.328708   0.000000   -242.326347   0.017670   0.017670   -242.344018   -242.344018
o  H2O                 -76.368128   0.020772    -76.343577   0.021458   0.021458    -76.365035    -76.365035
o  HCN_singlet         -93.358851   0.015978    -93.339373   0.022896   0.022896    -93.362269    -93.362269
o  HCN_triplet         -93.153787   0.012567    -93.137780   0.024070   0.024070    -93.161850    -93.161850
o  allene             -116.569605   0.053913   -116.510916   0.027618   0.027621   -116.538534   -116.538537
o  benzene            -232.227201   0.101377   -232.120521   0.032742   0.032745   -232.153263   -232.153265
o  ethane              -79.830421   0.075238    -79.750770   0.027523   0.027525    -79.778293    -79.778295
o  isobutane          -158.458811   0.132380   -158.319804   0.034241   0.034252   -158.354046   -158.354056
o  methylaniline      -326.664901   0.142118   -326.514489   0.039668   0.039535   -326.554157   -326.554024
o  neopentane         -197.772980   0.160311   -197.604824   0.036952   0.036966   -197.641776   -197.641791
   *********************************************************************************************************
TOTAL CPU      0 days  2 hrs 29 mins 28 secs

```
Wildcard characters (`*`) can be used to specify all output files in a directory.

##### Example 8: Entropic Symmetry Correction
```bash
goodvibes examples/allene.out examples/benzene.out examples/ethane.out examples/isobutane.out examples/neopentane.out --symm

   Structure                    E        ZPE             H        T.S     T.qh-S          G(T)       qh-G(T)  Point Group
   **********************************************************************************************************************
o  allene             -116.569605   0.053913   -116.510916   0.026309   0.026312   -116.537225   -116.537228          D2d
o  benzene            -232.227201   0.101377   -232.120521   0.030396   0.030399   -232.150917   -232.150919          D6h
o  ethane              -79.830421   0.075238    -79.750770   0.025831   0.025833    -79.776601    -79.776603          D3d
o  isobutane          -158.458811   0.132380   -158.319804   0.033204   0.033214   -158.353008   -158.353019          C3v
o  neopentane         -197.772980   0.160311   -197.604824   0.034606   0.034620   -197.639430   -197.639444           Td
   *********************************************************************************************************************************************
```

##### Example 9: Potential Energy Surface (PES) Comparison with Accessible Conformer Correction
```bash
goodvibes examples/gconf_ee_boltz/*.log --pes examples/gconf_ee_boltz/gconf_aminox_cat.yaml

   Structure                       E        ZPE             H        T.S     T.qh-S          G(T)       qh-G(T)
   ************************************************************************************************************
o  Aminoxylation_TS1_R   -879.405138   0.295352   -879.091374   0.063746   0.061481   -879.155120   -879.152855
o  Aminoxylation_TS2_S   -879.404445   0.295301   -879.090562   0.064366   0.061891   -879.154928   -879.152453
o  aminox_cat_conf212_S  -517.875165   0.200338   -517.662195   0.051817   0.049814   -517.714012   -517.712009
o  aminox_cat_conf280_R  -517.877308   0.200869   -517.664171   0.049996   0.048777   -517.714167   -517.712948
o  aminox_cat_conf65_S   -517.877161   0.200789   -517.664159   0.049790   0.048656   -517.713949   -517.712815
o  aminox_subs_conf713   -361.535757   0.095336   -361.433167   0.037824   0.037696   -361.470991   -361.470863
   ************************************************************************************************************

   Gconf correction requested to be applied to below relative values using quasi-harmonic Boltzmann factors

   RXN: Reaction (kcal/mol)       DE       DZPE            DH       T.DS    T.qh-DS         DG(T)      qh-DG(T)
   ************************************************************************************************************
o  Cat+Subs                     0.00       0.00          0.00       0.00       0.00          0.00          0.00
o  TS                           4.72      -0.46          3.53     -15.85     -16.37         19.39         19.90
   ************************************************************************************************************
```

##### Example 10: Stereoselectivity and Boltzmann populations

```bash
goodvibes examples/gconf_ee_boltz/Aminoxylation_TS1_R.log examples/gconf_ee_boltz/Aminoxylation_TS2_S.log \
          --boltz --label R='*_R*' --label S='*_S*'

   Structure                       E        ZPE             H        T.S     T.qh-S          G(T)       qh-G(T)  Boltz
   *******************************************************************************************************************
o  Aminoxylation_TS1_R   -879.405138   0.295352   -879.091374   0.063746   0.061481   -879.155120   -879.152855  0.605
o  Aminoxylation_TS2_S   -879.404445   0.295301   -879.090562   0.064366   0.061891   -879.154928   -879.152453  0.395
   *******************************************************************************************************************

   Selectivity, Boltzmann-averaged (gibbs, T = 298.15 K)
       Species   Files   Population (%)   ΔΔG (kcal/mol)
   ★   R             1            60.50            0.000
       S             1            39.50            0.249

   Ratio R:S = 60:40   Major: R   excess = 20.98%   ΔΔG = 0.25 kcal/mol
```

`--label NAME=PATTERN` is repeatable and supports any number of buckets (N=2 reports `excess` + `ΔΔG`; N>2 reports the ratio only). The legacy `--ee "*_R*:*_S*"` flag still works with a `DeprecationWarning` and will be removed in a future release.

#### CLI Reference

Run `goodvibes -h` for the full list of options. Key flags:

| Flag | Description | Default |
|------|-------------|---------|
| `-q` | Apply quasi-harmonic corrections (Grimme entropy + Head-Gordon enthalpy) | off |
| `-f FREQ` | Frequency cut-off for entropy and enthalpy (cm-1) | 100 |
| `--fs FREQ` | Frequency cut-off for entropy only (cm-1) | 100 |
| `--fh FREQ` | Frequency cut-off for enthalpy only (cm-1) | 100 |
| `--qs {grimme,truhlar}` | Quasi-harmonic entropy method | grimme |
| `--qh` | Apply Head-Gordon enthalpy correction only | off |
| `--temp TEMP` | Temperature in Kelvin | 298.15 |
| `--ti START,END,STEP` | Temperature interval scan | -- |
| `--conc CONC` | Concentration in mol/L (solution-phase entropy) | gas phase |
| `-v SCALE` | Vibrational frequency scaling factor (partition-function H/S) | auto |
| `--zpe-vscal SCALE` | Separate scaling factor for ZPE (Truhlar `zpe_fac`) | auto |
| `--spc SUFFIX` | Single-point energy correction (suffix or `link`) | -- |
| `--jobs N` | Parse and compute in parallel across N worker processes (`0` = all cores) | 1 |
| `--symm` | Apply symmetry correction to entropy (pymsym) | off |
| `--pg` | Display detected point group (no entropy correction applied) | off |
| `--boltz` | Print Boltzmann-weighted populations | off |
| `--label NAME=PATTERN` | N-way selectivity bucket (repeatable, fnmatch on basenames) | -- |
| `--selectivity FILE.yaml` | Selectivity spec via YAML (alternative to `--label`) | -- |
| `--ee PATTERNS` | (deprecated) Two-species selectivity, e.g. `"*_R*:*_S*"` — use `--label` | -- |
| `--pes FILE` | YAML-defined reaction pathway analysis (legacy + true YAML auto-detected) | -- |
| `--lowest-only` | PES tables: use only each species' lowest qh-G conformer | off |
| `--json PATH` | Write structured results (schema v1.0: qcdata + thermo + selectivity + pes blocks) | -- |
| `--export PATH` | Synonym for `--json`; same v1.0 payload, intended for `--import` re-use | -- |
| `--import PATH` | Read pre-parsed data from a v1.0 JSON file (or legacy cache envelope) instead of re-parsing | -- |
| `--csv PATH` | Write per-file thermochemistry to a CSV file (requires pandas; `goodvibes[full]`) | -- |
| `--parquet PATH` | Write per-file thermochemistry to a Parquet file (requires pyarrow; `goodvibes[full]`) | -- |
| `--strip-plot PATH` | Save a per-species ΔG strip plot (requires `--label`/`--selectivity`; `goodvibes[plot]`) | -- |
| `--pes-plot PATH` | Save a reaction-profile plot (requires `--pes`; `goodvibes[plot]`) | -- |
| `--media SOLVENT` | Solvent standard-state concentration correction | -- |
| `--freespace SOLVENT` | Free-space correction for solvent cavity | -- |
| `--invert [THRESH]` | Invert small imaginary frequencies to positive values | off |
| `--bav {global,conf}` | Moment of inertia for free-rotor entropy | global |
| `--sort [energy\|gibbs]` | Sort output by energy | -- |
| `--dedup` | Remove duplicate structures | off |
| `--dp N` | Decimal places for energy output | 6 |
| `--cache-save FILE` | (deprecated) alias for `--export`; still accepted, prefer `--export` | -- |
| `--cache-read FILE` | (deprecated) alias for `--import`; still accepted, prefer `--import` | -- |
| `--custom_ext EXTS` | Additional file extensions (comma-separated) | -- |
| `--exclude PATTERN` | Glob pattern to exclude files | -- |
| `--check` | Verify consistency across input files | off |
| `--cpu` | Print total CPU time | off |
| `--xyz` | Write Cartesian coordinates to .xyz file | off |
| `--imag` | Print imaginary frequencies | off |
| `--output NAME` | Output file base name | output |
| `--vmm SCALE` | Frequency scaling factor for ONIOM MM region | -- |
| `--nogconf` | Disable Gconf correction in PES analysis | off |
| `--graph FILE` | Graph a reaction profile from free energies | -- |

#### Dependencies

- **Python** >= 3.9
- **numpy** -- numerical computations
- **pymsym** -- point group detection and symmetry numbers
- **rich** >= 13 -- console table rendering

Optional:

- **ase** >= 3.22 (`goodvibes[ase]`) -- only needed to parse `.extxyz` inputs
- **pyyaml** -- needed at runtime when reading new-style PES YAML or `--selectivity FILE.yaml`; included in the `test` extra

Build requires setuptools >= 64. See `pyproject.toml` for details.

#### Contributing

Install for development:

```bash
pip install -e .
```

Run the test suite:

```bash
pytest -v
```

Test data is organized by program:
- `tests/g16/` -- Gaussian 16 output files
- `tests/orca6/` -- ORCA 6 output files (full coverage)
- `tests/orca5/` -- ORCA 5 output files (lightweight regression layer)
- `tests/qchem6/` -- QChem 6 output files
- `tests/xtb/` -- xTB output files
- `tests/ase/` -- ASE (extended XYZ) test files

Test helpers in `tests/conftest.py` provide path resolvers (`g16path()`, `orca_path()`, `orca5_path()`, `qchem_path()`, `xtb_path()`, `ase_path()`) and categorized file lists (`G16_FREQ_FILES`, `ORCA_FREQ_FILES`, `QCHEM_FREQ_FILES`, `XTB_FREQ_FILES`, `ASE_FREQ_FILES`, etc.) used by parametrized tests.

#### Citing GoodVibes
Luchini, G.; Alegre-Requena, J. V.; Funes-Ardoiz, I.; Paton, R. S. GoodVibes: Automated Thermochemistry for Heterogeneous Computational Chemistry Data. *F1000Research*, **2020**, *9*, 291 [**DOI:** 10.12688/f1000research.22758.1](https://doi.org/10.12688/f1000research.22758.1)

#### References
1. Ribeiro, R. F.; Marenich, A. V.; Cramer, C. J.; Truhlar, D. G. *J. Phys. Chem. B* **2011**, *115*, 14556-14562 [**DOI:** 10.1021/jp205508z](http://dx.doi.org/10.1021/jp205508z)
2. Grimme, S. *Chem. Eur. J.* **2012**, *18*, 9955-9964 [**DOI:** 10.1002/chem.201200497](http://dx.doi.org/10.1002/chem.201200497)
3. Li, Y.; Gomes, J.; Sharada, S. M.; Bell, A. T.; Head-Gordon, M. *J. Phys. Chem. C* **2015**, *119*, 1840-1850 [**DOI:** 10.1021/jp509921r](http://dx.doi.org/10.1021/jp509921r)
4. Alecu, I. M.; Zheng, J.; Zhao, Y.; Truhlar, D. G.; *J. Chem. Theory Comput.* **2010**, *6*, 2872-2887 [**DOI:** 10.1021/ct100326h](http://dx.doi.org/10.1021/ct100326h)
5. Mammen, M.; Shakhnovich, E. I.; Deutch, J. M.; Whitesides, G. M. *J. Org. Chem.* **1998**, *63*, 3821-3830 [**DOI:** 10.1021/jo970944f](http://dx.doi.org/10.1021/jo970944f)
6. Pracht, P.; Grimme, S. *Chem. Sci.* **2021**, *12*, 6551-6568 [**DOI:** 10.1039/D1SC00621E](https://doi.org/10.1039/D1SC00621E)

---
#### License

GoodVibes is freely available under an [MIT](https://opensource.org/licenses/MIT) License
