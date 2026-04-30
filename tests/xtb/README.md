# xtb Example Output Files

## Overview

42 example xtb output files covering common molecules, charge/spin states,
solvation models, and edge cases. xtb takes a `.xyz` coordinate file plus
command-line flags rather than an input deck, so each entry has a paired
`*.xyz` and `*.out`. All runs use xtb v6.7.0.

These outputs are parsed by `goodvibes.io.parse_xtb_thermo` (auto-detected
through `parse_qcdata`). For runs without `--opt`/`--ohess`, Cartesian
coordinates fall back to the paired `.xyz` file.

Files 01–41 are GFN-xTB calculations (almost all GFN2-xTB; #32 uses GFN1-xTB
to exercise alternate Hamiltonian parsing). File 42 is an empty `.xyz` that
xtb aborts on — used for edge-case parsing tests.

Job-type flags:

- `--ohess` — optimize then analytical Hessian (the default for full thermo)
- `--hess` — Hessian only (single-geometry frequency analysis)
- *(no flag)* — single-point SCC energy only
- `-c <n>` / `-u <n>` — net charge / number of unpaired electrons (Nα − Nβ)
- `--gfn <0|1|2>` — Hamiltonian level (default GFN2)
- `--cpcmx <solvent>` — CPCM-X implicit solvation
- `-g <solvent>` — GBSA implicit solvation

---

## File Index

| #  | File | Hamiltonian | Job | Key Feature |
|----|------|-------------|-----|-------------|
| 01 | `01_water.out` | GFN2-xTB | Opt+Hess | Symmetric (C2v), reference thermochemistry |
| 02 | `02_ethane.out` | GFN2-xTB | Opt+Hess | D3d staggered, internal-rotation low mode |
| 03 | `03_acetone.out` | GFN2-xTB | Opt+Hess | C2v carbonyl |
| 04 | `04_benzene_radical_cation.out` | GFN2-xTB | Opt+Hess | Charge=+1, Mult=2 (`-c 1 -u 1`) |
| 05 | `05_methylene_triplet_carbene.out` | GFN2-xTB | Opt+Hess | Mult=3 (`-u 2`) |
| 06 | `06_carbon_atom_single_point.out` | GFN2-xTB | SP | Single atom, no Hessian (0 modes) |
| 07 | `07_neon_atom_with_freq.out` | GFN2-xTB | Opt+Hess | Single atom with `--ohess` (0 modes) |
| 08 | `08_alanine.out` | GFN2-xTB | Opt+Hess | Non-symmetric C1 amino acid |
| 09 | `09_caffeine.out` | GFN2-xTB | Opt+Hess | Larger drug-like molecule (24 atoms) |
| 10 | `10_formaldehyde.out` | GFN2-xTB | Opt+Hess | C2v, four atoms |
| 11 | `11_hf_molecule.out` | GFN2-xTB | Opt+Hess | Diatomic; output contains UTF-8 box-drawing characters |
| 12 | `12_water_dimer.out` | GFN2-xTB | Opt+Hess | Hydrogen-bonded complex, low-frequency intermolecular modes |
| 13 | `13_methanol.out` | GFN2-xTB | Opt+Hess | OH torsion low mode |
| 14 | `14_o2_superoxide_anion.out` | GFN2-xTB | Opt+Hess | Charge=-1, Mult=2 (`-c -1 -u 1`), linear |
| 15 | `15_iron_complex_quintet.out` | GFN2-xTB | Opt+Hess | Transition metal, Mult=5 (`-u 4`) |
| 16 | `16_propane.out` | GFN2-xTB | Opt+Hess | `-ohess` (single-dash form, parser robustness) |
| 17 | `17_acetic_acid_dmso.out` | GFN2-xTB | SP+solv | CPCM-X DMSO (`--cpcmx dimethylsulfoxide`) |
| 18 | `18_benzene_singlepoint.out` | GFN2-xTB | SP | D6h, no Hessian |
| 19 | `19_naphthalene.out` | GFN2-xTB | Opt+Hess | D2h aromatic |
| 20 | `20_hcn_linear.out` | GFN2-xTB | Opt+Hess | Linear, C∞v |
| 21 | `21_cs2_linear.out` | GFN2-xTB | Opt+Hess | Linear, D∞h |
| 22 | `22_iodobenzene.out` | GFN2-xTB | Opt+Hess | Heavy halogen (I) |
| 23 | `23_pd_complex.out` | GFN2-xTB | Opt+Hess | 4d transition metal phosphine complex |
| 24 | `24_pt_complex.out` | GFN2-xTB | Opt+Hess | 5d transition metal complex |
| 25 | `25_ethane.out` | GFN2-xTB | Opt+Hess | Duplicate of #02 (different starting geometry, sanity check) |
| 26 | `26_pyridine_acetonitrile.out` | GFN2-xTB | Opt+Hess+solv | GBSA acetonitrile (`-g acetonitrile`) |
| 27 | `27_aniline_chloroform.out` | GFN2-xTB | Opt+Hess+solv | GBSA chloroform |
| 28 | `28_phenol_thf.out` | GFN2-xTB | Opt+Hess+solv | GBSA THF |
| 29 | `29_methylammonium_water.out` | GFN2-xTB | SP+solv | GBSA water, no Hessian |
| 30 | `30_cyclohexane.out` | GFN2-xTB | Opt+Hess | D3d chair, ring-puckering low modes |
| 31 | `31_methanol.out` | GFN2-xTB | Opt+Hess | Duplicate of #13 (different starting geometry, sanity check) |
| 32 | `32_butadiene.out` | **GFN1-xTB** | Opt+Hess | Alternate Hamiltonian (`--gfn 1`) |
| 33 | `33_furan.out` | GFN2-xTB | Opt+Hess | C2v aromatic heterocycle |
| 34 | `34_imidazole.out` | GFN2-xTB | Opt+Hess | Aromatic heterocycle |
| 35 | `35_planar_cyclohexane_3rd_order_saddle.out` | GFN2-xTB | Hess | **3rd-order saddle** (3 imaginary modes), `--hess` only |
| 36 | `36_naphthalene.out` | GFN2-xTB | Opt+Hess | Larger aromatic system |
| 37 | `37_oxazole_dcm.out` | GFN2-xTB | Opt+Hess+solv | GBSA dichloromethane (`-g CH2Cl2`) |
| 38 | `38_n2o_linear.out` | GFN2-xTB | Opt+Hess | Linear, C∞v (xyz filename retains `_highT` suffix) |
| 39 | `39_thiophene.out` | GFN2-xTB | Opt+Hess | C2v sulfur heterocycle |
| 40 | `40_dmso.out` | GFN2-xTB | Opt+Hess | Cs sulfoxide |
| 41 | `41_dmabn.out` | GFN2-xTB | Opt+Hess | Donor–acceptor aromatic (4-(dimethylamino)benzonitrile) |
| 42 | `42_empty.out` | — | — | Empty `.xyz` triggers a fatal xtb read error; edge-case parsing test |

---

## Notes

- **Solvation coverage.** CPCM-X (#17) and GBSA (#26–29, #37). Only #17
  exercises CPCM-X; the rest use GBSA. #29 has no Hessian (SP+solv only).
- **Charge / multiplicity coverage.** Cation (#04), anion (#14), triplet
  (#05), quintet (#15).
- **Linear molecules.** #14 (O2−), #20 (HCN), #21 (CS2), #38 (N2O).
- **Single-point only (no Hessian).** #06, #18, #29.
- **Saddle points / imaginary frequencies.** #35 has 3 imaginary modes
  (planar cyclohexane 3rd-order saddle).
- **Edge cases.** #11 has UTF-8 characters that trip plain `grep`; #42 is
  the abort-on-empty-input case.
