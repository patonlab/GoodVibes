# Q-Chem 6 test inputs

These `.qcin` files mirror the Gaussian fixtures in `../g16/` so the same
molecule + level-of-theory combinations can be run through Q-Chem 6 and
compared. Filenames are kept identical (only the extension differs) so each
Q-Chem input pairs with its g16 counterpart by name:

```
g16/01a_water_hf_freq.com   ↔   qchem6/01a_water_hf_freq.qcin
```

## How to run

Default: `qchem 01a_water_hf_freq.qcin 01a_water_hf_freq.out`. Multi-job
inputs (those that contain `@@@`) chain an Opt followed by a Freq in a
single submission — Q-Chem's standard pattern for opt+freq jobs. Memory and
parallelism are intentionally **not** set in the input; configure them at
submission time (`qchem -nt 16 …`, or via `QC_PARALLEL_NPROC`).

## Conventions

| Gaussian feature | Q-Chem encoding |
|------------------|-----------------|
| `Opt Freq` (single job) | `JOBTYPE = opt` then `@@@` then `JOBTYPE = freq` (geometry read) |
| `Opt=(TS,CalcFC,…)` | `JOBTYPE = ts` (Q-Chem computes initial Hessian by default) |
| `Freq=(Anharmonic)` | `ANHAR = true` in `$rem` |
| `Freq=(Temperature=…,Pressure=…)` | Run at default T/P; pass `--temperature` and `--conc` to GoodVibes |
| `SCRF=(CPCM,Solvent=X)` | `SOLVENT_METHOD = pcm` + `$pcm` (THEORY = cpcm) + `$solvent` (Dielectric) |
| `SCRF=(SMD,Solvent=X)` | `SOLVENT_METHOD = SMD` + `$smx` (solvent name) |
| `EmpiricalDispersion=GD3BJ` | `DFT_D = d3_bj` |
| `EmpiricalDispersion=GD3` | `DFT_D = d3_zero` |
| `Counterpoise=2` | `JOBTYPE = bsse` + `--` fragment separators in `$molecule` |
| `NMR=GIAO` | `JOBTYPE = nmr` |
| `td=(nstates=N,root=K)` | `CIS_N_ROOTS = N`, `CIS_STATE_DERIV = K` |
| `Gen` mixed basis + `ECP` | `BASIS = gen`, `ECP = gen`, `$basis` and `$ecp` blocks |
| Multi-step `--Link1--` | `@@@` separator with `READ` for the molecule block |

## Skipped fixtures

Two g16 fixtures have no clean Q-Chem 6 equivalent:

- `15_methanol_oniom_qmmm.com` — ONIOM (B3LYP:PM6) two-layer QM/MM. Q-Chem
  uses a different QM/MM machinery (PEqM / ChemShell) that doesn't read
  layer tags from a route line.
- `21_naphthalene_pm7_semiempirical.com` — Q-Chem doesn't ship PM7. The
  closest analogues (PM3, AM1) are not direct surrogates for thermochemistry
  benchmarking.

## Caveats

- These inputs were translated from Gaussian route lines; please run a few
  on your Q-Chem 6 install to spot-check. Report any syntax issues.
- `THERMO_TEMP` / `THERMO_PRES` are not set on opt+freq jobs that used a
  non-default T/P in g16 (`02_ethane`, `40_n2o`). Use GoodVibes
  `--temperature` / `--conc` to apply those at thermo time.
- `Freq=NoRaman` has no behavioural counterpart in Q-Chem; Q-Chem doesn't
  compute Raman intensities by default.
- `12_water_anharmonic_vpt2` and `23_cs2_linear_anharmonic` use `ANHAR = true`;
  consult the Q-Chem 6 manual for additional VPT2-specific REM variables
  (e.g. `VPT2_*`) if your install needs them.
- Error fixtures `51-60` are recreated to trigger Q-Chem-equivalent failure
  modes. The exact error message will differ from Gaussian.
