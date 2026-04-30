# GoodVibes ASE thermo extxyz format

ASE is a Python framework, not a quantum-chemistry program — it has no
canonical "thermo output file." When a user drives a calculation through
ASE (with ASE's own calculators or a wrapped QM code), GoodVibes consumes
the result via a small **Extended XYZ (`.extxyz`)** schema that bundles
electronic energy, frequencies, geometry, and metadata in one file.

The format is standard extxyz: line 1 is the atom count, line 2 is a
`key=value` comment line (ASE's `Atoms.info` round-trips through it via
`ase.io.read`/`write`), and the remaining lines are the atom block per the
`Properties=` schema.

## Example

```
3
Properties=species:S:1:pos:R:3 program=ase ase_version=3.22.1 \
  level_of_theory="B3LYP/6-31G*" \
  scf_energy=-76.40123456 scf_energy_units=Hartree \
  charge=0 multiplicity=1 \
  frequencies="1655.4 3826.7 3935.6" frequencies_units=cm-1 \
  zpe=0.021 point_group=C2v symmno=2 linear_mol=F \
  solvation_model="gas phase" empirical_dispersion="" job_type=Freq
O 0.000000  0.000000  0.117790
H 0.000000  0.755453 -0.471161
H 0.000000 -0.755453 -0.471161
```

## Required keys

- `program=ase` — drives auto-detection in `parse_qcdata`.
- `scf_energy` — electronic energy. Hartree by default; set
  `scf_energy_units=eV` (or `kcal/mol` / `kJ/mol`) to use other units.
- The atom block (3 columns: element + Cartesian xyz, in Ångström).

## Optional keys

| Key | Meaning |
|-----|---------|
| `frequencies` | Vibrational modes in cm⁻¹, space-separated, **negatives are imaginary** (split by sign into `frequency_wn` / `im_frequency_wn` by the parser). |
| `frequencies_units` | Always `cm-1` (only one unit currently supported). |
| `level_of_theory` | Free-form, e.g. `"B3LYP/6-31G*"`. Used for the scaling-factor lookup; omit to fall back to `--freq_scale_factor`. |
| `charge`, `multiplicity` | Default `0` / `1`. |
| `point_group`, `symmno` | Skip if you want pymsym (via `calc_bbe.ex_sym()`) to detect them. |
| `linear_mol` | `T` / `F`. Inferred from inertia eigenvalues if omitted. |
| `molecular_mass`, `roconst_cm`, `rotemp` | Override the geometry-derived defaults. |
| `zpe` | Zero-point energy in Hartree. Recomputed from frequencies at thermo time if omitted. |
| `solvation_model`, `empirical_dispersion` | Free-form metadata. |
| `applied_freq_scale_factor` | If the source program already scaled the frequencies, GoodVibes un-scales before re-applying its own factor. |
| `job_type` | `Freq`, `GSFreq`, `TS`, `SP`. Inferred from frequency presence/sign if omitted. |

## Writing a fixture from Python

The optional `goodvibes.ase_helper.write_thermo_extxyz` helper takes care of
everything; it requires `ase` to be installed (`pip install goodvibes[ase]`).

```python
from ase import Atoms
from goodvibes.ase_helper import write_thermo_extxyz

atoms = Atoms(symbols=['O', 'H', 'H'], positions=[...])
write_thermo_extxyz(
    'water.extxyz',
    atoms,
    energy=-76.40,           # Hartree by default
    frequencies=[1655.4, 3826.7, 3935.6],
    charge=0, multiplicity=1,
    level_of_theory='B3LYP/6-31G*',
)
```

## Regenerating these fixtures

The `.extxyz` files in this directory were generated from the matching
Gaussian logs in `tests/g16/` so that `tests/test_thermo_ase.py` can
cross-validate (same physical inputs → same H/S/G regardless of source
format). To regenerate:

```bash
python tests/ase/_generate.py
```

The output is deterministic — `git diff tests/ase/` is the regression
check.
