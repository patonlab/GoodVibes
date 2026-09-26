# GoodVibes roadmap

GoodVibes turns quantum-chemistry and MLIP frequency calculations into
quasi-harmonic thermochemistry. Its next job is to own two things:

1. **An open, versioned reaction-profile record**,
   [`reaction-profile/1.0`](docs/source/reaction_profile.md): a document a
   chemist can write by hand and the one GoodVibes deposits with every figure.
2. **The default publication figure and table** drawn from that record.

Thermochemistry from Gaussian, ORCA, xTB, Q-Chem, NWChem or an MLIP through
ASE is one way to fill the record; a table typed in from a paper's SI is
another. Post-hoc qRRHO on program outputs is no longer unique to GoodVibes
(ORCA 6.1, Shermo and pymatgen ship it). A reproducible profile record that
other tools read and write is a position nobody holds.

What has shipped is listed in [CHANGELOG.md](CHANGELOG.md). The previous
roadmap (v4.1 to v5.0 items 1 to 17, sub-plans A and B) is kept in git
history:
[ROADMAP.md at 407c14d](https://github.com/patonlab/GoodVibes/blob/407c14d/ROADMAP.md).

---

## Status

| Milestone | Release | State | Pull requests |
| --- | --- | --- | --- |
| **M0** Correctness and compatibility goldens | 5.0 | ✅ merged | #115, #116, #117 |
| **M1** The profile model and file-free MLIP input | 5.0 | ✅ merged | #118 |
| **M2a** The `reaction-profile/1.0` format, payload 1.1, `goodvibes-profile` | 5.0 | ✅ merged | #119 |
| **M2b** Figure polish and the gallery | 5.0 / 5.1 | 🟡 gallery done, rest open | |
| **M3** Methods, MLIP overlay, selectivity on the profile | 5.1 | open | |
| **M4** Adoption and polish | 5.2 | open | |
| **M5** Removals | 6.0 | open | |

Everything merged so far still carries the version string **4.4.0**. The
next release is **5.0**: it makes `docs/source/migration_v5.md` true and
ships M0 to M2a plus whatever of M2b is ready.

---

## Shipped

**M0: correctness and goldens.**
- `tests/compatibility/`: 29 CLI goldens (`.dat` and `--json`), under both
  `TERM=dumb` and a colour terminal.
- The quantity registry (`goodvibes/quantities.py`), `--pes-plot-quantity`
  (and its alias `--gtype`), and eV / hartree units.
- `--ti` selectivity recomputes G(T) at every temperature.
- The single-point downgrade is loud, with `--strict-spc` to make it fatal.
- Per-label dedup; the ASE / Q-Chem `TSFreq` job type; complex ASE frequencies.
- The `.dat` archive is free of terminal escape codes.
- `CHANGELOG.md`, `CITATION.cff` and `CONTRIBUTING.md`.

**M1: the profile model.**
- `ComputedEntry`: a structure plus its options, evaluable at any temperature.
- `ConformerSet` gains `from_results`, `populations`, `ensemble_free_energy`,
  `s_conf` and `dedup`.
- Point roles and display labels, pathway edges, and `Series` (computed or
  declared).
- `plot_profile`: merged x axis, overlays, panels, `ProfileAxes`.
- `--ti --pes` builds the model.
- `QCData.from_atoms` / `from_vibrations` and the mass table to Pu.
- Top-level exports.

**M2a: the format.**
- The JSON Schema (CC0) and its specification page.
- `goodvibes.profile`: `Profile`, `load_profile` and `validate_document`,
  with the conformance kit (`tests/profile_conformance/`).
- CSV/TSV tables of relative energies read as declared-only profiles.
- `--profile` and `--with-conformers`.
- Payload 1.1 with a `profile` block.
- `goodvibes-profile validate | plot | table | convert | evaluate`.

**M2b so far: the gallery.**
- [`goodvibes/examples/gallery`](goodvibes/examples/gallery) holds six figures
  rebuilt from compact committed inputs.
- `tests/test_gallery.py` rebuilds them and checks their numbers against the CLI.

---

## Next

### M2b: figure polish (5.0 or 5.1)

- [x] Gallery of reproducible examples (azabor DFT, temperature overlay,
      ΔE/ΔH/Δqh-G, R vs S panels, CSV table, computed with declared).
- [ ] Style presets (`single-column`, `double-column`, `slide`): the
      reserved `style.preset` key; figure size, fonts and line widths applied
      inside an `rc_context`, never set globally.
- [ ] SVG output with a `gid` on every element and the evaluated document in
      `<metadata>`, so a figure carries its own data and can be edited in
      Inkscape without redrawing.
- [ ] Uncertainty drawn as error bars. `series.uncertainty` is stored and
      tabulated but not drawn.
- [ ] Image baselines for the gallery (pytest-mpl or a tolerance-based
      comparison) so layout regressions fail CI.
- [ ] Cookbook opens with the CSV-to-figure recipe.
- [ ] Publish the schema at a stable URL (Read the Docs) with a Zenodo DOI
      per schema minor; the `$id` currently points at the raw file on
      `master`.

### M3: methods, MLIP overlay, selectivity on the profile (5.1)

- [ ] Per-method thermochemistry options. `goodvibes.sources` is already
      per method; a DFT and an MLIP method should also differ in scaling and
      qRRHO settings within one evaluation.
- [ ] `QCData.with_single_point(energy, units, method)` for DFT//MLIP
      composites without SPC files.
- [ ] Provenance on `ThermoResult`: temperature, options,
      `scale_factor_source` (`truhlar | user | mlip-unscaled | none-found`,
      the last a warning instead of a silent 1.0), `symmetry_source` and
      `n_imag`.
- [ ] Multi-frame `.xyz` / `.extxyz` reader yielding energy-only entries
      (CREST ensembles and MLIP sweeps alike).
- [ ] The reserved `selectivity:` block: competing points sharing a
      reference point, ΔG‡ and G_ensemble per branch, and `SelectivityResult`
      v2. It needs a Curtin–Hammett precondition warning, a documented ee
      sign and a `major` convention.
- [ ] `compute_selectivity_batch(jobs, temperatures)` returning a tidy
      DataFrame for prediction pipelines, plus cutoff and conformer-window
      sensitivity sweeps. This gives the "ee 92 % (88–94 % over cutoffs)"
      statement.
- [ ] `plot_boltzmann_histogram` and `plot_temperature_scan` over
      `ConformerSet` and `Series` (currently stubs).
- [ ] `goodvibes-profile diff` between two documents.

### M4: adoption and polish (5.2)

- [ ] Label de-overlap (value labels of close series still collide), y-axis
      break, optional RDKit depictions.
- [ ] Minimal `kinetics.py`: Eyring rate ratio, energy span, a step table and
      mikimo CSV export. No microkinetics.
- [ ] A per-structure SI table exporter (E, ZPE, H, T·S, qh-G, n_imag,
      lowest frequencies, scale factor, symmetry, xyz appendix).
- [ ] Outreach, sent as pull requests rather than waited for:
  - PESViewer writing the core format;
  - autodE exporting from `Reaction`;
  - a quacc `VibThermoSchema` importer here.
- [ ] Reproduce three published profiles from their SI tables as declared
      series for the gallery.
- [ ] Promote the schema from 1.0-draft to 1.0 after the first external
      round trip.

### M5: removals (6.0)

One release removes everything, with the goldens updated deliberately and a
changelog entry for each changed output.

Already deprecated, with notices in place that name 6.0:
- the legacy `--- # PES` text format (`pes_legacy.py`);
- `pes.py` (`get_pes`, `graph_reaction_profile`) and `--graph`;
- `--ee`, `--cache-save` and `--cache-read`;
- the 15-argument `calc_bbe` constructor.

Need a deprecation notice during 5.x first:
- the legacy `--ti` PES text path (the model already tabulates reaction-profile
  documents at every scan temperature);
- the `pes` payload block, in favour of `profile`;
- `--freespace`, if the maintainer agrees (proposed, not decided).

---

## Principles

- **Output compatibility.** The compatibility goldens pin the `.dat` and
  JSON output of the common flag combinations. Output changes only
  deliberately, and each change is listed under *Output changes* in the
  changelog.
- **One removal version.** Everything deprecated in 4.x is removed in
  **6.0**, and every message says so.
- **The format is small and additive.** The core is program-neutral;
  GoodVibes-specific recipe keys live under `goodvibes:`. Minors only add
  optional keys, keys reserved for a later minor are rejected rather than
  ignored, and `x-*` keys are free.
- **Declared values never mix with computed ones inside a point.** A literature value or
  a hand-typed number lives at the point level of a declared series and is
  never summed with absolute energies.
- **Examples without bloat.**
  - Commit the parsed record, never the program output: a reaction-profile
    document with `--with-conformers`, an `--export` payload, or `.extxyz`
    files.
  - Keep the script that made it next to it, and archive the raw outputs
    elsewhere (e.g. Zenodo).
  - Label illustrative values as illustrative.

  The azabor set is 380 KB this way instead of about 100 Gaussian outputs.
- **Docs.** Stay on Sphinx + MyST. Effort goes into the format page, the
  gallery and the cookbook.
- **CI.** Run on GitHub Actions: lint, Linux 3.9 to 3.13, Windows 3.12.
  - Raise the floor to 3.10 and add 3.14 at the 5.0 release.
  - CircleCI duplicates the Linux job and can be retired: delete
    `.circleci/config.yml`, a maintainer decision.

---

## Decided against

| Old roadmap item | Decision | Why |
| --- | --- | --- |
| CBS/Gn composite detection; Wigner tunnelling | Dropped | Niche; tunnelling belongs in kinetics tooling such as kinisot. |
| Ensemble container with lazy parsing and 10⁴-conformer streaming | Replaced | `ConformerSet` with recomputable entries is the species ensemble; no second container. |
| Hindered rotors (and glowfreq, issue #77) | Deferred indefinitely | Orthogonal science; MLIP Hessian noise makes torsion identification unreliable. |
| CENSO JSON import | Dropped | The multi-frame `.xyz` reader in M3 covers CREST ensembles. |
| Auto-SI generator with a methods paragraph and BibTeX | Replaced | The deposited profile document plus the per-structure SI table exporter in M4. |
| mkdocs migration | Dropped | Sphinx + MyST is enough; effort goes to content. |
| Performance targets (10³ files in 30 s, 10⁴ conformers in 200 MB) | Dropped | Not on the critical path; parsing is already parallel. |

Backlog, if asked for: heat capacities Cv / Cp (issue #66), a cheap
addition to the quantity registry.

---

## Risks

- **Scope.** M3 and M4 are gated on demand and on an external writer of the
  format appearing; M0 to M2a were the committed slice.
- **Schema churn.** Methods and selectivity arrive after 1.0. The core /
  namespace split, reserved keys and additive minors keep them 1.x. The
  format stays 1.0-draft until an external round trip.
- **Misleading overlays.** Mixing DFT qh-G, a literature ΔG at 1 atm and an
  MLIP ΔE on one axis invites bad figures. The legend names quantity and
  method, declared series are hollow, and M3 adds a warning when a series
  mixes scale-factor sources, SPC status or standard states.
- **MLIP frequency noise.** `from_vibrations` treats tiny imaginary modes as
  noise with a warning and checks the imaginary-mode count against the job
  type. M3 records `n_imag` on the result.
- **Standard-setting depends on others.** The format is small, the conformance
  kit and minimal page exist, and M4 sends the integration pull requests
  rather than waiting for them.
