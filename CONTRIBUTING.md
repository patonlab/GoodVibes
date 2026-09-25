# Contributing to GoodVibes

## Development install

```bash
git clone https://github.com/patonlab/GoodVibes
cd GoodVibes
pip install -e ".[test]"
pytest -q                 # about two minutes
ruff check goodvibes/ tests/
```

The test suite runs against real program outputs checked in under
`tests/<program>/`; see `tests/README.md` for the layout. Example data used
by the documentation lives under `goodvibes/examples/` and is not shipped
in the wheel.

## Output stability and goldens

Users parse GoodVibes' `.dat` archives and `--json` payloads, so their
layout is part of the interface. `tests/compatibility/` runs the common
flag combinations through the CLI and compares the output against
checked-in goldens. If your change legitimately alters an output:

1. regenerate the affected goldens with
   `GOODVIBES_UPDATE_GOLDENS=1 pytest tests/compatibility -q`,
2. look at `git diff tests/compatibility/goldens` and make sure only the
   intended lines changed,
3. describe the change under **Output changes** in `CHANGELOG.md`.

A golden that changes for a reason you did not intend is a bug in the
change, not in the golden.

## Changelog and versions

Every user-visible change gets a line in `CHANGELOG.md` under
`[Unreleased]` (Keep a Changelog format). Versions follow semantic
versioning: additive changes and bug fixes are minor/patch releases; the
next major (6.0) removes everything currently deprecated. Deprecated
surfaces emit a `DeprecationWarning` (API) or a `!` notice in the CLI
output, always naming the removal version. Do not introduce a new
removal version; use the next major.

## Adding an example set

Do not commit raw program outputs for new examples (the existing
`goodvibes/examples/pes` set already weighs 260 MB in history). Commit the
parsed record instead: the `--export` JSON payload for a set of
structures, or `.extxyz` files written with `goodvibes.ase_helper`. Keep
the script that generated them next to the data and archive the raw
outputs on Zenodo, cited from the example's README.

## Pull requests

Work on a branch, keep commits focused (one fix or feature per commit,
with its tests), and run the full suite before pushing. CI runs lint and
the suite on Linux (several Python versions) and Windows.
