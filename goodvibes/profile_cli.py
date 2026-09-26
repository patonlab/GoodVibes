"""``goodvibes-profile``: validate, plot, tabulate, convert and re-evaluate
reaction-profile documents. File-free: it reads profile documents (and
tables of relative energies), never quantum-chemistry output files.

    goodvibes-profile validate profile.yaml [--strict]
    goodvibes-profile plot profile.json -o profile.svg [-o profile.pdf] [--series G298,lit]
    goodvibes-profile table profile.json [-o table.csv|table.md] [--long]
    goodvibes-profile convert levels.csv -o profile.yaml --quantity gibbs --temperature 298.15
    goodvibes-profile evaluate profile.json -o hot.json --temperatures 298.15,373.15

A profile is a reaction-profile document (.yaml/.yml/.json), a GoodVibes
``--json`` payload with a ``profile`` block, a CSV/TSV table of relative
energies, a GoodVibes v2 PES YAML or a legacy ``--- # PES`` file (the last
two only for ``validate`` and ``convert``: they name output files and
carry no values).
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import yaml
from typing import List, Optional

from .constants import __version__


def _split(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    return items or None


def _temperatures(value: Optional[str]) -> Optional[List[float]]:
    items = _split(value)
    if items is None:
        return None
    try:
        temps = [float(v) for v in items]
    except ValueError:
        raise SystemExit(f"goodvibes-profile: error: --temperatures expects numbers, got {value!r}")
    if any(t <= 0 for t in temps):
        raise SystemExit("goodvibes-profile: error: temperatures must be positive")
    return temps


def _add_table_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("reading a CSV/TSV table of relative energies")
    g.add_argument("--quantity", default="gibbs", help="quantity of the table's values (default: gibbs)")
    g.add_argument("--temperature", type=float, default=298.15, help="temperature of the values, K (default 298.15)")
    g.add_argument("--units", default="kcal/mol", help="units of the values (default: kcal/mol)")
    g.add_argument("--table-layout", dest="table_layout", choices=("auto", "wide", "long"), default="auto",
                   help="table layout (default: auto)")
    g.add_argument("--method", default=None, help="method id to record for the table's series")
    g.add_argument("--doc-title", dest="table_title", default=None, help="title of the document made from the table")


def _load(path: str, args, *, require_values: bool = False):
    from .profile import load_profile
    table_options = {}
    if os.path.splitext(path)[1].lower() in (".csv", ".tsv"):
        table_options = dict(quantity=args.quantity, temperature=args.temperature, units=args.units,
                             layout=args.table_layout, method=args.method, title=args.table_title)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prof = load_profile(path, strict=getattr(args, "strict", False), **table_options)
    for w in prof.warnings:
        print(f"goodvibes-profile: warning: {path}: {w}", file=sys.stderr)
    if require_values and prof.upgraded_from is not None:
        raise SystemExit(f"goodvibes-profile: error: {path} is a GoodVibes PES file: it names output files but "
                         "carries no values. Evaluate it first: goodvibes *.log --pes "
                         f"{os.path.basename(path)} --profile profile.json")
    return prof


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_validate(args) -> int:
    import json
    from .profile import validate_document
    status = 0
    for path in args.files:
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in (".csv", ".tsv"):
                _load(path, args)
                print(f"{path}: valid table")
                continue
            text = open(path, encoding="utf-8").read()
            from .pes_loader import is_legacy_format
            if is_legacy_format(text):
                _load(path, args)
                print(f"{path}: valid (legacy '--- # PES' text; `goodvibes-profile convert` writes the explicit form)")
                continue
            if ext == ".json":
                data = json.loads(text)
            else:
                data = yaml.safe_load(text)
            if isinstance(data, dict) and "schema_version" in data and "profile" in data:
                data = data["profile"]
        except (OSError, ValueError, yaml.YAMLError) as exc:
            print(f"{path}: cannot read: {exc}", file=sys.stderr)
            status = 1
            continue
        errors, warns = validate_document(data, strict=args.strict)
        for w in warns:
            print(f"{path}: warning: {w}")
        if errors:
            status = 1
            print(f"{path}: INVALID ({len(errors)} error{'s' if len(errors) != 1 else ''})")
            for e in errors:
                print(f"  {e}")
        else:
            kind = "" if isinstance(data, dict) and "schema" in data else " (v2 PES YAML shorthand)"
            print(f"{path}: valid{kind}")
    return status


def cmd_plot(args) -> int:
    prof = _load(args.profile, args, require_values=True)
    temps = _temperatures(args.temperatures)
    if temps:
        prof = prof.evaluate(temperatures=temps)
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        raise SystemExit("goodvibes-profile: error: plotting needs matplotlib; "
                         "install with `pip install goodvibes[plot]`")
    fig = prof.plot(series=_split(args.series), pathways=_split(args.pathways), layout=args.layout,
                    label_points=True if args.label_points else None, connector=args.connector,
                    title=args.title, annotations=not args.no_annotations)
    fig.save(*args.output, dpi=args.dpi)
    fig.close()
    for out in args.output:
        print(f"wrote {out}")
    return 0


def cmd_table(args) -> int:
    prof = _load(args.profile, args, require_values=True)
    layout = "long" if args.long else "wide"
    if args.output:
        prof.write_table(args.output, layout=layout, decimals=args.decimals)
        print(f"wrote {args.output}")
        return 0
    rows = prof.to_rows(layout)
    if not rows:
        print("(no levels: evaluate the document first)")
        return 0
    decimals = args.decimals if args.decimals is not None else prof.style.get("decimals", 1)
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:                                     # pragma: no cover
        from .profile import _rows_to_markdown
        print(_rows_to_markdown(rows, decimals))
        return 0
    title = prof.title or "reaction profile"
    table = Table(title=f"{title}  ({prof.units})")
    cols = list(rows[0].keys())
    for c in cols:
        table.add_column(c, justify="left" if c in ("point", "role", "display", "pathway", "series",
                                                    "label", "quantity", "method", "units") else "right")

    def fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.{decimals}f}"
        return str(v)
    for r in rows:
        table.add_row(*(fmt(r.get(c)) for c in cols))
    # piped or redirected: don't squeeze the table into 80 columns
    Console(width=None if sys.stdout.isatty() else 200).print(table)
    return 0


def cmd_convert(args) -> int:
    prof = _load(args.input, args)
    prof.dump(args.output, include_conformers=not args.no_conformers)
    print(f"wrote {args.output}")
    return 0


def cmd_evaluate(args) -> int:
    prof = _load(args.profile, args, require_values=True)
    if not prof.namespace.get("conformers"):
        raise SystemExit("goodvibes-profile: error: the document has no embedded conformers; write it with "
                         "goodvibes ... --pes FILE --profile OUT --with-conformers")
    new = prof.evaluate(temperatures=_temperatures(args.temperatures),
                        invocation="goodvibes-profile " + " ".join(sys.argv[1:]))
    new.dump(args.output, include_conformers=not args.no_conformers)
    print(f"wrote {args.output}")
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="goodvibes-profile",
        description="Validate, plot, tabulate, convert and re-evaluate reaction-profile documents "
                    "(no quantum-chemistry output files needed).")
    parser.add_argument("--version", action="version", version=f"goodvibes-profile {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    p = sub.add_parser("validate", help="check documents against reaction-profile 1.0")
    p.add_argument("files", nargs="+", metavar="FILE")
    p.add_argument("--strict", action="store_true", help="unknown keys are errors")
    _add_table_options(p)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("plot", help="draw a profile (PNG/PDF/SVG by extension)")
    p.add_argument("profile", metavar="PROFILE")
    p.add_argument("-o", "--output", action="append", required=True, metavar="FILE",
                   help="output figure; repeat for several formats")
    p.add_argument("--series", default=None, help="comma-separated series ids (default: all)")
    p.add_argument("--pathways", default=None, help="comma-separated pathway names (default: all)")
    p.add_argument("--layout", choices=("overlay", "panels"), default=None)
    p.add_argument("--connector", choices=("bezier", "linear", "step"), default=None)
    p.add_argument("--label-points", action="store_true", help="print each level's value")
    p.add_argument("--no-annotations", action="store_true", help="omit the document's annotations")
    p.add_argument("--title", default=None)
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--temperatures", default=None, metavar="T1,T2",
                   help="re-evaluate at these temperatures first (needs embedded conformers)")
    _add_table_options(p)
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("table", help="print or write the levels (.csv, .md)")
    p.add_argument("profile", metavar="PROFILE")
    p.add_argument("-o", "--output", default=None, metavar="FILE")
    p.add_argument("--long", action="store_true", help="one row per pathway × point × series")
    p.add_argument("--decimals", type=int, default=None)
    _add_table_options(p)
    p.set_defaults(func=cmd_table)

    p = sub.add_parser("convert", help="write the explicit reaction-profile form (.yaml/.yml/.json)")
    p.add_argument("input", metavar="INPUT")
    p.add_argument("-o", "--output", required=True, metavar="FILE")
    p.add_argument("--no-conformers", action="store_true", help="drop embedded conformers")
    _add_table_options(p)
    p.set_defaults(func=cmd_convert)

    p = sub.add_parser("evaluate", help="re-evaluate a document from its embedded conformers")
    p.add_argument("profile", metavar="PROFILE")
    p.add_argument("-o", "--output", required=True, metavar="FILE")
    p.add_argument("--temperatures", default=None, metavar="T1,T2",
                   help="one series per temperature for each computed series")
    p.add_argument("--no-conformers", action="store_true", help="drop embedded conformers from the output")
    _add_table_options(p)
    p.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    from .profile import ProfileError
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ProfileError as exc:
        print(f"goodvibes-profile: error: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, ImportError, yaml.YAMLError) as exc:
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"goodvibes-profile: error: {msg}", file=sys.stderr)
        return 1


if __name__ == "__main__":                                  # pragma: no cover
    sys.exit(main())
