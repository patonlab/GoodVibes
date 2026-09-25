"""The .dat archive must never contain terminal escape codes.

Regression: _print_rich_table rendered the .dat copy through a console that
mirrored the target's terminal/colour settings, so with TERM=dumb (CircleCI)
Rich's default italic table title reached the archive as '\\x1b[3m ... \\x1b[0m'
and the PES compatibility goldens failed there while passing elsewhere.
"""
import os

import pytest
from rich import box
from rich.table import Table

from goodvibes.output import _print_rich_table
from goodvibes.utils import setup_logging

from test_cli_errors import gv_logger_cleanup  # noqa: F401  (fixture re-export)


@pytest.mark.parametrize("term", ["dumb", "linux", "xterm-256color", None])
def test_dat_copy_of_a_titled_table_has_no_escape_codes(tmp_path, monkeypatch, term, gv_logger_cleanup):  # noqa: F811
    if term is None:
        monkeypatch.delenv("TERM", raising=False)
    else:
        monkeypatch.setenv("TERM", term)
    monkeypatch.chdir(tmp_path)
    setup_logging("GoodVibes", "ansi")
    table = Table(title="RXN: Reaction  (kcal/mol)  at T = 298.15 K", box=box.SIMPLE, header_style="bold")
    table.add_column("Species")
    table.add_column("ΔE", justify="right")
    table.add_row("A", "0.00")
    _print_rich_table(table)
    text = (tmp_path / "GoodVibes_ansi.dat").read_text(encoding="utf-8")
    assert "\x1b" not in text
    assert "RXN: Reaction" in text and "─" in text and "ΔE" in text
    assert os.path.exists(tmp_path / "GoodVibes_ansi.dat")
