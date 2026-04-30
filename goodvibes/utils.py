"""Utility classes and functions for GoodVibes."""
import logging
import os.path
import sys
from datetime import datetime, timedelta
from typing import Optional

try:
    from rich.console import Console
except ImportError:
    Console = None


_console_stdout: Optional["Console"] = None
_console_dat: Optional["Console"] = None


def all_same(items):
    """Returns bool for checking if all items in a list are the same."""
    return all(x == items[0] for x in items)


def setup_logging(filein, append):
    """Configure the 'goodvibes' logger with dual output: stdout + .dat file.

    Creates a shared .dat file handle for both the logging FileHandler and a
    no-color Rich Console, ensuring ANSI codes don't corrupt the .dat output.
    The .dat Console uses box-drawing characters but no color.

    Also initializes module-level Rich Consoles for use by output.py functions.

    Parameters:
        filein (str): prefix for the output file (e.g. "GoodVibes").
        append (str): suffix for the output file (e.g. "output").
    """
    global _console_stdout, _console_dat

    logger = logging.getLogger('goodvibes')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(message)s')

    # stdout handler + terminal console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.terminator = ''
    logger.addHandler(console_handler)

    if Console is not None:
        _console_stdout = Console(highlight=False, force_terminal=True, width=200)

    # .dat file: shared handle for both logging and Rich output
    dat_path = f'{filein}_{append}.dat'
    dat_fp = open(dat_path, 'w')

    # Use StreamHandler with the open file, not FileHandler (avoids double-open)
    datfile_handler = logging.StreamHandler(dat_fp)
    datfile_handler.setFormatter(formatter)
    datfile_handler.terminator = ''
    logger.addHandler(datfile_handler)

    if Console is not None:
        _console_dat = Console(
            file=dat_fp,
            force_terminal=True,  # emit box-drawing chars even though file is not a TTY
            no_color=True,        # strip ANSI color codes
            highlight=False,      # don't auto-highlight tokens
        )


def fatal(message):
    """Log a critical message and exit."""
    log = logging.getLogger('goodvibes')
    log.critical(message + "\n")
    logging.shutdown()
    sys.exit(1)


def add_time(tm, cpu):
    """Calculate elapsed time."""
    [days, hrs, mins, secs, msecs] = cpu
    fulldate = datetime(100, 1, tm.day, tm.hour, tm.minute, tm.second, tm.microsecond)
    fulldate = fulldate + timedelta(days=days, hours=hrs, minutes=mins, seconds=secs, microseconds=msecs * 1000)
    return fulldate


def display_name(file):
    """Return the basename without extension, used for output display."""
    return os.path.splitext(os.path.basename(file))[0]


def natural_key(path):
    """Sort key that orders ``conf_2`` before ``conf_10`` (and before ``conf_a``).

    Splits on digit runs and treats them as integers so ordinary string
    comparison won't put ``conf_10`` between ``conf_1`` and ``conf_2``.
    Comparison uses the basename so files from different directories with the
    same name don't separate solely by directory path.
    """
    import re
    base = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', base)]


def get_console_stdout() -> "Console":
    """Return the Rich Console for stdout (with colors/formatting)."""
    if _console_stdout is None:
        raise RuntimeError("setup_logging() must be called before get_console_stdout()")
    return _console_stdout


def get_console_dat() -> "Console":
    """Return the Rich Console for the .dat file (no color, box-drawing chars)."""
    if _console_dat is None:
        raise RuntimeError("setup_logging() must be called before get_console_dat()")
    return _console_dat
