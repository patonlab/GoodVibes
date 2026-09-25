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


def _force_utf8(stream):
    """Best-effort switch a text stream to UTF-8.

    GoodVibes output contains Unicode (✔, box-drawing chars). On Windows the
    console/file default to cp1252, which raises UnicodeEncodeError mid-run
    (issue #102). Reconfigure to UTF-8 where possible; ``errors='replace'``
    keeps any exotic glyph from ever crashing the program.
    """
    if stream is None:
        return
    try:
        stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError, OSError):
        pass  # not a reconfigurable TextIOWrapper (e.g. already wrapped/redirected)


def all_same(items):
    """
    Determine whether every element of `items` equals the first element.

    Parameters:
        items (Sequence): A non-empty sequence of comparable elements.

    Returns:
        True if every element equals the first element, False otherwise.

    Raises:
        IndexError: If `items` is empty.
    """
    return all(x == items[0] for x in items)


def setup_logging(filein, append):
    """
    Configure the 'goodvibes' logger to write to both stdout and a .dat file and initialize module-level Rich consoles.
    
    Initializes the logger named 'goodvibes' to emit messages to standard output and to a file at "{filein}_{append}.dat". Opens the .dat file for writing and, if Rich is available, assigns module-level Console instances for stdout and the .dat file for later use.
    
    Parameters:
        filein (str): Prefix for the output file (e.g., "GoodVibes").
        append (str): Suffix for the output file (e.g., "output").
    """
    global _console_stdout, _console_dat

    # Ensure the console can encode GoodVibes' Unicode output (issue #102).
    _force_utf8(sys.stdout)
    _force_utf8(sys.stderr)

    logger = logging.getLogger('goodvibes')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(message)s')

    # stdout handler + terminal console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.terminator = ''
    logger.addHandler(console_handler)

    if Console is not None:
        # height MUST be set alongside width: Rich's Console.size only
        # short-circuits to the explicit width when height is also given.
        # With only width set, a dumb terminal (TERM=dumb, e.g. CircleCI)
        # runs terminal detection, ignores width=200 and clamps to 80 cols,
        # cropping the rightmost table columns (incl. qh-G).
        _console_stdout = Console(highlight=False, force_terminal=True,
                                  width=200, height=200)

    # .dat file: shared handle for both logging and Rich output
    dat_path = f'{filein}_{append}.dat'
    dat_fp = open(dat_path, 'w', encoding='utf-8')

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
            width=200,            # match stdout console: Rich defaults non-TTY
                                  # files to 80 cols and crops tables, dropping
                                  # the rightmost columns (incl. qh-G) from .dat
            height=200,           # required for width to be honored on dumb
                                  # terminals (see _console_stdout above)
        )


def fatal(message):
    """
    Log a critical error message and terminate the process.
    
    Shuts down the logging subsystem and exits the process with status code 1.
    
    Parameters:
        message (str): The message to emit at the critical level.
    """
    log = logging.getLogger('goodvibes')
    log.critical(message + "\n")
    logging.shutdown()
    sys.exit(1)


def add_time(tm, cpu):
    """
    Create a datetime representing tm's day/time advanced by an elapsed CPU-style interval.
    
    Parameters:
        tm (datetime): Source datetime whose day, hour, minute, second, and microsecond are used.
        cpu (Sequence[int]): Elapsed time as [days, hrs, mins, secs, msecs].
    
    Returns:
        datetime: A new datetime with year set to 100 and month set to 1, using tm's day/time
        plus the interval from `cpu` (milliseconds interpreted as 1/1000 second).
    """
    [days, hrs, mins, secs, msecs] = cpu
    fulldate = datetime(100, 1, tm.day, tm.hour, tm.minute, tm.second, tm.microsecond)
    fulldate = fulldate + timedelta(days=days, hours=hrs, minutes=mins, seconds=secs, microseconds=msecs * 1000)
    return fulldate


def parse_temperature_interval(spec):
    """Turn a ``--ti`` specification into the list of temperatures to scan.

    ``spec`` is ``"start,end"`` or ``"start,end,step"`` (kelvin). With two
    values the range is divided into ten steps. Values are floats: a step of
    0.5 K or a start of 298.15 K is honoured rather than truncated to
    integers (which used to turn ``--ti 200,201,0.5`` into a range() error and
    ``--ti 298.15,398.15,50`` into 298, 348, 398). The end temperature is
    included when it lies on the grid (within 1e-9 K).

    Returns:
        list[float]: temperatures in ascending order.

    Raises:
        ValueError: on a malformed spec, a non-positive step or end < start.
    """
    try:
        values = [float(x) for x in str(spec).split(',')]
    except ValueError:
        raise ValueError(f"--ti expects 'start,end[,step]' in kelvin, got {spec!r}") from None
    if len(values) == 2:
        values.append((values[1] - values[0]) / 10.0)
    if len(values) != 3:
        raise ValueError(f"--ti expects 'start,end[,step]' in kelvin, got {spec!r}")
    start, end, step = values
    if end < start:
        raise ValueError(f"--ti: end temperature {end} K is below start {start} K")
    if step <= 0:
        raise ValueError(f"--ti: temperature step must be positive, got {step}")
    temps = []
    n = 0
    while True:
        t = start + n * step
        if t > end + 1e-9:
            break
        temps.append(round(t, 10))
        n += 1
    return temps


def display_name(file):
    """
    Get the basename of a file path without its extension for display.
    
    Parameters:
        file (str): Path or filename from which to extract the display name.
    
    Returns:
        display_name (str): The filename portion of `file` with the final extension removed.
    """
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
    """
    Get the Rich Console configured for colored stdout output.
    
    Returns:
        Console: The Rich Console instance used for stdout.
    
    Raises:
        RuntimeError: If `setup_logging()` has not been called and the console is not initialized.
    """
    if _console_stdout is None:
        raise RuntimeError("setup_logging() must be called before get_console_stdout()")
    return _console_stdout


def get_console_dat() -> "Console":
    """Return the Rich Console for the .dat file (no color, box-drawing chars)."""
    if _console_dat is None:
        raise RuntimeError("setup_logging() must be called before get_console_dat()")
    return _console_dat
