"""Utility classes and functions for GoodVibes."""
import logging
import os.path
import sys
from datetime import datetime, timedelta


def all_same(items):
    """Returns bool for checking if all items in a list are the same."""
    return all(x == items[0] for x in items)


def setup_logging(filein, append):
    """Configure the 'goodvibes' logger with dual output: stdout + .dat file.

    Both handlers use a bare formatter (message only, no level/timestamp)
    and empty terminators (no auto-newlines) to preserve the partial-line
    write pattern used throughout the codebase.

    Parameters:
        filein (str): prefix for the output file (e.g. "GoodVibes").
        append (str): suffix for the output file (e.g. "output").
    """
    logger = logging.getLogger('goodvibes')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(message)s')

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.terminator = ''

    datfile = logging.FileHandler(f'{filein}_{append}.dat', mode='w')
    datfile.setFormatter(formatter)
    datfile.terminator = ''

    logger.addHandler(console)
    logger.addHandler(datfile)


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
