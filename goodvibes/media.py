"""Solvent database for media concentration corrections.

Solvent molecular weights [g/mol] and densities [g/mL] at 20 °C, loaded from
solvents.json alongside this module. Each solvent entry defines a canonical name
and one or more lookup aliases (abbreviations, full names).

Public API:
    solvents  -- dict mapping alias (lowercase str) -> (mw, density) tuple
"""

import json
import os


def _load_solvents():
    """Load solvent data from the JSON file and build a flat alias -> (mw, density) dict."""
    json_path = os.path.join(os.path.dirname(__file__), 'solvents.json')
    with open(json_path, 'r') as f:
        data = json.load(f)

    result = {}
    for entry in data['solvents']:
        value = (entry['mw'], entry['density'])
        for alias in entry['aliases']:
            result[alias.lower()] = value
    return result


solvents = _load_solvents()


def compute_media_conc(media, file):
    """Return the neat solvent concentration if the filename matches the solvent name.

    Used to replace the default concentration with the pure-solvent concentration
    when the calculation is for the solvent molecule itself.

    Parameters:
        media (str): solvent name from --media.
        file (str): output file path.

    Returns:
        float or None: concentration in mol/L, or None if the file doesn't match.
    """
    from .utils import display_name
    key = media.lower()
    if key in solvents and key == display_name(file).lower():
        mweight, density = solvents[key]
        return (density * 1000) / mweight
    return None
