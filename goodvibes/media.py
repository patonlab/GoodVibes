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
    """
    Load the bundled solvents.json and build a flat mapping from each lowercase alias to its (molecular weight, density) tuple.
    
    The JSON file is read from the same directory as this module. Each solvent entry's `aliases` list is expanded and normalized to lowercase.
    
    Returns:
        dict: Mapping where keys are lowercase alias strings and values are `(mw, density)` tuples — `mw` in g/mol and `density` in g/mL.
    """
    json_path = os.path.join(os.path.dirname(__file__), 'solvents.json')
    with open(json_path, 'r') as f:
        data = json.load(f)

    result = {}
    for entry in data['solvents']:
        canonical = entry['name']
        value = (entry['mw'], entry['density'], canonical)
        for alias in entry['aliases']:
            result[alias.lower()] = value
    return result


solvents = _load_solvents()


def compute_media_conc(media, file):
    """
    Compute the neat-solvent molar concentration when the output file corresponds to the specified solvent.
    
    Parameters:
        media (str): Solvent name as provided (e.g., from --media).
        file (str): Path to the output file used to infer the solvent name.
    
    Returns:
        float or None: Neat-solvent concentration in mol/L if the file's solvent matches `media`, `None` otherwise.
    """
    from .utils import display_name
    media_key = media.lower()
    file_key = display_name(file).lower()

    # Get canonical identities for both media and file
    if media_key not in solvents:
        return None

    mweight, density, media_canonical = solvents[media_key]

    # Check if file also references a known solvent
    if file_key in solvents:
        _, _, file_canonical = solvents[file_key]
        # Only apply concentration correction if both refer to the same canonical solvent
        if media_canonical == file_canonical:
            return (density * 1000) / mweight

    return None
