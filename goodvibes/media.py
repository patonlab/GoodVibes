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
        value = (entry['mw'], entry['density'])
        for alias in entry['aliases']:
            result[alias.lower()] = value
    return result


solvents = _load_solvents()


def lookup_solvent(name):
    """Look up a solvent by alias and return its (mw, density).

    Raises ValueError with a "did you mean ..." hint and a list of common
    aliases if `name` is unknown. Use this to validate user input (e.g.
    --media / --freespace) up-front; `compute_media_conc` and other call
    sites can then assume the alias is valid.
    """
    key = name.lower()
    if key not in solvents:
        import difflib
        suggestions = difflib.get_close_matches(key, solvents.keys(), n=5, cutoff=0.5)
        common = ['water', 'methanol', 'ethanol', 'acetone', 'dmso', 'dmf',
                  'thf', 'benzene', 'toluene', 'chloroform', 'dcm', 'acetonitrile']
        msg = f"Unknown solvent {name!r}."
        if suggestions:
            msg += f" Did you mean: {', '.join(suggestions)}?"
        msg += (f" Common aliases: {', '.join(common)} "
                f"({len(solvents)} total in goodvibes/solvents.json).")
        raise ValueError(msg)
    return solvents[key]


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
    key = media.lower()
    if key in solvents and key == display_name(file).lower():
        mweight, density = solvents[key]
        return (density * 1000) / mweight
    return None
