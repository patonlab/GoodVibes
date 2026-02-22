#!/usr/bin/python

# Comments and/or additions are welcome (send e-mail to:
# robert.paton@chem.ox.ac.uk

#####################################
#        vib_scale_factors.py       #
#####################################
###  Written by:  Rob Paton #########
###          and Guilian Luchini  ###
###  Last modified:  2025         ###
#####################################

"""Vibrational frequency scaling factors from the Truhlar group database.

Frequency scaling factors taken from version 4 of the Truhlar group database
(https://t1.chem.umn.edu/freqscale/index.html).

Alecu, I. M.; Zheng, J.; Zhao, Y.; Truhlar, D. G. J. Chem. Theory Comput.
2010, 6, 2872-2887.

The data is stored in scaling_factors.json alongside this module. Each entry
contains scaling factors for ZPEs, harmonic frequencies, and fundamentals.

Methods:
    D: Directly obtained from the ZPVE15/10 or F38/10 databases (Ref. 1).
    C: Obtained by applying a systematic correction of -0.0025 to preexisting
       scale factor.
    R: Obtained via the Reduced Scale Factor Optimization Model (Ref. 1).

Public API:
    scaling_data_dict    -- dict mapping canonicalized level/basis -> ScalingData
    scaling_refs         -- list of reference citation strings
    ScalingData          -- namedtuple for scaling factor data
    canonicalize_level   -- normalize level/basis strings for lookup
    FUNCTIONAL_ALIASES   -- dict of known functional name aliases
"""

import json
import os
from collections import namedtuple

ScalingData = namedtuple("ScalingData", [
    'level_basis', 'zpe_fac', 'zpe_ref', 'zpe_meth',
    'harm_fac', 'harm_ref', 'harm_meth',
    'fund_fac', 'fund_ref', 'fund_meth',
])

# Maps non-canonical functional names to the canonical name used in the
# database. All keys must be UPPERCASE. This handles differences between
# quantum chemistry programs (e.g. Gaussian's PBE1PBE vs ORCA's PBE0).
FUNCTIONAL_ALIASES = {
    # Gaussian uses PBE1PBE internally; database has PBE0
    "PBE1PBE": "PBE0",
    # Gaussian archive PBEPBE -> PBE (the pure GGA)
    "PBEPBE": "PBE",
    # HSE06 is the same as HSEh1PBE
    "HSE06": "HSEH1PBE",
    # Minnesota functionals: ORCA uses hyphens, database does not
    "M06-2X": "M062X",
    "M06-L": "M06L",
    "M06-HF": "M06HF",
    "M05-2X": "M052X",
    "M08-HX": "M08HX",
    "M08-SO": "M08SO",
    # omega-B97 variants: database has wB97XD (no hyphen)
    "WB97X-D": "WB97XD",
    "WB97X-D3": "WB97XD",
    "WB97XD3": "WB97XD",
    # B97-3 -> B973
    "B97-3": "B973",
    # CAM-B3LYP without hyphen
    "CAMB3LYP": "CAM-B3LYP",
}


def canonicalize_level(level_basis):
    """Normalize a level/basis string for consistent dictionary lookup.

    Applied to both database keys (at load time) and query strings (at lookup
    time) to ensure matching regardless of program-specific naming.

    Steps:
        1. Uppercase the entire string
        2. Split functional from basis set at the first '/'
        3. Apply functional alias mapping
        4. Rejoin
    """
    s = level_basis.strip().upper()
    if '/' in s:
        functional, basis = s.split('/', 1)
    else:
        functional = s
        basis = None
    functional = FUNCTIONAL_ALIASES.get(functional, functional)
    if basis is not None:
        return functional + '/' + basis
    return functional


def _load_scaling_data():
    """Load scaling factor data from the JSON file."""
    json_path = os.path.join(os.path.dirname(__file__), 'scaling_factors.json')
    with open(json_path, 'r') as f:
        data = json.load(f)

    refs = data['references']
    entries = {}
    for entry in data['entries']:
        sd = ScalingData(
            level_basis=entry['level_basis'],
            zpe_fac=entry['zpe_fac'],
            zpe_ref=entry['zpe_ref'],
            zpe_meth=entry['zpe_meth'],
            harm_fac=entry['harm_fac'],
            harm_ref=entry['harm_ref'],
            harm_meth=entry['harm_meth'],
            fund_fac=entry['fund_fac'],
            fund_ref=entry['fund_ref'],
            fund_meth=entry['fund_meth'],
        )
        key = canonicalize_level(entry['level_basis'])
        entries[key] = sd

    return refs, entries


scaling_refs, scaling_data_dict = _load_scaling_data()
