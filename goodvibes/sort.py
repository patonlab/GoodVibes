"""Structure sorting and duplicate detection for GoodVibes."""
import logging
import numpy as np

from .constants import KCAL_TO_AU

log = logging.getLogger('goodvibes')


def kabsch_rmsd(coords_a, coords_b):
    """
    Compute the RMSD between two Nx3 Cartesian coordinate sets after optimal rigid alignment using the Kabsch algorithm.
    
    Parameters:
        coords_a (array-like): Reference coordinates with shape (N, 3).
        coords_b (array-like): Mobile coordinates with shape (N, 3).
    
    Returns:
        float: Root-mean-square deviation between the aligned coordinates, in the same units as the inputs.
    """
    a = np.array(coords_a, dtype=float)
    b = np.array(coords_b, dtype=float)
    # Center both structures
    a -= a.mean(axis=0)
    b -= b.mean(axis=0)
    # Kabsch: find optimal rotation via SVD of cross-covariance matrix
    H = b.T @ a
    U, _, Vt = np.linalg.svd(H)
    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, d])
    R = Vt.T @ sign_matrix @ U.T
    b_aligned = b @ R.T
    return np.sqrt(np.mean((a - b_aligned) ** 2))


def deduplicate(thermo_data, *, e_cutoff=0.05, ro_cutoff=0.01,
                rmsd_cutoff=None):
    """Identify duplicate or enantiomeric structures by comparing energies,
    rotational constants, and optionally Cartesian RMSD.

    All active criteria must pass for a pair to be flagged as duplicate.

    Parameters:
        thermo_data (dict): file path → calc_bbe mapping.
        e_cutoff (float): max absolute SCF energy difference in kcal/mol (default 0.05).
        ro_cutoff (float): max relative difference in rotational constants as a
            fraction, e.g. 0.01 = 1% (default 0.01).
        rmsd_cutoff (float or None): max Cartesian RMSD in Angstrom.
            None (default) disables RMSD comparison; 0.125 matches CREST.

    Returns:
        list: pairs [file_i, file_j] flagged as duplicates.
    """
    files = list(thermo_data)
    dup_list = []
    e_cutoff_au = e_cutoff / KCAL_TO_AU  # Convert kcal/mol to Hartree
    cutoff_msg = "\n   Checking for duplicate structures. Applying: e_cutoff={} kcal/mol, ro_cutoff={}%".format(e_cutoff, ro_cutoff * 100)
    if rmsd_cutoff is not None:
        cutoff_msg += ", rmsd_cutoff={} A".format(rmsd_cutoff)
    log.info("\n" + cutoff_msg)
    for i, file in enumerate(files):
        for j in range(0, i):
            bbe_i, bbe_j = thermo_data[files[i]], thermo_data[files[j]]

            # Energy gate (cheap): reject the pair before computing ro_diff or RMSD.
            if not (hasattr(bbe_i, "scf_energy") and hasattr(bbe_j, "scf_energy")):
                continue
            if abs(bbe_i.scf_energy - bbe_j.scf_energy) >= e_cutoff_au:
                continue

            # Rotational-constants gate (microseconds): only run if energy passed.
            if not (hasattr(bbe_i, "roconst") and hasattr(bbe_j, "roconst")):
                continue
            if len(bbe_i.roconst) != len(bbe_j.roconst):
                continue
            ri = np.array(bbe_i.roconst)
            rj = np.array(bbe_j.roconst)
            avg = 0.5 * (ri + rj)
            nonzero = avg > 0
            if np.any(nonzero):
                ro_diff = np.max(np.abs(ri[nonzero] - rj[nonzero]) / avg[nonzero])
            else:
                # Both structures have all-zero rotational constants
                # (e.g. single atoms) — treat as matching
                ro_diff = 0.0
            if ro_diff >= ro_cutoff:
                continue

            # Kabsch RMSD gate (most expensive): only run if e and ro both passed.
            if rmsd_cutoff is not None:
                if not (hasattr(bbe_i, "cartesians") and hasattr(bbe_j, "cartesians")):
                    continue
                coords_i = np.array(bbe_i.cartesians)
                coords_j = np.array(bbe_j.cartesians)
                if coords_i.shape != coords_j.shape or len(coords_i) == 0:
                    continue
                if kabsch_rmsd(coords_i, coords_j) >= rmsd_cutoff:
                    continue

            dup_list.append([files[i], files[j]])
    return dup_list


SORT_KEYS = {
    'energy': 'scf_energy',
    'gibbs': 'qh_gibbs_free_energy',
}


def sort_thermo(thermo_data, key):
    """Return thermo_data reordered by the given energy attribute (lowest first).

    Entries with linear_warning, missing the attribute, or with a None value are placed at the end.

    Parameters:
        thermo_data (dict): file path → calc_bbe mapping.
        key (str): sort mode — 'energy' (scf_energy) or 'gibbs' (qh_gibbs_free_energy).

    Returns:
        dict: new dict with the same items in sorted order.
    """
    attr = SORT_KEYS[key]
    inf = float('inf')

    def sort_val(item):
        """
        Compute the sort key for a thermochemistry mapping item, treating flagged or missing values as infinite so they sort last.
        
        Parameters:
            item (tuple): A (key, calc_bbe) pair where `calc_bbe` is the object containing thermochemical attributes.
        
        Returns:
            float: The attribute value `getattr(calc_bbe, attr)` when present and not None; `float('inf')` if `calc_bbe.linear_warning` is truthy or the attribute is missing/None.
        """
        bbe = item[1]
        if getattr(bbe, 'linear_warning', False):
            return inf
        val = getattr(bbe, attr, None)
        return val if val is not None else inf

    return dict(sorted(thermo_data.items(), key=sort_val))
