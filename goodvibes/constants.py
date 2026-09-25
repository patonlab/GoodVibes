"""Constants and literature references for GoodVibes."""

# VERSION NUMBER
__version__ = "4.4.0"

SUPPORTED_EXTENSIONS = set(('.out', '.log', '.extxyz'))

# PHYSICAL CONSTANTS & UNITS
GAS_CONSTANT = 8.3144621  # J / K / mol
ATMOS = 101.325  # kPa; 1 atm
KCAL_TO_AU = 627.509541  # kcal/mol per Hartree
J_TO_AU = 4.184 * KCAL_TO_AU * 1000.0  # J/mol per Hartree
HARTREE_TO_EV = 27.211386245988  # eV per Hartree (CODATA 2018)

# Energy units accepted wherever a user chooses display units (PES tables,
# plots, JSON). Keys are the canonical spellings; the aliases map onto them.
UNIT_FACTORS = {
    'kcal/mol': KCAL_TO_AU,
    'kJ/mol': J_TO_AU / 1000.0,
    'eV': HARTREE_TO_EV,
    'hartree': 1.0,
}
_UNIT_ALIASES = {
    'kcal/mol': 'kcal/mol', 'kcal': 'kcal/mol', 'kcalmol': 'kcal/mol', 'kcal mol-1': 'kcal/mol',
    'kj/mol': 'kJ/mol', 'kj': 'kJ/mol', 'kjmol': 'kJ/mol', 'kj mol-1': 'kJ/mol',
    'ev': 'eV',
    'hartree': 'hartree', 'hartrees': 'hartree', 'eh': 'hartree', 'au': 'hartree', 'a.u.': 'hartree',
}
SUPPORTED_UNITS = tuple(UNIT_FACTORS)


def canonical_units(units):
    """Return the canonical spelling of an energy unit, or raise ValueError.

    ``canonical_units('kJ/mol') == 'kJ/mol'``, ``canonical_units('ev') == 'eV'``.
    """
    key = str(units).strip().lower()
    try:
        return _UNIT_ALIASES[key]
    except KeyError:
        raise ValueError(
            f"unknown energy units {units!r}; expected one of {', '.join(SUPPORTED_UNITS)}"
        ) from None


def hartree_factor(units):
    """Multiplier that converts a value in Hartree to ``units``."""
    return UNIT_FACTORS[canonical_units(units)]

# Some literature references
grimme_mRRHO_ref = "Grimme, S. Chem. Eur. J. 2012, 18, 9955-9964"
grimme_msRRHO_ref = "Grimme, S.; Pracht, P. Chem. Sci. 2021, 12, 6551-6568"
truhlar_ref = "Ribeiro, R. F.; Marenich, A. V.; Cramer, C. J.; Truhlar, D. G. J. Phys. Chem. B 2011, 115, 14556-14562"
head_gordon_ref = "Li, Y.; Gomes, J.; Sharada, S. M.; Bell, A. T.; Head-Gordon, M. J. Phys. Chem. C 2015, 119, 1840-1850"
goodvibes_ref = ("Luchini, G.; Alegre-Requena, J. V.; Funes-Ardoiz, I.; Paton, R. S. F1000Research, 2020, 9, 291."
                 "\n   DOI: 10.12688/f1000research.22758.1")

# Banner with version and citation info
gv_banner = ("      ________   ________   ________    _______   ________   ________   ________   ________   ________ \n"
            "     \u2571        \u2572 \u2571        \u2572 \u2571        \u2572 _\u2571       \u2572 \u2571    \u2571   \u2572 \u2571        \u2572 \u2571       \u2571  \u2571        \u2572 \u2571        \u2572\n"
            "    \u2571   G   __\u2571\u2571    O    \u2571\u2571    O    \u2571\u2571    D    \u2571\u2571    V    \u2571_\u2571   I   \u2571 \u2571    B   \u2572 \u2571    E    \u2571\u2571    S   _\u2571\n"
            "   \u2571       \u2571 \u2571\u2571         \u2571\u2571         \u2571\u2571         \u2571 \u2572        \u2571\u2571         \u2571\u2571         \u2571\u2571        _\u2571\u2571- v" + __version__ + " \u2571 \n"
            "   \u2572________\u2571 \u2572________\u2571 \u2572________\u2571 \u2572________\u2571   \u2572______\u2571 \u2572________\u2571 \u2572________\u2571 \u2572________\u2571 \u2572________\u2571\n"
            "\n   Citation: " + goodvibes_ref + "\n")
