"""Constants and literature references for GoodVibes."""

# VERSION NUMBER
__version__ = "4.2.0"

SUPPORTED_EXTENSIONS = set(('.out', '.log', '.extxyz'))

# PHYSICAL CONSTANTS & UNITS
GAS_CONSTANT = 8.3144621  # J / K / mol
ATMOS = 101.325  # UNIT CONVERSION
J_TO_AU = 4.184 * 627.509541 * 1000.0  # UNIT CONVERSION
KCAL_TO_AU = 627.509541  # UNIT CONVERSION

# Some literature references
grimme_mRRHO_ref = "Grimme, S. Chem. Eur. J. 2012, 18, 9955-9964"
grimme_msRRHO_ref = "Grimme, S.; Pracht, P. Chem. Sci. 2021, 12, 6551-6568"
truhlar_ref = "Ribeiro, R. F.; Marenich, A. V.; Cramer, C. J.; Truhlar, D. G. J. Phys. Chem. B 2011, 115, 14556-14562"
head_gordon_ref = "Li, Y.; Gomes, J.; Sharada, S. M.; Bell, A. T.; Head-Gordon, M. J. Phys. Chem. C 2015, 119, 1840-1850"
goodvibes_ref = ("Luchini, G.; Alegre-Requena, J. V.; Funes-Ardoiz, I.; Paton, R. S. F1000Research, 2020, 9, 291."
                 "\n   DOI: 10.12688/f1000research.22758.1")
oniom_scale_ref = "Simon, L.; Paton, R. S. J. Am. Chem. Soc. 2018, 140, 5412-5420"

# Banner with version and citation info
gv_banner = ("      ________   ________   ________    _______   ________   ________   ________   ________   ________ \n"
            "     \u2571        \u2572 \u2571        \u2572 \u2571        \u2572 _\u2571       \u2572 \u2571    \u2571   \u2572 \u2571        \u2572 \u2571       \u2571  \u2571        \u2572 \u2571        \u2572\n"
            "    \u2571   G   __\u2571\u2571    O    \u2571\u2571    O    \u2571\u2571    D    \u2571\u2571    V    \u2571_\u2571   I   \u2571 \u2571    B   \u2572 \u2571    E    \u2571\u2571    S   _\u2571\n"
            "   \u2571       \u2571 \u2571\u2571         \u2571\u2571         \u2571\u2571         \u2571 \u2572        \u2571\u2571         \u2571\u2571         \u2571\u2571        _\u2571\u2571-  v" + __version__ + "  \u2571 \n"
            "   \u2572________\u2571 \u2572________\u2571 \u2572________\u2571 \u2572________\u2571   \u2572______\u2571 \u2572________\u2571 \u2572________\u2571 \u2572________\u2571 \u2572________\u2571\n"
            "\n   Citation: " + goodvibes_ref + "\n")
