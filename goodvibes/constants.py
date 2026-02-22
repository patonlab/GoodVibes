"""Constants and literature references for GoodVibes."""

# VERSION NUMBER
__version__ = "4.0"

SUPPORTED_EXTENSIONS = set(('.out', '.log'))

# PHYSICAL CONSTANTS                                      UNITS
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
csd_ref = ("C. R. Groom, I. J. Bruno, M. P. Lightfoot and S. C. Ward, Acta Cryst. 2016, B72, 171-179"
           "\n   Cordero, B.; Gomez V.; Platero-Prats, A. E.; Reves, M.; Echeverria, J.; Cremades, E.; Barragan, F.; Alvarez, S. Dalton Trans. 2008, 2832-2838")
oniom_scale_ref = "Simon, L.; Paton, R. S. J. Am. Chem. Soc. 2018, 140, 5412-5420"
