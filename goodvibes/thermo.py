# -*- coding: utf-8 -*-

import math
import os.path
import sys
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .io import parse_qcdata, parse_data, sp_cpu as _sp_cpu, find_spc_file

# PHYSICAL CONSTANTS & UNITS
GAS_CONSTANT = 8.3144621  # J / K / mol
PLANCK_CONSTANT = 6.62606957e-34  # J * s
BOLTZMANN_CONSTANT = 1.3806488e-23  # J / K
SPEED_OF_LIGHT = 2.99792458e10  # cm / s
AVOGADRO_CONSTANT = 6.0221415e23  # 1 / mol
AMU_to_KG = 1.66053886E-27  # UNIT CONVERSION
J_TO_AU = 4.184 * 627.509541 * 1000.0  # UNIT CONVERSION
GRIMME_BAV = 1.00e-44  # Default average moment of inertia (kg m^2) from Grimme

# Solvents supported by --freespace (Shakhnovich & Whitesides free-volume model).
# Case-sensitive — these are the literal strings the user must pass.
FREESPACE_SOLVENTS = ("H2O", "toluene", "DMF", "AcOH", "chloroform")


# Symmetry numbers for different point groups
pg_sm = {"C1": 1, "Cs": 1, "Ci": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6, "C7": 7, "C8": 8, "D2": 4, "D3": 6,
         "D4": 8, "D5": 10, "D6": 12, "D7": 14, "D8": 16, "C2v": 2, "C3v": 3, "C4v": 4, "C5v": 5, "C6v": 6, "C7v": 7,
         "C8v": 8, "C2h": 2, "C3h": 3, "C4h": 4, "C5h": 5, "C6h": 6, "C7h": 7, "C8h": 8, "D2h": 4, "D3h": 6, "D4h": 8,
         "D5h": 10, "D6h": 12, "D7h": 14, "D8h": 16, "D2d": 4, "D3d": 6, "D4d": 8, "D5d": 10, "D6d": 12, "D7d": 14,
         "D8d": 16, "S4": 4, "S6": 6, "S8": 8, "T": 6, "Th": 12, "Td": 12, "O": 12, "Oh": 24, "Cinfv": 1, "Dinfh": 2,
         "I": 30, "Ih": 60, "Kh": 1}




def calc_translational_energy(temperature):
    """
    Compute the ideal-gas translational molar energy at a specified temperature.
    
    Parameters:
        temperature (float): Temperature in kelvin (K).
    
    Returns:
        float: Translational energy in joules per mole (J/mol), equal to 3/2 · R · T.
    """
    energy = 1.5 * GAS_CONSTANT * temperature
    return energy


def calc_rotational_energy(temperature, monatomic=False, linear=False):
    """
    Compute the rotational energy for a species at a given temperature.
    
    Returns 0 for monatomic species, R*T for linear molecules, and 3/2*R*T for non-linear molecules.
    
    Parameters:
        temperature (float): Temperature in kelvin.
        monatomic (bool): If True, treat as a single atom (zero rotational energy).
        linear (bool): If True, treat as a linear molecule (use R*T).
    
    Returns:
        float: Rotational energy in joules per mole.
    """
    if monatomic:
        energy = 0.0
    elif linear:
        energy = GAS_CONSTANT * temperature
    else:
        energy = 1.5 * GAS_CONSTANT * temperature
    return energy


def _coerce_scale_factor(freq_scale_factor, fract_modelsys):
    # Coerce to Python float(s) so numpy.float32 inputs can't force float32
    # arithmetic downstream, which under value-based scalar casting
    # (NumPy < 2.0) silently underflows expressions like h / (8 π² ν c s)
    # in calc_freerot_entropy to 0.0 and produces NaN entropies.
    if fract_modelsys is not None:
        return [float(x) for x in freq_scale_factor]
    return float(freq_scale_factor)


def calc_vibrational_energy(temperature, frequency_wn, freq_scale_factor=1.0, fract_modelsys=None):
    """
    Compute the vibrational energy (zero-point + thermal contributions) in joules per mole at a given temperature.
    
    Parameters:
        temperature (float): Temperature in kelvin; must be greater than 0.
        frequency_wn (list[float]): Vibrational mode wavenumbers in cm^-1.
        freq_scale_factor (float or list[float], optional): Frequency scaling factor. If ONIOM blending is used (fract_modelsys provided), supply [qm_scale, mm_scale]; otherwise a single scalar scale factor is applied to all modes.
        fract_modelsys (list[float] or None, optional): Per-mode ONIOM fractions (values between 0 and 1). When provided, per-mode scale factors are computed by blending the two entries of `freq_scale_factor` according to these fractions. If None, no ONIOM blending is applied.
    
    Returns:
        float: Total vibrational energy (J/mol), including zero-point energy and thermal excitations.
    
    Raises:
        ValueError: If `temperature` is not greater than 0, or if mode energies produce numerical overflow (indicative of too-low temperature).
    """
    if temperature <= 0:
        raise ValueError(
            "Temperature must be positive for vibrational energy calculation."
        )

    freq_scale_factor = _coerce_scale_factor(freq_scale_factor, fract_modelsys)
    if fract_modelsys is not None:
        s0, s1 = freq_scale_factor[0], freq_scale_factor[1]
        freq_scale_factor = [s0 * fm + s1 * (1.0 - fm) for fm in fract_modelsys]
        factor = [(PLANCK_CONSTANT * f * SPEED_OF_LIGHT * s) / (BOLTZMANN_CONSTANT * temperature)
                  for f, s in zip(frequency_wn, freq_scale_factor)]
    else:
        factor = [(PLANCK_CONSTANT * freq * SPEED_OF_LIGHT * freq_scale_factor) /
                  (BOLTZMANN_CONSTANT * temperature) for freq in frequency_wn]

    # Error occurs if T is too low when performing math.exp
    for entry in factor:
        if entry > math.log(sys.float_info.max):
            raise ValueError(
                "Temperature may be too low to calculate vibrational energy. "
                "Please adjust using the `-t` option and try again."
            )

    energy = [entry * GAS_CONSTANT * temperature * (0.5 + (1.0 / (math.exp(entry) - 1.0)))
              for entry in factor]

    return sum(energy)


def calc_zeropoint_energy(frequency_wn, freq_scale_factor = 1.0, fract_modelsys = None):
    """
    Compute the vibrational zero-point energy for a set of vibrational modes.
    
    When `fract_modelsys` is provided, `freq_scale_factor` is expected to be a two-element sequence
    `[qm_scale, mm_scale]`; per-mode scale factors are blended using the fractions in `fract_modelsys`.
    
    Parameters:
        frequency_wn (list): Vibrational wavenumbers (cm^-1).
        freq_scale_factor (float or sequence): Global scale factor or `[qm_scale, mm_scale]` for ONIOM blending.
        fract_modelsys (list, optional): Per-mode fractions for ONIOM blending; if given, a per-mode
            scale factor is computed by blending the two entries of `freq_scale_factor`.
    
    Returns:
        float: Vibrational zero-point energy (J/mol), computed as the sum over modes of 0.5 * h * nu.
    """
    freq_scale_factor = _coerce_scale_factor(freq_scale_factor, fract_modelsys)
    if fract_modelsys is not None:
        s0, s1 = freq_scale_factor[0], freq_scale_factor[1]
        freq_scale_factor = [s0 * fm + s1 * (1.0 - fm) for fm in fract_modelsys]
        factor = [(PLANCK_CONSTANT * f * SPEED_OF_LIGHT * s) / (BOLTZMANN_CONSTANT)
                  for f, s in zip(frequency_wn, freq_scale_factor)]
    else:
        factor = [(PLANCK_CONSTANT * freq * SPEED_OF_LIGHT * freq_scale_factor) / (BOLTZMANN_CONSTANT)
                  for freq in frequency_wn]
    energy = [0.5 * entry * GAS_CONSTANT for entry in factor]
    return sum(energy)


def get_free_space(solv):
    """
    Estimate the accessible free volume in a bulk solvent (mL per L).
    
    Computes the volume fraction of a litre of solvent that is not occupied by solvent
    molecules using literature molarities and molecular volumes (Shakhnovich & Whitesides).
    This function is deprecated and experimental.
    
    Parameters:
        solv (str): Solvent name (case-sensitive) expected in the supported set.
    
    Returns:
        float: Estimated accessible free volume in milliliters per liter.
    
    Notes:
        - If `solv` is not recognized a UserWarning is emitted and the function
          returns 1000.0 (gas-phase fallback).
        - The function emits a DeprecationWarning on use.
    """
    import warnings
    warnings.warn(
        "get_free_space is experimental and no longer recommended.",
        DeprecationWarning,
        stacklevel=2,
    )
    solvent_list = list(FREESPACE_SOLVENTS)
    molarity = [55.6, 9.4, 12.9, 17.4, 12.5]  # mol/l
    molecular_vol = [27.944, 149.070, 77.442, 86.10, 97.0]  # Angstrom^3

    if solv not in solvent_list:
        warnings.warn(
            f"Solvent '{solv}' not recognized. Supported solvents: "
            f"{', '.join(solvent_list)}. Returning gas-phase free space.",
            UserWarning,
            stacklevel=2,
        )
        return 1000.0

    idx = solvent_list.index(solv)
    solv_molarity = molarity[idx]
    solv_volume = molecular_vol[idx]
    v_free = 8 * ((1E27 / (solv_molarity * AVOGADRO_CONSTANT)) ** 0.333333 - solv_volume ** 0.333333) ** 3
    freespace = v_free * solv_molarity * AVOGADRO_CONSTANT * 1E-24
    return freespace


def calc_translational_entropy(molecular_mass, conc, temperature, solvent = None):
    """
    Calculate the translational entropy for a species at a given temperature and concentration.
    
    Parameters:
        molecular_mass (float): Molecular mass in atomic mass units (amu).
        conc (float): Concentration in moles per liter (mol/L).
        temperature (float): Temperature in kelvin (K).
        solvent (str, optional): If provided, adjusts the effective free volume using the solvent name via get_free_space; if None, assumes ideal-gas volume.
    
    Returns:
        float: Translational entropy in joules per mole per kelvin (J/(mol·K)).
    """
    lmda = ((2.0 * math.pi * molecular_mass * AMU_to_KG * BOLTZMANN_CONSTANT * temperature) ** 0.5) / PLANCK_CONSTANT
    if solvent is not None:
        freespace = get_free_space(solvent)
        ndens = conc * 1000 * AVOGADRO_CONSTANT / (freespace / 1000.0)
    else:
        ndens = conc * 1000 * AVOGADRO_CONSTANT # number per m^3
    
    entropy = GAS_CONSTANT * (2.5 + math.log(lmda ** 3 / ndens))
    return entropy


def calc_electronic_entropy(multiplicity):
    """
    Compute the electronic entropy contribution from spin multiplicity.
    
    Parameters:
    	multiplicity (int): Spin multiplicity (number of degenerate electronic states).
    
    Returns:
    	float: Electronic entropy in J/(mol*K), equal to R * ln(multiplicity).
    """
    entropy = GAS_CONSTANT * (math.log(multiplicity))
    return entropy


def calc_rotational_entropy(temperature, rotemp, symmno=1, monatomic=False, linear=False):
    """
    Calculate the rotational entropy of a species at a given temperature.
    
    Parameters:
        temperature (float): Temperature in kelvin.
        rotemp (Sequence[float]): Rotational temperatures (K). For linear molecules provide at least one value; for non-linear provide three principal rotational temperatures.
        symmno (int): Molecular symmetry number (default 1).
        monatomic (bool): If True, treat as a single atom (entropy = 0).
        linear (bool): If True, treat as a linear molecule.
    
    Returns:
        float: Rotational entropy in J/(mol·K).
    """
    if monatomic:
        return 0.0

    if linear:
        if rotemp is None or len(rotemp) < 1:
            raise ValueError(
                "calc_rotational_entropy: invalid `rotemp` for linear molecule "
                "(monatomic=False, linear=True). Expected len(rotemp) >= 1."
            )
        qrot = temperature / rotemp[0]
        return GAS_CONSTANT * (math.log(qrot / symmno) + 1)

    if rotemp is None or len(rotemp) < 3:
        raise ValueError(
            "calc_rotational_entropy: invalid `rotemp` for non-linear molecule "
            "(monatomic=False, linear=False). Expected len(rotemp) >= 3."
        )
    qrot = (math.pi * temperature ** 3 / (rotemp[0] * rotemp[1] * rotemp[2])) ** 0.5
    return GAS_CONSTANT * (math.log(qrot / symmno) + 1.5)


def calc_rrho_entropy(temperature, frequency_wn, freq_scale_factor = 1.0, fract_modelsys=None):
    """
    Compute per-mode vibrational entropies using the rigid-rotor harmonic-oscillator (RRHO) model.
    
    Supports ONIOM-style blending of QM/MM scale factors when `fract_modelsys` is provided: in that case `freq_scale_factor` is expected to be a two-element iterable [qm_scale, mm_scale] and per-mode scale factors are blended by the fractions in `fract_modelsys`.
    
    Parameters:
        temperature (float): Temperature in kelvin.
        frequency_wn (iterable): Vibrational wavenumbers in cm⁻¹.
        freq_scale_factor (float or iterable): Frequency scaling factor applied to each mode, or a two-element sequence [qm_scale, mm_scale] when using ONIOM blending.
        fract_modelsys (iterable, optional): Per-mode fractions for ONIOM blending; when provided, per-mode scale = qm_scale*frac + mm_scale*(1-frac).
    
    Returns:
        List of per-mode vibrational entropies in J/(mol*K).
    """
    freq_scale_factor = _coerce_scale_factor(freq_scale_factor, fract_modelsys)
    if fract_modelsys is not None:
        s0, s1 = freq_scale_factor[0], freq_scale_factor[1]
        freq_scale_factor = [s0 * fm + s1 * (1.0 - fm) for fm in fract_modelsys]
        factor = [(PLANCK_CONSTANT * f * SPEED_OF_LIGHT * s) / (BOLTZMANN_CONSTANT * temperature)
                  for f, s in zip(frequency_wn, freq_scale_factor)]
    else:
        factor = [(PLANCK_CONSTANT * freq * SPEED_OF_LIGHT * freq_scale_factor) / (BOLTZMANN_CONSTANT * temperature)
                  for freq in frequency_wn]
    entropy = [entry * GAS_CONSTANT / (math.exp(entry) - 1) - GAS_CONSTANT * math.log(1 - math.exp(-entry))
               for entry in factor]
    return entropy


def calc_qRRHO_energy(temperature, frequency_wn, freq_scale_factor = 1.0):
    """
    Compute per-mode quasi-rigid-rotor harmonic-oscillator (qRRHO) vibrational energy terms.
    
    Parameters:
        temperature (float): Temperature in kelvin used for thermal factors.
        frequency_wn (list[float]): Vibrational frequencies in cm⁻¹.
        freq_scale_factor (float | list[float], optional): Global frequency scaling factor (or list of per-mode factors) applied to each frequency.
    
    Returns:
        list[float]: Per-mode qRRHO vibrational energy terms in J/mol.
    """
    freq_scale_factor = float(freq_scale_factor)
    factor = [PLANCK_CONSTANT * freq * SPEED_OF_LIGHT * freq_scale_factor for freq in frequency_wn]
    energy = [0.5 * AVOGADRO_CONSTANT * entry + GAS_CONSTANT * temperature * entry / BOLTZMANN_CONSTANT
              / temperature * math.exp(-entry / BOLTZMANN_CONSTANT / temperature) /
              (1 - math.exp(-entry / BOLTZMANN_CONSTANT / temperature)) for entry in factor]
    return energy


def calc_avg_moment_of_inertia(roconst):
    """
    Compute the average moment of inertia from a sequence of rotational constants.
    
    Parameters:
        roconst (list[float]): Rotational constants in gigahertz (GHz).
    
    Returns:
        float: Average moment of inertia in kilogram square meters (kg·m^2).
    
    Raises:
        ValueError: If `roconst` is empty or if the mean rotational constant is not greater than zero.
    """
    if not roconst:
        raise ValueError("roconst list cannot be empty")
    av_roconst_ghz = sum(roconst) / len(roconst)  # GHz
    if av_roconst_ghz <= 0:
        raise ValueError("Average rotational constant must be positive")
    av_roconst_hz = av_roconst_ghz * 1e9  # Hz
    return PLANCK_CONSTANT / av_roconst_hz  # kg m^2


def calc_freerot_entropy(temperature, frequency_wn, bav=GRIMME_BAV, freq_scale_factor=1.0, fract_modelsys=None):
    """
    Compute per-mode free-rotor vibrational entropies for a list of vibrational modes.
    
    Parameters:
        temperature (float): Temperature in kelvin.
        frequency_wn (Sequence[float]): Vibrational frequencies in cm^-1.
        bav (float): Reference average moment of inertia in kg·m^2 (defaults to GRIMME_BAV).
        freq_scale_factor (float or Sequence[float]): Frequency scale factor applied to modes. If ONIOM blending is used (fract_modelsys provided), this should be a two-item sequence [qm_scale, mm_scale].
        fract_modelsys (Sequence[float] or None): Per-mode ONIOM fractions (values in [0,1]) to blend QM/MM scale factors; pass None to disable ONIOM blending.
    
    Returns:
        list[float]: Per-mode free-rotor entropies in J/(mol·K).
    """
    freq_scale_factor = _coerce_scale_factor(freq_scale_factor, fract_modelsys)
    if fract_modelsys is not None:
        s0, s1 = freq_scale_factor[0], freq_scale_factor[1]
        freq_scale_factor = [s0 * fm + s1 * (1.0 - fm) for fm in fract_modelsys]
        mu = [PLANCK_CONSTANT / (8 * math.pi ** 2 * f * SPEED_OF_LIGHT * s)
              for f, s in zip(frequency_wn, freq_scale_factor)]
    else:
        mu = [PLANCK_CONSTANT / (8 * math.pi ** 2 * freq * SPEED_OF_LIGHT * freq_scale_factor) for freq in frequency_wn]
    mu_primed = [entry * bav / (entry + bav) for entry in mu]
    factor = [8 * math.pi ** 3 * entry * BOLTZMANN_CONSTANT * temperature / PLANCK_CONSTANT ** 2 for entry in mu_primed]
    entropy = [(0.5 + math.log(entry ** 0.5)) * GAS_CONSTANT for entry in factor]
    return entropy


def calc_damp(frequency_wn, freq_cutoff):
    """
    Compute per-mode damping factors for quasi-harmonic interpolation between RRHO and free-rotor entropy regimes.
    
    Parameters:
    	frequency_wn (list[float]): Vibrational frequencies in cm^-1.
    	freq_cutoff (float): Cutoff frequency in cm^-1 at which the damping factor is approximately 0.5.
    
    Returns:
    	list[float]: Damping factors (0 to 1) for each input frequency; values near 1 indicate RRHO behavior, values near 0 indicate free-rotor behavior.
    """
    alpha = 4
    damp = [1 / (1 + (freq_cutoff / entry) ** alpha) for entry in frequency_wn]
    return damp


def _apply_frequency_inversion(raw_freqs, raw_im_freqs, invert, job_type):
    """
    Decide which parsed imaginary vibrational frequencies to keep as imaginary and which to invert to positive values based on the user-specified inversion policy.
    
    Parameters:
        raw_freqs (iterable[float]): Parsed non-imaginary (positive) vibrational frequencies (cm⁻¹).
        raw_im_freqs (iterable[float]): Parsed imaginary (negative) vibrational frequencies (cm⁻¹).
        invert (None | str | numeric): Inversion policy:
            - None: leave all imaginary frequencies as imaginary.
            - 'auto': invert imaginary frequencies to positive except that for job types containing "TSFreq" the single most-negative mode is retained as imaginary.
            - numeric (or numeric string): invert imaginary frequencies whose value is greater than this numeric threshold; others remain imaginary.
        job_type (str): Job type string used to detect transition-state frequency handling (looks for substring "TSFreq").
    
    Returns:
        tuple:
            frequency_wn (list[float]): Final list of positive frequencies (original positives plus any inverted modes).
            im_frequency_wn (list[float]): Imaginary frequencies retained as negative values.
            inverted_freqs (list[float]): Imaginary frequencies that were inverted (original negative values).
    """
    im_freq_cutoff = 0.0
    frequency_wn = []
    im_frequency_wn = []
    inverted_freqs = []

    # Copy positive frequencies directly
    frequency_wn = list(raw_freqs)

    # Apply inversion policy to imaginary frequencies
    # Determine the most negative frequency for TS auto-inversion
    all_raw = list(raw_freqs) + list(raw_im_freqs)
    most_low_freq = min(all_raw) if all_raw else 0.0

    for x in raw_im_freqs:
        if x < -1 * im_freq_cutoff:
            if invert is not None:
                if invert == 'auto':
                    if "TSFreq" in job_type:
                        if x == most_low_freq:
                            im_frequency_wn.append(x)
                        else:
                            frequency_wn.append(x * -1.)
                            inverted_freqs.append(x)
                    else:
                        frequency_wn.append(x * -1.)
                        inverted_freqs.append(x)
                elif x > float(invert):
                    frequency_wn.append(x * -1.)
                    inverted_freqs.append(x)
                else:
                    im_frequency_wn.append(x)
            else:
                im_frequency_wn.append(x)

    return frequency_wn, im_frequency_wn, inverted_freqs


@dataclass(frozen=True)
class ThermoOptions:
    """Bundle of thermochemistry options for `calc_bbe.from_options`.

    Frozen so it's safe to share across worker processes (parallel
    parsing) and across multiple `calc_bbe` calls without accidental
    mutation. Field names match the v4.2 façade `compute_thermo` kwargs;
    the older `calc_bbe.__init__` parameter names are mapped internally.
    """
    QS: str = "grimme"                          # 'grimme' or 'truhlar'
    QH: bool = False                            # Head-Gordon enthalpy correction
    s_freq_cutoff: float = 100.0                # entropy cutoff (cm⁻¹)
    h_freq_cutoff: float = 100.0                # enthalpy cutoff (cm⁻¹)
    temperature: float = 298.15                 # K
    concentration: Optional[float] = None       # mol/L; None → gas-phase 1 atm
    freq_scale_factor: Optional[float] = None   # None → auto-lookup harm_fac
    zpe_scale_factor: Optional[float] = None    # None → auto-lookup zpe_fac
    solv: Optional[str] = None                  # solvent name for free-space corr
    spc: Optional[str] = None                   # 'link' or filename suffix
    invert: Optional[float] = None              # imag → real cutoff (cm⁻¹)
    symm: bool = False                          # pymsym symmetry-number correction
    mm_freq_scale_factor: Optional[float] = None
    inertia: str = "global"

    def _to_calc_bbe_kwargs(self):
        """Map ThermoOptions fields onto the calc_bbe constructor's
        (older, less consistent) parameter names. Resolves
        concentration to the gas-phase 1 atm value when not supplied,
        so calc_bbe never receives None for `conc`."""
        conc = self.concentration
        if conc is None:
            # Inline the gas-phase reference to avoid a constants import here.
            ATMOS_kPa = 101.325
            R_J_per_K_per_mol = 8.3144621
            conc = ATMOS_kPa / (R_J_per_K_per_mol * self.temperature)
        return {
            "QS": self.QS, "QH": self.QH,
            "cutoff": self.s_freq_cutoff,
            "H_FREQ_CUTOFF": self.h_freq_cutoff,
            "temp": self.temperature,
            "conc": conc,
            "scale_fac": self.freq_scale_factor,
            "solv": self.solv,
            "spc": self.spc, "invert": self.invert,
            "symm": self.symm,
            "mm_freq_scale_factor": self.mm_freq_scale_factor,
            "inertia": self.inertia,
            "zpe_scale_fac": self.zpe_scale_factor,
        }


class calc_bbe:
    """
    Compute "black box" entropy and enthalpy values along with all
    other thermochemical quantities.

    Computes H, S from partition functions, applying quasi-harmonic corrections

    Attributes:
        xyz (QCData): contains Cartesian coordinates and atom data.
        job_type (str): contains information on the type of Gaussian
            job such as ground or transition state optimization, frequency.
        roconst (list): list of parsed rotational constants from Gaussian calculations.
        program (str): program used in chemical computation.
        version_program (str): program version used in chemical computation.
        solvation_model (str): solvation model used in chemical computation.
        file (str): input chemical computation output file.
        charge (int): overall charge of molecule.
        empirical_dispersion (str): empirical dispersion model used in computation.
        multiplicity (int): multiplicity of molecule or chemical system.
        mult (int): multiplicity of molecule or chemical system.
        point_group (str): point group of molecule or chemical system used for symmetry corrections.
        sp_energy (float): single-point energy parsed from output file.
        sp_program (str): program used for single-point energy calculation.
        sp_version_program (str): version of program used for single-point energy calculation.
        sp_solvation_model (str): solvation model used for single-point energy calculation.
        sp_file (str): single-point energy calculation output file.
        sp_charge (int): overall charge of molecule in single-point energy calculation.
        sp_empirical_dispersion (str): empirical dispersion model used in single-point energy computation.
        sp_multiplicity (int): multiplicity of molecule or chemical system in single-point energy computation.
        cpu (list): days, hours, mins, secs, msecs of computation time.
        scf_energy (float): self-consistent field energy.
        frequency_wn (list): frequencies parsed from chemical computation output file.
        im_freq (list): imaginary frequencies parsed from chemical computation output file.
        inverted_freqs (list): frequencies inverted from imaginary to real numbers.
        zero_point_corr (float): thermal corrections for zero-point energy parsed from file.
        zpe (float): vibrational zero point energy computed from frequencies.
        enthalpy (float): enthalpy computed from partition functions.
        qh_enthalpy (float): enthalpy computed from partition functions, quasi-harmonic corrections applied.
        entropy (float): entropy of chemical system computed from partition functions.
        qh_entropy (float): entropy of chemical system computed from
            partition functions, quasi-harmonic corrections applied.
        gibbs_free_energy (float): Gibbs free energy of chemical system
            computed from enthalpy and entropy.
        qh_gibbs_free_energy (float): Gibbs free energy of chemical system
            computed from quasi-harmonic enthalpy and/or entropy.
        linear_warning (bool): flag for linear molecules, may be missing a rotational constant.
    """
    def __init__(self, file, QS = "grimme", QH=False, cutoff=100.0, H_FREQ_CUTOFF=100.0, temp=298.15, conc=None, scale_fac=None, solv=None, spc=None,
                 invert=None, symm=False, mm_freq_scale_factor=None, inertia='global', qcdata=None,
                 zpe_scale_fac=None, _from_options=False):
        """
        Initialize a calc_bbe instance by parsing QC output (or using provided qcdata) and computing thermochemical quantities (enthalpy, entropy, Gibbs free energy, ZPE, frequency lists, and related intermediate values).

        Parameters:
            file (str): Path to the quantum-chemistry output file used for parsing when qcdata is not supplied.
            QS (str): Vibrational entropy scheme; either "grimme" or "truhlar".
            QH (bool): If True, compute quasi-harmonic (qRRHO/qRRHO enthalpy) corrections.
            cutoff (float): Frequency cutoff (cm⁻¹) used for quasi-RRHO damping / RRQHO selection.
            H_FREQ_CUTOFF (float): Frequency cutoff (cm⁻¹) used for quasi-harmonic enthalpy damping.
            temp (float): Temperature in kelvin for all thermal calculations.
            conc (float or None): Concentration in mol/L used for translational entropy; None leaves behavior to defaults.
            scale_fac (float or list or None): Frequency scaling factor (or list for ONIOM: [qm_scale, mm_scale]).
            solv (str or None): Solvent identifier; if None or 'none', translational entropy is computed as ideal gas.
            spc (None, bool, or str): Single-point correction control. If not None and not 'link', a single-point file is sought/applied.
            invert (None, str, or numeric): Policy for handling imaginary frequencies; passed to _apply_frequency_inversion.
            symm (bool): If True, attempt a symmetry entropy correction via pymsym and add it to computed entropies.
            mm_freq_scale_factor (float or None): MM frequency scale factor for ONIOM blending; when provided, enables ONIOM blending.
            inertia (str): 'global' to use the global GRIMME_BAV moment of inertia, otherwise attempt to derive from rotational constants.
            qcdata (QCData or None): Pre-parsed QC data object; when provided, parsing of file is skipped.

        Notes:
            - On successful initialization the instance will have computed attributes including (but not limited to)
              enthalpy, qh_enthalpy, entropy, qh_entropy, gibbs_free_energy, qh_gibbs_free_energy, zpe,
              frequency_wn, im_frequency_wn, inverted_freqs, and various intermediate thermal terms.
            - If required frequency/ZPE/rotational data are missing or unparsable, thermal quantities remain unset or zeroed
              according to the module's behavior.
        """
        # The 15-arg constructor is deprecated in v5.0+; new code should use
        # `calc_bbe.from_options(qcdata_or_path, ThermoOptions(...))` or, at
        # the highest level, `goodvibes.compute_thermo(...)`. Internal
        # callers (compute_thermo, the parallel worker) set
        # `_from_options=True` to suppress the warning.
        if not _from_options:
            warnings.warn(
                "Calling calc_bbe(file, QS, QH, ...) directly is deprecated. "
                "Use calc_bbe.from_options(qcdata_or_path, ThermoOptions(...)) "
                "or goodvibes.compute_thermo(path, ...) instead. The legacy "
                "constructor will be removed in v6.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        # 1. Parse all data from file (program-agnostic), or use provided cache
        if qcdata is None:
            qcdata = parse_qcdata(file)

        # 2. Geometry data (from native parser via qcdata)
        self.xyz = qcdata
        self.atom_types = qcdata.atom_types
        self.atom_nums = qcdata.atom_nums
        self.cartesians = qcdata.cartesians

        # 3. Populate self attributes from qcdata
        self.program = qcdata.program
        self.version_program = qcdata.version_program
        self.solvation_model = qcdata.solvation_model
        self.file = qcdata.file
        self.charge = qcdata.charge
        self.empirical_dispersion = qcdata.empirical_dispersion
        self.multiplicity = qcdata.multiplicity
        self.scf_energy = qcdata.scf_energy
        self.sp_energy = qcdata.scf_energy
        self.zero_point_corr = qcdata.zero_point_corr
        self.job_type = qcdata.job_type
        self.roconst = qcdata.roconst
        self.point_group = qcdata.point_group
        self.cpu = qcdata.cpu
        self.sp_cpu = None

        molecular_mass = qcdata.molecular_mass
        symmno = qcdata.symmno
        linear_mol = 1 if qcdata.linear_mol else 0
        rotemp = qcdata.rotemp
        linear_warning = qcdata.linear_warning

        # ONIOM fract_modelsys setup
        if mm_freq_scale_factor is None:
            fract_modelsys = None
        else:
            fract_modelsys = qcdata.fract_modelsys if qcdata.fract_modelsys else []
            scale_fac = [scale_fac, mm_freq_scale_factor]

        # 4. Read any single point energies if requested.
        #
        # SPC cache: when --spc was used in a previous run that produced
        # the QCData we're now reading from, the parsed SPC numbers are
        # carried on qcdata.sp_*. Reuse them when the suffix matches so
        # `--import` doesn't re-parse the SPC file from disk.
        if spc and spc != 'link':
            cached_sp_hit = (
                qcdata is not None
                and qcdata.sp_energy is not None
                and qcdata.sp_suffix == spc
            )
            if cached_sp_hit:
                self.sp_energy = qcdata.sp_energy
                self.sp_version_program = qcdata.sp_version_program
                self.sp_solvation_model = qcdata.sp_solvation_model
                self.sp_charge = qcdata.sp_charge
                self.sp_empirical_dispersion = qcdata.sp_empirical_dispersion
                self.sp_multiplicity = qcdata.sp_multiplicity
            else:
                name, _ = os.path.splitext(file)
                sp_file = find_spc_file(name, spc)
                if sp_file is None:
                    self.sp_energy = '!'
                else:
                    try:
                        (self.sp_energy, _,
                         self.sp_version_program, self.sp_solvation_model,
                         _, self.sp_charge,
                         self.sp_empirical_dispersion,
                         self.sp_multiplicity) = parse_data(sp_file)
                        self.sp_cpu = _sp_cpu(sp_file)
                        # Persist into qcdata so a subsequent --export
                        # captures the SPC alongside the parent parse.
                        if qcdata is not None:
                            qcdata.sp_energy = self.sp_energy
                            qcdata.sp_version_program = self.sp_version_program
                            qcdata.sp_solvation_model = self.sp_solvation_model
                            qcdata.sp_charge = self.sp_charge
                            qcdata.sp_empirical_dispersion = self.sp_empirical_dispersion
                            qcdata.sp_multiplicity = self.sp_multiplicity
                            qcdata.sp_suffix = spc
                            qcdata.sp_file = os.path.abspath(sp_file)
                    except ValueError:
                        self.sp_energy = '!'
        elif qcdata is not None and not os.path.exists(file):
            # Cache-only mode: derive sp_* fields from cached QCData
            self.sp_energy = qcdata.scf_energy
            self.sp_version_program = qcdata.version_program
            self.sp_solvation_model = qcdata.solvation_model
            self.sp_charge = qcdata.charge
            self.sp_empirical_dispersion = qcdata.empirical_dispersion
            self.sp_multiplicity = qcdata.multiplicity
        else:
            (self.sp_energy, _,
             self.sp_version_program, self.sp_solvation_model,
             _, self.sp_charge,
             self.sp_empirical_dispersion,
             self.sp_multiplicity) = parse_data(file)

        # 5. Apply frequency inversion policy (user option, not file parsing)
        frequency_wn, im_frequency_wn, inverted_freqs = _apply_frequency_inversion(
            qcdata.frequency_wn, qcdata.im_frequency_wn,
            invert, self.job_type)

        self.inverted_freqs = inverted_freqs

        # Skip the calculation if unable to parse the frequencies or zpe from the output file
        if self.zero_point_corr is not None and rotemp and self.scf_energy is not None:
            cutoffs = [cutoff for freq in frequency_wn]

            # Translational and electronic contributions to the energy and entropy do not depend on frequencies
            u_trans = calc_translational_energy(temp)
            solvent = solv if solv not in (None, 'none') else None
            s_trans = calc_translational_entropy(molecular_mass, conc, temp, solvent)
            s_elec = calc_electronic_entropy(self.multiplicity)

            # Rotational and Vibrational contributions to the energy entropy.
            #
            # The Truhlar database (Alecu et al., JCTC 2010) calibrates ZPE
            # and harmonic-frequency scale factors *separately* — they
            # correct for slightly different systematic errors. When
            # `zpe_scale_fac` is supplied, ZPE uses it; the partition-function
            # frequencies (H_vib, S_vib) keep using `scale_fac`. When None,
            # ZPE falls back to `scale_fac` (preserves v4.x.0 behavior).
            effective_zpe_scale = zpe_scale_fac if zpe_scale_fac is not None else scale_fac
            if len(frequency_wn) > 0:
                zpe = calc_zeropoint_energy(frequency_wn, effective_zpe_scale, fract_modelsys)
                u_rot = calc_rotational_energy(temp, monatomic=(self.zero_point_corr == 0.0), linear=(linear_mol == 1))
                u_vib = calc_vibrational_energy(temp, frequency_wn, scale_fac, fract_modelsys)
                s_rot = calc_rotational_entropy(temp, rotemp,
                                                symmno=symmno,
                                                monatomic=(self.zero_point_corr == 0.0),
                                                linear=(linear_mol == 1))

                # Calculate harmonic entropy, free-rotor entropy and damping function for each frequency
                Svib_rrho = calc_rrho_entropy(temp, frequency_wn, scale_fac, fract_modelsys)

                if cutoff > 0.0:
                    Svib_rrqho = calc_rrho_entropy(temp, cutoffs, scale_fac, fract_modelsys)
                if inertia != "global" and len(self.roconst) > 0 and any(x != 0.0 for x in self.roconst):
                    try:
                        bav = calc_avg_moment_of_inertia(self.roconst)
                    except (ValueError, ZeroDivisionError):
                        bav = GRIMME_BAV
                else:
                    bav = GRIMME_BAV
                Svib_free_rot = calc_freerot_entropy(
                    temp, frequency_wn, bav,
                    scale_fac, fract_modelsys)
                S_damp = calc_damp(frequency_wn, cutoff)

                # check for qh
                if QH:
                    Uvib_qrrho = calc_qRRHO_energy(temp, frequency_wn, scale_fac)
                    H_damp = calc_damp(frequency_wn, H_FREQ_CUTOFF)

                # Compute entropy (cal/mol/K) using the two values and damping function
                vib_entropy = []
                vib_energy = []
                for j in range(0, len(frequency_wn)):
                    # Entropy correction
                    if QS == "grimme":
                        vib_entropy.append(Svib_rrho[j] * S_damp[j] + (1 - S_damp[j]) * Svib_free_rot[j])
                    elif QS == "truhlar":
                        if cutoff > 0.0:
                            if frequency_wn[j] > cutoff:
                                vib_entropy.append(Svib_rrho[j])
                            else:
                                vib_entropy.append(Svib_rrqho[j])
                        else:
                            vib_entropy.append(Svib_rrho[j])
                    # Enthalpy correction
                    if QH:
                        vib_energy.append(
                            H_damp[j] * Uvib_qrrho[j] +
                            (1 - H_damp[j]) * 0.5 * GAS_CONSTANT * temp)

                qh_s_vib, h_s_vib = sum(vib_entropy), sum(Svib_rrho)
                if QH:
                    qh_u_vib = sum(vib_energy)
            else:
                zpe, u_rot, u_vib, qh_u_vib, s_rot, h_s_vib, qh_s_vib = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            # Add terms (converted to au) to get Free energy - perform separately
            # for harmonic and quasi-harmonic values out of interest
            self.enthalpy = self.scf_energy + (u_trans + u_rot + u_vib + GAS_CONSTANT * temp) / J_TO_AU
            self.qh_enthalpy = 0.0
            if QH:
                self.qh_enthalpy = self.scf_energy + (u_trans + u_rot + qh_u_vib + GAS_CONSTANT * temp) / J_TO_AU
            # Single point correction replaces energy from optimization with single point value
            if spc is not None:
                try:
                    spc_correction = self.sp_energy - self.scf_energy
                    self.enthalpy += spc_correction
                    if QH:
                        self.qh_enthalpy += spc_correction
                except TypeError:
                    pass

            self.zpe = zpe / J_TO_AU
            self.entropy = (s_trans + s_rot + h_s_vib + s_elec) / J_TO_AU
            self.qh_entropy = (s_trans + s_rot + qh_s_vib + s_elec) / J_TO_AU

            # Symmetry - entropy correction for molecular symmetry
            if symm:
                sym_entropy_correction, pgroup = self.sym_correction(file.split('.')[0].replace('/', '_'))
                self.point_group = pgroup
                self.entropy += sym_entropy_correction
                self.qh_entropy += sym_entropy_correction

            # Calculate Free Energy
            if QH:
                self.gibbs_free_energy = self.enthalpy - temp * self.entropy
                self.qh_gibbs_free_energy = self.qh_enthalpy - temp * self.qh_entropy
            else:
                self.gibbs_free_energy = self.enthalpy - temp * self.entropy
                self.qh_gibbs_free_energy = self.enthalpy - temp * self.qh_entropy

        self.frequency_wn = frequency_wn
        self.im_frequency_wn = im_frequency_wn
        self.linear_warning = linear_warning

    @classmethod
    def from_options(cls, qcdata_or_path, options):
        """Construct a `calc_bbe` from a `ThermoOptions` bundle.

        Recommended v5.0+ entry point — replaces the legacy 15-argument
        positional constructor (which still works but emits a
        `DeprecationWarning`). Accepts either a parsed `QCData`
        instance (skips re-parsing) or a filesystem path (parses
        internally, like the old constructor).

        When `options.freq_scale_factor` and/or `options.zpe_scale_factor`
        are `None`, this method auto-looks them up from the file's
        level of theory via the Truhlar database (mirroring the CLI's
        behavior). Pass explicit floats to skip the lookup.
        """
        if isinstance(qcdata_or_path, str):
            file = qcdata_or_path
            qcdata = None
        else:
            qcdata = qcdata_or_path
            file = qcdata.file
        # Resolve scale factors. Three cases, matching the documented
        # CLI semantics:
        #   neither set      → auto-lookup harm_fac (freq) and zpe_fac
        #                      (zpe) from the level of theory
        #   freq set, zpe None → zpe inherits freq (v3.x back-compat:
        #                        --vscal X scales everything by X)
        #   freq None, zpe set → freq auto-lookup, zpe explicit
        #   both set         → use as-is
        if options.freq_scale_factor is None or options.zpe_scale_factor is None:
            from dataclasses import replace
            harm = options.freq_scale_factor
            zpe = options.zpe_scale_factor
            if harm is None and zpe is None:
                from .io import read_initial
                from .vib_scale_factors import canonicalize_level, scaling_data_dict
                entry = None
                try:
                    lot = read_initial(file)[0]
                    if lot and lot != "none":
                        entry = scaling_data_dict.get(canonicalize_level(lot))
                except (IOError, OSError):
                    pass
                harm = entry.harm_fac if entry is not None else 1.0
                zpe = entry.zpe_fac if entry is not None else 1.0
            elif harm is not None and zpe is None:
                # --vscal alone: ZPE inherits (matches v3.x and v4.x.0
                # behaviour where vscal was the single scale factor).
                zpe = harm
            elif harm is None and zpe is not None:
                from .io import read_initial
                from .vib_scale_factors import canonicalize_level, scaling_data_dict
                entry = None
                try:
                    lot = read_initial(file)[0]
                    if lot and lot != "none":
                        entry = scaling_data_dict.get(canonicalize_level(lot))
                except (IOError, OSError):
                    pass
                harm = entry.harm_fac if entry is not None else 1.0
            options = replace(options, freq_scale_factor=harm, zpe_scale_factor=zpe)
        return cls(
            file,
            qcdata=qcdata,
            _from_options=True,
            **options._to_calc_bbe_kwargs(),
        )

    # Get external symmetry number and point group using pymsym, if available, for symmetry corrections to entropy
    def ex_sym(self, file):
        """
        Determine the molecular point group and symmetry number using pymsym.
        
        Parameters:
            file (str): Ignored by this method; present for API compatibility.
        
        Returns:
            tuple: (symmetry_number, point_group) where `symmetry_number` is an int and `point_group` is a string.
        
        Raises:
            RuntimeError: If the `pymsym` package is not installed.
        """
        try:
            import pymsym
        except ImportError as e:
            raise RuntimeError(
                "pymsym is required for symmetry detection but is not installed. "
                "Install it with: pip install pymsym"
            ) from e
        atom_nums = np.array(self.xyz.atom_nums)
        positions = np.array(self.xyz.cartesians)
        pgroup = pymsym.get_point_group(atom_nums, positions)
        sym_num = pymsym.get_symmetry_number(atom_nums, positions)
        return sym_num, pgroup

    def sym_correction(self, file):
        """
        Compute and return the symmetry entropy correction and detected point group for the structure identified by file.
        
        Parameters:
            file (str): Filename or identifier for the structure passed to ex_sym to determine symmetry number and point group.
        
        Returns:
            tuple:
                sym_correction (float): Symmetry entropy correction converted to internal energy units (J/mol divided by J_TO_AU, i.e., same units used elsewhere in the module).
                pgroup (str): Detected molecular point-group symbol.
        """
        ex_sym, pgroup = self.ex_sym(file)
        sym_correction = (-GAS_CONSTANT * math.log(ex_sym)) / J_TO_AU
        return sym_correction, pgroup
