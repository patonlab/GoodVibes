# -*- coding: utf-8 -*-

import math
import os.path
import sys
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


# Symmetry numbers for different point groups
pg_sm = {"C1": 1, "Cs": 1, "Ci": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6, "C7": 7, "C8": 8, "D2": 4, "D3": 6,
         "D4": 8, "D5": 10, "D6": 12, "D7": 14, "D8": 16, "C2v": 2, "C3v": 3, "C4v": 4, "C5v": 5, "C6v": 6, "C7v": 7,
         "C8v": 8, "C2h": 2, "C3h": 3, "C4h": 4, "C5h": 5, "C6h": 6, "C7h": 7, "C8h": 8, "D2h": 4, "D3h": 6, "D4h": 8,
         "D5h": 10, "D6h": 12, "D7h": 14, "D8h": 16, "D2d": 4, "D3d": 6, "D4d": 8, "D5d": 10, "D6d": 12, "D7d": 14,
         "D8d": 16, "S4": 4, "S6": 6, "S8": 8, "T": 6, "Th": 12, "Td": 12, "O": 12, "Oh": 24, "Cinfv": 1, "Dinfh": 2,
         "I": 30, "Ih": 60, "Kh": 1}




def calc_translational_energy(temperature):
    """
    Translational energy evaluation

    Calculates the translational energy (J/mol) of an ideal gas.
    i.e. non-interacting molecules so molar energy = Na * atomic energy.
    This approximation applies to all energies and entropies computed within.
    Etrans = 3/2 RT!

    Parameter:
    temperature (float): temperature for calculations to be performed at (K)

    Returns:
    float: translational energy of chemical system.
    """
    energy = 1.5 * GAS_CONSTANT * temperature
    return energy


def calc_rotational_energy(temperature, monatomic=False, linear=False):
    """
    Rotational energy evaluation

    Calculates the rotational energy (J/mol)
    Erot = 0 (atomic) ; RT (linear); 3/2 RT (non-linear)

    Parameters:
    temperature (float): temperature for calculations to be performed at (K)
    monatomic (bool): True for single atoms (zero rotational energy).
    linear (bool): True for linear molecules (RT instead of 3/2 RT).

    Returns:
    float: rotational energy of chemical system.
    """
    if monatomic:
        energy = 0.0
    elif linear:
        energy = GAS_CONSTANT * temperature
    else:
        energy = 1.5 * GAS_CONSTANT * temperature
    return energy


def calc_vibrational_energy(temperature, frequency_wn, freq_scale_factor=1.0, fract_modelsys=None):
    """
    Vibrational energy evaluation.

    Calculates the vibrational energy contribution (J/mol).
    Includes ZPE (0K) and thermal contributions.
    Evib = R * Sum(0.5 hv/k + (hv/k)/(e^(hv/KT)-1))

    Parameters:
    temperature (float): temperature for calculations to be performed at K)
    frequency_wn (list): list of frequencies parsed from file (cm^-1).
    freq_scale_factor (float): frequency scaling factor based on level of theory
        and basis set used. When fract_modelsys is provided, this should be a
        2-element list [qm_scale, mm_scale].
    fract_modelsys (list or False): per-frequency ONIOM model-system fractions.
        False (default) disables ONIOM blending.

    Returns:
    float: vibrational energy of chemical system.
    """
    if temperature <= 0:
        raise ValueError(
            "Temperature must be positive for vibrational energy calculation."
        )

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
    Vibrational Zero point energy evaluation.

    Calculates the vibrational ZPE (J/mol)
    EZPE = Sum(0.5 hv/k)

    Parameters:
    frequency_wn (list): list of frequencies parsed from file (cm^-1).
    freq_scale_factor (float): frequency scaling factor based on level of theory and basis set used.
    fract_modelsys (list): MM frequency scale factors obtained from ONIOM calculations.

    Returns:
    float: zero point energy of chemical system.
    """
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
    Computed the amount of accessible free space (ml per L) in solution.

    .. deprecated::
        This function is experimental and no longer recommended for use.

    Calculates the free space in a litre of bulk solvent, based on
    Shakhnovich and Whitesides (J. Org. Chem. 1998, 63, 3821-3830).
    Free space based on accessible to a solute immersed in bulk solvent,
    i.e. this is the volume not occupied by solvent molecules, calculated using
    literature values for molarity and B3LYP/6-31G* computed molecular volumes.

    Parameter:
    solv (str): solvent used in chemical calculation.

    Returns:
    float: accessible free space in solution.
    """
    import warnings
    warnings.warn(
        "get_free_space is experimental and no longer recommended.",
        DeprecationWarning,
        stacklevel=2,
    )
    solvent_list = ["H2O", "toluene", "DMF", "AcOH", "chloroform"]
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
    Translational entropy evaluation.

    Calculates the translational entropic contribution (J/(mol*K)) of an ideal gas.
    Needs the molecular mass. Convert mass in amu to kg; conc in mol/l to number per m^3
    Strans = R(Ln(2pimkT/h^2)^3/2(1/C)) + 1 + 3/2)

    Parameters:
    molecular_mass (float): total molecular mass of chemical system (amu).
    conc (float): concentration (mol/L).
    temperature (float): temperature for calculations to be performed at (K).
    solvent (str): solvent used in chemical calculation.

    Returns:
    float: translational entropy of chemical system.
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
    Electronic entropy evaluation.

    Calculates the electronic entropic contribution (J/(mol*K)) of the molecule
    Selec = R(Ln(multiplicity)

    Parameter:
    multiplicity (int): multiplicity of chemical system.

    Returns:
    float: electronic entropy of chemical system.
    """
    entropy = GAS_CONSTANT * (math.log(multiplicity))
    return entropy


def calc_rotational_entropy(temperature, rotemp, symmno=1, monatomic=False, linear=False):
    """
    Rotational entropy evaluation.

    Calculates the rotational entropy (J/(mol*K))
    Srot = 0 (atomic) ; R(Ln(q)+1) (linear); R(Ln(q)+3/2) (non-linear)

    Parameters:
    temperature (float): temperature for calculations to be performed at (K).
    symmno (float): symmetry number of chemical system (default 1 = no symmetry).
    rotemp (list): list of parsed rotational temperatures of chemical system (K).
    monatomic (bool): True for single atoms (zero rotational entropy).
    linear (bool): True for linear molecules.

    Returns:
    float: rotational entropy of chemical system.
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
    Rigid rotor harmonic oscillator (RRHO) entropy evaluation - this is the default treatment

    Entropic contributions (J/(mol*K)) according to a rigid-rotor
    harmonic-oscillator description for a list of vibrational modes
    Sv = RSum(hv/(kT(e^(hv/kT)-1) - ln(1-e^(-hv/kT)))

    Parameters:
    frequency_wn (list): list of frequencies parsed from file.
    temperature (float): temperature for calculations to be performed at (K).
    freq_scale_factor (float): frequency scaling factor based on level of theory and basis set used.
    fract_modelsys (list): MM frequency scale factors obtained from ONIOM calculations.

    Returns:
    float: RRHO entropy of chemical system.
    """
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
    Quasi-rigid rotor harmonic oscillator energy evaluation.

    Head-Gordon RRHO-vibrational energy contribution (J/mol*K) of
    vibrational modes described by a rigid-rotor harmonic approximation.
    V_RRHO = 1/2(Nhv) + RT(hv/kT)e^(-hv/kT)/(1-e^(-hv/kT))

    Parameters:
    frequency_wn (list): list of frequencies parsed from file.
    temperature (float): temperature for calculations to be performed at.
    freq_scale_factor (float): frequency scaling factor based on level of theory and basis set used.

    Returns:
    float: quasi-RRHO energy of chemical system.
    """
    factor = [PLANCK_CONSTANT * freq * SPEED_OF_LIGHT * freq_scale_factor for freq in frequency_wn]
    energy = [0.5 * AVOGADRO_CONSTANT * entry + GAS_CONSTANT * temperature * entry / BOLTZMANN_CONSTANT
              / temperature * math.exp(-entry / BOLTZMANN_CONSTANT / temperature) /
              (1 - math.exp(-entry / BOLTZMANN_CONSTANT / temperature)) for entry in factor]
    return energy


def calc_avg_moment_of_inertia(roconst):
    """
    Compute average moment of inertia from rotational constants.

    Parameters:
    roconst (list): rotational constants (GHz).

    Returns:
    float: average moment of inertia (kg m^2).
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
    Entropic contributions (J/(mol*K)) according to a free-rotor
    description for a list of vibrational modes
    Sr = R(1/2 + 1/2ln((8pi^3u'kT/h^2))

    Parameters:
    temperature (float): temperature for calculations to be performed at (K).
    frequency_wn (list): list of frequencies parsed from file (cm^-1).
    bav (float): average moment of inertia (kg m^2). Defaults to Grimme's
        global value of 1e-44.
    freq_scale_factor (float): frequency scaling factor based on level of theory and basis set used.
    fract_modelsys (list or False): per-frequency ONIOM model-system fractions.
        False (default) disables ONIOM blending.

    Returns:
    float: free rotor entropy of chemical system.
    """
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
    Damping function for quasi-harmonic entropy interpolation.

    Compute the Head-Gordon damping factor w(v) = 1 / (1 + (v0/v)^4) used to
    interpolate between rigid-rotor harmonic oscillator (RRHO) and free-rotor
    entropy for each vibrational mode. Returns ~1 for frequencies well above
    the cutoff (RRHO regime) and ~0 for frequencies well below (free-rotor regime).

    Parameters:
    frequency_wn (list): vibrational frequencies (cm^-1).
    freq_cutoff (float): cutoff frequency (cm^-1) where the damping factor is 0.5.

    Returns:
    list: damping factor for each frequency (values between 0 and 1).
    """
    alpha = 4
    damp = [1 / (1 + (freq_cutoff / entry) ** alpha) for entry in frequency_wn]
    return damp


def _apply_frequency_inversion(raw_freqs, raw_im_freqs, invert, job_type):
    """Apply frequency inversion policy to raw parsed frequencies.

    This is a user-controlled policy decision, not file parsing.
    Raw positive frequencies stay in frequency_wn.
    Raw negative frequencies are either kept as imaginary or inverted to positive
    depending on the invert policy and job type.

    Returns:
        (frequency_wn, im_frequency_wn, inverted_freqs)
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
                 invert=None, symm=False, mm_freq_scale_factor=None, inertia='global', qcdata=None):
        
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

        # 4. Read any single point energies if requested
        if spc and spc != 'link':
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
                    self.cpu = _sp_cpu(sp_file)
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

            # Rotational and Vibrational contributions to the energy entropy
            if len(frequency_wn) > 0:
                zpe = calc_zeropoint_energy(frequency_wn, scale_fac, fract_modelsys)
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

    # Get external symmetry number and point group using pymsym, if available, for symmetry corrections to entropy
    def ex_sym(self, file):
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
        ex_sym, pgroup = self.ex_sym(file)
        sym_correction = (-GAS_CONSTANT * math.log(ex_sym)) / J_TO_AU
        return sym_correction, pgroup
