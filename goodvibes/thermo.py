''' Functions to compute various contributions to the molecular 
partition function used by goodvibes'''

# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import math
import sys
import numpy as np
from molmass import Formula

# Importing regardless of relative import
from goodvibes.utils import ATMOS, GAS_CONSTANT, PLANCK_CONSTANT, SPEED_OF_LIGHT, BOLTZMANN_CONSTANT, AVOGADRO_CONSTANT, AMU_TO_KG, GHZ_TO_K, EV_TO_H, J_TO_AU, periodictable, pg_sm

def calc_translational_energy(temperature):
    """
    Translational energy evaluation

    Calculates the translational energy (J/mol) of an ideal gas.
    i.e. non-interacting molecules so molar energy = Na * atomic energy.
    This approximation applies to all energies and entropies computed within.
    Etrans = 3/2 RT!

    Parameter:
    temperature (float): temperature for calculations to be performed at.

    Returns:
    float: translational energy of chemical system.
    """
    energy = 1.5 * GAS_CONSTANT * temperature
    return energy

def calc_rotational_energy(zpe, temperature, linear):
    """
    Rotational energy evaluation

    Calculates the rotational energy (J/mol)
    Etrans = 0 (atomic) ; RT (linear); 3/2 RT (non-linear)

    Parameters:
    zpe (float): zero point energy of chemical system.
    temperature (float): temperature for calculations to be performed at.
    linear (bool): flag for linear molecules, changes how calculation is performed.

    Returns:
    float: rotational energy of chemical system.
    """
    if zpe == 0.0:
        energy = 0.0
    elif linear == 1:
        energy = GAS_CONSTANT * temperature
    else:
        energy = 1.5 * GAS_CONSTANT * temperature
    return energy

def calc_vibrational_energy(frequency_wn, temperature, freq_scale_factor, fract_modelsys):
    """
    Vibrational energy evaluation.

    Calculates the vibrational energy contribution (J/mol).
    Includes ZPE (0K) and thermal contributions.
    Evib = R * Sum(0.5 hv/k + (hv/k)/(e^(hv/KT)-1))

    Parameters:
    frequency_wn (list): list of frequencies parsed from file.
    temperature (float): temperature for calculations to be performed at.
    freq_scale_factor (float): frequency scaling factor based on level of theory and basis set used.
    fract_modelsys (list): MM frequency scale factors obtained from ONIOM calculations.

    Returns:
    float: vibrational energy of chemical system.
    """
    if fract_modelsys is not False:
        freq_scale_factor = [freq_scale_factor[0] * fract_modelsys[i] + freq_scale_factor[1] * (1.0 - fract_modelsys[i])
                             for i in range(len(fract_modelsys))]
        factor = [(PLANCK_CONSTANT * frequency_wn[i] * SPEED_OF_LIGHT * freq_scale_factor[i]) / (BOLTZMANN_CONSTANT * temperature)
                  for i in range(len(frequency_wn))]
    else:
        factor = [(PLANCK_CONSTANT * freq * SPEED_OF_LIGHT * freq_scale_factor) / (BOLTZMANN_CONSTANT * temperature) for freq in frequency_wn]
    # Error occurs if T is too low when performing math.exp
    for entry in factor:
        if entry > math.log(sys.float_info.max):
            sys.exit("\nx  Warning! Temperature may be too low to calculate vibrational energy. Please adjust using the `-t` option and try again.\n")

    energy = [entry * GAS_CONSTANT * temperature * (0.5 + (1.0 / (math.exp(entry) - 1.0)))
              for entry in factor]

    return sum(energy)

def calc_zeropoint_energy(frequency_wn, freq_scale_factor, fract_modelsys):
    """
    Vibrational Zero point energy evaluation.

    Calculates the vibrational ZPE (J/mol)
    EZPE = Sum(0.5 hv/k)

    Parameters:
    frequency_wn (list): list of frequencies parsed from file.
    freq_scale_factor (float): frequency scaling factor based on level of theory and basis set used.
    fract_modelsys (list): MM frequency scale factors obtained from ONIOM calculations.

    Returns:
    float: zero point energy of chemical system.
    """
    if fract_modelsys is not False:
        freq_scale_factor = [freq_scale_factor[0] * fract_modelsys[i] + freq_scale_factor[1] * (1.0 - fract_modelsys[i])
                             for i in range(len(fract_modelsys))]
        factor = [(PLANCK_CONSTANT * frequency_wn[i] * SPEED_OF_LIGHT * freq_scale_factor[i]) / (BOLTZMANN_CONSTANT)
                  for i in range(len(frequency_wn))]
    else:
        factor = [(PLANCK_CONSTANT * freq * SPEED_OF_LIGHT * freq_scale_factor) / (BOLTZMANN_CONSTANT)
                  for freq in frequency_wn]
    energy = [0.5 * entry * GAS_CONSTANT for entry in factor]
    return sum(energy)

def calc_translational_entropy(molecular_mass, conc, temperature):
    """
    Translational entropy evaluation.

    Calculates the translational entropic contribution (J/(mol*K)) of an ideal gas.
    Needs the molecular mass. Convert mass in amu to kg; conc in mol/l to number per m^3
    Strans = R(Ln(2pimkT/h^2)^3/2(1/C)) + 1 + 3/2)

    Parameters:
    molecular_mass (float): total molecular mass of chemical system.
    conc (float): concentration to perform calculations at.
    temperature (float): temperature for calculations to be performed at.

    Returns:
    float: translational entropy of chemical system.
    """
    lmda = ((2.0 * math.pi * molecular_mass * AMU_TO_KG * BOLTZMANN_CONSTANT * temperature) ** 0.5) / PLANCK_CONSTANT
    ndens = conc * 1000 * AVOGADRO_CONSTANT
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

def calc_rotational_entropy(zpe, linear, symmno, rotemp, temperature):
    """
    Rotational entropy evaluation.

    Calculates the rotational entropy (J/(mol*K))
    Strans = 0 (atomic) ; R(Ln(q)+1) (linear); R(Ln(q)+3/2) (non-linear)

    Parameters:
    zpe (float): zero point energy of chemical system.
    linear (bool): flag for linear molecules.
    symmno (float): symmetry number of chemical system.
    rotemp (list): list of parsed rotational temperatures of chemical system.
    temperature (float): temperature for calculations to be performed at.

    Returns:
    float: rotational entropy of chemical system.
    """
    if rotemp == [0.0, 0.0, 0.0] or zpe == 0.0:  # Monatomic
        entropy = 0.0
    else:
        if len(rotemp) == 1:  # Diatomic or linear molecules
            linear = 1
            qrot = temperature / rotemp[0]
        elif len(rotemp) == 2:  # Possible gaussian problem with linear triatomic
            linear = 2
        else:
            qrot = math.pi * temperature ** 3 / (rotemp[0] * rotemp[1] * rotemp[2])
            qrot = qrot ** 0.5
        if linear == 1:
            entropy = GAS_CONSTANT * (math.log(qrot / symmno) + 1)
        elif linear == 2:
            entropy = 0.0
        else:
            entropy = GAS_CONSTANT * (math.log(qrot / symmno) + 1.5)
    return entropy

def calc_rrho_entropy(frequency_wn, temperature, freq_scale_factor, fract_modelsys):
    """
    Rigid rotor harmonic oscillator (RRHO) entropy evaluation - this is the default treatment

    Entropic contributions (J/(mol*K)) according to a rigid-rotor
    harmonic-oscillator description for a list of vibrational modes
    Sv = RSum(hv/(kT(e^(hv/kT)-1) - ln(1-e^(-hv/kT)))

    Parameters:
    frequency_wn (list): list of frequencies parsed from file.
    temperature (float): temperature for calculations to be performed at.
    freq_scale_factor (float): frequency scaling factor based on level of theory and basis set used.
    fract_modelsys (list): MM frequency scale factors obtained from ONIOM calculations.

    Returns:
    float: RRHO entropy of chemical system.
    """
    if fract_modelsys is not False:
        freq_scale_factor = [freq_scale_factor[0] * fract_modelsys[i] + freq_scale_factor[1] * (1.0 - fract_modelsys[i])
                             for i in range(len(fract_modelsys))]
        factor = [(PLANCK_CONSTANT * frequency_wn[i] * SPEED_OF_LIGHT * freq_scale_factor[i]) /
                  (BOLTZMANN_CONSTANT * temperature) for i in range(len(frequency_wn))]
    else:
        factor = [(PLANCK_CONSTANT * freq * SPEED_OF_LIGHT * freq_scale_factor) / (BOLTZMANN_CONSTANT * temperature)
                  for freq in frequency_wn]
    entropy = [entry * GAS_CONSTANT / (math.exp(entry) - 1) - GAS_CONSTANT * math.log(1 - math.exp(-entry))
               for entry in factor]
    return entropy

def calc_qrrho_energy(frequency_wn, temperature, freq_scale_factor):
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

def calc_freerot_entropy(frequency_wn, temperature, freq_scale_factor, fract_modelsys, inertia, roconst):
    """
    Free rotor entropy evaluation.

    Entropic contributions (J/(mol*K)) according to a free-rotor
    description for a list of vibrational modes
    Sr = R(1/2 + 1/2ln((8pi^3u'kT/h^2))

    Parameters:
    frequency_wn (list): list of frequencies parsed from file.
    temperature (float): temperature for calculations to be performed at.
    freq_scale_factor (float): frequency scaling factor based on level of theory and basis set used.
    fract_modelsys (list): MM frequency scale factors obtained from ONIOM calculations.
    inertia (str): flag for choosing global average moment of inertia for all molecules or computing individually from parsed rotational constants
    roconst (list): list of parsed rotational constants for computing the average moment of inertia.

    Returns:
    float: free rotor entropy of chemical system.
    """
    # This is the average moment of inertia used by Grimme
    if inertia == "global" or len(roconst) == 0:
        bav = 1.00e-44
    else:
        av_roconst_ghz = sum(roconst)/len(roconst)  #GHz
        av_roconst_hz = av_roconst_ghz * 1000000000 #Hz
        av_roconst_s = 1 / av_roconst_hz            #s
        av_roconst = av_roconst_s * PLANCK_CONSTANT #kg m^2
        bav = av_roconst

    if fract_modelsys is not False:
        freq_scale_factor = [freq_scale_factor[0] * fract_modelsys[i] + freq_scale_factor[1] * (1.0 - fract_modelsys[i])
                             for i in range(len(fract_modelsys))]
        mu = [PLANCK_CONSTANT / (8 * math.pi ** 2 * frequency_wn[i] * SPEED_OF_LIGHT * freq_scale_factor[i]) for i in
              range(len(frequency_wn))]
    else:
        mu = [PLANCK_CONSTANT / (8 * math.pi ** 2 * freq * SPEED_OF_LIGHT * freq_scale_factor) for freq in frequency_wn]
    mu_primed = [entry * bav / (entry + bav) for entry in mu]
    factor = [8 * math.pi ** 3 * entry * BOLTZMANN_CONSTANT * temperature / PLANCK_CONSTANT ** 2 for entry in mu_primed]
    entropy = [(0.5 + math.log(entry ** 0.5)) * GAS_CONSTANT for entry in factor]
    return entropy

def calc_damp(frequency_wn, freq_cutoff):
    """A damping function to interpolate between RRHO and free rotor vibrational entropy values"""
    alpha = 4
    damp = [1 / (1 + (freq_cutoff / entry) ** alpha) for entry in frequency_wn]
    return damp

class QrrhoThermo:
    """
    The function to compute quasi-RRHO entropy and Gibbs energy values.
    Computes H, S from partition functions, applying qhasi-harmonic corrections, COSMO-RS solvation corrections,
    considering frequency scaling factors from detected level of theory/basis set, and optionally ONIOM frequency scaling.

    Attributes:
        roconst (list): list of rotational constants from compchem calculations.
        mult (int): multiplicity of molecule or chemical system.
        point_group (str): point group of molecule or chemical system used for symmetry corrections.
        symm_no (int): symmetry number of molecule or chemical system.
        sp_energy (float): single-point energy parsed from output file.
        scf_energy (float): self-consistent field energy.
        frequency_wn (list): frequencies parsed from chemical computation output file.
        im_freq (list): imaginary frequencies parsed from chemical computation output file.
        inverted_freqs (list): frequencies inverted from imaginary to real numbers.
        zero_point_corr (float): thermal corrections for zero-point energy parsed from file.
        zpe (float): vibrational zero point energy computed from frequencies.
        enthalpy (float): enthalpy computed from partition functions.
        qh_enthalpy (float): enthalpy computed from partition functions, quasi-harmonic corrections applied.
        entropy (float): entropy of chemical system computed from partition functions.
        qh_entropy (float): entropy of chemical system computed from partition functions, quasi-harmonic corrections applied.
        gibbs_free_energy (float): Gibbs free energy of chemical system computed from enthalpy and entropy.
        qh_gibbs_free_energy (float): Gibbs free energy of chemical system computed from quasi-harmonic enthalpy and/or entropy.
        cosmo_qhg (float): quasi-harmonic Gibbs free energy with COSMO-RS correction for Gibbs free energy of solvation
        linear_warning (bool): flag for linear molecules, may be missing a rotational constant.
    """
    def __init__(self, species, qs="grimme", qh=False, s_freq_cutoff=100.0, h_freq_cutoff=100.0, temperature=298.15, conc=False, freq_scale_factor=1.0, spc=False,
                 invert=False, cosmo=None, mm_freq_scale_factor=False, inertia='global'):

        if not conc: # if the concentration is not provided, assume 1 atm
            conc = ATMOS / (GAS_CONSTANT * temperature)

        im_freq_cutoff = 0.0 # can be increased to discard low lying imaginary frequencies

        try:
            self.nbasis = species.nbasis
        except AttributeError:
            pass

        # the following molecule attributes are essential for the thermochemistry calculations
        try:
            self.name = species.name
            self.scf_energy = species.scfenergies[-1] * EV_TO_H
            self.atomnos = species.atomnos # atomic numbers
            self.mult = species.mult # molecular multiplicity
        except AttributeError:
            print(f"x  Unable to extract molecular data from {species.name}\n")

        # calculate the molecular mass and formula
        if hasattr(self, 'atomnos'):
            self.natoms = len(self.atomnos) # num. atoms
            self.atomtypes = [periodictable[at] for at in self.atomnos] # atom types
            mol_formula = ''.join(self.atomtypes)
            formula = Formula(mol_formula)
            self.monoisotopic_mass = formula.monoisotopic_mass # molecular mass

        if not hasattr(self, 'monoisotopic_mass'):
            print(f"x  Unable to compute molecular mass for {self.name}")

        # not strictly necessary for thermochemistry but required for printing
        try:
            self.charge = species.charge # molecular charge
        except AttributeError:
            self.charge = np.nan

        try:
            self.cartesians = species.atomcoords[-1] # cartesian coordinates
        except AttributeError:
            pass

        if spc is not False:
            if species.name+'_' not in spc.name:
                print(f"x  Species name mismatch: {species.name} vs {spc.name}")
            try:
                self.spc_name = spc.name
                self.sp_energy = spc.scfenergies[-1] * EV_TO_H
            except AttributeError:
                self.sp_energy = np.nan
            try:
                self.sp_nbasis = spc.nbasis
            except AttributeError:
                pass

        if not hasattr(species, 'point_group'): # inherit point group otherwise assign as C1
            try:
                self.point_group = species.metadata['symmetry_detected'].capitalize()
                self.symm_no = pg_sm.get(self.point_group)
            except KeyError:
                self.point_group = 'C1'
                self.symm_no = 1
        else:
            self.point_group = species.point_group
            self.symm_no = species.symm_no

        try: # most important attributes for thermochemistry!
            self.zpve = species.zpve # ZPE
            self.vibfreqs = species.vibfreqs # frequencies
            self.rotconsts = species.rotconsts[-1] # rotational constants
        except AttributeError:
            if self.natoms > 1: # for single atoms there are no vibrational modes to parse
                print(f"x  Unable to extract frequency information from {species.name}")

        if hasattr(self, 'rotconsts'):
            self.rotemps = [GHZ_TO_K * roconst for roconst in self.rotconsts] # rotational temperatures

        if self.point_group in ('D*h', 'C*v', 'Cinfv', 'Dinfh'):
            linear_mol = 1
            self.rotconsts = self.rotconsts[2:]
            self.rotemps = self.rotemps[2:]
        else:
            linear_mol = 0
        if mm_freq_scale_factor is False:
            fract_modelsys = False
        else:
            fract_modelsys = []
            freq_scale_factor = [freq_scale_factor, mm_freq_scale_factor]

        # separate frequencies into real and imaginary
        frequency_wn = []
        im_frequency_wn = []
        inverted_freqs = []

        if hasattr(self, 'vibfreqs'):
            for freq in self.vibfreqs:
                # Only deal with real frequencies
                if freq > 0.00:
                    frequency_wn.append(freq)
                # Check if we want to make any low lying imaginary frequencies positive
                elif freq < 1 * im_freq_cutoff:
                    if invert is not False:
                        if freq > float(invert):
                            frequency_wn.append(freq * -1.)
                            inverted_freqs.append(freq)
                        else:
                            im_frequency_wn.append(freq)
                    else:
                        im_frequency_wn.append(freq)

        # Skip the calculation if unable to parse the frequencies or zpe from the output file
        if hasattr(self, "zpve") and hasattr(self, 'atomnos'):
            # Translational and electronic contributions to the energy and entropy do not depend on frequencies
            u_trans = calc_translational_energy(temperature)
            s_trans = calc_translational_entropy(self.monoisotopic_mass, conc, temperature)
            s_elec = calc_electronic_entropy(self.mult)

            # Rotational and Vibrational contributions to the energy entropy
            if len(frequency_wn) > 0 and hasattr(self, "rotemps"):
                cutoffs = [s_freq_cutoff for freq in frequency_wn]
                zpe = calc_zeropoint_energy(frequency_wn, freq_scale_factor, fract_modelsys)
                u_rot = calc_rotational_energy(self.zpve, temperature, linear_mol)
                u_vib = calc_vibrational_energy(frequency_wn, temperature, freq_scale_factor, fract_modelsys)
                s_rot = calc_rotational_entropy(self.zpve, linear_mol, self.symm_no, self.rotemps, temperature)

                # Calculate harmonic entropy, free-rotor entropy and damping function for each frequency
                s_vib_rrho = calc_rrho_entropy(frequency_wn, temperature, freq_scale_factor, fract_modelsys)

                if s_freq_cutoff > 0.0:
                    s_vib_rrqho = calc_rrho_entropy(cutoffs, temperature, freq_scale_factor, fract_modelsys)
                s_vib_free_rot = calc_freerot_entropy(frequency_wn, temperature, freq_scale_factor, fract_modelsys, inertia, self.rotconsts[-1])
                s_damp = calc_damp(frequency_wn, s_freq_cutoff)

                # check for qh
                if qh:
                    u_vib_qrrho = calc_qrrho_energy(frequency_wn, temperature, freq_scale_factor)
                    h_damp = calc_damp(frequency_wn, h_freq_cutoff)

                # Compute entropy (cal/mol/K) using the two values and damping function
                vib_entropy = []
                vib_energy = []
                for j in range(0, len(frequency_wn)):
                    # Entropy correction
                    if qs == "grimme":
                        vib_entropy.append(s_vib_rrho[j] * s_damp[j] + (1 - s_damp[j]) * s_vib_free_rot[j])
                    elif qs == "truhlar":
                        if s_freq_cutoff > 0.0:
                            if self.vibfreqs[j] > s_freq_cutoff:
                                vib_entropy.append(s_vib_rrho[j])
                            else:
                                vib_entropy.append(s_vib_rrqho[j])
                        else:
                            vib_entropy.append(s_vib_rrho[j])
                    # Enthalpy correction
                    if qh:
                        vib_energy.append(h_damp[j] * u_vib_qrrho[j] + (1 - h_damp[j]) * 0.5 * GAS_CONSTANT * temperature)

                qh_s_vib, h_s_vib = sum(vib_entropy), sum(s_vib_rrho)
                if qh:
                    qh_u_vib = sum(vib_energy)
            else:
                zpe, u_rot, u_vib, qh_u_vib, s_rot, h_s_vib, qh_s_vib = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            # Add terms (converted to au) to get Free energy - perform separately
            # for harmonic and quasi-harmonic values out of interest
            self.enthalpy = self.scf_energy + (u_trans + u_rot + u_vib + GAS_CONSTANT * temperature) / J_TO_AU
            self.qh_enthalpy = 0.0
            if qh:
                self.qh_enthalpy = self.scf_energy + (u_trans + u_rot + qh_u_vib + GAS_CONSTANT * temperature) / J_TO_AU
            # Single point correction replaces energy from optimization with single point value
            if spc is not False:
                #try:
                #    self.enthalpy = self.enthalpy - self.scf_energy + self.sp_energy
                #except TypeError:
                #    pass
                if qh:
                    try:
                        self.qh_enthalpy = self.qh_enthalpy - self.scf_energy + self.sp_energy
                    except TypeError:
                        pass

            self.zpe = zpe / J_TO_AU
            self.entropy = (s_trans + s_rot + h_s_vib + s_elec) / J_TO_AU
            self.qh_entropy = (s_trans + s_rot + qh_s_vib + s_elec) / J_TO_AU

            # Calculate Free Energy
            if qh:
                self.gibbs_free_energy = self.enthalpy - temperature * self.entropy
                self.qh_gibbs_free_energy = self.qh_enthalpy - temperature * self.qh_entropy
            else:
                self.gibbs_free_energy = self.enthalpy - temperature * self.entropy
                self.qh_gibbs_free_energy = self.enthalpy - temperature * self.qh_entropy

            if cosmo:
                self.cosmo_qhg = self.qh_gibbs_free_energy + cosmo
            self.im_freq = []

            for freq in im_frequency_wn:
                if freq < -1 * im_freq_cutoff:
                    self.im_freq.append(freq)

        self.frequency_wn = frequency_wn

        try:
            self.ts = self.entropy * temperature
            self.qhts = self.qh_entropy * temperature
        except AttributeError:
            self.ts = np.nan
            self.qhts = np.nan

        if hasattr(self, 'sp_energy'):
            try:
                self.sp_enthalpy = self.enthalpy - self.scf_energy + self.sp_energy
                self.sp_gibbs_free_energy = self.gibbs_free_energy - self.scf_energy + self.sp_energy
                self.sp_qh_gibbs_free_energy = self.qh_gibbs_free_energy - self.scf_energy + self.sp_energy
            except TypeError:
                pass
        if not hasattr(self,'scf_energy'):
            self.scf_energy = np.nan
        if not hasattr(self,'enthalpy'):
            self.enthalpy = np.nan
        if not hasattr(self,'qh_enthalpy'):
            self.qh_enthalpy = np.nan
        if not hasattr(self,'entropy'):
            self.entropy = np.nan
        if not hasattr(self,'qh_entropy'):
            self.qh_entropy = np.nan
        if not hasattr(self,'gibbs_free_energy'):
            self.gibbs_free_energy = np.nan
        if not hasattr(self,'qh_gibbs_free_energy'):
            self.qh_gibbs_free_energy = np.nan
        if not hasattr(self, 'zpe'):
            self.zpe = np.nan
        if not hasattr(self, 'im_freq'):
            self.im_freq = []
        if not hasattr(self, 'mult'):
            self.mult = np.nan
