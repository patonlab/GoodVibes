# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import ctypes, math, os.path, sys, cclib, json
import numpy as np

# Importing regardless of relative import
try:
    from .io import *
except:
    from io import *

# PHYSICAL CONSTANTS                                      UNITS
ATMOS = 101.325  # UNIT CONVERSION
GAS_CONSTANT = 8.3144621  # J / K / mol
PLANCK_CONSTANT = 6.62606957e-34  # J * s
BOLTZMANN_CONSTANT = 1.3806488e-23  # J / K
SPEED_OF_LIGHT = 2.99792458e10  # cm / s
AVOGADRO_CONSTANT = 6.0221415e23  # 1 / mol
AMU_to_KG = 1.66053886E-27  # UNIT CONVERSION
J_TO_AU = 4.184 * 627.509541 * 1000.0  # UNIT CONVERSION

# Symmetry numbers for different point groups
pg_sm = {"C1": 1, "Cs": 1, "Ci": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6, "C7": 7, "C8": 8, "D2": 4, "D3": 6,
         "D4": 8, "D5": 10, "D6": 12, "D7": 14, "D8": 16, "C2v": 2, "C3v": 3, "C4v": 4, "C5v": 5, "C6v": 6, "C7v": 7,
         "C8v": 8, "C2h": 2, "C3h": 3, "C4h": 4, "C5h": 5, "C6h": 6, "C7h": 7, "C8h": 8, "D2h": 4, "D3h": 6, "D4h": 8,
         "D5h": 10, "D6h": 12, "D7h": 14, "D8h": 16, "D2d": 4, "D3d": 6, "D4d": 8, "D5d": 10, "D6d": 12, "D7d": 14,
         "D8d": 16, "S4": 4, "S6": 6, "S8": 8, "T": 6, "Th": 12, "Td": 12, "O": 12, "Oh": 24, "Cinfv": 1, "Dinfh": 2,
         "I": 30, "Ih": 60, "Kh": 1}

def sharepath(filename):
    """
    Get absolute pathway to GoodVibes project.

    Used in finding location of compiled C files used in symmetry corrections.

    Parameter:
    filename (str): name of compiled C file, OS specific.

    Returns:
    str: absolute path on machine to compiled C file.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, 'share', filename)

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

def calc_vibrational_energy(frequency_wn, temperature, freq_scale_factor):
    """
    Vibrational energy evaluation.

    Calculates the vibrational energy contribution (J/mol).
    Includes ZPE (0K) and thermal contributions.
    Evib = R * Sum(0.5 hv/k + (hv/k)/(e^(hv/KT)-1))

    Parameters:
    frequency_wn (list): list of frequencies parsed from file.
    temperature (float): temperature for calculations to be performed at.
    freq_scale_factor (float): frequency scaling factor based on level of theory and basis set used.

    Returns:
    float: vibrational energy of chemical system.
    """

    factor = [(PLANCK_CONSTANT * freq * SPEED_OF_LIGHT * freq_scale_factor) / (BOLTZMANN_CONSTANT * temperature) for freq in frequency_wn]
    
    # Error occurs if T is too low when performing math.exp
    for entry in factor:
        if entry > math.log(sys.float_info.max):
            sys.exit("\nx  Warning! Temperature may be too low to calculate vibrational energy. Please adjust using the `-t` option and try again.\n")

    energy = [entry * GAS_CONSTANT * temperature * (0.5 + (1.0 / (math.exp(entry) - 1.0)))
              for entry in factor]

    return sum(energy)

def calc_zeropoint_energy(frequency_wn, freq_scale_factor):
    """
    Vibrational Zero point energy evaluation.

    Calculates the vibrational ZPE (J/mol)
    EZPE = Sum(0.5 hv/k)

    Parameters:
    frequency_wn (list): list of frequencies parsed from file.
    freq_scale_factor (float): frequency scaling factor based on level of theory and basis set used.

    Returns:
    float: zerp point energy of chemical system.
    """

    factor = [(PLANCK_CONSTANT * freq * SPEED_OF_LIGHT * freq_scale_factor) / (BOLTZMANN_CONSTANT)
                  for freq in frequency_wn]
    energy = [0.5 * entry * GAS_CONSTANT for entry in factor]
    return sum(energy)

def get_free_space(solv):
    """
    Computed the amount of accessible free space (ml per L) in solution.

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
    solvent_list = ["none", "H2O", "toluene", "DMF", "AcOH", "chloroform"]
    molarity = [1.0, 55.6, 9.4, 12.9, 17.4, 12.5]  # mol/l
    molecular_vol = [1.0, 27.944, 149.070, 77.442, 86.10, 97.0]  # Angstrom^3

    nsolv = 0
    for i in range(0, len(solvent_list)):
        if solv == solvent_list[i]:
            nsolv = i
    solv_molarity = molarity[nsolv]
    solv_volume = molecular_vol[nsolv]
    if nsolv > 0:
        v_free = 8 * ((1E27 / (solv_molarity * AVOGADRO_CONSTANT)) ** 0.333333 - solv_volume ** 0.333333) ** 3
        freespace = v_free * solv_molarity * AVOGADRO_CONSTANT * 1E-24
    else:
        freespace = 1000.0
    return freespace

def calc_translational_entropy(molecular_mass, conc, temperature, solv):
    """
    Translational entropy evaluation.

    Calculates the translational entropic contribution (J/(mol*K)) of an ideal gas.
    Needs the molecular mass. Convert mass in amu to kg; conc in mol/l to number per m^3
    Strans = R(Ln(2pimkT/h^2)^3/2(1/C)) + 1 + 3/2)

    Parameters:
    molecular_mass (float): total molecular mass of chemical system.
    conc (float): concentration to perform calculations at.
    temperature (float): temperature for calculations to be performed at.
    solv (str): solvent used in chemical calculation.

    Returns:
    float: translational entropy of chemical system.
    """
    lmda = ((2.0 * math.pi * molecular_mass * AMU_to_KG * BOLTZMANN_CONSTANT * temperature) ** 0.5) / PLANCK_CONSTANT
    freespace = get_free_space(solv)
    ndens = conc * 1000 * AVOGADRO_CONSTANT / (freespace / 1000.0)
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

def calc_rrho_entropy(frequency_wn, temperature, freq_scale_factor):
    """
    Rigid rotor harmonic oscillator (RRHO) entropy evaluation - this is the default treatment

    Entropic contributions (J/(mol*K)) according to a rigid-rotor
    harmonic-oscillator description for a list of vibrational modes
    Sv = RSum(hv/(kT(e^(hv/kT)-1) - ln(1-e^(-hv/kT)))

    Parameters:
    frequency_wn (list): list of frequencies parsed from file.
    temperature (float): temperature for calculations to be performed at.
    freq_scale_factor (float): frequency scaling factor based on level of theory and basis set used.

    Returns:
    float: RRHO entropy of chemical system.
    """
    
    factor = [(PLANCK_CONSTANT * freq * SPEED_OF_LIGHT * freq_scale_factor) / (BOLTZMANN_CONSTANT * temperature)
                  for freq in frequency_wn]
    entropy = [entry * GAS_CONSTANT / (math.exp(entry) - 1) - GAS_CONSTANT * math.log(1 - math.exp(-entry))
               for entry in factor]
    return entropy

def calc_qRRHO_energy(frequency_wn, temperature, freq_scale_factor):
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

def calc_freerot_entropy(frequency_wn, temperature, freq_scale_factor, file, inertia, roconst):
    """
    Free rotor entropy evaluation.

    Entropic contributions (J/(mol*K)) according to a free-rotor
    description for a list of vibrational modes
    Sr = R(1/2 + 1/2ln((8pi^3u'kT/h^2))

    Parameters:
    frequency_wn (list): list of frequencies parsed from file.
    temperature (float): temperature for calculations to be performed at.
    freq_scale_factor (float): frequency scaling factor based on level of theory and basis set used.
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

def get_energy(dict_data):
    if 'TD energy' in dict_data['properties']['energy']:
        energy_val = cclib.parser.utils.convertor(dict_data['properties']['energy']['TD energy'], "eV", "hartree")
    elif 'moller plesset' in dict_data['properties']['energy']:
        energy_val = cclib.parser.utils.convertor(dict_data['properties']['energy']['moller plesset'][-1][-1], "eV", "hartree")
    elif 'G4 energy' in dict_data['properties']['energy']:
        energy_val = cclib.parser.utils.convertor(dict_data['properties']['energy']['G4 energy'], "eV", "hartree")
    elif 'ONIOM energy' in dict_data['properties']['energy']:
        energy_val = cclib.parser.utils.convertor(dict_data['properties']['energy']['ONIOM energy'], "eV", "hartree")
    else:
        energy_val = cclib.parser.utils.convertor(dict_data['properties']['energy']['total'], "eV", "hartree")

    return energy_val

class calc_bbe:
    """
    The function to compute the "black box" entropy and enthalpy values along with all other thermochemical quantities.

    Parses energy, program version, frequencies, charge, multiplicity, solvation model, computation time.
    Computes H, S from partition functions, applying qhasi-harmonic corrections, COSMO-RS solvation corrections,
    considering frequency scaling factors from detected level of theory/basis set, and optionally ONIOM frequency scaling.

    Attributes:
        xyz (getoutData object): contains Cartesian coordinates, atom connectivity.
        job_type (str): contains information on the type of Gaussian job such as ground or transition state optimization, frequency.
        roconst (list): list of parsed rotational constants from Gaussian calculations.
        program (str): program used in chemical computation.
        file (str): input chemical computation output file.
        charge (int): overall charge of molecule.
        multiplicity (int): multiplicity of molecule or chemical system.
        mult (int): multiplicity of molecule or chemical system.
        point_group (str): point group of molecule or chemical system used for symmetry corrections.
        sp_energy (float): single-point energy parsed from output file.
        sp_program (str): program used for single-point energy calculation.
        sp_file (str): single-point energy calculation output file.
        sp_charge (int): overall charge of molecule in single-point energy calculation.
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
        qh_entropy (float): entropy of chemical system computed from partition functions, quasi-harmonic corrections applied.
        gibbs_free_energy (float): Gibbs free energy of chemical system computed from enthalpy and entropy.
        qh_gibbs_free_energy (float): Gibbs free energy of chemical system computed from quasi-harmonic enthalpy and/or entropy.
        cosmo_qhg (float): quasi-harmonic Gibbs free energy with COSMO-RS correction for Gibbs free energy of solvation
        linear_warning (bool): flag for linear molecules, may be missing a rotational constant.
    """

    def __init__(self, file, cclib_data, sp_file=None, QS='grimme', QH=False, s_freq_cutoff=100.0, H_FREQ_CUTOFF=100.0, temperature=298.15, conc=0.04087404707082671, freq_scale_factor=1.0, solv=None, spc=False,
                 invert=False, d3_correction=0.0, nosymm=False, cosmo=None, inertia='global', g4=False, noStrans=False, noEtrans=False):

        # List of frequencies and default values
        im_freq_cutoff, frequency_wn, im_frequency_wn, rotemp, roconst, linear_mol, symmno, self.cpu, inverted_freqs = 0.0, [], [], [
            0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0, 1, [0, 0, 0, 0, 0], []
        linear_warning = False

        self.xyz = getoutData(cclib_data)
        self.job_type = cclib_data['metadata']['ground or transition state']

        self.cosmo_qhg = 0.0

        # get frequency information
        all_freqs = cclib_data['vibrations']['frequencies']
        for x in all_freqs:
            # Only deal with real frequencies
            if x > 0.00:
                frequency_wn.append(x)
            # Check if we want to make any low lying imaginary frequencies positive
            elif x < -1 * im_freq_cutoff:
                if invert is not False:
                    if invert == 'auto':
                        if "TSFreq" in self.job_type:
                            if x == min(all_freqs):
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

        self.inverted_freqs = inverted_freqs

        # energy of the file and external single-point energy correction (if any)
        self.scf_energy = get_energy(cclib_data)

        if spc != False and spc != 'link':
            if sp_file:
                with open(f'{sp_file.split(".")[0]}.json') as json_file:
                    cclib_data_spc = json.load(json_file)
            else:
                name, ext = os.path.splitext(file)
                file_spc = name + '_' + spc + ext
                with open(f'{file_spc.split(".")[0]}.json') as json_file:
                    cclib_data_spc = json.load(json_file)
            
            self.sp_energy = get_energy(cclib_data_spc)

        else:
            self.sp_energy = self.scf_energy

        # Skip the calculation if unable to parse the frequencies or zpe from the output file
        if "zero-point correction" in cclib_data['properties']['energies'] and rotemp:
            cutoffs = [s_freq_cutoff for freq in frequency_wn]

            # Translational and electronic contributions to the energy and entropy do not depend on frequencies
            molecular_mass = cclib_data['properties']['molecular mass']
 
            if not noEtrans:
                u_trans = calc_translational_energy(temperature)
            else:
                u_trans = 0
            if not noStrans:
                s_trans = calc_translational_entropy(molecular_mass, conc, temperature, solv)
            else:
                s_trans = 0
            s_elec = calc_electronic_entropy(cclib_data['properties']['multiplicity'])

            # Rotational and Vibrational contributions to the energy entropy
            if len(frequency_wn) > 0:
                rotemp = cclib_data['properties']['rotational']['rotational temperatures']
                roconst = cclib_data['properties']['rotational']['rotational constants']

                # calculate new zpe with selected scaling factor
                zpe = calc_zeropoint_energy(frequency_wn, freq_scale_factor)
                u_rot = calc_rotational_energy(zpe, temperature, linear_mol)
                u_vib = calc_vibrational_energy(frequency_wn, temperature, freq_scale_factor)
                # this option uses symmetry number 1 since the symmetry correction to entropy is applied below
                s_rot = calc_rotational_entropy(zpe, linear_mol, symmno, rotemp, temperature)

                # Calculate harmonic entropy, free-rotor entropy and damping function for each frequency
                Svib_rrho = calc_rrho_entropy(frequency_wn, temperature, freq_scale_factor)

                if s_freq_cutoff > 0.0:
                    Svib_rrqho = calc_rrho_entropy(cutoffs, temperature, freq_scale_factor)
                Svib_free_rot = calc_freerot_entropy(frequency_wn, temperature, freq_scale_factor,file, inertia, roconst)
                S_damp = calc_damp(frequency_wn, s_freq_cutoff)

                # check for qh
                if QH:
                    Uvib_qrrho = calc_qRRHO_energy(frequency_wn, temperature, freq_scale_factor)
                    H_damp = calc_damp(frequency_wn, H_FREQ_CUTOFF)

                # Compute entropy (cal/mol/K) using the two values and damping function
                vib_entropy = []
                vib_energy = []
                for j in range(0, len(frequency_wn)):
                    # Entropy correction
                    if QS == "grimme":
                        vib_entropy.append(Svib_rrho[j] * S_damp[j] + (1 - S_damp[j]) * Svib_free_rot[j])
                    elif QS == "truhlar":
                        if s_freq_cutoff > 0.0:
                            if frequency_wn[j] > s_freq_cutoff:
                                vib_entropy.append(Svib_rrho[j])
                            else:
                                vib_entropy.append(Svib_rrqho[j])
                        else:
                            vib_entropy.append(Svib_rrho[j])
                    # Enthalpy correction
                    if QH:
                        vib_energy.append(H_damp[j] * Uvib_qrrho[j] + (1 - H_damp[j]) * 0.5 * GAS_CONSTANT * temperature)

                qh_s_vib, h_s_vib = sum(vib_entropy), sum(Svib_rrho)
                if QH:
                    qh_u_vib = sum(vib_energy)
            else:
                zpe, u_rot, u_vib, qh_u_vib, s_rot, h_s_vib, qh_s_vib = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            # The D3 term is added to the energy term here. If not requested then this term is zero
            # It is added to the SPC energy if defined (instead of the SCF energy)
            if spc is False:
                self.scf_energy += d3_correction
            else:
                self.sp_energy += d3_correction

            # Add terms (converted to au) to get Free energy - perform separately
            # for harmonic and quasi-harmonic values out of interest
            self.enthalpy = self.scf_energy + (u_trans + u_rot + u_vib + GAS_CONSTANT * temperature) / J_TO_AU
            self.qh_enthalpy = 0.0
            if QH:
                self.qh_enthalpy = self.scf_energy + (u_trans + u_rot + qh_u_vib + GAS_CONSTANT * temperature) / J_TO_AU
            # Single point correction replaces energy from optimization with single point value
            if spc is not False:
                try:
                    self.enthalpy = self.enthalpy - self.scf_energy + self.sp_energy
                except TypeError:
                    pass
                if QH:
                    try:
                        self.qh_enthalpy = self.qh_enthalpy - self.scf_energy + self.sp_energy
                    except TypeError:
                        pass

            self.zpe = zpe / J_TO_AU
            self.entropy = (s_trans + s_rot + h_s_vib + s_elec) / J_TO_AU
            self.qh_entropy = (s_trans + s_rot + qh_s_vib + s_elec) / J_TO_AU

            # Symmetry - entropy correction for molecular symmetry
            if not nosymm:
                sym_entropy_correction, pgroup = self.sym_correction(file.split('.')[0].replace('/', '_'))
                self.point_group = pgroup
                self.entropy += sym_entropy_correction
                self.qh_entropy += sym_entropy_correction

            # Calculate Free Energy
            if QH:
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
        self.im_frequency_wn = im_frequency_wn
        self.linear_warning = linear_warning

    # Get external symmetry number
    def ex_sym(self, file):
        coords_string = self.xyz.coords_string()
        coords = coords_string.encode('utf-8')
        c_coords = ctypes.c_char_p(coords)

        # Determine OS with sys.platform to see what compiled symmetry file to use
        platform = sys.platform
        if platform.startswith('linux'):  # linux - .so file
            path1 = sharepath('symmetry_linux.so')
            newlib = 'lib_' + file + '.so'
            path2 = sharepath(newlib)
            copy = 'cp ' + path1 + ' ' + path2
            os.popen(copy).close()
            symmetry = ctypes.CDLL(path2)
        elif platform.startswith('darwin'):  # macOS - .dylib file
            try:
                path1 = sharepath('symmetry_mac.dylib')
                newlib = 'lib_' + file + '.dylib'
                path2 = sharepath(newlib)
                copy = 'cp ' + path1 + ' ' + path2
                os.popen(copy).close()
                symmetry = ctypes.CDLL(path2)
            except:
                path1 = sharepath('symmetry_arm64.dylib')
                newlib = 'lib_' + file + '.dylib'
                path2 = sharepath(newlib)
                copy = 'cp ' + path1 + ' ' + path2
                os.popen(copy).close()
                symmetry = ctypes.CDLL(path2)
        elif platform.startswith('win'):  # windows - .dll file
            path1 = sharepath('symmetry_windows.dll')
            newlib = 'lib_' + file + '.dll'
            path2 = sharepath(newlib)
            copy = f'copy "{path1}" "{path2}"'
            os.popen(copy).close()
            symmetry = ctypes.cdll.LoadLibrary(path2)

        symmetry.symmetry.restype = ctypes.c_char_p
        pgroup = symmetry.symmetry(c_coords).decode('utf-8')
        ex_sym = pg_sm.get(pgroup)

        # Remove file
        if platform.startswith('linux'):  # linux - .so file
            remove = 'rm ' + path2
            os.popen(remove).close()
        elif platform.startswith('darwin'):  # macOS - .dylib file
            remove = 'rm ' + path2
            os.popen(remove).close()
        elif platform.startswith('win'):  # windows - .dll file
            handle = symmetry._handle
            del symmetry
            ctypes.windll.kernel32.FreeLibrary(ctypes.c_void_p(handle))
            remove = 'Del /F "' + path2 + '"'
            os.popen(remove).close()

        return ex_sym, pgroup

    def int_sym(self):
        self.xyz.get_connectivity()
        cap = [1, 9, 17]
        neighbor = [5, 6, 7, 8, 14, 15, 16]
        int_sym = 1

        for i, row in enumerate(self.xyz.connectivity):
            if self.xyz.atom_nums[i] != 6: continue
            As = np.array(self.xyz.atom_nums)[row]
            if len(As == 4):
                neighbors = [x for x in As if x in neighbor]
                caps = [x for x in As if x in cap]
                if (len(neighbors) == 1) and (len(set(caps)) == 1):
                    int_sym *= 3
        return int_sym

    def sym_correction(self, file):
        ex_sym, pgroup = self.ex_sym(file)
        int_sym = self.int_sym()
        #override int_sym
        int_sym = 1
        sym_num = ex_sym * int_sym
        sym_correction = (-GAS_CONSTANT * math.log(sym_num)) / J_TO_AU
        return sym_correction, pgroup
