# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import ctypes, math, os.path, sys
import numpy as np

try:
    from pyDFTD3 import dftd3 as D3
except:
    try:
        from dftd3 import dftd3 as D3
    except:
        print('D3 import failed')

# PHYSICAL CONSTANTS                                      UNITS
GAS_CONSTANT = 8.3144621  # J / K / mol
PLANCK_CONSTANT = 6.62606957e-34  # J * s
BOLTZMANN_CONSTANT = 1.3806488e-23  # J / K
SPEED_OF_LIGHT = 2.99792458e10  # cm / s
ATMOS = 101.325  # UNIT CONVERSION
AVOGADRO_CONSTANT = 6.0221415e23  # 1 / mol
AMU_to_KG = 1.66053886E-27  # UNIT CONVERSION
J_TO_AU = 4.184 * 627.509541 * 1000.0  # UNIT CONVERSION
KCAL_TO_AU = 627.509541  # UNIT CONVERSION

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

def calc_rotational_energy(zpe, symmno, temperature, linear):
    """
    Rotational energy evaluation

    Calculates the rotational energy (J/mol)
    Etrans = 0 (atomic) ; RT (linear); 3/2 RT (non-linear)

    Parameters:
    zpe (float): zero point energy of chemical system.
    symmno (float): symmetry number, used for adding a symmetry correction.
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
    float: zerp point energy of chemical system.
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

def calc_freerot_entropy(frequency_wn, temperature, freq_scale_factor, fract_modelsys, file, inertia, roconst):
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

class calc_bbe:
    def __init__(self, file, options, ssymm=False, cosmo=None, mm_freq_scale_factor=False):
        ''' the thermochemistry calculation using quasi RRHO'''

        # Careful with single atoms!
        if file.natom == 1:
            file.symmno, file.rotemp, file.roconst, file.vibfreqs = 1, [], [], []

        #if not hasattr(file, 'rotemp'): print('\nx  Missing rotemp in', file.name)
        if not hasattr(file, 'roconst'): pass #print('x  Missing roconst in', file.name)
        else: self.roconst = file.roconst
        #if not hasattr(file, 'symmno'): print('x  Missing symmno in', file.name)
        #if not hasattr(file, 'cpu'): print('x  Missing cpu in', file.name)
        #if not hasattr(file, 'linear_mol'): print('x  Missing linear_mol in', file.name)
        #if not hasattr(file, 'vibfreqs'): print('x  Missing vibfreqs in', file.name)
        #if not hasattr(file, 'atomcoords'): print('x  Missing atomcoords in', file.name)
        #if not hasattr(file, 'point_group'): print('x  Missing point_group in', file.name)

        frequency_wn, im_frequency_wn, inverted_freqs = [], [], []

        # time taken for calcs
        if hasattr(file, 'cpu'): self.cpu = file.cpu
        else: self.cpu = [0,0,0,0,0]

        if hasattr(options, 'spc'):
            if options.spc is not False:
                # find the single point energy
                if hasattr(file, 'sp_energy'):
                    self.sp_energy = file.sp_energy
                else:
                    self.sp_energy = 0.0
                # adds the time spend doing single point calculation to total
                if hasattr(file, 'sp_cpu'):
                    self.cpu = [cpu + sp_cpu for cpu, sp_cpu in zip(self.cpu, file.sp_cpu)]

        if hasattr(file, 'vibfreqs'):
            for freq in file.vibfreqs:
                if freq > 0.00:
                    frequency_wn.append(freq)
                elif freq < 0.00:
                    if options.invert is not False:
                        if abs(freq) < abs(float(options.invert)):
                            frequency_wn.append(freq * -1.0)
                            inverted_freqs.append(freq)
                        else: im_frequency_wn.append(freq)
                    else: im_frequency_wn.append(freq)

        linear_warning = False

        if mm_freq_scale_factor is False:
            fract_modelsys = False
        else:
            fract_modelsys = []
            freq_scale_factor = [freq_scale_factor, mm_freq_scale_factor]

        self.inverted_freqs = inverted_freqs

        # Symmetry - entropy correction for molecular symmetry
        # if requested these values are obtained and override those parsed from output file(s)
        self.sym_correction, self.point_group = 0, file.point_group
        if hasattr(options, 'ssymm'):
            if options.ssymm:
                try:
                    self.symmno, self.point_group = file.ex_sym, file.ex_pgroup
                    self.sym_correction = (-GAS_CONSTANT * math.log(self.symmno)) / J_TO_AU
                except: pass

        # electronic energy term
        self.scf_energy = 0.0
        if hasattr(file, 'scfenergies'): self.scf_energy += file.scfenergies[-1]

        # Skip the calculation if unable to parse the output file
        if hasattr(file, 'molecular_mass') and hasattr(file, 'mult'):

            cutoffs = [options.S_freq_cutoff for freq in frequency_wn]

            # Translational and electronic contributions to the energy and entropy do not depend on frequencies
            u_trans = calc_translational_energy(options.temperature)
            s_trans = calc_translational_entropy(file.molecular_mass, options.conc, options.temperature, options.freespace)
            s_elec = calc_electronic_entropy(file.mult)

        else:
            u_trans, s_trans, s_elec = 0.0, 0.0, 0.0

        if hasattr(file, 'vibfreqs') and hasattr(file, 'rotemp') and hasattr(file, 'symmno') and hasattr(file, 'linear_mol'):
            # Rotational and Vibrational contributions to the energy entropy
            if len(frequency_wn) > 0:
                zpe = calc_zeropoint_energy(frequency_wn, options.freq_scale_factor, fract_modelsys)
                u_rot = calc_rotational_energy(zpe, file.symmno, options.temperature, file.linear_mol)
                u_vib = calc_vibrational_energy(frequency_wn, options.temperature, options.freq_scale_factor, fract_modelsys)
                s_rot = calc_rotational_entropy(zpe, file.linear_mol, file.symmno, file.rotemp, options.temperature)

                # Calculate harmonic entropy, free-rotor entropy and damping function for each frequency
                Svib_rrho = calc_rrho_entropy(frequency_wn, options.temperature, options.freq_scale_factor, fract_modelsys)

                if options.S_freq_cutoff > 0.0:
                    Svib_rrqho = calc_rrho_entropy(cutoffs, options.temperature, options.freq_scale_factor, fract_modelsys)
                Svib_free_rot = calc_freerot_entropy(frequency_wn, options.temperature, options.freq_scale_factor, fract_modelsys, file.int_sym, options.inertia, self.roconst)
                S_damp = calc_damp(frequency_wn, options.S_freq_cutoff)

                # check for qh
                if options.QH:
                    Uvib_qrrho = calc_qRRHO_energy(frequency_wn, options.temperature, options.freq_scale_factor)
                    H_damp = calc_damp(frequency_wn, options.H_freq_cutoff)

                # Compute entropy (cal/mol/K) using the two values and damping function
                vib_entropy, vib_energy = [], []

                for j in range(0, len(frequency_wn)):
                    # Entropy correction
                    if options.QS == "grimme":
                        vib_entropy.append(Svib_rrho[j] * S_damp[j] + (1 - S_damp[j]) * Svib_free_rot[j])
                    elif options.QS == "truhlar":
                        if options.S_freq_cutoff > 0.0:
                            if frequency_wn[j] > options.S_freq_cutoff:
                                vib_entropy.append(Svib_rrho[j])
                            else:
                                vib_entropy.append(Svib_rrqho[j])
                        else:
                            vib_entropy.append(Svib_rrho[j])
                    # Enthalpy correction
                    if options.QH:
                        vib_energy.append(H_damp[j] * Uvib_qrrho[j] + (1 - H_damp[j]) * 0.5 * GAS_CONSTANT * options.temperature)

                qh_s_vib, h_s_vib = sum(vib_entropy), sum(Svib_rrho)

                if options.QH:
                    qh_u_vib = sum(vib_energy)
            else:
                zpe, u_rot, u_vib, qh_u_vib, s_rot, h_s_vib, qh_s_vib = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            # The D3 term is added to the energy term here. If not requested then this term is zero
            # It is added to the SPC energy if defined (instead of the SCF energy)
            # computes D3 term if requested, which is then sent to calc_bbe as a correction
            if hasattr(options, 'D3') or hasattr(options, 'D3BJ'):
                if options.D3 or options.D3BJ:
                    verbose, intermolecular, pairwise, abc_term = False, False, False, False
                    s6, rs6, s8, bj_a1, bj_a2 = 0.0, 0.0, 0.0, 0.0, 0.0

                    if options.D3: damp = 'zero'
                    elif options.D3BJ: damp = 'bj'
                    if options.ATM: abc_term = True

                    try:
                        d3_calc = D3.calcD3(file, file.functional, s6, rs6, s8, bj_a1, bj_a2, damp, abc_term, intermolecular, pairwise, verbose)
                        if options.ATM: d3_term = (d3_calc.attractive_r6_vdw + d3_calc.attractive_r8_vdw + d3_calc.repulsive_abc) / KCAL_TO_AU
                        else: d3_term = (d3_calc.attractive_r6_vdw + d3_calc.attractive_r8_vdw) / KCAL_TO_AU
                    except:
                        print('   ! Dispersion Correction Failed for {}'.format(file.name))
                        d3_term = 0.0

                    if options.spc is False:
                        self.scf_energy += d3_term
                    elif hasattr(self, "sp_energy"):
                        if self.sp_energy != '!':
                            self.sp_energy += d3_term

            # Add terms (converted to au) to get Free energy - perform separately
            # for harmonic and quasi-harmonic values out of interest
            self.enthalpy = self.scf_energy + (u_trans + u_rot + u_vib + GAS_CONSTANT * options.temperature) / J_TO_AU

            if options.QH:
                self.qh_enthalpy = self.scf_energy + (u_trans + u_rot + qh_u_vib + GAS_CONSTANT * options.temperature) / J_TO_AU
            else: self.qh_enthalpy = 0.0

            # Single point correction replaces energy from optimization with single point value
            if hasattr(options, 'spc'):
                if options.spc is not False:
                    if hasattr(self, "sp_energy"):
                        if self.sp_energy != '!':
                            try:
                                self.enthalpy = self.enthalpy - self.scf_energy + self.sp_energy
                            except TypeError:
                                pass
                            if options.QH:
                                try:
                                    self.qh_enthalpy = self.qh_enthalpy - self.scf_energy + self.sp_energy
                                except TypeError:
                                    pass

            self.zpe = zpe / J_TO_AU
            self.entropy = (s_trans + s_rot + h_s_vib + s_elec) / J_TO_AU + self.sym_correction
            self.qh_entropy = (s_trans + s_rot + qh_s_vib + s_elec) / J_TO_AU + self.sym_correction

            # Calculate Free Energy
            if options.QH:
                self.gibbs_free_energy = self.enthalpy - options.temperature * self.entropy
                self.qh_gibbs_free_energy = self.qh_enthalpy - options.temperature * self.qh_entropy
            else:
                self.gibbs_free_energy = self.enthalpy - options.temperature * self.entropy
                self.qh_gibbs_free_energy = self.enthalpy - options.temperature * self.qh_entropy

            if options.cosmo:
                self.solv_qhg = self.qh_gibbs_free_energy + cosmo
            else:
                self.solv_qhg = self.qh_gibbs_free_energy

            self.im_freq = []
            for freq in im_frequency_wn:
                self.im_freq.append(freq)

        self.frequency_wn = frequency_wn
        self.im_frequency_wn = im_frequency_wn
        self.linear_warning = linear_warning
