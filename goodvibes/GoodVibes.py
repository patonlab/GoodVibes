
#!/usr/bin/python
from __future__ import print_function, absolute_import

#######################################################################
#                              GoodVibes.py                           #
#  Evaluation of quasi-harmonic thermochemistry from Gaussian.        #
#  Partion functions are evaluated from vibrational frequencies       #
#  and rotational temperatures from the standard output.              #
#######################################################################
#  The rigid-rotor harmonic oscillator approximation is used as       #
#  standard for all frequencies above a cut-off value. Below this,    #
#  two treatments can be applied to entropic values:                  #
#    (a) low frequencies are shifted to the cut-off value (as per     #
#    Cramer-Truhlar)                                                  #
#    (b) a free-rotor approximation is applied below the cut-off (as  #
#    per Grimme). In this approach, a damping function interpolates   #
#    between the RRHO and free-rotor entropy treatment of Svib to     #
#    avoid a discontinuity.                                           #
#  Both approaches avoid infinitely large values of Svib as wave-     #
#  numbers tend to zero. With a cut-off set to 0, the results will be #
#  identical to standard values output by the Gaussian program.       #
#######################################################################
#  Enthalpy values below the cutoff value are treated similarly to    #
#  Grimme's method (as per Head-Gordon) where below the cutoff value, #
#  a damping function is applied as the value approaches a value of   #
#  0.5RT.                                                             #
#######################################################################
#  The free energy can be evaluated for variable temperature,         #
#  concentration, vibrational scaling factor, and with a haptic       #
#  correction of the translational entropy in different solvents,     #
#  according to the amount of free space available.                   #
#######################################################################
#  A potential energy surface may be evaluated for a given set of     #
#  structures or conformers, in which case a correction to the free-  #
#  energy due to multiple conformers is applied.                      #
#  Enantiomeric excess and ddG can also be calculated to show         #
#  preference of R or S enantiomers.                                  #
#######################################################################
#  Careful checks may be applied to compare variables between         #
#  multiple files such as Gaussian version, solvation models, levels  #
#  of theory, charge and multiplicity, potential duplicate structures #
#  errors in potentail linear molecules, correct or incorrect         #
#  transition states, and empirical dispersion models.                #
#######################################################################


#######################################################################
#######  Written by:  Rob Paton, Ignacio Funes-Ardoiz  ################
#######               Guilian Luchini, Juanvi Alegre   ################
#######  Last modified:  2019                          ################
#######################################################################

import os.path
import sys
import math
import time
from datetime import datetime, timedelta
from glob import glob
from argparse import ArgumentParser

# Importing regardless of relative import
try:
    from .vib_scale_factors import scaling_data_dict, scaling_data_dict_mod, scaling_refs
except:
    from vib_scale_factors import scaling_data_dict, scaling_data_dict_mod, scaling_refs

# VERSION NUMBER
__version__ = "3.0.0"

SUPPORTED_EXTENSIONS = set(('.out', '.log'))

# PHYSICAL CONSTANTS
GAS_CONSTANT =          8.3144621                       # J / K / mol
PLANCK_CONSTANT =       6.62606957e-34                  # J * s
BOLTZMANN_CONSTANT =    1.3806488e-23                   # J / K
SPEED_OF_LIGHT =        2.99792458e10                   # cm / s
AVOGADRO_CONSTANT =     6.0221415e23                    # 1 / mol
AMU_to_KG =             1.66053886E-27                  # UNIT CONVERSION
ATMOS =                 101.325                         # UNIT CONVERSION
J_TO_AU =               4.184 * 627.509541 * 1000.0     # UNIT CONVERSION
KCAL_TO_AU =            627.509541                      # UNIT CONVERSION

# Some literature references
grimme_ref = "Grimme, S. Chem. Eur. J. 2012, 18, 9955-9964"
truhlar_ref = "Ribeiro, R. F.; Marenich, A. V.; Cramer, C. J.; Truhlar, D. G. J. Phys. Chem. B 2011, 115, 14556-14562"
head_gordon_ref = "Li, Y.; Gomes, J.; Sharada, S. M.; Bell, A. T.; Head-Gordon, M. J. Phys. Chem. C 2015, 119, 1840-1850"
goodvibes_ref = "Funes-Ardoiz, I.; Paton, R. S. (2018). GoodVibes: GoodVibes "+__version__+" http://doi.org/10.5281/zenodo.595246"

# Some useful arrays
periodictable = ["", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si",
    "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
    "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Uub", "Uut", "Uuq", "Uup", "Uuh", "Uus", "Uuo"]

def elementID(massno):
    try:
        return periodictable[massno]
    except IndexError:
        return "XX"

alphabet = 'abcdefghijklmnopqrstuvwxyz'


# Enables output to terminal and to text file
class Logger:
    def __init__(self, filein, append, csv):
        self.csv = csv
        if self.csv == False:
            suffix = 'dat'
        else:
            suffix = 'csv'
        self.log = open('{}_{}.{}'.format(filein, append, suffix), 'w' )

    def Write(self, message, thermodata=False):
        self.thermodata = thermodata
        print(message, end='')
        if self.csv == True and self.thermodata==True:
            items = message.split()
            message = ",".join(items)
            message = message + ","
        self.log.write(message)

    def Fatal(self, message):
        print(message+"\n")
        self.log.write(message + "\n"); self.Finalize()
        sys.exit(1)

    def Finalize(self):
        self.log.close()

# Enables output of optimized coordinates to a single xyz-formatted file
class XYZout:
    def __init__(self, filein, suffix, append):
        self.xyz = open('{}_{}.{}'.format(filein, append, suffix), 'w')

    def Writetext(self, message):
        self.xyz.write(message + "\n")

    def Writecoords(self, atoms, coords):
        for n, carts in enumerate(coords):
            self.xyz.write('{:>1}'.format(atoms[n]))
            for cart in carts:
                self.xyz.write('{:13.6f}'.format(cart))
            self.xyz.write('\n')

    def Finalize(self):
        self.xyz.close()


# The funtion to compute the "black box" entropy and enthalpy values (along with all other thermochemical quantities)
class calc_bbe:
    def __init__(self, file, QS, QH, S_FREQ_CUTOFF, H_FREQ_CUTOFF, temperature, conc, freq_scale_factor, solv, spc, invert):
        # List of frequencies and default values
        im_freq_cutoff, frequency_wn, im_frequency_wn, rotemp, linear_mol, link, freqloc, linkmax, symmno, self.cpu = 0.0, [], [], [0.0,0.0,0.0], 0, 0, 0, 0, 1, [0,0,0,0,0]
        linear_warning = ""
        inverted_freqs = []
        with open(file) as f:
            g_output = f.readlines()

        # read any single point energies if requested
        if spc != False and spc != 'link':
            name, ext = os.path.splitext(file)
            try:
                self.sp_energy = sp_energy(name+'_'+spc+ext)[0]
                self.cpu = sp_cpu(name+'_'+spc+ext)
            except ValueError:
                self.sp_energy = '!'; pass
        if spc == 'link':
            self.sp_energy = sp_energy(file)[0]

        #count number of links
        for line in g_output:
            # only read first link + freq not other link jobs
            if "Normal termination" in line:
                linkmax += 1
            else:
                frequency_wn = []
            if 'Frequencies --' in line:
                freqloc = linkmax

        # Iterate over output
        if freqloc == 0:
            freqloc = len(g_output)
        for line in g_output:
            # link counter
            if "Normal termination" in line:
                link += 1
                # reset frequencies if in final freq link
                if link == freqloc: frequency_wn = []
            # if spc specified will take last Energy from file, otherwise will break after freq calc
            if link > freqloc:
                break
          	# Iterate over output: look out for low frequencies
            if line.strip().startswith('Frequencies -- '):
                for i in range(2,5):
                    try:
                        x = float(line.strip().split()[i])
                        # only deal with real frequencies
                        if x > 0.00:
                            frequency_wn.append(x)
                        # check if we want to make any low lying imaginary frequencies positive
                        elif x < -1 * im_freq_cutoff:
                            if invert is not False:
                                if x > float(invert):
                                    frequency_wn.append(x * -1.)
                                    inverted_freqs.append(x)
                                else:
                                    im_frequency_wn.append(x)
                            else:
                                im_frequency_wn.append(x)
                    except IndexError:
                        pass
            # For QM calculations look for SCF energies, last one will be the optimized energy
            elif line.strip().startswith('SCF Done:'):
                self.scf_energy = float(line.strip().split()[4])
            # For Counterpoise calculations the corrected energy value will be taken
            elif line.strip().startswith('Counterpoise corrected energy'):
                self.scf_energy = float(line.strip().split()[4])
            # For MP2 calculations replace with EUMP2
            elif 'EUMP2 =' in line.strip():
                self.scf_energy = float((line.strip().split()[5]).replace('D', 'E'))
            # For ONIOM calculations use the extrapolated value rather than SCF value
            elif "ONIOM: extrapolated energy" in line.strip():
                self.scf_energy = (float(line.strip().split()[4]))
            # For Semi-empirical or Molecular Mechanics calculations
            elif "Energy= " in line.strip() and "Predicted" not in line.strip() and "Thermal" not in line.strip():
                self.scf_energy = (float(line.strip().split()[1]))
            # look for thermal corrections, paying attention to point group symmetry
            elif line.strip().startswith('Zero-point correction='):
                self.zero_point_corr = float(line.strip().split()[2])
            elif 'Multiplicity' in line.strip():
                try:
                    mult = float(line.split('=')[-1].strip().split()[0])
                except:
                    mult = float(line.split()[-1])
                self.mult = mult

            elif line.strip().startswith('Molecular mass:'):
                molecular_mass = float(line.strip().split()[2])
            elif line.strip().startswith('Rotational symmetry number'):
                symmno = int((line.strip().split()[3]).split(".")[0])
            elif line.strip().startswith('Full point group'):
                if line.strip().split()[3] == 'D*H' or line.strip().split()[3] == 'C*V':
                    linear_mol = 1
            elif line.strip().startswith('Rotational temperature '):
                rotemp = [float(line.strip().split()[3])]
            elif line.strip().startswith('Rotational temperatures'):
                try:
                    rotemp = [float(line.strip().split()[3]), float(line.strip().split()[4]), float(line.strip().split()[5])]
                except ValueError:
                    rotemp = None
                    if line.strip().find('********'):
                        linear_warning = ["Warning! Potential invalid calculation of linear molecule from Gaussian."]
                        rotemp = [float(line.strip().split()[4]), float(line.strip().split()[5])]
            if "Job cpu time" in line.strip():
                days = int(line.split()[3]) + self.cpu[0]
                hours = int(line.split()[5]) + self.cpu[1]
                mins = int(line.split()[7]) + self.cpu[2]
                secs = 0 + self.cpu[3]
                msecs = int(float(line.split()[9])*1000.0) + self.cpu[4]
                self.cpu = [days,hours,mins,secs,msecs]
        self.inverted_freqs = inverted_freqs

        # skip the calculation if unable to parse the frequencies or zpe from the output file
        if hasattr(self, "zero_point_corr") and rotemp:
            # create a list of frequencies equal to cut-off value
            cutoffs = [S_FREQ_CUTOFF for freq in frequency_wn]

            # Translational and electronic contributions to the energy and entropy do not depend on frequencies
            Utrans = calc_translational_energy(temperature)
            Strans = calc_translational_entropy(molecular_mass, conc, temperature, solv)
            Selec = calc_electronic_entropy(mult)

            # Rotational and Vibrational contributions to the energy entropy
            if len(frequency_wn) > 0:
                ZPE = calc_zeropoint_energy(frequency_wn, freq_scale_factor)
                Urot = calc_rotational_energy(self.zero_point_corr, symmno, temperature, linear_mol)
                Uvib = calc_vibrational_energy(frequency_wn, temperature, freq_scale_factor)
                Srot = calc_rotational_entropy(self.zero_point_corr, linear_mol, symmno, rotemp, temperature)

                # Calculate harmonic entropy, free-rotor entropy and damping function for each frequency
                Svib_rrho = calc_rrho_entropy(frequency_wn, temperature, freq_scale_factor)
                if S_FREQ_CUTOFF > 0.0:
                    Svib_rrqho = calc_rrho_entropy(cutoffs, temperature, 1.0)
                Svib_free_rot = calc_freerot_entropy(frequency_wn, temperature, freq_scale_factor)
                S_damp = calc_damp(frequency_wn, S_FREQ_CUTOFF)

                #check for qh
                if QH:
                    Uvib_qrrho = calc_qRRHO_energy(frequency_wn, temperature, freq_scale_factor)
                    H_damp = calc_damp(frequency_wn, H_FREQ_CUTOFF)

                # Compute entropy (cal/mol/K) using the two values and damping function
                vib_entropy = []
                vib_energy = []
                for j in range(0,len(frequency_wn)):
                    #entropy correction
                    if QS == "grimme":
                        vib_entropy.append(Svib_rrho[j] * S_damp[j] + (1-S_damp[j]) * Svib_free_rot[j])
                    elif QS == "truhlar":
                        if S_FREQ_CUTOFF > 0.0:
                            if frequency_wn[j] > S_FREQ_CUTOFF:
                                vib_entropy.append(Svib_rrho[j])
                            else:
                                vib_entropy.append(Svib_rrqho[j])
                        else:
                            vib_entropy.append(Svib_rrho[j])
                    #enthalpy correction
                    if QH:
                        vib_energy.append(H_damp[j] * Uvib_qrrho[j] + (1-H_damp[j]) * 0.5 * GAS_CONSTANT * temperature)

                qh_Svib, h_Svib = sum(vib_entropy), sum(Svib_rrho)
                if QH:
                    qh_Uvib = sum(vib_energy)
            # monatomic species have no vibrational or rotational degrees of freedom
            else:
                ZPE, Urot, Uvib, qh_Uvib, Srot, h_Svib, qh_Svib = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            # Add terms (converted to au) to get Free energy - perform separately
            # for harmonic and quasi-harmonic values out of interest
            self.enthalpy = self.scf_energy + (Utrans + Urot + Uvib + GAS_CONSTANT * temperature) / J_TO_AU
            self.qh_enthalpy = 0.0
            if QH:
                self.qh_enthalpy = self.scf_energy + (Utrans + Urot + qh_Uvib + GAS_CONSTANT * temperature) / J_TO_AU
            # single point correction replaces energy from optimization with single point value
            if hasattr(self, 'sp_energy'):
                try:
                    self.enthalpy = self.enthalpy - self.scf_energy + self.sp_energy
                except TypeError:
                    pass
                if QH:
                    try:
                        self.qh_enthalpy = self.qh_enthalpy - self.scf_energy + self.sp_energy
                    except TypeError:
                        pass
            self.zpe = ZPE / J_TO_AU
            self.entropy = (Strans + Srot + h_Svib + Selec) / J_TO_AU
            self.qh_entropy = (Strans + Srot + qh_Svib + Selec) / J_TO_AU

            #Calculate Free Energy
            if QH:
                self.gibbs_free_energy = self.enthalpy - temperature * self.entropy
                self.qh_gibbs_free_energy = self.qh_enthalpy - temperature * self.qh_entropy
            else:
                self.gibbs_free_energy = self.enthalpy - temperature * self.entropy
                self.qh_gibbs_free_energy = self.enthalpy - temperature * self.qh_entropy

            self.im_freq = []
            for freq in im_frequency_wn:
                if freq < -1 * im_freq_cutoff:
                    self.im_freq.append(freq)
        self.frequency_wn = frequency_wn
        self.im_frequency_wn = im_frequency_wn
        self.linear_warning = linear_warning


# Obtain relative thermochemistry between species and for reactions
class get_pes:
    def __init__(self, file, thermo_data, log, options):
        # defaults
        self.dec = 2
        self.units = 'kcal/mol'
        self.boltz = False

        with open(file) as f:
            data = f.readlines()
        folder, program, names, files = None, None, [], []
        for i, line in enumerate(data):
            if line.strip().find('SPECIES') > -1:
                for j, line in enumerate(data[i+1:]):
                    if line.strip().startswith('---') == True:
                        break
                    else:
                        if line.lower().strip().find('folder') > -1:
                            try:
                                folder = line.strip().replace('#','=').split("=")[1].strip()
                            except IndexError:
                                pass
                        else:
                            try:
                                n, f = (line.strip().replace(':','=').split("="))
                                # check the specified filename is also one that GoodVibes has thermochemistry for:
                                if f.find('*') == -1:
                                    match = None
                                    for key in thermo_data:
                                        if os.path.splitext(os.path.basename(key))[0] == f.strip():
                                            match = key
                                    if match:
                                        names.append(n.strip())
                                        files.append(match)
                                    else:
                                        log.Write("   Warning! "+f.strip()+' is specified in '+file+' but no thermochemistry data found\n')
                                else:
                                    match = []
                                    for key in thermo_data:
                                        if os.path.splitext(os.path.basename(key))[0].find(f.strip().strip('*')) == 0:
                                            match.append(key)
                                    if len(match) > 0:
                                       names.append(n.strip())
                                       files.append(match)
                                    else:
                                        log.Write("   Warning! "+f.strip()+' is specified in '+file+' but no thermochemistry data found\n')
                            except ValueError:
                                if len(line) > 2: 
                                    log.Write("   Warning! "+file+' input is incorrectly formatted!\n')

            if line.strip().find('FORMAT') > -1:
                for j, line in enumerate(data[i+1:]):
                    if line.strip().find('zero') > -1:
                        try:
                            zero = line.strip().replace(':','=').split("=")[1].strip()
                        except IndexError:
                            pass
                    if line.strip().find('dec') > -1:
                        try:
                            self.dec = int(line.strip().replace(':','=').split("=")[1].strip())
                        except IndexError:
                            pass
                    if line.strip().find('units') > -1:
                        try:
                            self.units = line.strip().replace(':','=').split("=")[1].strip()
                        except IndexError:
                            pass
                    if line.strip().find('boltz') > -1:
                        try:
                            self.boltz = line.strip().replace(':','=').split("=")[1].strip()
                        except IndexError:
                            pass

        if options.gconf:
            log.Write('\n   Gconf correction applied to below values using quasi-harmonic Boltzmann factors\n')

        for i in range(len(files)):
            if len(files[i]) is 1:
                files[i] = files[i][0]  
        species = dict(zip(names, files))

        self.path, self.species = [], []
        self.spc_abs, self.e_abs, self.zpe_abs, self.h_abs, self.qh_abs, self.s_abs, self.qs_abs, self.g_abs, self.qhg_abs =  [], [], [], [], [], [], [], [], []
        self.spc_zero, self.e_zero, self.zpe_zero, self.h_zero, self.qh_zero, self.ts_zero, self.qhts_zero, self.g_zero, self.qhg_zero =  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        min_conf = False
        h_conf, h_tot, s_conf, s_tot, qh_conf, qh_tot, qs_conf, qs_tot = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        zero_structures = zero.replace(' ','').split('+')
        for structure in zero_structures:
            try:
                if not isinstance(species[structure], list):
                    if hasattr(thermo_data[species[structure]], "sp_energy"):
                        self.spc_zero += thermo_data[species[structure]].sp_energy
                    self.e_zero += thermo_data[species[structure]].scf_energy
                    self.zpe_zero += thermo_data[species[structure]].zpe
                    self.h_zero += thermo_data[species[structure]].enthalpy
                    self.qh_zero += thermo_data[species[structure]].qh_enthalpy
                    self.ts_zero += thermo_data[species[structure]].entropy
                    self.g_zero += thermo_data[species[structure]].gibbs_free_energy
                    self.qhts_zero += thermo_data[species[structure]].qh_entropy
                    self.qhg_zero += thermo_data[species[structure]].qh_gibbs_free_energy
                else: #if we have a list of different kinds of structures: loop over conformers
                    g_min, boltz_sum = sys.float_info.max, 0.0
                    for conformer in species[structure]:#find minimum G, along with associated enthalpy and entropy
                        if thermo_data[conformer].qh_gibbs_free_energy <= g_min:
                            min_conf = thermo_data[conformer]
                            g_min = thermo_data[conformer].qh_gibbs_free_energy
                    for conformer in species[structure]:#get a Boltzmann sum for conformers
                        g_rel = thermo_data[conformer].qh_gibbs_free_energy - g_min
                        boltz_fac = math.exp(-g_rel*J_TO_AU/GAS_CONSTANT/options.temperature)
                        boltz_sum += boltz_fac
                    for conformer in species[structure]:#calculate relative data based on Gmin and the Boltzmann sum
                        g_rel = thermo_data[conformer].qh_gibbs_free_energy - g_min
                        boltz_fac = math.exp(-g_rel*J_TO_AU/GAS_CONSTANT/options.temperature)
                        boltz_prob = boltz_fac / boltz_sum
                        if hasattr(thermo_data[conformer], "sp_energy") and thermo_data[conformer].sp_energy is not '!':
                            self.spc_zero += thermo_data[conformer].sp_energy * boltz_prob
                        if hasattr(thermo_data[conformer], "sp_energy") and thermo_data[conformer].sp_energy is '!':
                            sys.exit("Not all files contain a SPC value, relative values will not be calculated.")
                        self.e_zero += thermo_data[conformer].scf_energy * boltz_prob
                        self.zpe_zero += thermo_data[conformer].zpe * boltz_prob
                        if options.gconf: #default calculate gconf correction for conformers
                            h_conf += thermo_data[conformer].enthalpy * boltz_prob
                            s_conf += thermo_data[conformer].entropy * boltz_prob
                            s_conf += -GAS_CONSTANT / J_TO_AU * boltz_prob * math.log(boltz_prob)

                            qh_conf += thermo_data[conformer].qh_enthalpy * boltz_prob
                            qs_conf += thermo_data[conformer].qh_entropy * boltz_prob
                            qs_conf += -GAS_CONSTANT / J_TO_AU * boltz_prob * math.log(boltz_prob)
                        else:
                            self.h_zero += thermo_data[conformer].enthalpy * boltz_prob
                            self.ts_zero += thermo_data[conformer].entropy * boltz_prob
                            self.g_zero += thermo_data[conformer].gibbs_free_energy * boltz_prob

                            self.qh_zero += thermo_data[conformer].qh_enthalpy * boltz_prob
                            self.qhts_zero += thermo_data[conformer].qh_entropy * boltz_prob
                            self.qhg_zero += thermo_data[conformer].qh_gibbs_free_energy * boltz_prob
                if options.gconf and isinstance(species[structure], list):
                    h_adj = h_conf - min_conf.enthalpy
                    h_tot = min_conf.enthalpy + h_adj
                    s_adj = s_conf - min_conf.entropy
                    s_tot = min_conf.entropy + s_adj
                    g_corr = h_tot - options.temperature * s_tot
                    self.h_zero += h_tot
                    self.ts_zero += s_tot
                    self.g_zero += g_corr

                    qh_adj = qh_conf - min_conf.qh_enthalpy
                    qh_tot = min_conf.qh_enthalpy + qh_adj
                    qs_adj = qs_conf - min_conf.qh_entropy
                    qs_tot = min_conf.qh_entropy + qs_adj
                    if options.QH:
                        qg_corr = qh_tot - options.temperature * qs_tot
                    else:
                        qg_corr = h_tot - options.temperature * qs_tot
                    self.qh_zero = qh_tot
                    self.qhts_zero = qs_tot
                    self.qhg_zero = qg_corr
            except KeyError:
                log.Write("   Warning! Structure "+structure+' has not been defined correctly as energy-zero in '+file+'\n')
                log.Write("   Make sure this structure matches one of the SPECIES defined in the same file\n")
                sys.exit("   Please edit "+file+" and try again\n")

        with open(file) as f:
            data = f.readlines()
        for i, line in enumerate(data):
            if line.strip().find('PES') > -1:
                n = 0
                for j, line in enumerate(data[i+1:]):
                    if line.strip().startswith('#') == True:
                        pass
                    elif len(line) < 2:
                        pass
                    elif line.strip().startswith('---') == True:
                        break
                    else:
                        try:
                            self.species.append([]); self.e_abs.append([]); self.spc_abs.append([]); self.zpe_abs.append([]); self.h_abs.append([])
                            self.qh_abs.append([]); self.s_abs.append([]); self.g_abs.append([]); self.qs_abs.append([]); self.qhg_abs.append([])
                            pathway, pes = line.strip().replace(':','=').split("=")
                            pes = pes.strip()
                            points = [entry.strip() for entry in pes.lstrip('[').rstrip(']').split(',')]
                            self.path.append(pathway.strip())
                            for point in points:
                                if point != '':
                                    point_structures = point.replace(' ','').split('+')
                                    e_abs, spc_abs, zpe_abs, h_abs, qh_abs, s_abs, g_abs, qs_abs, qhg_abs = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                                    qh_conf, qh_tot, qs_conf, qs_tot, h_conf, h_tot, s_conf, s_tot, g_corr, qg_corr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                                    min_conf = False
                                    try:
                                        for structure in point_structures:#loop over structures, structures are species specified
                                            if not isinstance(species[structure], list):
                                                e_abs += thermo_data[species[structure]].scf_energy
                                                if hasattr(thermo_data[species[structure]], "sp_energy"):
                                                    spc_abs += thermo_data[species[structure]].sp_energy
                                                zpe_abs += thermo_data[species[structure]].zpe
                                                h_abs += thermo_data[species[structure]].enthalpy
                                                qh_abs += thermo_data[species[structure]].qh_enthalpy
                                                s_abs += thermo_data[species[structure]].entropy
                                                g_abs += thermo_data[species[structure]].gibbs_free_energy
                                                qs_abs += thermo_data[species[structure]].qh_entropy
                                                qhg_abs += thermo_data[species[structure]].qh_gibbs_free_energy
                                            else: #if we have a list of different kinds of structures: loop over conformers
                                                g_min, boltz_sum = sys.float_info.max, 0.0
                                                for conformer in species[structure]:#find minimum G, along with associated enthalpy and entropy
                                                    if thermo_data[conformer].qh_gibbs_free_energy <= g_min:
                                                        min_conf = thermo_data[conformer]
                                                        g_min = thermo_data[conformer].qh_gibbs_free_energy
                                                for conformer in species[structure]:#get a Boltzmann sum for conformers
                                                    g_rel = thermo_data[conformer].qh_gibbs_free_energy - g_min
                                                    boltz_fac = math.exp(-g_rel*J_TO_AU/GAS_CONSTANT/options.temperature)
                                                    boltz_sum += boltz_fac
                                                for conformer in species[structure]:#calculate relative data based on Gmin and the Boltzmann sum
                                                    g_rel = thermo_data[conformer].qh_gibbs_free_energy - g_min
                                                    boltz_fac = math.exp(-g_rel*J_TO_AU/GAS_CONSTANT/options.temperature)
                                                    boltz_prob = boltz_fac / boltz_sum
                                                    if hasattr(thermo_data[conformer], "sp_energy") and thermo_data[conformer].sp_energy is not '!':
                                                        spc_abs += thermo_data[conformer].sp_energy * boltz_prob
                                                    if hasattr(thermo_data[conformer], "sp_energy") and thermo_data[conformer].sp_energy is '!':
                                                        sys.exit("\n   Not all files contain a SPC value, relative values will not be calculated.\n")
                                                    e_abs += thermo_data[conformer].scf_energy * boltz_prob
                                                    zpe_abs += thermo_data[conformer].zpe * boltz_prob
                                                    if options.gconf: #default calculate gconf correction for conformers
                                                        h_conf += thermo_data[conformer].enthalpy * boltz_prob
                                                        s_conf += thermo_data[conformer].entropy *  boltz_prob
                                                        s_conf += -GAS_CONSTANT / J_TO_AU * boltz_prob * math.log(boltz_prob)

                                                        qh_conf += thermo_data[conformer].qh_enthalpy * boltz_prob
                                                        qs_conf += thermo_data[conformer].qh_entropy * boltz_prob
                                                        qs_conf += -GAS_CONSTANT / J_TO_AU * boltz_prob * math.log(boltz_prob)
                                                    else:
                                                        h_abs += thermo_data[conformer].enthalpy * boltz_prob
                                                        s_abs += thermo_data[conformer].entropy *  boltz_prob
                                                        g_abs += thermo_data[conformer].gibbs_free_energy * boltz_prob

                                                        qh_abs += thermo_data[conformer].qh_enthalpy * boltz_prob
                                                        qs_abs += thermo_data[conformer].qh_entropy * boltz_prob
                                                        qhg_abs += thermo_data[conformer].qh_gibbs_free_energy * boltz_prob
                                                if options.gconf:
                                                    h_adj = h_conf - min_conf.enthalpy
                                                    h_tot = min_conf.enthalpy + h_adj
                                                    s_adj = s_conf - min_conf.entropy
                                                    s_tot = min_conf.entropy + s_adj
                                                    g_corr = h_tot - options.temperature * s_tot
                                                    qh_adj = qh_conf - min_conf.qh_enthalpy
                                                    qh_tot = min_conf.qh_enthalpy + qh_adj
                                                    qs_adj = qs_conf - min_conf.qh_entropy
                                                    qs_tot = min_conf.qh_entropy + qs_adj
                                                    if options.QH:
                                                        qg_corr = qh_tot - options.temperature * qs_tot
                                                    else:
                                                        qg_corr = h_tot - options.temperature * qs_tot
                                    except KeyError:
                                        log.Write("   Warning! Structure "+structure+' has not been defined correctly in '+file+'\n')
                                        sys.exit("   Please edit "+file+" and try again\n")

                                    self.species[n].append(point); self.e_abs[n].append(e_abs); self.spc_abs[n].append(spc_abs); self.zpe_abs[n].append(zpe_abs)
                                    conformers, single_structure, mix = False,False,False
                                    for structure in point_structures:
                                        if not isinstance(species[structure], list):
                                            single_structure = True
                                        else:
                                            conformers = True
                                    if conformers and single_structure:
                                        mix = True
                                    
                                    # print(point,conformers,single_structure,mix)
                                    if options.gconf and min_conf is not False:
                                        if mix:
                                            h_mix = h_tot+h_abs
                                            s_mix = s_tot+s_abs
                                            g_mix = g_corr+g_abs
                                            qh_mix = qh_tot+qh_abs
                                            qs_mix = qs_tot+qs_abs
                                            qg_mix = qg_corr+qhg_abs
                                            self.h_abs[n].append(h_mix)
                                            self.s_abs[n].append(s_mix)
                                            self.g_abs[n].append(g_mix)
                                            self.qh_abs[n].append(qh_mix)
                                            self.qs_abs[n].append(qs_mix)
                                            self.qhg_abs[n].append(qg_mix)
                                        elif conformers:
                                            self.h_abs[n].append(h_tot)
                                            self.s_abs[n].append(s_tot)
                                            self.g_abs[n].append(g_corr)
                                            self.qh_abs[n].append(qh_tot)
                                            self.qs_abs[n].append(qs_tot)
                                            self.qhg_abs[n].append(qg_corr)
                                    else:
                                        self.h_abs[n].append(h_abs)
                                        self.s_abs[n].append(s_abs)
                                        self.g_abs[n].append(g_abs)

                                        self.qh_abs[n].append(qh_abs)
                                        self.qs_abs[n].append(qs_abs)
                                        self.qhg_abs[n].append(qhg_abs)
                                else:
                                    self.species[n].append('none')
                                    self.e_abs[n].append(float('nan'))
                            n = n + 1
                        except IndexError:
                            pass


# Read molecule data from a compchem output file
class getoutData:
    def __init__(self, file):
        with open(file) as f:
            data = f.readlines()
        program = 'none'

        for line in data:
           if "Gaussian" in line:
               program = "Gaussian"
               break
           if "* O   R   C   A *" in line:
               program = "Orca"
               break

        def getATOMTYPES(self, outlines, program):
            if program == "Gaussian":
                for i, line in enumerate(outlines):
                    if "Input orientation" in line or "Standard orientation" in line:
                        self.ATOMTYPES, self.CARTESIANS, self.ATOMICTYPES, carts = [], [], [], outlines[i+5:]
                        for j, line in enumerate(carts):
                            if "-------" in line :
                                break
                            self.ATOMTYPES.append(elementID(int(line.split()[1])))
                            self.ATOMICTYPES.append(int(line.split()[2]))
                            if len(line.split()) > 5:
                                self.CARTESIANS.append([float(line.split()[3]),float(line.split()[4]),float(line.split()[5])])
                            else:
                                self.CARTESIANS.append([float(line.split()[2]),float(line.split()[3]),float(line.split()[4])])
            if program == "Orca":
                for i, line in enumerate(outlines):
                    if "*" in line and ">" in line and "xyz" in line:
                        self.ATOMTYPES, self.CARTESIANS, carts = [], [], outlines[i+1:]
                        for j, line in enumerate(carts):
                            if ">" in line and "*" in line:
                                break
                            if len(line.split()) > 5:
                                self.CARTESIANS.append([float(line.split()[3]),float(line.split()[4]),float(line.split()[5])])
                                self.ATOMTYPES.append(line.split()[2])
                            else:
                                self.CARTESIANS.append([float(line.split()[2]),float(line.split()[3]),float(line.split()[4])])
                                self.ATOMTYPES.append(line.split()[1])

        getATOMTYPES(self, data, program)


#graph a reaction profile
def graph_reaction_profile(graph_data,log,options,plt):
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches

    log.Write("\n   Graphing Reaction Profile\n")
    data,yaxis,color = {},None,None
    #get pes data
    for i, path in enumerate(graph_data.path):
        g_data = []
        zero_val = graph_data.qhg_zero
        for j, e_abs in enumerate(graph_data.e_abs[i]):
            species = graph_data.qhg_abs[i][j]
            relative = species-zero_val
            if graph_data.units == 'kJ/mol':
                formatted_g = J_TO_AU / 1000.0 * relative
            else:
                formatted_g = KCAL_TO_AU * relative # defaults to kcal/mol
            g_data.append(formatted_g)
        data[path]=g_data

    #grab any other formatting for graph
    with open(options.graph) as f:
        yaml= f.readlines()
    folder, program, names, files, label_g, dpi, dec, legend = None, None, [], [], True, False, 2, True
    for i, line in enumerate(yaml):
        if line.strip().find('FORMAT') > -1:
            for j, line in enumerate(yaml[i+1:]):
                if line.strip().find('yaxis') > -1:
                    try:
                        yaxis = line.strip().replace(':','=').split("=")[1].strip().split(',')
                    except IndexError:
                        pass
                if line.strip().find('color') > -1:
                    try:
                        colors = line.strip().replace(':','=').split("=")[1].strip().split(',')
                    except IndexError:
                        pass
                if line.strip().find('dec') > -1:
                    try:
                        dec = int(line.strip().replace(':','=').split("=")[1].strip().split(',')[0])
                    except IndexError:
                        pass
                if line.strip().find('label') > -1:
                    try:
                        label_input = line.strip().replace(':','=').split("=")[1].strip().split(',')[0].lower()
                        if label_input == 'false':
                            label_g = False
                    except IndexError:
                        pass
                if line.strip().find('dpi') > -1:
                    try:
                        dpi = int(line.strip().replace(':','=').split("=")[1].strip().split(',')[0])
                    except IndexError:
                        pass
                if line.strip().find('legend') > -1:
                    try:
                        legend_input = line.strip().replace(':','=').split("=")[1].strip().split(',')[0].lower()
                        if legend_input == 'false':
                            legend = False
                    except IndexError:
                        pass
    #do some graphing
    Path = mpath.Path
    fig, ax = plt.subplots()

    for i, path in enumerate(graph_data.path):
        for j in range(len(data[path])-1):
            if colors is not None:
                if len(colors) > 1:
                    color = colors[i]
                else:
                    color = colors[0]
            if j == 0:
                path_patch = mpatches.PathPatch(
                    Path([(j, data[path][j]), (j+0.5,data[path][j]), (j+0.5,data[path][j+1]), (j+1,data[path][j+1])],
                         [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]),
                         label=path,fc="none", transform=ax.transData,color=color)

            else:
                path_patch = mpatches.PathPatch(
                    Path([(j, data[path][j]), (j+0.5,data[path][j]), (j+0.5,data[path][j+1]), (j+1,data[path][j+1])],
                         [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]),
                         fc="none", transform=ax.transData,color=color)
            ax.add_patch(path_patch)
            plt.hlines(data[path][j],j-0.15,j+0.15)
        plt.hlines(data[path][-1],len(data[path])-1.15,len(data[path])-.85)
    
    if legend:
        plt.legend()
    if label_g:
        for i, path in enumerate(graph_data.path):
            #annotate points with energy level
            for i, point in enumerate(data[path]):
                if dec is 1:
                    ax.annotate("{:.1f}".format(point),(i,point-fig.get_figheight()*fig.dpi*0.025),horizontalalignment='center')
                else:
                    ax.annotate("{:.2f}".format(point),(i,point-fig.get_figheight()*fig.dpi*0.025),horizontalalignment='center')

    if yaxis is not None:
        ax.set_ylim(float(yaxis[0]),float(yaxis[1]))
    ax.set_ylabel(r"$G_{rel}$ (kcal / mol)")

    #label structureswas
    plt.subplots_adjust(bottom=0.1*(len(data)-1))

    ax_label = []
    xaxis_text=[]
    newax_text_list=[]

    for i, path in enumerate(graph_data.path):
        newax_text = []
        ax_label.append(path)
        for j, e_abs in enumerate(graph_data.e_abs[i]):
            if i is 0:
                xaxis_text.append(graph_data.species[i][j])
            else:
                newax_text.append(graph_data.species[i][j])
        newax_text_list.append(newax_text)

    plt.xticks(range(len(xaxis_text)),xaxis_text)
    locs,labels = plt.xticks()
    newax = []
    for i in range(len(ax_label)):
        if i > 0:
            y = ax.twiny()
            newax.append(y)

    for i in range(len(newax)):
        newax[i].set_xticks(locs)
        newax[i].set_xlim(ax.get_xlim())
        if color is not None:
            newax[i].tick_params(axis='x',colors=colors[i+1])
        newax[i].set_xticklabels(newax_text_list[i+1])
        newax[i].xaxis.set_ticks_position('bottom')
        newax[i].xaxis.set_label_position('bottom')
        newax[i].xaxis.set_ticks_position('none')
        newax[i].spines['bottom'].set_position(('outward', 15*(i+1)))
        newax[i].spines['bottom'].set_visible(False)

    ax.set_title("Reaction Profile")
    if dpi is not False:
        plt.savefig('Rxn_profile_'+options.graph.split('.')[0]+'.png', dpi=dpi)
    plt.show()


# Read solvation free energies from a COSMO-RS dat file
def COSMORSout(datfile, names):
    GSOLV = {}
    if os.path.exists(datfile):
        with open(datfile) as f:
            data = f.readlines()
    else:
        raise ValueError("File {} does not exist".format(datfile))

    for i, line in enumerate(data):
        for name in names:
            if line.find(name.split('.')[0]) > -1:
                gsolv = float(line.split(' ')[-1]) / KCAL_TO_AU
                GSOLV[name] = gsolv
    return GSOLV


# Read gaussian output for a single point energy
def sp_energy(file):
    spe, program, data, version_program, solvation_model, keyword_line, a, charge = 'none', 'none', [], '', '', '', 0, []

    if os.path.exists(os.path.splitext(file)[0]+'.log'):
        with open(os.path.splitext(file)[0]+'.log') as f:
            data = f.readlines()
    elif os.path.exists(os.path.splitext(file)[0]+'.out'):
        with open(os.path.splitext(file)[0]+'.out') as f:
            data = f.readlines()
    else:
        raise ValueError("File {} does not exist".format(file))

    for line in data:
        if "Gaussian" in line:
            program = "Gaussian"
            break
        if "* O   R   C   A *" in line:
            program = "Orca"
            break
    repeated_link1 = 0
    for line in data:
        if program == "Gaussian":
            if line.strip().startswith('SCF Done:'):
                spe = float(line.strip().split()[4])
            if line.strip().startswith('Counterpoise corrected energy'):
                spe = float(line.strip().split()[4])
            # For MP2 calculations replace with EUMP2
            if 'EUMP2 =' in line.strip():
                spe = float((line.strip().split()[5]).replace('D', 'E'))
            # For ONIOM calculations use the extrapolated value rather than SCF value
            if "ONIOM: extrapolated energy" in line.strip():
                spe = (float(line.strip().split()[4]))
            # For Semi-empirical or Molecular Mechanics calculations
            if "Energy= " in line.strip() and "Predicted" not in line.strip() and "Thermal" not in line.strip():
                spe = (float(line.strip().split()[1]))
            if "Gaussian" in line and "Revision" in line and repeated_link1 == 0:
                for i in range(len(line.strip(",").split(","))-1):
                    line.strip(",").split(",")[i]
                    version_program += line.strip(",").split(",")[i]
                    repeated_link1 = 1
                version_program = version_program[1:]
            if "Charge" in line.strip() and "Multiplicity" in line.strip():
                charge = line.strip("=").split()[2]
        if program == "Orca":
            if line.strip().startswith('FINAL SINGLE POINT ENERGY'):
                spe = float(line.strip().split()[4])
            if 'Program Version' in line.strip():
                version_program = "ORCA version " + line.split()[2]
            if "Total Charge" in line.strip() and "...." in line.strip():
                charge = line.strip("=").split()[-1]

    # Solvation model detection
    if 'Gaussian' in version_program.strip():
        for i, line in enumerate(data):
            if '#' in line.strip() and a == 0:
                for j, line in enumerate(data[i:i+10]):
                    if '--' in line.strip():
                        a = a + 1
                        break
                    if a != 0:
                        break
                    else:
                        for k in range(len(line.strip().split("\n"))):
                            line.strip().split("\n")[k]
                            keyword_line += line.strip().split("\n")[k]
        keyword_line = keyword_line.lower()
        if 'scrf' not in keyword_line.strip():
            solvation_model = "gas phase"
        else:
            start_scrf = keyword_line.strip().find('scrf') + 5
            if keyword_line[start_scrf] == "(":
                end_scrf = keyword_line.find(")",start_scrf)
                solvation_model = "scrf=" + keyword_line[start_scrf:end_scrf]
                if solvation_model[-1] != ")":
                    solvation_model = solvation_model + ")"
            else:
                start_scrf2 = keyword_line.strip().find('scrf') + 4
                if keyword_line.find(" ",start_scrf) > -1:
                    end_scrf = keyword_line.find(" ",start_scrf)
                else:
                    end_scrf = len(keyword_line)
                if keyword_line[start_scrf2] == "(":
                    solvation_model = "scrf=(" + keyword_line[start_scrf:end_scrf]
                    if solvation_model[-1] != ")":
                        solvation_model = solvation_model + ")"
                else:
                    if keyword_line.find(" ",start_scrf) > -1:
                        end_scrf = keyword_line.find(" ",start_scrf)
                    else:
                        end_scrf = len(keyword_line)
                    solvation_model = "scrf=" + keyword_line[start_scrf:end_scrf]
        #For empirical dispersion
        empirical_dispersion = ''
        if keyword_line.strip().find('empiricaldispersion') == -1 and keyword_line.strip().find('emp=') == -1 and keyword_line.strip().find('emp(') == -1:
            empirical_dispersion = "No empirical dispersion detected"
        elif keyword_line.strip().find('empiricaldispersion') > -1:
            start_empirical_dispersion = keyword_line.strip().find('empiricaldispersion') + 20
            if keyword_line[start_empirical_dispersion] == "(":
                end_empirical_dispersion = keyword_line.find(")",start_empirical_dispersion)
                empirical_dispersion = "empiricaldispersion=" + keyword_line[start_empirical_dispersion+1:end_empirical_dispersion]
                if empirical_dispersion[-1] != ")":
                    empirical_dispersion = empirical_dispersion + ")"
            else:
                start_empirical_dispersion2 = keyword_line.strip().find('empiricaldispersion') + 19
                if keyword_line.find(" ",start_empirical_dispersion) > -1:
                    end_empirical_dispersion = keyword_line.find(" ",start_empirical_dispersion)
                else:
                    end_empirical_dispersion = len(keyword_line)
                if keyword_line[start_empirical_dispersion2] == "(":
                    empirical_dispersion = "empiricaldispersion=" + keyword_line[start_empirical_dispersion:end_empirical_dispersion-1]
                else:
                    empirical_dispersion = "empiricaldispersion=" + keyword_line[start_empirical_dispersion:end_empirical_dispersion]
        elif keyword_line.strip().find('emp=') > -1:
            start_empirical_dispersion = keyword_line.strip().find('emp=') + 4
            if keyword_line[start_empirical_dispersion] == "(":
                end_empirical_dispersion = keyword_line.find(")",start_empirical_dispersion)
                empirical_dispersion = "empiricaldispersion=" + keyword_line[start_empirical_dispersion+1:end_empirical_dispersion]
            else:
                start_empirical_dispersion2 = keyword_line.strip().find('emp=') + 3
                if keyword_line.find(" ",start_empirical_dispersion) > -1:
                    end_empirical_dispersion = keyword_line.find(" ",start_empirical_dispersion)
                else:
                    end_empirical_dispersion = len(keyword_line)
                if keyword_line[start_empirical_dispersion2] == "(":
                    empirical_dispersion2 = "empiricaldispersion=(" + keyword_line[start_empirical_dispersion:end_empirical_dispersion]
                else:
                    empirical_dispersion = "empiricaldispersion=" + keyword_line[start_empirical_dispersion:end_empirical_dispersion]
        elif keyword_line.strip().find('emp(') > -1:
            start_empirical_dispersion = keyword_line.strip().find('emp(') + 3
            end_empirical_dispersion = keyword_line.find(")",start_empirical_dispersion)
            empirical_dispersion = "empiricaldispersion=" + keyword_line[start_empirical_dispersion+1:end_empirical_dispersion]


    if 'ORCA' in version_program.strip():
        keyword_line_1 = "gas phase"
        keyword_line_2 = ''
        keyword_line_3 = ''
        for i, line in enumerate(data):
            if 'CPCM SOLVATION MODEL' in line.strip():
                keyword_line_1 = "CPCM,"
            if 'SMD CDS free energy correction energy' in line.strip():
                keyword_line_2 = "SMD,"
            if "Solvent:              " in line.strip():
                keyword_line_3 = line.strip().split()[-1]
        solvation_model = keyword_line_1 + keyword_line_2 + keyword_line_3
        empirical_dispersion1 = 'No empirical dispersion detected'
        empirical_dispersion2 = ''
        empirical_dispersion3 = ''
        for i, line in enumerate(data):
            if keyword_line.strip().find('DFT DISPERSION CORRECTION') > -1:
                empirical_dispersion1 = ''
            if keyword_line.strip().find('DFTD3') > -1:
                empirical_dispersion2 = "D3"
            if keyword_line.strip().find('USING zero damping') > -1:
                empirical_dispersion3 = ' with zero damping'
        empirical_dispersion = empirical_dispersion1 + empirical_dispersion2 + empirical_dispersion3


    return spe, program, version_program, solvation_model, file, charge, empirical_dispersion


# Read single-point output for cpu time
def sp_cpu(file):
    spe, program, data, cpu = None, None, [], None

    if os.path.exists(os.path.splitext(file)[0]+'.log'):
        with open(os.path.splitext(file)[0]+'.log') as f:
            data = f.readlines()
    elif os.path.exists(os.path.splitext(file)[0]+'.out'):
        with open(os.path.splitext(file)[0]+'.out') as f:
            data = f.readlines()
    else:
        raise ValueError("File {} does not exist".format(file))

    for line in data:
        if line.find("Gaussian") > -1:
            program = "Gaussian"
            break
        if line.find("* O   R   C   A *") > -1:
            program = "Orca"
            break

    for line in data:
        if program == "Gaussian":
            if line.strip().startswith('SCF Done:'):
                spe = float(line.strip().split()[4])
            if line.strip().find("Job cpu time") > -1:
                days = int(line.split()[3])
                hours = int(line.split()[5])
                mins = int(line.split()[7])
                secs = 0
                msecs = int(float(line.split()[9])*1000.0)
                cpu = [days,hours,mins,secs,msecs]
        if program == "Orca":
            if line.strip().startswith('FINAL SINGLE POINT ENERGY'):
                spe = float(line.strip().split()[4])
            if line.strip().find("TOTAL RUN TIME") > -1:
                days = int(line.split()[3])
                hours = int(line.split()[5])
                mins = int(line.split()[7])
                secs = int(line.split()[9])
                msecs = float(line.split()[11])
                cpu = [days,hours,mins,secs,msecs]

    return cpu


# Read output for the level of theory and basis set used
def level_of_theory(file):
    repeated_theory = 0
    with open(file) as f:
        data = f.readlines()
    level, bs = 'none', 'none'
    for line in data:
        if line.strip().find('External calculation') > -1:
            level, bs = 'ext', 'ext'
            break
        if '\\Freq\\' in line.strip() and repeated_theory == 0:
            try:
                level, bs = (line.strip().split("\\")[4:6])
                repeated_theory = 1
            except IndexError:
                pass
        elif '|Freq|' in line.strip() and repeated_theory == 0:
            try:
                level, bs = (line.strip().split("|")[4:6])
                repeated_theory = 1
            except IndexError:
                pass
        if '\\SP\\' in line.strip() and repeated_theory == 0:
            try:
                level, bs = (line.strip().split("\\")[4:6])
                repeated_theory = 1
            except IndexError:
                pass
        elif '|SP|' in line.strip() and repeated_theory == 0:
            try:
                level, bs = (line.strip().split("|")[4:6])
                repeated_theory = 1
            except IndexError:
                pass

    for line in data:
        if 'DLPNO BASED TRIPLES CORRECTION' in line.strip():
            level = 'DLPNO-CCSD(T)'
        if 'Estimated CBS total energy' in line.strip():
            try:
                bs = ("Extrapol."+line.strip().split()[4])
            except IndexError:
                pass
        # remove the restricted R or unrestricted U label
        if level[0] in ('R', 'U'):
            level = level[1:]

    return '/'.join([level, bs])


# Calculate elapsed time
def addTime(tm, cpu):
    [days, hrs, mins, secs, msecs] = cpu
    fulldate = datetime(100, 1, tm.day, tm.hour, tm.minute, tm.second, tm.microsecond)
    fulldate = fulldate + timedelta(days=days, hours=hrs, minutes=mins, seconds=secs, microseconds=msecs*1000)

    return fulldate


# Translational energy evaluation (depends on temperature)
def calc_translational_energy(temperature):
    """
    Calculates the translational energy (J/mol) of an ideal gas
    i.e. non-interactiing molecules so molar energy = Na * atomic energy.
    This approximation applies to all energies and entropies computed within
    Etrans = 3/2 RT!
    """
    energy = 1.5 * GAS_CONSTANT * temperature
    return energy


# Rotational energy evaluation (depends on molecular shape and temperature)
def calc_rotational_energy(zpe, symmno, temperature, linear):
    """
    Calculates the rotaional energy (J/mol)
    Etrans = 0 (atomic) ; RT (linear); 3/2 RT (non-linear)
    """
    if zpe == 0.0:
        energy = 0.0
    elif linear == 1:
        energy = GAS_CONSTANT * temperature
    else:
        energy = 1.5 * GAS_CONSTANT * temperature

    return energy


# Vibrational energy evaluation (depends on frequencies, temperature and scaling factor: default = 1.0)
def calc_vibrational_energy(frequency_wn, temperature, freq_scale_factor):
    """
    Calculates the vibrational energy contribution (J/mol). Includes ZPE (0K) and thermal contributions
    Evib = R * Sum(0.5 hv/k + (hv/k)/(e^(hv/KT)-1))
    """
    factor = [(PLANCK_CONSTANT * freq * SPEED_OF_LIGHT * freq_scale_factor) / (BOLTZMANN_CONSTANT * temperature)
                for freq in frequency_wn]
    energy = [entry * GAS_CONSTANT * temperature * (0.5 + (1.0 / (math.exp(entry) - 1.0)))
                for entry in factor]

    return sum(energy)


# Vibrational Zero point energy evaluation (depends on frequencies and scaling factor: default = 1.0)
def calc_zeropoint_energy(frequency_wn, freq_scale_factor):
    """
    Calculates the vibrational ZPE (J/mol)
    EZPE = Sum(0.5 hv/k)
    """
    factor = [PLANCK_CONSTANT * freq * SPEED_OF_LIGHT * freq_scale_factor / BOLTZMANN_CONSTANT
                for freq in frequency_wn]
    energy = [0.5 * entry * GAS_CONSTANT for entry in factor]

    return sum(energy)


# Computed the amount of accessible free space (ml per L) in solution
# accessible to a solute immersed in bulk solvent, i.e. this is the volume
# not occupied by solvent molecules, calculated using literature values for
# molarity and B3LYP/6-31G* computed molecular volumes.
def get_free_space(solv):
    """
    Calculates the free space in a litre of bulk solvent, based on
    Shakhnovich and Whitesides (J. Org. Chem. 1998, 63, 3821-3830)
    """
    solvent_list = ["none", "H2O", "toluene", "DMF", "AcOH", "chloroform"]
    molarity = [1.0, 55.6, 9.4, 12.9, 17.4, 12.5] #mol/l
    molecular_vol = [1.0, 27.944, 149.070, 77.442, 86.10, 97.0] #Angstrom^3

    nsolv = 0
    for i in range(0,len(solvent_list)):
        if solv == solvent_list[i]:
            nsolv = i

    solv_molarity = molarity[nsolv]
    solv_volume = molecular_vol[nsolv]

    if nsolv > 0:
        V_free = 8 * ((1E27/(solv_molarity * AVOGADRO_CONSTANT)) ** 0.333333 - solv_volume ** 0.333333) ** 3
        freespace = V_free * solv_molarity * AVOGADRO_CONSTANT * 1E-24
    else:
        freespace = 1000.0

    return freespace


# Translational entropy evaluation (depends on mass, concentration, temperature, solvent free space: default = 1000.0)
def calc_translational_entropy(molecular_mass, conc, temperature, solv):
    """
    Calculates the translational entropic contribution (J/(mol*K)) of an ideal gas.
    Needs the molecular mass. Convert mass in amu to kg; conc in mol/l to number per m^3
    Strans = R(Ln(2pimkT/h^2)^3/2(1/C)) + 1 + 3/2)
    """
    lmda = ((2.0 * math.pi * molecular_mass * AMU_to_KG * BOLTZMANN_CONSTANT * temperature)**0.5) / PLANCK_CONSTANT
    freespace = get_free_space(solv)
    Ndens = conc * 1000 * AVOGADRO_CONSTANT / (freespace/1000.0)
    entropy = GAS_CONSTANT * (2.5 + math.log(lmda**3 / Ndens))

    return entropy


# Electronic entropy evaluation (depends on multiplicity)
def calc_electronic_entropy(multiplicity):
    """
    Calculates the electronic entropic contribution (J/(mol*K)) of the molecule
    Selec = R(Ln(multiplicity)
    """
    entropy = GAS_CONSTANT * (math.log(multiplicity))
    return entropy


# Rotational entropy evaluation (depends on molecular shape and temp.)
def calc_rotational_entropy(zpe, linear, symmno, rotemp, temperature):
    """
    Calculates the rotational entropy (J/(mol*K))
    Strans = 0 (atomic) ; R(Ln(q)+1) (linear); R(Ln(q)+3/2) (non-linear)
    """

    if rotemp == [0.0,0.0,0.0] or zpe == 0.0: # monatomic
        entropy = 0.0
    else:
        if len(rotemp) == 1: # diatomic or linear molecules
            linear = 1
            qrot = temperature/rotemp[0]
        elif len(rotemp) == 2: # possible gaussian problem with linear triatomic
            linear = 2
        else:
            qrot = math.pi*temperature**3/(rotemp[0]*rotemp[1]*rotemp[2])
            qrot = qrot ** 0.5

        if linear == 1:
            entropy = GAS_CONSTANT * (math.log(qrot / symmno) + 1)
        elif linear == 2:
            entropy = 0.0
        else:
            entropy = GAS_CONSTANT * (math.log(qrot / symmno) + 1.5)

    return entropy


# Rigid rotor harmonic oscillator (RRHO) entropy evaluation - this is the default treatment
def calc_rrho_entropy(frequency_wn, temperature, freq_scale_factor):
    """
    Entropic contributions (J/(mol*K)) according to a rigid-rotor
    harmonic-oscillator description for a list of vibrational modes
    Sv = RSum(hv/(kT(e^(hv/kT)-1) - ln(1-e^(-hv/kT)))
    """
    factor = [PLANCK_CONSTANT * freq * SPEED_OF_LIGHT * freq_scale_factor / BOLTZMANN_CONSTANT / temperature
                for freq in frequency_wn]
    entropy = [entry * GAS_CONSTANT / (math.exp(entry) - 1) - GAS_CONSTANT * math.log(1 - math.exp(-entry))
                for entry in factor]
    return entropy


# Quasi-rigid rotor harmonic oscillator energy evaluation used for calculating quasi-harmonic enthalpy
def calc_qRRHO_energy(frequency_wn, temperature, freq_scale_factor):
    """
    Head-Gordon RRHO-vibrational energy contribution (J/mol*K) of
    vibrational modes described by a rigid-rotor harmonic approximation
    V_RRHO = 1/2(Nhv) + RT(hv/kT)e^(-hv/kT)/(1-e^(-hv/kT))
    """
    factor = [PLANCK_CONSTANT * freq * SPEED_OF_LIGHT * freq_scale_factor
                for freq in frequency_wn]
    energy = [0.5 * AVOGADRO_CONSTANT * entry + GAS_CONSTANT * temperature * entry / BOLTZMANN_CONSTANT
                / temperature * math.exp(-entry / BOLTZMANN_CONSTANT / temperature) /
                (1 - math.exp(-entry / BOLTZMANN_CONSTANT / temperature))
                for entry in factor]

    return energy


# Free rotor entropy evaluation - used for low frequencies below the cut-off if qs=grimme is specified
def calc_freerot_entropy(frequency_wn, temperature, freq_scale_factor):
    """
    Entropic contributions (J/(mol*K)) according to a free-rotor
    description for a list of vibrational modes
    Sr = R(1/2 + 1/2ln((8pi^3u'kT/h^2))
    """
    # This is the average moment of inertia used by Grimme
    Bav = 10.0e-44
    mu = [PLANCK_CONSTANT / (8 * math.pi**2 * freq * SPEED_OF_LIGHT * freq_scale_factor) for freq in frequency_wn]
    mu_primed = [entry * Bav /(entry + Bav) for entry in mu]
    factor = [8 * math.pi**3 * entry * BOLTZMANN_CONSTANT * temperature / PLANCK_CONSTANT**2 for entry in mu_primed]
    entropy = [(0.5 + math.log(entry**0.5)) * GAS_CONSTANT for entry in factor]
    return entropy


# A damping function to interpolate between RRHO and free rotor vibrational entropy values
def calc_damp(frequency_wn, FREQ_CUTOFF):
    alpha = 4
    damp = [1 / (1+(FREQ_CUTOFF/entry)**alpha) for entry in frequency_wn]
    return damp


# Calculate enantioselectivity based on boltzmann factors of given R and S enantiomers
def get_ee(name,files,boltz_facs,boltz_sum,temperature,log):
    R_files,S_files = [], []
    R_sum,S_sum = 0.0, 0.0
    failed = False
    RS_pref = ''
    for file in files:
        if file.lower().startswith(name.lower()):
            if file.find('_R.') > -1:
                R_files.append(file)
                R_sum += boltz_facs[file]/boltz_sum
            elif file.find('_S.') > -1:
                S_files.append(file)
                S_sum += boltz_facs[file]/boltz_sum
            else:
                log.Write("\n   Warning! Filename "+file+' has not been formatted correctly for determining enantioselectivity\n')
                log.Write("   Make sure the filename ends in either '_R' or '_S' \n")
                sys.exit("   Please edit "+file+" and try again\n")

    ee = (R_sum - S_sum) * 100.

    #if ee is negative, more in favor of S
    if ee == 0:
        log.Write("\n   Warning! No files found for an enantioselectivity analysis, adjust the stereodetermining step name and try again.\n")
        failed = True
    elif ee > 0:
        RS_pref = 'R'
    else:
        RS_pref = 'S'

    dd_free_energy = GAS_CONSTANT / J_TO_AU * temperature * math.log((50 + abs(ee) / 2.0) / (50 - abs(ee) / 2.0)) * KCAL_TO_AU
    return abs(ee), dd_free_energy, failed, RS_pref


# Obtain Boltzmann factors, Boltzmann sums, and weighted free energy values, used for --ee and --boltz options
def get_boltz(files,thermo_data,clustering,temperature):
    boltz_facs, weighted_free_energy, e_rel, e_min, boltz_sum = {}, {}, {}, sys.float_info.max, 0.0

    for file in files: # Need the most stable structure
        bbe = thermo_data[file]
        if hasattr(bbe,"qh_gibbs_free_energy"):
            if bbe.qh_gibbs_free_energy != None:
                if bbe.qh_gibbs_free_energy < e_min:
                    e_min = bbe.qh_gibbs_free_energy

    if clustering == True:
        for n, cluster in enumerate(clusters):
            boltz_facs['cluster-'+alphabet[n].upper()] = 0.0
            weighted_free_energy['cluster-'+alphabet[n].upper()] = 0.0
    for file in files: # Now calculate E_rel and Boltzmann factors
        bbe = thermo_data[file]
        if hasattr(bbe,"qh_gibbs_free_energy"):
            if bbe.qh_gibbs_free_energy != None:
                e_rel[file] = bbe.qh_gibbs_free_energy - e_min
                boltz_facs[file] = math.exp(-e_rel[file]*J_TO_AU/GAS_CONSTANT/temperature)

                if clustering == True:
                   for n, cluster in enumerate(clusters):
                       for structure in cluster:
                           if structure == file:
                               boltz_facs['cluster-'+alphabet[n].upper()] += math.exp(-e_rel[file]*J_TO_AU/GAS_CONSTANT/temperature)
                               weighted_free_energy['cluster-'+alphabet[n].upper()] += math.exp(-e_rel[file]*J_TO_AU/GAS_CONSTANT/temperature) * bbe.qh_gibbs_free_energy
                boltz_sum += math.exp(-e_rel[file]*J_TO_AU/GAS_CONSTANT/temperature)

    return boltz_facs, weighted_free_energy, boltz_sum


def main():
    files = []; bbe_vals = []; command = '   Requested: '; clustering = False
    # get command line inputs. Use -h to list all possible arguments and default values
    parser = ArgumentParser()
    parser.add_argument("-q", dest="Q", action="store_true", default=False,
                        help="Quasi-harmonic entropy correction and enthalpy correction applied (default S=Grimme, H=Head-Gordon)")
    parser.add_argument("--qs", dest="QS", default="grimme", type=str.lower, metavar="QS",choices=('grimme', 'truhlar'),
                        help="Type of quasi-harmonic entropy correction (Grimme or Truhlar) (default Grimme)",)
    parser.add_argument("--qh", dest="QH", action="store_true", default=False,
                        help="Type of quasi-harmonic enthalpy correction (Head-Gordon)")
    parser.add_argument("-f", dest="freq_cutoff", default=100, type=float, metavar="FREQ_CUTOFF",
                        help="Cut-off frequency for both entropy and enthalpy (wavenumbers) (default = 100)",)
    parser.add_argument("--fs", dest="S_freq_cutoff", default=100.0, type=float, metavar="S_FREQ_CUTOFF",
                        help="Cut-off frequency for entropy (wavenumbers) (default = 100)")
    parser.add_argument("--fh", dest="H_freq_cutoff", default=100.0, type=float, metavar="H_FREQ_CUTOFF",
                        help="Cut-off frequency for enthalpy (wavenumbers) (default = 100)")
    parser.add_argument("-t", dest="temperature", default=298.15, type=float, metavar="TEMP",
                        help="Temperature (K) (default 298.15)")
    parser.add_argument("-c", dest="conc", default=False, type=float, metavar="CONC",
                        help="Concentration (mol/l) (default 1 atm)")
    parser.add_argument("--ti", dest="temperature_interval", default=False, metavar="TI",
                        help="Initial temp, final temp, step size (K)")
    parser.add_argument("--ci", dest="conc_interval", default=False, metavar="CI",
                        help="Initial conc, final conc, step size (mol/l)")
    parser.add_argument("-v", dest="freq_scale_factor", default=False, type=float, metavar="SCALE_FACTOR",
                        help="Frequency scaling factor. If not set, try to find a suitable value in database. If not found, use 1.0")
    parser.add_argument("--spc", dest="spc", type=str, default=False, metavar="SPC",
                        help="Indicates single point corrections (default False)")
    parser.add_argument("--boltz", dest="boltz", action="store_true", default=False,
                        help="Show Boltzmann factors")
    parser.add_argument("--cpu", dest="cputime", action="store_true", default=False,
                        help="Total CPU time")
    parser.add_argument("--xyz", dest="xyz", action="store_true", default=False,
                        help="Write Cartesians to a .xyz file (default False)")
    parser.add_argument("--csv", dest="csv", action="store_true", default=False,
                        help="Write .csv output file format")
    parser.add_argument("--imag", dest="imag_freq", action="store_true", default=False,
                        help="Print imaginary frequencies (default False)")
    parser.add_argument("--invertifreq", dest="invert", nargs='?', const=True, default=False,
                        help="Make low lying imaginary frequencies positive (cutoff > -50.0 wavenumbers)")
    parser.add_argument("--freespace", dest="freespace", default="none", type=str, metavar="FREESPACE",
                        help="Solvent (H2O, toluene, DMF, AcOH, chloroform) (default none)")
    parser.add_argument("--cosmo", dest="cosmo", default=False, metavar="COSMO-RS",
                        help="Filename of a COSMO-RS out file")
    parser.add_argument("--output", dest="output", default="output", metavar="OUTPUT",
                        help="Change the default name of the output file to GoodVibes_\"output\".dat")
    parser.add_argument("--pes", dest="pes", default=False, metavar="PES",
                        help="Tabulate relative values")
    parser.add_argument("--nogconf", dest="gconf", action="store_false", default=True,
                        help="Calculate a free-energy correction related to multi-configurational space (default calculate Gconf)")
    parser.add_argument("--ee", dest="ee", default=False, type=str,
                        help="Tabulate %% enantiomeric excess value of a mixture, provide name of stereodetermining step")
    parser.add_argument("--check", dest="check", action="store_true", default=False,
                        help="Checks if calculations were done with the same program, level of theory and solvent, as well as detects potential duplicates")
    parser.add_argument("--media", dest="media", default=False, metavar="MEDIA",
                        help="Correction for standard concentration of solvents")
    parser.add_argument("--custom_ext", type=str, default='',
                        help="List of additional file extensions to support, separated by commas (ie, '.qfi,.gaussian'). " +
                            "It can also be specified with environment variable GOODVIBES_CUSTOM_EXT")
    parser.add_argument("--graph", dest='graph', default=False, metavar="GRAPH",
                        help="Graph a reaction profile based on free energies calculated. ")

    # Parse Arguments
    (options, args) = parser.parse_known_args()
    # If requested, turn on head-gordon enthalpy correction
    if options.Q:
        options.QH = True
    if options.QH:
        STARS = "   " + "*" * 142
    else:
        STARS = "   " + "*" * 128

    # If necessary, create an xyz file for Cartesians
    if options.xyz:
        xyz = XYZout("Goodvibes","xyz", "output")

    # If user has specified different file extensions
    if options.custom_ext or os.environ.get('GOODVIBES_CUSTOM_EXT', ''):
        custom_extensions = options.custom_ext.split(',') + os.environ.get('GOODVIBES_CUSTOM_EXT', '').split(',')
        for ext in custom_extensions:
            SUPPORTED_EXTENSIONS.add(ext.strip())

    # Start a log for the results
    log = Logger("Goodvibes", options.output, options.csv)

    # Initialize the total CPU time
    total_cpu_time = datetime(100, 1, 1, 00, 00, 00, 00)
    add_days = 0

    if len(args) > 1:
        for elem in args:
            if elem == 'clust:':
                clustering = True; options.boltz = True
                clusters = []; nclust = -1

    # Get the filenames from the command line prompt
    args = sys.argv[1:]
    for elem in args:
        if clustering == True:
            if elem == 'clust:':
                clusters.append([]); nclust += 0
        try:
            if os.path.splitext(elem)[1] in SUPPORTED_EXTENSIONS: # look for file names
                for file in glob(elem):
                    if options.spc is False or options.spc is 'link':
                        files.append(file)
                        if clustering == True:
                            clusters[nclust].append(file)
                    else:
                        if file.find('_'+options.spc+".") == -1:
                            files.append(file)
                            if clustering == True:
                                clusters[nclust].append(file)
            elif elem != 'clust:': # look for requested options
                command += elem + ' '
        except IndexError:
            pass

    # Start printing results
    start = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    log.Write("   GoodVibes v" + __version__ + " " + start + "\n   REF: " + goodvibes_ref +"\n")
    if clustering ==True:
        command += '(clustering active)'
    log.Write(command+'\n\n')
    if options.temperature_interval is False:
        log.Write("   Temperature = "+str(options.temperature)+" Kelvin")
    # If not at standard temp, need to correct the molarity of 1 atmosphere (assuming Pressure is still 1 atm)
    if options.conc != False:
        log.Write("   Concentration = "+str(options.conc)+" mol/l")
    else:
        options.conc = ATMOS/(GAS_CONSTANT*options.temperature); log.Write("   Pressure = 1 atm")

    # Attempt to automatically obtain frequency scale factor
    # Requires all outputs to be same level of theory
    l_o_t = [level_of_theory(file) for file in files]
    def all_same(items):
        return all(x == items[0] for x in items)
    if options.freq_scale_factor is not False:
        log.Write("\n   User-defined vibrational scale factor "+str(options.freq_scale_factor) + " for " + l_o_t[0] + " level of theory" )
    else:
        filter_of_scaling_f = 0
        if all_same(l_o_t) is True:
            level = l_o_t[0].upper()
            for data in (scaling_data_dict, scaling_data_dict_mod):
                if level in data:
                    options.freq_scale_factor = data[level].zpe_fac
                    ref = scaling_refs[data[level].zpe_ref]
                    log.Write("\n\no  Found vibrational scaling factor of {:.3f} for {} level of theory\n"
                              "   REF: {}".format(options.freq_scale_factor, l_o_t[0], ref))
                    break
        elif all_same(l_o_t) is False:
            files_l_o_t,levels_l_o_t,filtered_calcs_l_o_t = [],[],[]
            for file in files:
                files_l_o_t.append(file)
            for i in l_o_t:
                levels_l_o_t.append(i)
            filtered_calcs_l_o_t.append(files_l_o_t)
            filtered_calcs_l_o_t.append(levels_l_o_t)
            l_o_t_freq_print = "Caution! Different levels of theory found - " + filtered_calcs_l_o_t[1][0] + " (" + filtered_calcs_l_o_t[0][0]
            for i in range(len(filtered_calcs_l_o_t[1])):
                if filtered_calcs_l_o_t[1][i] == filtered_calcs_l_o_t[1][0] and i != 0:
                    l_o_t_freq_print += ", " + filtered_calcs_l_o_t[0][i]
            l_o_t_freq_print += ")"
            for i in range(len(filtered_calcs_l_o_t[1])):
                if filtered_calcs_l_o_t[1][i] != filtered_calcs_l_o_t[1][0] and i != 0:
                    l_o_t_freq_print += ", " + filtered_calcs_l_o_t[1][i] + " (" + filtered_calcs_l_o_t[0][i] + ")"
                    filter_of_scaling_f = filter_of_scaling_f + 1
            log.Write("\nx  " + l_o_t_freq_print)

    if options.freq_scale_factor is False:
        options.freq_scale_factor = 1.0 # if no scaling factor is found use 1.0
        if filter_of_scaling_f == 0:
            log.Write("\n   Using vibrational scale factor "+str(options.freq_scale_factor) + " for " + l_o_t[0] + " level of theory")
        else:
            log.Write("\n   Using vibrational scale factor "+str(options.freq_scale_factor) + ": differing levels of theory detected.")
    # checks to see whether the available free space of a requested solvent is defined
    freespace = get_free_space(options.freespace)
    if freespace != 1000.0:
        log.Write("\n   Specified solvent "+options.freespace+": free volume "+str("%.3f" % (freespace/10.0))+" (mol/l) corrects the translational entropy")

    # read from COSMO-RS output
    if options.cosmo is not False:
        try:
            cosmo_solv = COSMORSout(options.cosmo, files)
            log.Write('\n\n   Reading COSMO-RS file: '+options.cosmo)
        except ValueError:
            log.Write('\n\n   Warning! COSMO-RS file '+options.cosmo+' requested but not found')
            cosmo_solv = None

    if options.freq_cutoff != 100.0:
        options.S_freq_cutoff = options.freq_cutoff
        options.H_freq_cutoff = options.freq_cutoff

    # Summary of the quasi-harmonic treatment; print out the relevant reference
    log.Write("\n\n   Entropic quasi-harmonic treatment: frequency cut-off value of "+str(options.S_freq_cutoff)+" wavenumbers will be applied.")
    if options.QS == "grimme":
        log.Write("\n   QS = Grimme: Using a mixture of RRHO and Free-rotor vibrational entropies."); qs_ref = grimme_ref
    elif options.QS == "truhlar":
        log.Write("\n   QS = Truhlar: Using an RRHO treatment where low frequencies are adjusted to the cut-off value."); qs_ref = truhlar_ref
    else:
        log.Fatal("\n   FATAL ERROR: Unknown quasi-harmonic model "+options.QS+" specified (QS must = grimme or truhlar).")
    log.Write("\n   REF: " + qs_ref)

    # Check if qh correction should be applied
    if options.QH:
        log.Write("\n\n   Enthalpy quasi-harmonic treatment: frequency cut-off value of "+str(options.H_freq_cutoff)+" wavenumbers will be applied.")
        log.Write("\n   QH = Head-Gordon: Using an RRHO treatement with an approximation term for vibrational energy.")
        qh_ref = head_gordon_ref
        log.Write("\n   REF: " + qh_ref)
    else:
        log.Write("\n\n   No quasi-harmonic enthalpy correction will be applied.")

    # Whether linked single-point energies are to be used
    if options.spc is "True":
        log.Write("\n   Link job: combining final single point energy with thermal corrections.")

    # Check if user has specified any files, if not quit now
    if len(files) == 0:
        sys.exit("\nWarning! No calculation output file specified to run with GoodVibes.\n")

    inverted_freqs, inverted_files = [], []
    for file in files: # loop over all specified output files and compute thermochemistry
        bbe = calc_bbe(file, options.QS, options.QH, options.S_freq_cutoff, options.H_freq_cutoff, options.temperature,
                        options.conc, options.freq_scale_factor, options.freespace, options.spc, options.invert)
        bbe_vals.append(bbe)

    fileList = [file for file in files]
    thermo_data = dict(zip(fileList, bbe_vals)) # the collected thermochemical data for all files

    inverted_freqs, inverted_files = [], []
    for file in files:
        if len(thermo_data[file].inverted_freqs) > 0:
            inverted_freqs.append(thermo_data[file].inverted_freqs)
            inverted_files.append(file)

    # Check if user has chosen to make any low lying imaginary frequencies positive
    if options.invert is not False:
        for i,file in enumerate(inverted_files):
            if len(inverted_freqs[i]) == 1:
                log.Write("\n\n   The following frequency was made positive and used in calculations: " + str(inverted_freqs[i][0]) + " from " + file)
            elif len(inverted_freqs[i]) > 1:
                log.Write("\n\n   The following frequencies were made positive and used in calculations: " + str(inverted_freqs[i]) + " from " + file)

    # Adjust printing according to options requested
    if options.spc is not False:
        STARS += '*' * 14
    if options.cosmo is not False:
        STARS += '*' * 30
    if options.imag_freq is True:
        STARS += '*' * 9
    if options.boltz is True:
        STARS += '*' * 7

    # Standard mode: tabulate thermochemistry ouput from file(s) at a single temperature and concentration
    if options.temperature_interval is False and options.conc_interval is False:
        if options.spc is False:
            log.Write("\n\n   ")
            if options.QH:
                log.Write('{:<39} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} {:>13}'.format("Structure", "E", "ZPE", "H", "qh-H", "T.S", "T.qh-S", "G(T)", "qh-G(T)"),thermodata=True)
            else:
                log.Write('{:<39} {:>13} {:>10} {:>13} {:>10} {:>10} {:>13} {:>13}'.format("Structure", "E", "ZPE", "H", "T.S", "T.qh-S", "G(T)", "qh-G(T)"),thermodata=True)
        else:
            log.Write("\n\n   ")
            if options.QH:
                log.Write('{:<39} {:>13} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} {:>13}'.format("Structure", "E_"+options.spc, "E", "ZPE", "H_"+options.spc, "qh-H_"+options.spc, "T.S", "T.qh-S", "G(T)_"+options.spc, "qh-G(T)_"+options.spc),thermodata=True)
            else:
                log.Write('{:<39} {:>13} {:>13} {:>10} {:>13} {:>10} {:>10} {:>13} {:>13}'.format("Structure", "E_"+options.spc, "E", "ZPE", "H_"+options.spc, "T.S", "T.qh-S", "G(T)_"+options.spc, "qh-G(T)_"+options.spc),thermodata=True)
        if options.cosmo is not False:
            log.Write('{:>13} {:>16}'.format("COSMO-RS","COSMO-qh-G(T)"))
        if options.boltz is True:
            log.Write('{:>7}'.format("Boltz"),thermodata=True)
        if options.imag_freq is True:
            log.Write('{:>9}'.format("im freq"),thermodata=True)
        log.Write("\n"+STARS+"")

        # Boltzmann factors and averaging over clusters
        if options.boltz != False:
            boltz_facs, weighted_free_energy, boltz_sum = get_boltz(files,thermo_data,clustering,options.temperature)

        Gqh_duplic, H_duplic, qh_entropy_duplic = [], [], []
        for file in files: # Loop over the output files and compute thermochemistry
            bbe = thermo_data[file]

            if options.cputime != False: # Add up CPU times
                if hasattr(bbe,"cpu"):
                    if bbe.cpu != None:
                        total_cpu_time = addTime(total_cpu_time, bbe.cpu)
                if hasattr(bbe,"sp_cpu"):
                    if bbe.sp_cpu != None:
                        total_cpu_time = addTime(total_cpu_time, bbe.sp_cpu)
            if total_cpu_time.month > 1:
                add_days += 31

            if options.xyz: # Write Cartesians
                xyzdata = getoutData(file)
                xyz.Writetext(str(len(xyzdata.ATOMTYPES)))
                if hasattr(bbe, "scf_energy"):
                    xyz.Writetext('{:<39} {:>13} {:13.6f}'.format(os.path.splitext(os.path.basename(file))[0], 'Eopt', bbe.scf_energy))
                else:
                    xyz.Writetext('{:<39}'.format(os.path.splitext(os.path.basename(file))[0]))
                if hasattr(xyzdata, 'CARTESIANS') and hasattr(xyzdata, 'ATOMTYPES'):
                    xyz.Writecoords(xyzdata.ATOMTYPES, xyzdata.CARTESIANS)
            warning_linear = calc_bbe(file, options.QS, options.QH, options.S_freq_cutoff, options.H_freq_cutoff, options.temperature,
                                        options.conc, options.freq_scale_factor, options.freespace, options.spc, options.invert)
            linear_warning = []
            linear_warning.append(warning_linear.linear_warning)
            if linear_warning == [['Warning! Potential invalid calculation of linear molecule from Gaussian.']]:
                log.Write("\nx  "+'{:<39}'.format(os.path.splitext(os.path.basename(file))[0]))
                log.Write('          ----   Caution! Potential invalid calculation of linear molecule from Gaussian')
                if options.check != False:
                    Gqh_duplic.append(0.0)
                    H_duplic.append(0.0)
                    qh_entropy_duplic.append(0.0)
            else:
                if hasattr(bbe, "gibbs_free_energy"):
                    if options.spc is not False:
                        if bbe.sp_energy != '!':
                            log.Write("\no  ")
                            log.Write('{:<39}'.format(os.path.splitext(os.path.basename(file))[0]),thermodata=True)
                            log.Write(' {:13.6f}'.format(bbe.sp_energy),thermodata=True)
                        if bbe.sp_energy == '!':
                            log.Write("\nx  ")
                            log.Write('{:<39}'.format(os.path.splitext(os.path.basename(file))[0]),thermodata=True)
                            log.Write(' {:>13}'.format('----'),thermodata=True)
                    else:
                        log.Write("\no  ")
                        log.Write('{:<39}'.format(os.path.splitext(os.path.basename(file))[0]),thermodata=True)
                
                if hasattr(bbe, "scf_energy") and not hasattr(bbe,"gibbs_free_energy"):#gaussian spc files
                    log.Write("\nx  "+'{:<39}'.format(os.path.splitext(os.path.basename(file))[0]))
                elif not hasattr(bbe, "scf_energy") and not hasattr(bbe,"gibbs_free_energy"): #orca spc files
                    log.Write("\nx  "+'{:<39}'.format(os.path.splitext(os.path.basename(file))[0]))
                if hasattr(bbe, "scf_energy"):
                    log.Write(' {:13.6f}'.format(bbe.scf_energy),thermodata=True)
                if not hasattr(bbe,"gibbs_free_energy"):
                    log.Write("   Warning! Couldn't find frequency information ...")
                    if options.check != False:
                        Gqh_duplic.append(0.0)
                        H_duplic.append(0.0)
                        qh_entropy_duplic.append(0.0)

                else:
                    if options.media == False:
                        if all(getattr(bbe, attrib) for attrib in ["enthalpy", "entropy", "qh_entropy", "gibbs_free_energy", "qh_gibbs_free_energy"]):
                            if options.QH:
                                log.Write(' {:10.6f} {:13.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(bbe.zpe, bbe.enthalpy, bbe.qh_enthalpy, (options.temperature * bbe.entropy), (options.temperature * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy))
                            else:
                                log.Write(' {:10.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(bbe.zpe, bbe.enthalpy, (options.temperature * bbe.entropy), (options.temperature * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy))
                            if options.check != False:
                                Gqh_duplic.append(bbe.qh_gibbs_free_energy)
                                H_duplic.append(bbe.enthalpy)
                                qh_entropy_duplic.append((options.temperature * bbe.qh_entropy))
                    else:
                        try:
                            from .media import solvents
                        except:
                            from media import solvents
                        if options.media.lower() in solvents and options.media.lower() == os.path.splitext(os.path.basename(file))[0].lower():
                            MW_solvent = solvents[options.media.lower()][0]
                            density_solvent = solvents[options.media.lower()][1]
                            concentration_solvent = (density_solvent*1000)/MW_solvent
                            media_correction = -(GAS_CONSTANT/J_TO_AU)*math.log(concentration_solvent)
                            if all(getattr(bbe, attrib) for attrib in ["enthalpy", "entropy", "qh_entropy", "gibbs_free_energy", "qh_gibbs_free_energy"]):
                                if options.QH:
                                    log.Write(' {:10.6f} {:13.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(bbe.zpe, bbe.enthalpy, bbe.qh_enthalpy, (options.temperature * (bbe.entropy+media_correction)), (options.temperature * (bbe.qh_entropy+media_correction)), bbe.gibbs_free_energy+(options.temperature * (-media_correction)), bbe.qh_gibbs_free_energy+(options.temperature * (-media_correction))))
                                    log.Write("  Solvent")
                                else:
                                    log.Write(' {:10.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(bbe.zpe, bbe.enthalpy, (options.temperature * (bbe.entropy+media_correction)), (options.temperature * (bbe.qh_entropy+media_correction)), bbe.gibbs_free_energy+(options.temperature * (-media_correction)), bbe.qh_gibbs_free_energy+(options.temperature * (-media_correction))))
                                    log.Write("  Solvent")
                                if options.check != False:
                                    Gqh_duplic.append(bbe.qh_gibbs_free_energy)
                                    H_duplic.append(bbe.enthalpy)
                                    qh_entropy_duplic.append((options.temperature * bbe.qh_entropy))
                        else:
                            if all(getattr(bbe, attrib) for attrib in ["enthalpy", "entropy", "qh_entropy", "gibbs_free_energy", "qh_gibbs_free_energy"]):
                                if options.QH:
                                    log.Write(' {:10.6f} {:13.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(bbe.zpe, bbe.enthalpy, bbe.qh_enthalpy, (options.temperature * bbe.entropy), (options.temperature * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy))
                                else:
                                    log.Write(' {:10.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(bbe.zpe, bbe.enthalpy, (options.temperature * bbe.entropy), (options.temperature * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy))
                                if options.check != False:
                                    Gqh_duplic.append(bbe.qh_gibbs_free_energy)
                                    H_duplic.append(bbe.enthalpy)
                                    qh_entropy_duplic.append((options.temperature * bbe.qh_entropy))

            if options.cosmo is not False and cosmo_solv != None:
                log.Write('{:13.6f} {:16.6f}'.format(cosmo_solv[file],bbe.qh_gibbs_free_energy+cosmo_solv[file]))
            if options.boltz is True:
                log.Write('{:7.3f}'.format(boltz_facs[file]/boltz_sum),thermodata=True)
            if options.imag_freq is True and hasattr(bbe, "im_frequency_wn") == True:
                for freq in bbe.im_frequency_wn:
                    log.Write('{:9.2f}'.format(freq),thermodata=True)

            if clustering == True:
                for n, cluster in enumerate(clusters):
                    for id, structure in enumerate(cluster):
                        if structure == file:
                            if id == len(cluster)-1:
                                if options.spc is not False:
                                    log.Write("\no  "+'{:<39} {:>13} {:>13} {:>10} {:9} {:>13} {:>10} {:>10} {:>13} {:13.6f} {:6.2f}'.format('Boltzmann-weighted Cluster '+alphabet[n].upper(), '***', '***', '***', '***', '***', '***', '***', '***', weighted_free_energy['cluster-'+alphabet[n].upper()] / boltz_facs['cluster-'+alphabet[n].upper()] , 100 * boltz_facs['cluster-'+alphabet[n].upper()]/boltz_sum))
                                else:
                                    log.Write("\no  "+'{:<39} {:>13} {:>10} {:>13} {:9} {:>10} {:>10} {:>13} {:13.6f} {:6.2f}'.format('Boltzmann-weighted Cluster '+alphabet[n].upper(), '***', '***', '***', '***', '***', '***', '***', weighted_free_energy['cluster-'+alphabet[n].upper()] / boltz_facs['cluster-'+alphabet[n].upper()] , 100 * boltz_facs['cluster-'+alphabet[n].upper()]/boltz_sum))

        log.Write("\n"+STARS+"\n")

        # Check checks
        if options.check != False:
            log.Write("\n   Checks for thermochemistry calculations (frequency calculations):")
            log.Write("\n"+STARS)
            version_check = [sp_energy(file)[2] for file in files]
            file_version = [sp_energy(file)[4] for file in files]
            if all_same(version_check) != False:
                log.Write("\no  Using "+version_check[0]+" in all the calculations.")
            else:
                version_check_print = "Caution! Different programs or versions found - " + version_check[0] + " (" + file_version[0]
                for i in range(len(version_check)):
                    if version_check[i] == version_check[0] and i != 0:
                        version_check_print += ", " + file_version[i]
                version_check_print += ")"
                for i in range(len(version_check)):
                    if version_check[i] != version_check[0] and i != 0:
                        version_check_print += ", " + version_check[i] + " (" + file_version[i] + ")"
                log.Write("\nx  " + version_check_print + ".")
            solvent_check = [sp_energy(file)[3] for file in files]
            if all_same(solvent_check) != False:
                log.Write("\no  Using "+solvent_check[0]+" in all the calculations.")
            else:
                solvent_check_print = "Caution! Different solvation models found - " + solvent_check[0] + " (" + file_version[0]
                filtered_calcs = []
                for i in range(len(solvent_check)):
                    if i != 0:
                        filter_num = 0
                        for j in range(len(solvent_check[0].replace("(",",").replace(")","").split(","))):
                            for k in range(len(solvent_check[i].replace("(",",").replace(")","").split(","))):
                                if solvent_check[0].replace("(",",").replace(")","").split(",")[j] == solvent_check[i].replace("(",",").replace(")","").split(",")[k]:
                                    filter_num = filter_num + 1
                                    if filter_num == len(solvent_check[0].replace("(",",").replace(")","").split(",")):
                                        solvent_check_print += ", " + file_version[i]
                                        filtered_calcs.append(solvent_check[i])
                solvent_check_print += ")"
                solvent_different,file_different = [],[]
                for i in range(len(solvent_check)):
                    if solvent_check[i] != solvent_check[0]:
                        solvent_different.append(solvent_check[i])
                        file_different.append(file_version[i])
                for i in range(len(solvent_different)):
                    for j in range(len(filtered_calcs)):
                        if solvent_different[i] == filtered_calcs[j]:
                            solvent_different.remove(solvent_different[i])
                            file_different.remove(file_different[i])
                            break
                for i in range(len(solvent_different)):
                    solvent_check_print += ", " + solvent_different[i] + " (" + file_different[i] + ")"
                log.Write("\nx  " + solvent_check_print + '.')

            # Check level of theory
            if all_same(l_o_t) is not False:
                log.Write("\no  Using "+l_o_t[0]+" in all the calculations.")
            elif all_same(l_o_t) is False:
                l_o_t_print = "Caution! Different levels of theory found - " + l_o_t[0] + " (" + file_version[0]
                for i in range(len(l_o_t)):
                    if l_o_t[i] == l_o_t[0] and i != 0:
                        l_o_t_print += ", " + file_version[i]
                l_o_t_print += ")"
                for i in range(len(l_o_t)):
                    if l_o_t[i] != l_o_t[0] and i != 0:
                        l_o_t_print += ", " + l_o_t[i] + " (" + file_version[i] + ")"
                log.Write("\nx  " + l_o_t_print + '.')

            # Check charge and multiplicity
            charge_check = [sp_energy(file)[5] for file in files]
            multiplicity_check = []
            for file in files:
                multiplicity_calc = calc_bbe(file, options.QS, options.QH, options.S_freq_cutoff, options.H_freq_cutoff, options.temperature,
                                                options.conc, options.freq_scale_factor, options.freespace, options.spc, options.invert)
                multiplicity_check.append(str(int(multiplicity_calc.mult)))
            if all_same(charge_check) != False and all_same(multiplicity_check) != False:
                log.Write("\no  Using charge and multiplicity "+charge_check[0]+ " " + multiplicity_check[0] + " in all the calculations.")
            else:
                charge_check_print = "Caution! Different charge and multiplicity found - " + charge_check[0] + " " + multiplicity_check[0] + " (" + file_version[0]
                for i in range(len(charge_check)):
                    if charge_check[i] == charge_check[0] and multiplicity_check[i] == multiplicity_check[0] and i != 0:
                        charge_check_print += ", " + file_version[i]
                charge_check_print += ")"
                for i in range(len(charge_check)):
                    if charge_check[i] != charge_check[0] or multiplicity_check[i] != multiplicity_check[0] and i != 0:
                        charge_check_print += ", " + charge_check[i] + " " + multiplicity_check[i] + " (" + file_version[i] + ")"
                log.Write("\nx  " + charge_check_print+ '.')

            # Check for duplicate structures
            energy_duplic,files_duplic = [],[]
            for file in files:
                energy_duplic.append(sp_energy(file)[0])
                files_duplic.append(file)
            info_duplic = []
            info_duplic.append(energy_duplic)
            info_duplic.append(Gqh_duplic)
            info_duplic.append(H_duplic)
            info_duplic.append(qh_entropy_duplic)
            info_duplic.append(files_duplic)
            #Add thermodynamic FILTERS
            duplicates = "Caution! Potential duplicates or enantiomeric conformations found (based on E, H, qh_T.S and qh_G) - "
            for i in range(len(files)):
                for j in range(len(files)):
                    if j > i:
                        if info_duplic[0][i] > info_duplic[0][j]-0.00016 and info_duplic[0][i] < info_duplic[0][j]+0.00016:
                            if info_duplic[1][i] > info_duplic[1][j]-0.00016 and info_duplic[1][i] < info_duplic[1][j]+0.00016:
                                if info_duplic[2][i] > info_duplic[2][j]-0.00016 and info_duplic[2][i] < info_duplic[2][j]+0.00016:
                                    if info_duplic[3][i] > info_duplic[3][j]-0.00016 and info_duplic[3][i] < info_duplic[3][j]+0.00016:
                                        duplicates += ", " + info_duplic[4][i] + " and " + info_duplic[4][j]
            if duplicates == "Caution! Potential duplicates or enantiomeric conformations found (based on E, H, qh_T.S and qh_G) - ":
                log.Write("\no  No potential duplicates or enantiomeric conformations found (based on E, H, qh_T.S and qh_G).")
            else:
                duplicates1 = duplicates[:101]
                duplicates2 = duplicates[103:]
                log.Write("\nx  " + duplicates1 + duplicates2 + '.')

            # Check for linear molecules with incorrect number of vibrational modes
            linear_fails,linear_fails_atom,linear_fails_cart,linear_fails_files,linear_fails_list = [],[],[],[],[]
            frequency_list, im_frequency_list, frequency_get= [],[],[]
            for file in files:
                linear_fails = getoutData(file)
                linear_fails_cart.append(linear_fails.CARTESIANS)
                linear_fails_atom.append(linear_fails.ATOMTYPES)
                linear_fails_files.append(file)
                frequency_get = calc_bbe(file, options.QS, options.QH, options.S_freq_cutoff, options.H_freq_cutoff, options.temperature,
                                            options.conc, options.freq_scale_factor, options.freespace, options.spc, options.invert)
                frequency_list.append(frequency_get.frequency_wn)
                im_frequency_list.append(frequency_get.im_frequency_wn)
            linear_fails_list.append(linear_fails_atom)
            linear_fails_list.append(linear_fails_cart)
            linear_fails_list.append(frequency_list)
            linear_fails_list.append(linear_fails_files)
            im_freq_separated = []
            im_freq_separated.append(im_frequency_list)
            im_freq_separated.append(linear_fails_files)

            linear_mol_correct,linear_mol_wrong = [],[]
            for i in range(len(linear_fails_list[0])):
                count_linear = 0
                if len(linear_fails_list[0][i]) == 2:
                    if len(linear_fails_list[2][i]) == 1:
                        linear_mol_correct.append(linear_fails_list[3][i])
                    else:
                        linear_mol_wrong.append(linear_fails_list[3][i])
                if len(linear_fails_list[0][i]) == 3:
                    if linear_fails_list[0][i] == ['I', 'I', 'I'] or linear_fails_list[0][i] == ['O', 'O', 'O'] or linear_fails_list[0][i] == ['N', 'N', 'N'] or linear_fails_list[0][i] == ['H', 'C', 'N'] or linear_fails_list[0][i] == ['H', 'N', 'C'] or linear_fails_list[0][i] == ['C', 'H', 'N'] or linear_fails_list[0][i] == ['C', 'N', 'H'] or linear_fails_list[0][i] == ['N', 'H', 'C'] or linear_fails_list[0][i] == ['N', 'C', 'H']:
                        if len(linear_fails_list[2][i]) == 4:
                            linear_mol_correct.append(linear_fails_list[3][i])
                        else:
                            linear_mol_wrong.append(linear_fails_list[3][i])
                    else:
                        for j in range(len(linear_fails_list[0][i])):
                            for k in range(len(linear_fails_list[0][i])):
                                if k > j:
                                    for l in range(len(linear_fails_list[1][i][j])):
                                        if linear_fails_list[0][i][j] == linear_fails_list[0][i][k]:
                                            if linear_fails_list[1][i][j][l] > (-linear_fails_list[1][i][k][l]-0.1) and linear_fails_list[1][i][j][l] < (-linear_fails_list[1][i][k][l]+0.1):
                                                count_linear = count_linear + 1
                                                if count_linear == 3:
                                                    if len(linear_fails_list[2][i]) == 4:
                                                        linear_mol_correct.append(linear_fails_list[3][i])
                                                    else:
                                                        linear_mol_wrong.append(linear_fails_list[3][i])
                if len(linear_fails_list[0][i]) == 4:
                    if linear_fails_list[0][i] == ['C', 'C', 'H', 'H'] or linear_fails_list[0][i] == ['C', 'H', 'C', 'H'] or linear_fails_list[0][i] == ['C', 'H', 'H', 'C'] or linear_fails_list[0][i] == ['H', 'C', 'C', 'H'] or linear_fails_list[0][i] == ['H', 'C', 'H', 'C'] or linear_fails_list[0][i] == ['H', 'H', 'C', 'C']:
                        if len(linear_fails_list[2][i]) == 7:
                            linear_mol_correct.append(linear_fails_list[3][i])
                        else:
                            linear_mol_wrong.append(linear_fails_list[3][i])
            linear_correct_print,linear_wrong_print = "",""
            for i in range(len(linear_mol_correct)):
                linear_correct_print += ', ' + linear_mol_correct[i]
            for i in range(len(linear_mol_wrong)):
                linear_wrong_print += ', ' + linear_mol_wrong[i]
            linear_correct_print = linear_correct_print[1:]
            linear_wrong_print = linear_wrong_print[1:]
            if len(linear_mol_correct) == 0:
                if len(linear_mol_wrong) == 0:
                    log.Write("\n-  No linear molecules found.")
                if len(linear_mol_wrong) >= 1:
                    log.Write("\nx  Caution! Potential linear molecules with wrong number of frequencies found (correct number = 3N-5) -"
                                + linear_wrong_print + ".")
            elif len(linear_mol_correct) >= 1:
                if len(linear_mol_wrong) == 0:
                    log.Write("\no  All the linear molecules have the correct number of frequencies -" + linear_correct_print + '.')
                if len(linear_mol_wrong) >= 1:
                    log.Write("\nx  Caution! Potential linear molecules with wrong number of frequencies found -" + linear_wrong_print
                                + ". Correct number of frequencies (3N-5) found in other calculations -" + linear_correct_print + '.')
            # Check for false TS
            false_TS_list,right_TS_list = [],[]
            for i in range(len(im_freq_separated[0])):
                TS_neg_freq, TS_false_freq = 0,0
                if im_freq_separated[1][i].strip().startswith('TS-') or im_freq_separated[1][i].strip().startswith('TS_'):
                    for j in range(len(im_freq_separated[0][i])):
                        if im_freq_separated[0][i][j] < -50.0:
                                TS_neg_freq = TS_neg_freq + 1
                                if TS_neg_freq == 2 and im_freq_separated[1][i] not in false_TS_list:
                                    false_TS_list.append(im_freq_separated[1][i])
                        if im_freq_separated[0][i][j] > -50.0 and im_freq_separated[0][i][j] < -1:
                            TS_false_freq = TS_false_freq + 1
                            if TS_false_freq == 1 and im_freq_separated[1][i] not in false_TS_list:
                                false_TS_list.append(im_freq_separated[1][i])
                    if TS_neg_freq != 1 and im_freq_separated[1][i] not in false_TS_list:
                        false_TS_list.append(im_freq_separated[1][i])
                    if TS_neg_freq == 1 and TS_false_freq == 0 and im_freq_separated[1][i] not in false_TS_list:
                        right_TS_list.append(im_freq_separated[1][i])
            false_TS_print = ""
            for i in false_TS_list:
                false_TS_print += ", " + i
            if len(false_TS_print) == 0 and len(right_TS_list) == 0:
                log.Write("\n-  No transition states found (Warning! If you have any TS, rename the files to start with \"TS-\" or \"TS_\").")
            if len(right_TS_list) > 0 and len(false_TS_print) == 0:
                log.Write("\no  All the transition states have only 1 imaginary frequency lower than -50 cm-1.")
            if len(false_TS_print) != 0:
                log.Write("\nx  Caution! Potential transition states with wrong number of negative frequencies found -"+ false_TS_print[1:] + ".")

            #Check for empirical dispersion
            dispersion_check = [sp_energy(file)[6] for file in files]
            if all_same(dispersion_check) != False:
                if dispersion_check[0] == 'No empirical dispersion detected':
                    log.Write("\n-  No empirical dispersion detected in any of the calculations.")
                else:
                    log.Write("\no  Using "+dispersion_check[0]+" in all the calculations.")
            else:
                dispersion_check_print = "Caution! Different dispersion models found - " + dispersion_check[0] + " (" + file_version[0]
                for i in range(len(dispersion_check)):
                    if dispersion_check[i] == dispersion_check[0] and i != 0:
                        dispersion_check_print += ", " + file_version[i]
                dispersion_check_print += ")"
                for i in range(len(dispersion_check)):
                    if dispersion_check[i] != dispersion_check[0] and i != 0:
                        dispersion_check_print += ", " + dispersion_check[i] + " (" + file_version[i] + ")"
                log.Write("\nx  " + dispersion_check_print + ".")

            log.Write("\n"+STARS+"\n")

            #Check for single-point corrections
            if options.spc is not False:
                log.Write("\n   Checks for single-point corrections:")
                log.Write("\n"+STARS)
                names_spc, version_check_spc = [], []
                for file in files:
                    name, ext = os.path.splitext(file)
                    if os.path.exists(name+'_'+options.spc+'.log'):
                        names_spc.append(name+'_'+options.spc+'.log')
                    elif os.path.exists(name+'_'+options.spc+'.out'):
                        names_spc.append(name+'_'+options.spc+'.out')

                # Check program versions
                version_check_spc = [sp_energy(name)[2] for name in names_spc]
                if all_same(version_check_spc) != False:
                    log.Write("\no  Using "+version_check_spc[0]+" in all the single-point corrections.")
                else:
                    version_check_spc_print = "Caution! Different programs or versions found - " + version_check_spc[0] + " (" + names_spc[0]
                    for i in range(len(version_check_spc)):
                        if version_check_spc[i] == version_check_spc[0] and i != 0:
                            version_check_spc_print += ", " + names_spc[i]
                    version_check_spc_print += ")"
                    for i in range(len(version_check_spc)):
                        if version_check_spc[i] != version_check_spc[0] and i != 0:
                            version_check_spc_print += ", " + version_check_spc[i] + " (" + names_spc[i] + ")"
                    log.Write("\nx  " + version_check_spc_print + ".")
                solvent_check_spc = [sp_energy(name)[3] for name in names_spc]
                if all_same(solvent_check_spc) != False:
                    log.Write("\no  Using "+solvent_check_spc[0]+" in all the single-point corrections.")
                else:
                    solvent_check_spc_print = "Caution! Different solvation models found - " + solvent_check_spc[0] + " (" + names_spc[0]
                    filtered_calcs_spc = []
                    for i in range(len(solvent_check_spc)):
                        if i != 0:
                            filter_num_spc = 0
                            for j in range(len(solvent_check_spc[0].replace("(",",").replace(")","").split(","))):
                                for k in range(len(solvent_check_spc[i].replace("(",",").replace(")","").split(","))):
                                    if solvent_check_spc[0].replace("(",",").replace(")","").split(",")[j] == solvent_check_spc[i].replace("(",",").replace(")","").split(",")[k]:
                                        filter_num_spc = filter_num_spc + 1
                                        if filter_num_spc == len(solvent_check_spc[0].replace("(",",").replace(")","").split(",")):
                                            solvent_check_spc_print += ", " + names_spc[i]
                                            filtered_calcs_spc.append(solvent_check_spc[i])
                    solvent_check_spc_print += ")"
                    solvent_different_spc,file_different_spc = [],[]
                    for i in range(len(solvent_check_spc)):
                        if solvent_check_spc[i] != solvent_check_spc[0]:
                            solvent_different_spc.append(solvent_check_spc[i])
                            file_different_spc.append(names_spc[i])
                    for i in range(len(solvent_different_spc)):
                        for j in range(len(filtered_calcs_spc)):
                            if solvent_different_spc[i] == filtered_calcs_spc[j]:
                                solvent_different_spc.remove(solvent_different_spc[i])
                                file_different_spc.remove(file_different_spc[i])
                                break
                    for i in range(len(solvent_different_spc)):
                        solvent_check_spc_print += ", " + solvent_different_spc[i] + " (" + file_different_spc[i] + ")"
                    log.Write("\nx  " + solvent_check_spc_print + '.')
                l_o_t_spc = [level_of_theory(name) for name in names_spc]
                if all_same(l_o_t_spc) != False:
                    log.Write("\no  Using "+l_o_t_spc[0]+" in all the single-point corrections.")
                elif all_same(l_o_t_spc) == False:
                    l_o_t_spc_print = "Caution! Different levels of theory found - " + l_o_t_spc[0] + " (" + names_spc[0]
                    for i in range(len(l_o_t_spc)):
                        if l_o_t_spc[i] == l_o_t_spc[0] and i != 0:
                            l_o_t_spc_print += ", " + names_spc[i]
                    l_o_t_spc_print += ")"
                    for i in range(len(l_o_t_spc)):
                        if l_o_t_spc[i] != l_o_t_spc[0] and i != 0:
                            l_o_t_spc_print += ", " + l_o_t_spc[i] + " (" + names_spc[i] + ")"
                    log.Write("\nx  " + l_o_t_spc_print + '.')
                charge_spc_check = [sp_energy(name)[5] for name in names_spc]
                multiplicity_spc_check = []
                for name in names_spc:
                     multiplicity_spc_calc = calc_bbe(name, options.QS, options.QH, options.S_freq_cutoff, options.H_freq_cutoff, options.temperature,
                                                        options.conc, options.freq_scale_factor, options.freespace, options.spc, options.invert)
                     multiplicity_spc_check.append(str(int(multiplicity_spc_calc.mult)))
                if all_same(charge_spc_check) != False and all_same(multiplicity_spc_check) != False:
                    log.Write("\no  Using charge and multiplicity "+charge_spc_check[0]+ " " + multiplicity_spc_check[0] + " in all the single-point corrections.")
                else:
                    charge_spc_check_print = "Caution! Different charge and multiplicity found - " + charge_spc_check[0] + " " + multiplicity_spc_check[0] + " (" + names_spc[0]
                    for i in range(len(charge_check)):
                        if charge_spc_check[i] == charge_spc_check[0] and multiplicity_spc_check[i] == multiplicity_spc_check[0] and i != 0:
                            charge_spc_check_print += ", " + names_spc[i]
                    charge_spc_check_print += ")"
                    for i in range(len(charge_spc_check)):
                        if charge_spc_check[i] != charge_spc_check[0] or multiplicity_spc_check[i] != multiplicity_spc_check[0] and i != 0:
                            charge_spc_check_print += ", " + charge_spc_check[i] + " " + multiplicity_spc_check[i] + " (" + names_spc[i] + ")"
                    log.Write("\nx  " + charge_spc_check_print + '.')
                #Check if the geometries of freq calculations match their corresponding structures in single-point calculations
                geom_duplic_list,geom_duplic_list_spc,geom_duplic_cart,geom_duplic_files,geom_duplic_cart_spc,geom_duplic_files_spc = [],[],[],[],[],[]
                for file in files:
                    geom_duplic = getoutData(file)
                    geom_duplic_cart.append(geom_duplic.CARTESIANS)
                    geom_duplic_files.append(file)
                geom_duplic_list.append(geom_duplic_cart)
                geom_duplic_list.append(geom_duplic_files)

                   #geom_duplic_list.append(round(geom_duplic.CARTESIANS, 4))
                for name in names_spc:
                    geom_duplic_spc = getoutData(name)
                    geom_duplic_cart_spc.append(geom_duplic_spc.CARTESIANS)
                    geom_duplic_files_spc.append(name)
                geom_duplic_list_spc.append(geom_duplic_cart_spc)
                geom_duplic_list_spc.append(geom_duplic_files_spc)
                spc_mismatching = "Caution! Potential differences found between frequency and single-point geometries -"
                if len(geom_duplic_list[0]) == len(geom_duplic_list_spc[0]):
                    for i in range(len(files)):
                        count = 1
                        for j in range(len(geom_duplic_list[0][i])):
                            if count == 1:
                                if geom_duplic_list[0][i][j] == geom_duplic_list_spc[0][i][j]:
                                    count = count
                                elif '{0:.3f}'.format(geom_duplic_list[0][i][j][0]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][0]*(-1)) or '{0:.3f}'.format(geom_duplic_list[0][i][j][0]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][0]):
                                    if '{0:.3f}'.format(geom_duplic_list[0][i][j][1]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][1]*(-1)) or '{0:.3f}'.format(geom_duplic_list[0][i][j][1]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][1]*(-1)):
                                        count = count
                                    if '{0:.3f}'.format(geom_duplic_list[0][i][j][2]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][2]*(-1)) or '{0:.3f}'.format(geom_duplic_list[0][i][j][2]) == '{0:.3f}'.format(geom_duplic_list_spc[0][i][j][2]*(-1)):
                                        count = count
                                else:
                                    spc_mismatching += ", " + geom_duplic_list[1][i]
                                    count = count + 1
                    if spc_mismatching == "Caution! Potential differences found between frequency and single-point geometries -":
                        log.Write("\no  No potential differences found between frequency and single-point geometries (based on input coordinates).")
                    else:
                        spc_mismatching_1 = spc_mismatching[:84]
                        spc_mismatching_2 = spc_mismatching[85:]
                        log.Write("\nx  " + spc_mismatching_1 + spc_mismatching_2 + '.')
                else:
                    log.Write("\nx  One or more geometries from single-point corrections are missing.")

                # Check for dispersion
                dispersion_check_spc = [sp_energy(name)[6] for name in names_spc]
                if all_same(dispersion_check_spc) != False:
                    if dispersion_check_spc[0] == 'No empirical dispersion detected':
                        log.Write("\n-  No empirical dispersion detected in any of the calculations.")
                    else:
                        log.Write("\no  Using "+dispersion_check_spc[0]+" in all the singe-point calculations.")
                else:
                  dispersion_check_spc_print = "Caution! Different dispersion models found - " + dispersion_check_spc[0] + " (" + names_spc[0]
                  for i in range(len(dispersion_check_spc)):
                     if dispersion_check_spc[i] == dispersion_check_spc[0] and i != 0:
                        dispersion_check_spc_print += ", " + names_spc[i]
                  dispersion_check_spc_print += ")"
                  for i in range(len(dispersion_check_spc)):
                     if dispersion_check_spc[i] != dispersion_check_spc[0] and i != 0:
                        dispersion_check_spc_print += ", " + dispersion_check_spc[i] + " (" + names_spc[i] + ")"
                  log.Write("\nx  " + dispersion_check_spc_print + ".")

                log.Write("\n"+STARS+"\n")

    # Running a variable temperature analysis of the enthalpy, entropy and the free energy
    elif options.temperature_interval != False:
        temperature_interval = [float(temp) for temp in options.temperature_interval.split(',')]
        # If no temperature step was defined, divide the region into 10
        if len(temperature_interval) == 2:
            temperature_interval.append((temperature_interval[1]-temperature_interval[0])/10.0)

        log.Write("\n\n   Variable-Temperature analysis of the enthalpy, entropy and the entropy at a constant pressure between")
        log.Write("\n   T_init:  %.1f,  T_final:  %.1f,  T_interval: %.1f" % (temperature_interval[0], temperature_interval[1], temperature_interval[2]))
        if options.QH:
            if options.spc is False:
                log.Write("\n\n   " + '{:<39} {:>13} {:>24} {:>13} {:>10} {:>10} {:>13} {:>13}'.format("Structure", "Temp/K", "H", "qh-H", "T.S", "T.qh-S", "G(T)", "qh-G(T)"),thermodata=True)
            else:
                log.Write("\n\n   " + '{:<39} {:>13} {:>24} {:>13} {:>10} {:>10} {:>13} {:>13}'.format("Structure", "Temp/K", "H_"+options.spc, "qh-H_"+options.spc, "T.S", "T.qh-S", "G(T)_"+options.spc, "qh-G(T)_"+options.spc),thermodata=True)
        else:
            if options.spc is False:
                log.Write("\n\n   " + '{:<39} {:>13} {:>24} {:>10} {:>10} {:>13} {:>13}'.format("Structure", "Temp/K", "H", "T.S", "T.qh-S", "G(T)", "qh-G(T)"),thermodata=True)
            else:
                log.Write("\n\n   " + '{:<39} {:>13} {:>24} {:>10} {:>10} {:>13} {:>13}'.format("Structure", "Temp/K", "H_"+options.spc, "T.S", "T.qh-S", "G(T)_"+options.spc, "qh-G(T)_"+options.spc),thermodata=True)

        for file in files: # loop over the output files
            log.Write("\n"+STARS)
            for i in range(int(temperature_interval[0]), int(temperature_interval[1]+1), int(temperature_interval[2])): # run through the temperature range
                temp, conc,linear_warning = float(i), ATMOS / GAS_CONSTANT / float(i),[]
                bbe = calc_bbe(file, options.QS, options.QH, options.S_freq_cutoff,options.H_freq_cutoff, temp,
                                conc, options.freq_scale_factor, options.freespace, options.spc, options.invert)
                linear_warning.append(bbe.linear_warning)
                if linear_warning == [['Warning! Potential invalid calculation of linear molecule from Gaussian.']]:
                    log.Write("\nx  ")
                    log.Write('{:<39}'.format(os.path.splitext(os.path.basename(file))[0]),thermodata=True)
                    log.Write('             Warning! Potential invalid calculation of linear molecule from Gaussian ...')
                else:
                    if hasattr(bbe, "scf_energy") and not hasattr(bbe,"gibbs_free_energy"):#gaussian spc files
                        log.Write("\nx  "+'{:<39}'.format(os.path.splitext(os.path.basename(file))[0]))
                    elif not hasattr(bbe, "scf_energy") and not hasattr(bbe,"gibbs_free_energy"): #orca spc files
                        log.Write("\nx  "+'{:<39}'.format(os.path.splitext(os.path.basename(file))[0]))
                    if not hasattr(bbe,"gibbs_free_energy"):
                        log.Write("Warning! Couldn't find frequency information ...")
                    else:
                        log.Write("\no  ")
                        log.Write('{:<39} {:13.1f}'.format(os.path.splitext(os.path.basename(file))[0], temp),thermodata=True)
                        if options.media == False:
                            if all(getattr(bbe, attrib) for attrib in ["enthalpy", "entropy", "qh_entropy", "gibbs_free_energy", "qh_gibbs_free_energy"]):
                                if options.QH:
                                    log.Write(' {:24.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(bbe.enthalpy, bbe.qh_enthalpy, (options.temperature * bbe.entropy), (options.temperature * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy))
                                else:
                                    log.Write(' {:24.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(bbe.enthalpy, (options.temperature * bbe.entropy), (options.temperature * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy))
                                if options.check != False:
                                    Gqh_duplic.append(bbe.qh_gibbs_free_energy)
                                    H_duplic.append(bbe.enthalpy)
                                    qh_entropy_duplic.append((options.temperature * bbe.qh_entropy))
                        else:
                            try:
                                from .media import solvents
                            except:
                                from media import solvents
                            if options.media.lower() in solvents and options.media.lower() == os.path.splitext(os.path.basename(file))[0].lower():
                                MW_solvent = solvents[options.media.lower()][0]
                                density_solvent = solvents[options.media.lower()][1]
                                concentration_solvent = (density_solvent*1000)/MW_solvent
                                media_correction = -(GAS_CONSTANT/J_TO_AU)*math.log(concentration_solvent)
                                if all(getattr(bbe, attrib) for attrib in ["enthalpy", "entropy", "qh_entropy", "gibbs_free_energy", "qh_gibbs_free_energy"]):
                                    if options.QH:
                                        log.Write(' {:10.6f} {:13.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(bbe.zpe, bbe.enthalpy, bbe.qh_enthalpy, (options.temperature * (bbe.entropy+media_correction)), (options.temperature * (bbe.qh_entropy+media_correction)), bbe.gibbs_free_energy+(options.temperature * (-media_correction)), bbe.qh_gibbs_free_energy+(options.temperature * (-media_correction))))
                                        log.Write("  Solvent")
                                else:
                                    log.Write(' {:10.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(bbe.zpe, bbe.enthalpy, (options.temperature * (bbe.entropy+media_correction)), (options.temperature * (bbe.qh_entropy+media_correction)), bbe.gibbs_free_energy+(options.temperature * (-media_correction)), bbe.qh_gibbs_free_energy+(options.temperature * (-media_correction))))
                                    log.Write("  Solvent")
                                if options.check != False:
                                    Gqh_duplic.append(bbe.qh_gibbs_free_energy)
                                    H_duplic.append(bbe.enthalpy)
                                    qh_entropy_duplic.append((options.temperature * bbe.qh_entropy))
                            else:
                                if all(getattr(bbe, attrib) for attrib in ["enthalpy", "entropy", "qh_entropy", "gibbs_free_energy", "qh_gibbs_free_energy"]):
                                    if options.QH:
                                        log.Write(' {:10.6f} {:13.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(bbe.zpe, bbe.enthalpy, bbe.qh_enthalpy, (options.temperature * bbe.entropy), (options.temperature * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy))
                                    else:
                                        log.Write(' {:10.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(bbe.zpe, bbe.enthalpy, (options.temperature * bbe.entropy), (options.temperature * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy))
                                    if options.check != False:
                                        Gqh_duplic.append(bbe.qh_gibbs_free_energy)
                                        H_duplic.append(bbe.enthalpy)
                                        qh_entropy_duplic.append((options.temperature * bbe.qh_entropy))
            log.Write("\n"+STARS+"\n")

    # Print CPU usage if requested
    if options.cputime != False:
        log.Write('   {:<13} {:>2} {:>4} {:>2} {:>3} {:>2} {:>4} {:>2} {:>4}\n'.format('TOTAL CPU', total_cpu_time.day + add_days - 1, 'days', total_cpu_time.hour, 'hrs', total_cpu_time.minute, 'mins', total_cpu_time.second, 'secs'))

    # Tabulate relative values
    if options.pes != False:
        for key in thermo_data:
            if not hasattr(thermo_data[key], "qh_gibbs_free_energy"):
                pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
                sys.exit(pes_error)
            if not hasattr(thermo_data[key], "sp_energy") and options.spc is not False:
                pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
                sys.exit(pes_error)


        PES = get_pes(options.pes, thermo_data, log, options)
        # Output the relative energy data
        if options.QH:
            zero_vals = [PES.spc_zero, PES.e_zero, PES.zpe_zero, PES.h_zero, PES.qh_zero, options.temperature * PES.ts_zero, options.temperature * PES.qhts_zero, PES.g_zero, PES.qhg_zero]
        else:
            zero_vals = [PES.spc_zero, PES.e_zero, PES.zpe_zero, PES.h_zero, options.temperature * PES.ts_zero, options.temperature * PES.qhts_zero, PES.g_zero, PES.qhg_zero]
        for i, path in enumerate(PES.path):
            if PES.boltz != False:
                e_sum, h_sum, g_sum, qhg_sum = 0.0, 0.0, 0.0, 0.0; sels = []
                for j, e_abs in enumerate(PES.e_abs[i]):
                    if options.QH:
                        species = [PES.spc_abs[i][j], PES.e_abs[i][j], PES.zpe_abs[i][j], PES.h_abs[i][j], PES.qh_abs[i][j], options.temperature * PES.s_abs[i][j], options.temperature * PES.qs_abs[i][j], PES.g_abs[i][j], PES.qhg_abs[i][j]]
                    else:
                        species = [PES.spc_abs[i][j], PES.e_abs[i][j], PES.zpe_abs[i][j], PES.h_abs[i][j], options.temperature * PES.s_abs[i][j], options.temperature * PES.qs_abs[i][j], PES.g_abs[i][j], PES.qhg_abs[i][j]]
                    relative = [species[x]-zero_vals[x] for x in range(len(zero_vals))]
                    e_sum += math.exp(-relative[1]*J_TO_AU/GAS_CONSTANT/options.temperature)
                    h_sum += math.exp(-relative[3]*J_TO_AU/GAS_CONSTANT/options.temperature)
                    g_sum += math.exp(-relative[7]*J_TO_AU/GAS_CONSTANT/options.temperature)
                    qhg_sum += math.exp(-relative[8]*J_TO_AU/GAS_CONSTANT/options.temperature)

            if options.spc is False:
                if options.QH:
                    log.Write("\n   " + '{:<39} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} {:>13}'.format("RXN:" + path + "(" + PES.units + ")", "DE", "DZPE", "DH", "qh-DH", "T.DS", "T.qh-DS", "DG(T)", "qh-DG(T)" ), thermodata=True)
                else:
                    log.Write("\n   " + '{:<39} {:>13} {:>10} {:>13} {:>10} {:>10} {:>13} {:>13}'.format("RXN:" + path + "(" + PES.units + ")", "DE", "DZPE", "DH", "T.DS", "T.qh-DS", "DG(T)", "qh-DG(T)" ), thermodata=True)
            else:
                if options.QH:
                    log.Write("\n   " + '{:<39} {:>13} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>14} {:>14}'.format("RXN: "+path+" ("+PES.units+")", "DE_"+options.spc, "DE", "DZPE", "DH_"+options.spc, "qh-DH_"+options.spc, "T.DS", "T.qh-DS", "DG(T)_"+options.spc, "qh-DG(T)_"+options.spc), thermodata=True)
                else:
                    log.Write("\n   " + '{:<39} {:>13} {:>13} {:>10} {:>13} {:>10} {:>10} {:>14} {:>14}'.format("RXN: "+path+" ("+PES.units+")", "DE_"+options.spc, "DE", "DZPE", "DH_"+options.spc, "T.DS", "T.qh-DS", "DG(T)_"+options.spc, "qh-DG(T)_"+options.spc), thermodata=True)
            log.Write("\n"+STARS)

            for j, e_abs in enumerate(PES.e_abs[i]):
                if options.QH:
                    species = [PES.spc_abs[i][j], PES.e_abs[i][j], PES.zpe_abs[i][j], PES.h_abs[i][j], PES.qh_abs[i][j], options.temperature * PES.s_abs[i][j], options.temperature * PES.qs_abs[i][j], PES.g_abs[i][j], PES.qhg_abs[i][j]]
                else:
                    species = [PES.spc_abs[i][j], PES.e_abs[i][j], PES.zpe_abs[i][j], PES.h_abs[i][j], options.temperature * PES.s_abs[i][j], options.temperature * PES.qs_abs[i][j], PES.g_abs[i][j], PES.qhg_abs[i][j]]
                relative = [species[x]-zero_vals[x] for x in range(len(zero_vals))]
                if PES.units == 'kJ/mol':
                    formatted_list = [J_TO_AU / 1000.0 * x for x in relative]
                else:
                    formatted_list = [KCAL_TO_AU * x for x in relative] # defaults to kcal/mol
                log.Write("\no  ")
                if options.spc is False:
                    formatted_list = formatted_list[1:]
                    if options.QH:
                        if PES.dec == 1:
                            log.Write('{:<39} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} {:13.1f}'.format(PES.species[i][j], *formatted_list), thermodata=True)
                        if PES.dec == 2:
                            log.Write('{:<39} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} {:13.2f}'.format(PES.species[i][j], *formatted_list), thermodata=True)
                    else:
                        if PES.dec == 1:
                            log.Write('{:<39} {:13.1f} {:10.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} {:13.1f}'.format(PES.species[i][j], *formatted_list), thermodata=True)
                        if PES.dec == 2:
                            log.Write('{:<39} {:13.2f} {:10.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} {:13.2f}'.format(PES.species[i][j], *formatted_list), thermodata=True)
                else:
                    if options.QH:
                        if PES.dec == 1:
                            log.Write('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} {:13.1f}'.format(PES.species[i][j], *formatted_list), thermodata=True)
                        if PES.dec == 2:
                            log.Write('{:<39} {:13.1f} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} {:13.2f}'.format(PES.species[i][j], *formatted_list), thermodata=True)
                    else:
                        if PES.dec == 1:
                            log.Write('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} {:13.1f}'.format(PES.species[i][j], *formatted_list), thermodata=True)
                        if PES.dec == 2:
                            log.Write('{:<39} {:13.1f} {:13.2f} {:10.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} {:13.2f}'.format(PES.species[i][j], *formatted_list), thermodata=True)
                if PES.boltz != False:
                    boltz = [math.exp(-relative[1]*J_TO_AU/GAS_CONSTANT/options.temperature)/e_sum, math.exp(-relative[3]*J_TO_AU/GAS_CONSTANT/options.temperature)/h_sum, math.exp(-relative[6]*J_TO_AU/GAS_CONSTANT/options.temperature)/g_sum, math.exp(-relative[7]*J_TO_AU/GAS_CONSTANT/options.temperature)/qhg_sum]
                    selectivity = [boltz[x]*100.0 for x in range(len(boltz))]
                    log.Write("\n  "+'{:<39} {:13.2f}%{:24.2f}%{:35.2f}%{:13.2f}%'.format('', *selectivity))
                    sels.append(selectivity)

            if PES.boltz == 'ee' and len(sels) == 2:
                ee = [sels[0][x]-sels[1][x] for x in range(len(sels[0]))]
                if options.spc is False:
                    log.Write("\n"+STARS+"\n   "+'{:<39} {:13.1f}%{:24.1f}%{:35.1f}%{:13.1f}%'.format('ee (%)', *ee))
                else:
                    log.Write("\n"+STARS+"\n   "+'{:<39} {:27.1f} {:24.1f} {:35.1f} {:13.1f} '.format('ee (%)', *ee))
            log.Write("\n"+STARS+"\n")

    if options.ee is not False:#compute enantiomeric excess
        EE_STARS = "   " + '*' * 81
        boltz_facs, weighted_free_energy, boltz_sum = get_boltz(files,thermo_data,clustering,options.temperature)
        ee, dd_free_energy,failed,RS_preference = get_ee(options.ee,files,boltz_facs,boltz_sum,options.temperature,log)
        if not failed:
            log.Write("\n   " + '{:<39} {:>13} {:>13} {:>13}'.format("Enantioselectivity" , "%ee", "ddG", "Abs. Config."), thermodata=True)
            log.Write("\n"+EE_STARS)
            log.Write("\no  ")
            log.Write('{:<39} {:13.2f} {:13.2f} {:>13}'.format(options.ee,ee,dd_free_energy,RS_preference), thermodata=True)
            log.Write("\n"+EE_STARS+"\n")

    #graph reaction profiles
    if options.graph is not False:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            log.Write("\n\n   Warning! matplotlib module is not installed, reaction profile will not be graphed.")
            log.Write("\n   To install matplotlib, run the following commands: \n\t   python -m pip install -U pip" +
                        "\n\t   python -m pip install -U matplotlib\n\n")

        for key in thermo_data:
            if not hasattr(thermo_data[key], "qh_gibbs_free_energy"):
                pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
                sys.exit(pes_error)
            if not hasattr(thermo_data[key], "sp_energy") and options.spc is not False:
                pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
                sys.exit(pes_error)

        graph_data = get_pes(options.graph, thermo_data, log, options)
        graph_reaction_profile(graph_data,log,options,plt)


    # Close the log
    log.Finalize()
    if options.xyz:
        xyz.Finalize()

if __name__ == "__main__":
    main()
