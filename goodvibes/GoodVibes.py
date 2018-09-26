
#!/usr/bin/python
from __future__ import print_function, absolute_import

#######################################################################
#                              GoodVibes.py                           #
#  Evaluation of quasi-harmonic thermochemistry from Gaussian.        #
#  Partion functions are evaluated from vibrational frequencies       #
#  and rotational temperatures from the standard output.              #
#  The rigid-rotor harmonic oscillator approximation is used as       #
#  standard for all frequencies above a cut-off value. Below this,    #
#  two treatments can be applied:                                     #
#    (a) low frequencies are shifted to the cut-off value (as per     #
#    Cramer-Truhlar)                                                  #
#    (b) a free-rotor approximation is applied below the cut-off (as  #
#    per Grimme). In this approach, a damping function interpolates   #
#    between the RRHO and free-rotor entropy treatment of Svib to     #
#    avoid a discontinuity.                                           #
#  Both approaches avoid infinitely large values of Svib as wave-     #
#  numbers tend to zero. With a cut-off set to 0, the results will be #
#  identical to standard values output by the Gaussian program.       #
#  The free energy can be evaluated for variable temperature,         #
#  concentration, vibrational scaling factor, and with a haptic       #
#  correction of the translational entropy in different solvents,     #
#  according to the amount of free space available.                   #
#######################################################################
#######  Written by:  Rob Paton and Ignacio Funes-Ardoiz ##############
#######  Last modified:   May 16, 2018 ################################
#######################################################################

import os.path, sys, math, textwrap, time
from datetime import datetime, timedelta
from glob import glob
from optparse import OptionParser

from .vib_scale_factors import scaling_data, scaling_refs

# PHYSICAL CONSTANTS
GAS_CONSTANT, PLANCK_CONSTANT, BOLTZMANN_CONSTANT, SPEED_OF_LIGHT, AVOGADRO_CONSTANT, AMU_to_KG, atmos = 8.3144621, 6.62606957e-34, 1.3806488e-23, 2.99792458e10, 6.0221415e23, 1.66053886E-27, 101.325
# UNIT CONVERSION
j_to_au = 4.184 * 627.509541 * 1000.0
kcal_to_au = 627.509541

# version number
__version__ = "2.0.2"
stars = "   " + "*" * 128

# some literature references
grimme_ref = "Grimme, S. Chem. Eur. J. 2012, 18, 9955-9964"
truhlar_ref = "Ribeiro, R. F.; Marenich, A. V.; Cramer, C. J.; Truhlar, D. G. J. Phys. Chem. B 2011, 115, 14556-14562"
goodvibes_ref = "Funes-Ardoiz, I.; Paton, R. S. (2016). GoodVibes: GoodVibes "+__version__+" http://doi.org/10.5281/zenodo.595246"

#Some useful arrays
periodictable = ["","H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr",
    "Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl",
    "Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Uub","Uut","Uuq","Uup","Uuh","Uus","Uuo"]

def elementID(massno):
    if massno < len(periodictable): return periodictable[massno]
    else: return "XX"

alphabet = 'abcdefghijklmnopqrstuvwxyz'

 # Enables output to terminal and to text file
class Logger:
   def __init__(self, filein, suffix, append):
      self.log = open(filein+"_"+append+"."+suffix, 'w' )

   def Write(self, message):
      print(message, end='')
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
      self.xyz = open(filein+"_"+append+"."+suffix, 'w' )

   def Writetext(self, message):
      self.xyz.write(message + "\n")

   def Writecoords(self, atoms, coords):
      for n, carts in enumerate(coords):
          self.xyz.write('{:>1}'.format(atoms[n]))
          for cart in carts: self.xyz.write('{:13.6f}'.format(cart))
          self.xyz.write('\n')

   def Finalize(self):
      self.xyz.close()

# Read solvation free energies from a COSMO-RS dat file
def COSMORSout(datfile, names):

   GSOLV = {}
   if os.path.exists(os.path.splitext(datfile)[0]+'.out'):
       with open(os.path.splitext(datfile)[0]+'.out') as f: data = f.readlines()
   else:
       raise ValueError("File {} does not exist".format(datfile))

   for i, line in enumerate(data):
       for name in names:
           if line.find('('+name.split('.')[0]+')') > -1 and line.find('Compound') > -1:
               if data[i+10].find('Gibbs') > -1:
                   gsolv = float(data[i+10].split()[6].strip()) / kcal_to_au
                   GSOLV[name] = gsolv
   return GSOLV

#Read molecule data from a compchem output file
class getoutData:
    def __init__(self, file):
        with open(file) as f: data = f.readlines()
        program = 'none'

        for line in data:
           if line.find("Gaussian") > -1: program = "Gaussian"; break

        def getATOMTYPES(self, outlines, program):
            if program == "Gaussian":
                for i, line in enumerate(outlines):
                    if line.find("Input orientation") >-1 or line.find("Standard orientation") > -1:
                        self.ATOMTYPES, self.CARTESIANS, self.ATOMICTYPES, carts = [], [], [], outlines[i+5:]
                        for j, line in enumerate(carts):
                            if line.find("-------") > -1: break
                            self.ATOMTYPES.append(elementID(int(line.split()[1])))
                            self.ATOMICTYPES.append(int(line.split()[2]))
                            if len(line.split()) > 5: self.CARTESIANS.append([float(line.split()[3]),float(line.split()[4]),float(line.split()[5])])
                            else: self.CARTESIANS.append([float(line.split()[2]),float(line.split()[3]),float(line.split()[4])])

        getATOMTYPES(self, data, program)

# Read gaussian output for a single point energy
def sp_energy(file):
   spe, program, data = 'none', 'none', []

   if os.path.exists(os.path.splitext(file)[0]+'.log'):
       with open(os.path.splitext(file)[0]+'.log') as f: data = f.readlines()
   elif os.path.exists(os.path.splitext(file)[0]+'.out'):
       with open(os.path.splitext(file)[0]+'.out') as f: data = f.readlines()
   else:
       raise ValueError("File {} does not exist".format(file))

   for line in data:
       if line.find("Gaussian") > -1: program = "Gaussian"; break
       if line.find("* O   R   C   A *") > -1: program = "Orca"; break

   for line in data:
       if program == "Gaussian":
           if line.strip().startswith('SCF Done:'): spe = float(line.strip().split()[4])
           if line.strip().startswith('Counterpoise corrected energy'): spe = float(line.strip().split()[4])
           # For MP2 calculations replace with EUMP2
           if line.strip().find('EUMP2 =') > -1: spe = float((line.strip().split()[5]).replace('D', 'E'))
           # For ONIOM calculations use the extrapolated value rather than SCF value
           if line.strip().find("ONIOM: extrapolated energy") > -1: spe = (float(line.strip().split()[4]))
           # For Semi-empirical or Molecular Mechanics calculations
           if line.strip().find("Energy= ") > -1 and line.strip().find("Predicted")==-1 and line.strip().find("Thermal")==-1: spe = (float(line.strip().split()[1]))
       if program == "Orca":
           if line.strip().startswith('FINAL SINGLE POINT ENERGY'): spe = float(line.strip().split()[4])
   return spe

# Read single-point output for cpu time
def sp_cpu(file):
   #print(file)
   spe, program, data, cpu = None, None, [], None

   if os.path.exists(os.path.splitext(file)[0]+'.log'):
       with open(os.path.splitext(file)[0]+'.log') as f: data = f.readlines()
   elif os.path.exists(os.path.splitext(file)[0]+'.out'):
       with open(os.path.splitext(file)[0]+'.out') as f: data = f.readlines()
   else:
       raise ValueError("File {} does not exist".format(file))

   for line in data:
       if line.find("Gaussian") > -1: program = "Gaussian"; break
       if line.find("* O   R   C   A *") > -1: program = "Orca"; break

   for line in data:
       if program == "Gaussian":
           if line.strip().startswith('SCF Done:'): spe = float(line.strip().split()[4])
           if line.strip().find("Job cpu time") > -1:
              days = int(line.split()[3]); hours = int(line.split()[5]); mins = int(line.split()[7]); secs = 0; msecs = int(float(line.split()[9])*1000.0)
              cpu = [days,hours,mins,secs,msecs]
       if program == "Orca":
           if line.strip().startswith('FINAL SINGLE POINT ENERGY'): spe = float(line.strip().split()[4])
           if line.strip().find("TOTAL RUN TIME") > -1:
               days = int(line.split()[3]); hours = int(line.split()[5]); mins = int(line.split()[7]); secs = int(line.split()[9]); msecs = float(line.split()[11])
               cpu = [days,hours,mins,secs,msecs]
   #print('orca', cpu)
   return cpu

# Read output for the level of theory and basis set used
def level_of_theory(file):
   with open(file) as f: data = f.readlines()
   level, bs = 'none', 'none'
   for line in data:
      if line.strip().find('External calculation') > -1:
          level, bs = 'ext', 'ext'
          break
      if line.strip().find('\\Freq\\') > -1:
          try: level, bs = (line.strip().split("\\")[4:6])
          except IndexError: pass
   for line in data:
      if line.strip().find('\\DLPNO BASED TRIPLES CORRECTION\\') > -1: level = 'DLPNO-CCSD(T)'
      if line.strip().find('\\Estimated CBS total energy\\') > -1: bs = 'CBS(2/3)'
      # remove the restricted R or unrestricted U label
      if level[0] == 'R' or level[0] == 'U': level = level[1:]
   return level+"/"+bs

def addTime(tm, cpu):
    [days, hrs, mins, secs, msecs] = cpu
    fulldate = datetime(100, 1, tm.day, tm.hour, tm.minute, tm.second, tm.microsecond)
    fulldate = fulldate + timedelta(days=days, hours=hrs, minutes=mins, seconds=secs, microseconds=msecs*1000)
    return fulldate

# translational energy evaluation (depends on temperature)
def calc_translational_energy(temperature):
   """
   Calculates the translational energy (J/mol) of an ideal gas - i.e. non-interactiing molecules so molar energy = Na * atomic energy
   This approximation applies to all energies and entropies computed within
   Etrans = 3/2 RT!
   """
   energy = 1.5 * GAS_CONSTANT * temperature
   return energy

# rotational energy evaluation (depends on molecular shape and temperature)
def calc_rotational_energy(zpe, symmno, temperature, linear):
   """
   Calculates the rotaional energy (J/mol)
   Etrans = 0 (atomic) ; RT (linear); 3/2 RT (non-linear)
   """
   if zpe == 0.0: energy = 0.0
   elif linear == 1: energy = GAS_CONSTANT * temperature
   else: energy = 1.5 * GAS_CONSTANT * temperature
   return energy

# vibrational energy evaluation (depends on frequencies, temperature and scaling factor: default = 1.0)
def calc_vibrational_energy(frequency_wn, temperature, freq_scale_factor):
   """
   Calculates the vibrational energy contribution (J/mol). Includes ZPE (0K) and thermal contributions
   Evib = R * Sum(0.5 hv/k + (hv/k)/(e^(hv/KT)-1))
   """
   factor = [(PLANCK_CONSTANT * freq * SPEED_OF_LIGHT * freq_scale_factor) / (BOLTZMANN_CONSTANT * temperature) for freq in frequency_wn]
   energy = [entry * GAS_CONSTANT * temperature * (0.5 + (1.0 / (math.exp(entry) - 1.0))) for entry in factor]
   return sum(energy)

# vibrational Zero point energy evaluation (depends on frequencies and scaling factor: default = 1.0)
def calc_zeropoint_energy(frequency_wn, freq_scale_factor):
   """
   Calculates the vibrational ZPE (J/mol)
   EZPE = Sum(0.5 hv/k)
   """
   factor = [PLANCK_CONSTANT * freq * SPEED_OF_LIGHT * freq_scale_factor / BOLTZMANN_CONSTANT for freq in frequency_wn]
   energy = [0.5 * entry * GAS_CONSTANT for entry in factor]
   return sum(energy)

# Computed the amount of accessible free space (ml per L) in solution accesible to a solute immersed in bulk solvent, i.e. this is the volume not occupied by solvent molecules, calculated using literature values for molarity and B3LYP/6-31G* computed molecular volumes.
def get_free_space(solv):
   """
   Calculates the free space in a litre of bulk solvent, based on Shakhnovich and Whitesides (J. Org. Chem. 1998, 63, 3821-3830)
   """
   solvent_list = ["none", "H2O", "toluene", "DMF", "AcOH", "chloroform"]
   molarity = [1.0, 55.6, 9.4, 12.9, 17.4, 12.5] #mol/l
   molecular_vol = [1.0, 27.944, 149.070, 77.442, 86.10, 97.0] #Angstrom^3

   nsolv = 0
   for i in range(0,len(solvent_list)):
      if solv == solvent_list[i]: nsolv = i

   solv_molarity = molarity[nsolv]
   solv_volume = molecular_vol[nsolv]

   if nsolv > 0:
      V_free = 8 * ((1E27/(solv_molarity * AVOGADRO_CONSTANT)) ** 0.333333 - solv_volume ** 0.333333) ** 3
      freespace = V_free * solv_molarity * AVOGADRO_CONSTANT * 1E-24
   else: freespace = 1000.0
   return freespace

# translational entropy evaluation (depends on mass, concentration, temperature, solvent free space: default = 1000.0)
def calc_translational_entropy(molecular_mass, conc, temperature, solv):
   """
   Calculates the translational entropic contribution (J/(mol*K)) of an ideal gas. Needs the molecular mass. Convert mass in amu to kg; conc in mol/l to number per m^3
   Strans = R(Ln(2pimkT/h^2)^3/2(1/C)) + 1 + 3/2)
   """
   lmda = ((2.0 * math.pi * molecular_mass * AMU_to_KG * BOLTZMANN_CONSTANT * temperature)**0.5) / PLANCK_CONSTANT
   freespace = get_free_space(solv)
   Ndens = conc * 1000 * AVOGADRO_CONSTANT / (freespace/1000.0)
   entropy = GAS_CONSTANT * (2.5 + math.log(lmda**3 / Ndens))
   return entropy

# electronic entropy evaluation (depends on multiplicity)
def calc_electronic_entropy(multiplicity):
   """
   Calculates the electronic entropic contribution (J/(mol*K)) of the molecule
   Selec = R(Ln(multiplicity)
   """
   entropy = GAS_CONSTANT * (math.log(multiplicity))
   return entropy

# rotational entropy evaluation (depends on molecular shape and temp.)
def calc_rotational_entropy(zpe, linear, symmno, rotemp, temperature):
   """
   Calculates the rotational entropy (J/(mol*K))
   Strans = 0 (atomic) ; R(Ln(q)+1) (linear); R(Ln(q)+3/2) (non-linear)
   """
   # monatomic
   if rotemp == [0.0,0.0,0.0] or zpe == 0.0: entropy = 0.0
   else:
      if len(rotemp) == 1: # diatomic or linear
              qrot = temperature/rotemp[0]
      else:
         qrot = math.pi*temperature**3/(rotemp[0]*rotemp[1]*rotemp[2])
         qrot = qrot ** 0.5

      if linear == 1: entropy = GAS_CONSTANT * (math.log(qrot / symmno) + 1)
      else: entropy = GAS_CONSTANT * (math.log(qrot / symmno) + 1.5)
   return entropy

# rigid rotor harmonic oscillator (RRHO) entropy evaluation - this is the default treatment
def calc_rrho_entropy(frequency_wn, temperature, freq_scale_factor):
   """
   Entropic contributions (J/(mol*K)) according to a rigid-rotor harmonic-oscillator description for a list of vibrational modes
   Sv = RSum(hv/(kT(e^(hv/KT)-1) - ln(1-e^(-hv/kT)))
   """
   factor = [PLANCK_CONSTANT * freq * SPEED_OF_LIGHT * freq_scale_factor / BOLTZMANN_CONSTANT / temperature for freq in frequency_wn]
   entropy = [entry * GAS_CONSTANT / (math.exp(entry) - 1) - GAS_CONSTANT * math.log(1 - math.exp(-entry)) for entry in factor]
   return entropy

# free rotor entropy evaluation - used for low frequencies below the cut-off if qh=grimme is specified
def calc_freerot_entropy(frequency_wn, temperature, freq_scale_factor):
   """
   Entropic contributions (J/(mol*K)) according to a free-rotor description for a list of vibrational modes
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

# The funtion to compute the "black box" entropy values (and all other thermochemical quantities)
class calc_bbe:
   def __init__(self, file, QH, FREQ_CUTOFF, temperature, conc, freq_scale_factor, solv, spc):
      # List of frequencies and default values
      im_freq_cutoff, frequency_wn, im_frequency_wn, rotemp, linear_mol, link, freqloc, linkmax, symmno, self.cpu = 0.0, [], [], [0.0,0.0,0.0], 0, 0, 0, 0, 1, [0,0,0,0,0]

      with open(file) as f: g_output = f.readlines()

      # read any single point energies if requested
      if spc != False and spc != 'link':
         name, ext = os.path.splitext(file)
         try:
             self.sp_energy = sp_energy(name+'_'+spc+ext)
             self.cpu = sp_cpu(name+'_'+spc+ext)
         except ValueError:
             self.sp_energy = '!'; pass
      if spc == 'link':
          self.sp_energy = sp_energy(file)

      #count number of links
      for line in g_output:
         # only read first link + freq not other link jobs
         if line.find("Normal termination") != -1: linkmax += 1
         if line.find('Frequencies --') != -1: freqloc = linkmax

      # Iterate over output
      if freqloc == 0: freqloc = len(g_output)
      for line in g_output:
         # link counter
         if line.find("Normal termination")!= -1:
            link += 1
            # reset frequencies if in final freq link
            if link == freqloc: frequency_wn = []
         # if spc specified will take last Energy from file, otherwise will break after freq calc
         if link > freqloc: break

      	 # Iterate over output: look out for low frequencies
         if line.strip().startswith('Frequencies -- '):
            for i in range(2,5):
               try:
                  x = float(line.strip().split()[i])
                  #  only deal with real frequencies
                  if x > 0.00: frequency_wn.append(x)
                  if x < 0.00: im_frequency_wn.append(x)
               except IndexError: pass

         # For QM calculations look for SCF energies, last one will be the optimized energy
         if line.strip().startswith('SCF Done:'): self.scf_energy = float(line.strip().split()[4])
         # For Counterpoise calculations the corrected energy value will be taken
         if line.strip().startswith('Counterpoise corrected energy'): self.scf_energy = float(line.strip().split()[4])
         # For MP2 calculations replace with EUMP2
         if line.strip().find('EUMP2 =') > -1: self.scf_energy = float((line.strip().split()[5]).replace('D', 'E'))
         # For ONIOM calculations use the extrapolated value rather than SCF value
         if line.strip().find("ONIOM: extrapolated energy") > -1: self.scf_energy = (float(line.strip().split()[4]))
         # For Semi-empirical or Molecular Mechanics calculations
         if line.strip().find("Energy= ") > -1 and line.strip().find("Predicted")==-1 and line.strip().find("Thermal")==-1: self.scf_energy = (float(line.strip().split()[1]))
         # look for thermal corrections, paying attention to point group symmetry
         if line.strip().startswith('Zero-point correction='): self.zero_point_corr = float(line.strip().split()[2])
         if line.strip().find('Multiplicity') > -1: mult = float(line.split('=')[-1].strip())
         if line.strip().startswith('Molecular mass:'): molecular_mass = float(line.strip().split()[2])
         if line.strip().startswith('Rotational symmetry number'): symmno = int((line.strip().split()[3]).split(".")[0])
         if line.strip().startswith('Full point group'):
            if line.strip().split()[3] == 'D*H' or line.strip().split()[3] == 'C*V': linear_mol = 1
         if line.strip().startswith('Rotational temperature '): rotemp = [float(line.strip().split()[3])]
         if line.strip().startswith('Rotational temperatures'):
             try: rotemp = [float(line.strip().split()[3]), float(line.strip().split()[4]), float(line.strip().split()[5])]
             except ValueError: rotemp = None
             #else: rotemp = [1E10, float(line.strip().split()[4]), float(line.strip().split()[5])]
         if line.strip().find("Job cpu time") > -1:
             days = int(line.split()[3]) + self.cpu[0]; hours = int(line.split()[5]) + self.cpu[1]; mins = int(line.split()[7]) + self.cpu[2]; secs = 0 + self.cpu[3]; msecs = int(float(line.split()[9])*1000.0) + self.cpu[4]
             self.cpu = [days,hours,mins,secs,msecs]
             #print('CPU', self.cpu)

      # skip the next steps if unable to parse the frequencies or zpe from the output file
      if hasattr(self, "zero_point_corr") and rotemp:
         # create a list of frequencies equal to cut-off value
         cutoffs = [FREQ_CUTOFF for freq in frequency_wn]

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
             if FREQ_CUTOFF > 0.0: Svib_rrqho = calc_rrho_entropy(cutoffs, temperature, 1.0)
             Svib_free_rot = calc_freerot_entropy(frequency_wn, temperature, freq_scale_factor)
             damp = calc_damp(frequency_wn, FREQ_CUTOFF)

             # Compute entropy (cal/mol/K) using the two values and damping function
             vib_entropy = []
             for j in range(0,len(frequency_wn)):
                if QH == "grimme": vib_entropy.append(Svib_rrho[j] * damp[j] + (1-damp[j]) * Svib_free_rot[j])
                elif QH == "truhlar":
                   if FREQ_CUTOFF > 0.0:
                      if frequency_wn[j] > FREQ_CUTOFF: vib_entropy.append(Svib_rrho[j])
                      else: vib_entropy.append(Svib_rrqho[j])
                   else: vib_entropy.append(Svib_rrho[j])
             qh_Svib, h_Svib = sum(vib_entropy), sum(Svib_rrho)

         # monatomic species have no vibrational or rotational degrees of freedom
         else: ZPE, Urot, Uvib, Srot, h_Svib, qh_Svib = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

         # Add terms (converted to au) to get Free energy - perform separately for harmonic and quasi-harmonic values out of interest
         self.enthalpy = self.scf_energy + (Utrans + Urot + Uvib + GAS_CONSTANT * temperature) / j_to_au
         # single point correction replaces energy from optimization with single point value
         if hasattr(self, 'sp_energy'):
            try: self.enthalpy = self.enthalpy - self.scf_energy + self.sp_energy
            except TypeError: pass
         self.zpe = ZPE / j_to_au
         self.entropy, self.qh_entropy = (Strans + Srot + h_Svib + Selec) / j_to_au, (Strans + Srot + qh_Svib + Selec) / j_to_au
         self.gibbs_free_energy, self.qh_gibbs_free_energy = self.enthalpy - temperature * self.entropy, self.enthalpy - temperature * self.qh_entropy
         self.im_freq = []
         for freq in im_frequency_wn:
             if freq < -1 * im_freq_cutoff: self.im_freq.append(freq)

def main():
   # Start a log for the results
   log = Logger("Goodvibes","dat", "output")

   # get command line inputs. Use -h to list all possible arguments and default values
   parser = OptionParser(usage="Usage: %prog [options] <input1>.log <input2>.log ...")
   parser.add_option("-t", dest="temperature", action="store", help="temperature (K) (default 298.15)", default="298.15", type="float", metavar="TEMP")
   parser.add_option("-q", dest="QH", action="store", help="Type of quasi-harmonic correction (Grimme or Truhlar) (default Grimme)", default="grimme", type="string", metavar="QH")
   parser.add_option("-f", dest="freq_cutoff", action="store", help="Cut-off frequency (wavenumbers) (default = 100)", default="100.0", type="float", metavar="FREQ_CUTOFF")
   parser.add_option("-c", dest="conc", action="store", help="concentration (mol/l) (default 1 atm)", default="0.040876", type="float", metavar="CONC")
   parser.add_option("-v", dest="freq_scale_factor", action="store", help="Frequency scaling factor (default 1)", default=False, type="float", metavar="SCALE_FACTOR")
   parser.add_option("-s", dest="solv", action="store", help="Solvent (H2O, toluene, DMF, AcOH, chloroform) (default none)", default="none", type="string", metavar="SOLV")
   parser.add_option("--spc", dest="spc", action="store", help="Indicates single point corrections (default False)", type="string", default=False, metavar="SPC")
   parser.add_option("--boltz", dest="boltz", action="store_true", help="Show Boltzmann factors", default=False, metavar="BOLTZ")
   parser.add_option("--cpu", dest="cputime", action="store_true", help="Total CPU time", default=False, metavar="CPU")
   parser.add_option("--ti", dest="temperature_interval", action="store", help="initial temp, final temp, step size (K)", default=False, metavar="TI")
   parser.add_option("--ci", dest="conc_interval", action="store", help="initial conc, final conc, step size (mol/l)", default=False, metavar="CI")
   parser.add_option("--xyz", dest="xyz", action="store_true", help="write Cartesians to an xyz file (default False)", default=False, metavar="XYZ")
   parser.add_option("--imag", dest="imag_freq", action="store_true", help="print imaginary frequencies (default False)", default=False, metavar="IMAG_FREQ")
   parser.add_option("--cosmo", dest="cosmo", action="store", help="filename of a COSMO-RS out file", default=False, metavar="COSMO-RS")
   parser.add_option("--csv", dest="csv", action="store_true", help="print CSV format", default=False, metavar="CSV")

   (options, args) = parser.parse_args()
   options.QH = options.QH.lower() # case insensitive

   # if necessary create an xyz file for Cartesians
   if options.xyz == True: xyz = XYZout("Goodvibes","xyz", "output")

   # initialize the total CPU time
   total_cpu_time = datetime(100, 1, 1, 00, 00, 00, 00)
   add_days = 0

   command = '   Requested: '

   clustering = False
   if len(sys.argv) > 1:
      for elem in sys.argv[1:]:
          if elem == 'clust:':
              clustering = True; options.boltz = True
              clusters = []; nclust = -1

   # Get the filenames from the command line prompt
   files = []
   if len(sys.argv) > 1:
      for elem in sys.argv[1:]:
         if clustering == True:
            if elem == 'clust:':
               clusters.append([]); nclust += 0
         try:
            if os.path.splitext(elem)[1] in [".out", ".log"]:
               for file in glob(elem):
                   if options.spc == False or options.spc == 'link':
                       files.append(file)
                       if clustering == True: clusters[nclust].append(file)
                   else:
                       if file.find('_'+options.spc+".") == -1:
                           files.append(file)
                           if clustering == True: clusters[nclust].append(file)
            elif elem != 'clust:': command += elem + ' '
         except IndexError: pass

      # Start printing results
      start = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
      log.Write("   GoodVibes v" + __version__ + " " + start + "\n   REF: " + goodvibes_ref +"\n")
      if clustering ==True: command += '(clustering active)'
      log.Write(command+'\n\n')
      if options.temperature_interval == False: log.Write("   Temperature = "+str(options.temperature)+" Kelvin")
      # If not at standard temp, need to correct the molarity of 1 atmosphere (assuming Pressure is still 1 atm)
      if options.conc == 0.040876:
          options.conc = atmos/(GAS_CONSTANT*options.temperature); log.Write("   Pressure = 1 atm")
      else: log.Write("   Concentration = "+str(options.conc)+" mol/l")

      # attempt to automatically obtain frequency scale factor. Requires all outputs to be same level of theory
      if options.freq_scale_factor == False:
          l_o_t = [level_of_theory(file) for file in files]
          def all_same(items): return all(x == items[0] for x in items)

          if all_same(l_o_t) == True:
             for scal in scaling_data: # search through database of scaling factors
                if l_o_t[0].upper() == scal['level'].upper() or l_o_t[0].upper() == scal['level'].replace("-","").upper():
                   options.freq_scale_factor = scal['zpe_fac']; ref = scaling_refs[scal['zpe_ref']]
                   log.Write("\n\n   " + "Found vibrational scaling factor for " + l_o_t[0] + " level of theory" + "\n   REF: " + ref)
          elif all_same(l_o_t) == False: log.Write("\n   " + (textwrap.fill("CAUTION: different levels of theory found - " + '|'.join(l_o_t), 128, subsequent_indent='   ')))

      if options.freq_scale_factor == False: options.freq_scale_factor = 1.0 # if no scaling factor is found use 1.0
      log.Write("\n   Frequency scale factor "+str(options.freq_scale_factor))

      # checks to see whether the available free space of a requested solvent is defined
      freespace = get_free_space(options.solv)
      if freespace != 1000.0: log.Write("\n   Specified solvent "+options.solv+": free volume "+str("%.3f" % (freespace/10.0))+" (mol/l) corrects the translational entropy")

      # read from COSMO-RS output
      if options.cosmo != False:
          try:
              cosmo_solv = COSMORSout(options.cosmo, files)
              log.Write('\n\n   Reading COSMO-RS file: '+options.cosmo+'.out')
          except ValueError:
              log.Write('\n\n   Warning: COSMO-RS file '+options.cosmo+'.out requested but not found')
              cosmo_solv = None

      # summary of the quasi-harmonic treatment; print out the relevant reference
      log.Write("\n\n   Quasi-harmonic treatment: frequency cut-off value of "+str(options.freq_cutoff)+" wavenumbers will be applied")
      if options.QH == "grimme": log.Write("\n   QH = Grimme: Using a mixture of RRHO and Free-rotor vibrational entropies"); qh_ref = grimme_ref
      elif options.QH == "truhlar": log.Write("\n   QH = Truhlar: Using an RRHO treatment where low frequencies are adjusted to the cut-off value"); qh_ref = truhlar_ref
      else: log.Fatal("\n   FATAL ERROR: Unknown quasi-harmonic model "+options.QH+" specified (QH must = grimme or truhlar)")
      log.Write("\n   REF: " + qh_ref)

      # whether linked single-point energies are to be used
      if options.spc == "True": log.Write("\n   Link job: combining final single point energy with thermal corrections")

   # Standard mode: tabulate thermochemistry ouput from file(s) at a single temperature and concentration
   if options.temperature_interval == False and options.conc_interval == False:
      if options.csv == False:
          if options.spc == False: log.Write("\n\n   " + '{:<39} {:>13} {:>10} {:>13} {:>10} {:>10} {:>13} {:>13}'.format("Structure", "E", "ZPE", "H", "T.S", "T.qh-S", "G(T)", "qh-G(T)"))
          else: log.Write("\n\n   " + '{:<39} {:>13} {:>13} {:>10} {:>13} {:>10} {:>10} {:>13} {:>13}'.format("Structure", "E_SPC", "E", "ZPE", "H_SPC", "T.S", "T.qh-S", "G(T)_SPC", "qh-G(T)_SPC"))
          if options.cosmo != False: log.Write('{:>13}'.format("COSMO-RS"))
          if options.boltz == True: log.Write('{:>7}'.format("Boltz"))
          if clustering == True: log.Write('{:>7}'.format("Clust"))
          if options.imag_freq == True: log.Write('{:>9}'.format("im freq"))

      if options.csv == True:
          if options.spc == False: log.Write("\n\n   " + '{:<39} {:>13},{:>10},{:>13},{:>10},{:>10},{:>13},{:>13}'.format("Structure,", "E", "ZPE", "H", "T.S", "T.qh-S", "G(T)", "qh-G(T)"))
          else: log.Write("\n\n   " + '{:<39} {:>13},{:>13},{:>10},{:>13},{:>10},{:>10},{:>13},{:>13}'.format("Structure,", "E_SPC", "E", "ZPE", "H_SPC", "T.S", "T.qh-S", "G(T)_SPC", "qh-G(T)_SPC"))
          if options.cosmo != False: log.Write(',{:>12}'.format("COSMO-RS"))
          if options.boltz == True: log.Write(',{:>6}'.format("Boltz"))
          if clustering == True: log.Write(',{:>6}'.format("Clust"))
          if options.imag_freq == True: log.Write(',{:>8}'.format("im freq"))

      log.Write("\n"+stars)
      if options.spc != False: log.Write('*'*14)
      if options.cosmo != False: log.Write('*'*13)
      if options.imag_freq == True: log.Write('*'*9)
      if options.boltz == True: log.Write('*'*7)
      if clustering == True: log.Write('*'*7)
      log.Write("")

      if options.boltz != False:
         boltz_facs = {}
         e_rel = {}
         e_min = 999999.99999
         boltz_sum = 0.0

         for file in files: # Need the most stable structure
            bbe = calc_bbe(file, options.QH, options.freq_cutoff, options.temperature, options.conc, options.freq_scale_factor, options.solv, options.spc)
            if hasattr(bbe,"qh_gibbs_free_energy"):
                if bbe.qh_gibbs_free_energy != None:
                    if bbe.qh_gibbs_free_energy < e_min: e_min = bbe.qh_gibbs_free_energy

         if clustering == True:
            for n, cluster in enumerate(clusters): boltz_facs['cluster-'+alphabet[n].upper()] = 0.0
         for file in files: # Now calculate E_rel and Boltzmann factors
            bbe = calc_bbe(file, options.QH, options.freq_cutoff, options.temperature, options.conc, options.freq_scale_factor, options.solv, options.spc)
            if hasattr(bbe,"qh_gibbs_free_energy"):
                if bbe.qh_gibbs_free_energy != None:
                    e_rel[file] = bbe.qh_gibbs_free_energy - e_min
                    boltz_facs[file] = math.exp(-e_rel[file]*j_to_au/GAS_CONSTANT/options.temperature)
                    if clustering == True:
                       for n, cluster in enumerate(clusters):
                           for structure in cluster:
                               if structure == file: boltz_facs['cluster-'+alphabet[n].upper()] += math.exp(-e_rel[file]*j_to_au/GAS_CONSTANT/options.temperature)

                    boltz_sum += math.exp(-e_rel[file]*j_to_au/GAS_CONSTANT/options.temperature)

      for file in files: # loop over the output files and compute thermochemistry
         bbe = calc_bbe(file, options.QH, options.freq_cutoff, options.temperature, options.conc, options.freq_scale_factor, options.solv, options.spc)

         # Add CPU times
         if options.cputime != False:
             if hasattr(bbe,"cpu"):
                 if bbe.cpu != None: total_cpu_time = addTime(total_cpu_time, bbe.cpu)
             if hasattr(bbe,"sp_cpu"):
                 if bbe.sp_cpu != None: total_cpu_time = addTime(total_cpu_time, bbe.sp_cpu)

         if total_cpu_time.month > 1: add_days += 31

         if options.xyz == True: # write Cartesians
             xyzdata = getoutData(file)
             xyz.Writetext(str(len(xyzdata.ATOMTYPES)))
             if hasattr(bbe, "scf_energy"): xyz.Writetext('{:<39} {:>13} {:13.6f}'.format(os.path.splitext(os.path.basename(file))[0], 'Eopt', bbe.scf_energy))
             else: xyz.Writetext('{:<39}'.format(os.path.splitext(os.path.basename(file))[0]))
             if hasattr(xyzdata, 'CARTESIANS') and hasattr(xyzdata, 'ATOMTYPES'): xyz.Writecoords(xyzdata.ATOMTYPES, xyzdata.CARTESIANS)

         if options.csv == False: log.Write("\no  "+'{:<39}'.format(os.path.splitext(os.path.basename(file))[0]))
         else: log.Write("\no  "+'{:<39}'.format(os.path.splitext(os.path.basename(file))[0]+','))
         if options.spc != False:
            try: log.Write(' {:13.6f}'.format(bbe.sp_energy))
            except ValueError: log.Write(' {:>13}'.format('----'))
         if hasattr(bbe, "scf_energy"):
             if options.csv == False: log.Write(' {:13.6f}'.format(bbe.scf_energy))
             else: log.Write(' {:13.6f},'.format(bbe.scf_energy))
         if not hasattr(bbe,"gibbs_free_energy"): log.Write("   Warning! Couldn't find frequency information ...")
         else:
            if all(getattr(bbe, attrib) for attrib in ["enthalpy", "entropy", "qh_entropy", "gibbs_free_energy", "qh_gibbs_free_energy"]):
                if options.csv == False: log.Write(' {:10.6f} {:13.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(bbe.zpe, bbe.enthalpy, (options.temperature * bbe.entropy), (options.temperature * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy))
                else: log.Write('{:10.6f},{:13.6f},{:10.6f},{:10.6f},{:13.6f},{:13.6f}'.format(bbe.zpe, bbe.enthalpy, (options.temperature * bbe.entropy), (options.temperature * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy))
         if options.cosmo != False and cosmo_solv != None:
             log.Write('{:13.6f}'.format(cosmo_solv[file]))
         if options.boltz == True:
             if options.csv == False: log.Write('{:7.3f}'.format(boltz_facs[file]/boltz_sum))
             else: log.Write(',{:6.3f}'.format(boltz_facs[file]/boltz_sum))
         if clustering == True:
             for n, cluster in enumerate(clusters):
                 for id, structure in enumerate(cluster):
                     if structure == file:
                         if id == len(cluster)-1:
                             if options.csv == False: log.Write('{:7.1f}'.format(100 * boltz_facs['cluster-'+alphabet[n].upper()]/boltz_sum))
                             else: log.Write(',{:6.1f}'.format(100 * boltz_facs['cluster-'+alphabet[n].upper()]/boltz_sum))
                         else:
                             if options.csv == False: log.Write('{:>7}'.format(''))
                             else: log.Write(',{:>6}'.format(''))

         if options.imag_freq == True and hasattr(bbe, "im_freq") == True:
             for freq in bbe.im_freq:
                 if options.csv == False: log.Write('{:9.2f}'.format(freq))
                 else: log.Write(',{:8.2f}'.format(freq))

      log.Write("\n"+stars)
      if options.spc != False: log.Write('*'*14)
      if options.cosmo != False: log.Write('*'*13)
      if options.imag_freq == True: log.Write('*'*9)
      if options.boltz == True: log.Write('*'*7)
      if clustering == True: log.Write('*'*7)
      log.Write("\n")

   #Running a variable temperature analysis of the enthalpy, entropy and the free energy
   elif options.temperature_interval != False:
      temperature_interval = [float(temp) for temp in options.temperature_interval.split(',')]
      # If no temperature step was defined, divide the region into 10
      if len(temperature_interval) == 2: temperature_interval.append((temperature_interval[1]-temperature_interval[0])/10.0)

      log.Write("\n\n   Variable-Temperature analysis of the enthalpy, entropy and the entropy at a constant pressure between")
      log.Write("\n   T_init:  %.1f,  T_final:  %.1f,  T_interval: %.1f" % (temperature_interval[0], temperature_interval[1], temperature_interval[2]))
      log.Write("\n\n   " + '{:<39} {:>13} {:>24} {:>10} {:>10} {:>13} {:>13}'.format("Structure", "Temp/K", "H/au", "T.S/au", "T.qh-S/au", "G(T)/au", "qh-G(T)/au"))

      for file in files: # loop over the output files
         log.Write("\n"+stars)

         for i in range(int(temperature_interval[0]), int(temperature_interval[1]+1), int(temperature_interval[2])): # run through the temperature range
            temp, conc = float(i), atmos / GAS_CONSTANT / float(i)
            log.Write("\no  "+'{:<39} {:13.1f}'.format(os.path.basename(file), temp))
            bbe = calc_bbe(file, options.QH, options.freq_cutoff, temp, conc, options.freq_scale_factor, options.solv, options.spc)

            if not hasattr(bbe,"gibbs_free_energy"): log.Write("Warning! Couldn't find frequency information ...\n")
            else:
                if all(getattr(bbe, attrib) for attrib in ["enthalpy", "entropy", "qh_entropy", "gibbs_free_energy", "qh_gibbs_free_energy"]):
                    log.Write(' {:24.6f} {:10.6f} {:10.6f} {:13.6f} {:13.6f}'.format(bbe.enthalpy, (temp * bbe.entropy), (temp * bbe.qh_entropy), bbe.gibbs_free_energy, bbe.qh_gibbs_free_energy))
         log.Write("\n"+stars+"\n")

   #close the log
   if options.cputime != False: log.Write('   {:<13} {:>2} {:>4} {:>2} {:>3} {:>2} {:>4} {:>2} {:>4}\n'.format('TOTAL CPU', total_cpu_time.day + add_days - 1, 'days', total_cpu_time.hour, 'hrs', total_cpu_time.minute, 'mins', total_cpu_time.second, 'secs'))
   log.Finalize()
   if options.xyz == True: xyz.Finalize()

if __name__ == "__main__":
    main()
