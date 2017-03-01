#!/usr/bin/python
from __future__ import print_function

# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Comments and/or additions are welcome (send e-mail to:
# robert.paton@chem.ox.ac.uk

#######################################################################
#                              GoodVibes.py                           #
#  Evaluation of quasi-harmonic thermochemistry from Gaussian 09.     #
#  The partion functions are evaluated from vibrational frequencies   #
#  and rotational temperatures from the standard output.              #
#  The rigid-rotor harmonic oscillator approximation is used as       #
#  standard for all frequencies above a cut-off value. Below this,    #
#  two treatments can be applied: either low frequencies can be set   #
#  to a value of 100 cm-1 (as advocated by Cramer-Truhlar), or the    #
#  free-rotor approximation is applied below the cut-off, (proposed   #
#  by Grimme). A damping function interpolates between the RRHO and   #
#  free-rotor entropy treatment for  Svib to avoid a discontinuity.   #
#  Both approached avoide infinite values  of Svib as frequencies     #
#  tend to zero.                                                      #
#  The free energy can be evaluated for variable temperature,         #
#  concentration, vibrational scaling factor, and with a haptic       #
#  correction of the translational entropy in different solvents,     #
#  according to the amount of free space available. With a freq.      #
#  cut-off set to 0, the results will be identical to the standard    #
#  values output by the Gaussian program.                             #
#######################################################################
#######  Written by:  Rob Paton #######################################
#######  Modified by:  Ignacio Funes-Ardoiz ###########################
#######  Last modified:  Apr 04, 2016 #################################
#######################################################################

import sys, math, time

# PHYSICAL CONSTANTS
GAS_CONSTANT = 8.3144621
PLANCK_CONSTANT = 6.62606957e-34 
BOLTZMANN_CONSTANT = 1.3806488e-23
SPEED_OF_LIGHT = 2.99792458e10
AVOGADRO_CONSTANT = 6.0221415e23
AMU_to_KG = 1.66053886E-27
autokcal = 627.509541
kjtokcal = 4.184
atmos = 101.325
def_cut = 100.0

stars = "   *******************************************************************************************************************************"

# Enables output to terminal and to text file
class Logger:
   # Designated initializer
   def __init__(self,filein,suffix,append):
      # Create the log file at the input path
      self.log = open(filein+"_"+append+"."+suffix, 'w' )

   # Write a message to the log
   def Write(self, message):
      # Print the message
      print(message, end=' ') 
      # Write to log
      self.log.write(message)

   # Write a message only to the log and not to the terminal
   def Writeonlyfile(self, message):
      # Write to log
      self.log.write("\n"+message+"\n")

   # Write a fatal error, finalize and terminate the program
   def Fatal(self, message):
      # Print the message
      print(message+"\n")
      # Write to log
      self.log.write(message + "\n")
      # Finalize the log
      self.Finalize()
      # End the program
      sys.exit(1)

   # Finalize the log file
   def Finalize(self):
      self.log.close()


# translational energy evaluation (depends on temperature)
def calc_translational_energy(temperature):
   """
   Calculates the translational energy (kcal/mol) of an ideal gas - i.e. 
   non-interactiing molecules so molar energy = Na * atomic energy
   This approximxation applies to all energies and entropies computed within
   Etrans = 3/2 RT!
   """
   energy = 1.5 * GAS_CONSTANT * temperature
   energy = energy/kjtokcal/1000.0
   #print "\nH_trans", energy
   return energy

# rotational energy evaluation (depends on molecular shape and temperature)
def calc_rotational_energy(zpe, symmno, temperature, linear):
   """
   Calculates the rotaional energy (kcal/mol)
   Etrans = 0 (atomic) ; RT (linear); 3/2 RT (non-linear)
   """
   if zpe == 0.0: energy = 0.0
   elif linear == 1: energy = GAS_CONSTANT * temperature 
   else: energy = 1.5 * GAS_CONSTANT * temperature
   energy = energy/kjtokcal/1000.0
   #print "H_rot", energy
   return energy

# vibrational energy evaluation (depends on frequencies, temperature and scaling factor: default = 1.0)
def calc_vibrational_energy(frequency_wn, temperature,freq_scale_factor):
   """
   Calculates the vibrational energy contribution (kcal/mol)
   Includes ZPE (0K) and thermal contributions
   Evib = R * Sum(0.5 hv/k + (hv/k)/(e^(hv/KT)-1))
   """
   energy = 0.0
   frequency = [entry * SPEED_OF_LIGHT for entry in frequency_wn]
   for entry in frequency:
      factor = ((PLANCK_CONSTANT*entry*freq_scale_factor)/(BOLTZMANN_CONSTANT*temperature))
      temp = factor*temperature*(0.5 + (1/(math.exp(factor)-1)))
      temp = temp*GAS_CONSTANT
      energy = energy + temp
   energy = energy/kjtokcal/1000.0
   #print "H_vib", energy
   return energy

# vibrational Zero point energy evaluation (depends on frequencies and scaling factor: default = 1.0)
def calc_zeropoint_energy(frequency_wn,freq_scale_factor):
   """
   Calculates the vibrational ZPE (kcal/mol)
   EZPE = Sum(0.5 hv/k)
   """
   energy = 0.0
   frequency = [entry * SPEED_OF_LIGHT for entry in frequency_wn]
   for entry in frequency:
      factor = ((PLANCK_CONSTANT*entry*freq_scale_factor)/(BOLTZMANN_CONSTANT))
      temp = 0.5*factor
      temp = temp*GAS_CONSTANT
      energy = energy + temp
   energy = energy/kjtokcal/1000.0
   #print "H_zpe", energy
   return energy

# Computed the amount of accessible free space (ml per L) in solution accesible to a solute immersed in bulk solvent, i.e. this is the volume not occupied by solvent molecules, calculated using literature values for molarity and B3LYP/6-31G* computed molecular volumes.
def get_free_space(solv):
   """
   Calculates the free space in a litre of bulk solvent,
   based on Shakhnovich and Whitesides (J. Org. Chem. 1998, 63, 3821-3830)
   """
   solvent_list = ["none", "H2O", "Toluene", "DMF", "AcOH", "Chloroform"]
   molarity = [1.0, 55.6, 9.4, 12.9, 17.4, 12.5] #mol/l
   molecular_vol = [1.0, 27.944, 149.070, 77.442, 86.10, 97.0] #Angstrom^3

   nsolv = 0
   for i in range(0,len(solvent_list)):
      if solv == solvent_list[i]: nsolv = i

   solv_molarity = molarity[nsolv]
   solv_volume = molecular_vol[nsolv]

   if nsolv > 0:
      V_free = 8 * ((1E27/(solv_molarity*AVOGADRO_CONSTANT)) ** 0.333333 - solv_volume ** 0.333333) ** 3
      freespace = V_free * solv_molarity * AVOGADRO_CONSTANT * 1E-24
   else: freespace = 1000.0

#   print "free space", freespace
   return freespace

# translational entropy evaluation (depends on mass, concentration, temperature, solvent free space: default = 1000.0)
def calc_translational_entropy(molecular_mass, conc, temperature, solv):
   """
   Calculates the translational entropic contribution (cal/(mol*K)) of an ideal gas
   needs the molecular mass
   Convert mass in amu to kg; conc in mol/l to number per m^3
   Strans = R(Ln(2pimkT/h^2)^3/2(1/C)) + 1 + 3/2)
   """
   simass = molecular_mass*AMU_to_KG
   lmda = ((2.0*math.pi*simass*BOLTZMANN_CONSTANT*temperature)**0.5)/PLANCK_CONSTANT
   Ndens = conc*1000*AVOGADRO_CONSTANT
   freespace = get_free_space(solv)
   Ndens = Ndens / (freespace/1000.0)
   entropy = GAS_CONSTANT*(2.5+math.log(lmda**3/Ndens))/4.184
   #print "S_trans", entropy
   return entropy

# electronic entropy evaluation (depends on multiplicity)
def calc_electronic_entropy(multiplicity):
   """
   Calculates the electronic entropic contribution (cal/(mol*K)) of the molecule
   Selec = R(Ln(multiplicity)
   """
   entropy = GAS_CONSTANT*(math.log(multiplicity))/4.184
   #print "S_elec", entropy
   return entropy


# rotational entropy evaluation (depends on molecular shape and temp.)
def calc_rotational_entropy(zpe, linear, symmno, roconst, temperature):
   """
   Calculates the rotational entropy (cal/(mol*K))
   Strans = 0 (atomic) ; R(Ln(q)+1) (linear); R(Ln(q)+3/2) (non-linear)
   """
   # monatomic
   if roconst == [0.0,0.0,0.0]: return 0.0
   rotemp = [roconst[0]*PLANCK_CONSTANT*1000000000/BOLTZMANN_CONSTANT,roconst[1]*PLANCK_CONSTANT*1000000000/BOLTZMANN_CONSTANT,roconst[2]*PLANCK_CONSTANT*1000000000/BOLTZMANN_CONSTANT]

   # diatomic
   if 0.0 in rotemp:
           rotemp.remove(0.0)
           qrot = temperature/rotemp[0]
   else:
      qrot = math.pi*temperature**3/(rotemp[0]*rotemp[1]*rotemp[2])
      qrot = qrot ** 0.5

   qrot = qrot/symmno

   if zpe == 0.0: entropy = 0.0 # monatomic

   if linear == 1: entropy = GAS_CONSTANT * (math.log(qrot) + 1) 
   else: entropy = GAS_CONSTANT * (math.log(qrot) + 1.5)

   entropy = entropy/kjtokcal
   return entropy

# rigid rotor harmonic oscillator (RRHO) entropy evaluation - this is the default treatment
def calc_rrho_entropy(frequency_wn, temperature,freq_scale_factor):
   """
   Calculates the entropic contribution (cal/(mol*K)) of a harmonic oscillator for
   a list of frequencies of vibrational modes
   Sv = RSum(hv/(kT(e^(hv/KT)-1) - ln(1-e^(-hv/kT)))       
   """
   entropy = []
   frequency = [entry * SPEED_OF_LIGHT for entry in frequency_wn]
   for entry in frequency:
      factor = ((PLANCK_CONSTANT*entry*freq_scale_factor)/(BOLTZMANN_CONSTANT*temperature))
      temp = factor*(1/(math.exp(factor)-1)) - math.log(1-math.exp(-factor))
      temp = temp*GAS_CONSTANT/4.184
      entropy.append(temp)
   return entropy


# free rotor entropy evaluation - used for low frequencies below the cut-off if qh=grimme is specified
def calc_freerot_entropy(frequency_wn, temperature,freq_scale_factor):
   """
   Calculates the entropic contribution (cal/(mol*K)) of a rigid-rotor harmonic oscillator for
   a list of frequencies of vibrational modes
   Sr = R(1/2 + 1/2ln((8pi^3u'kT/h^2))
   """
   #??This is the average moment of inertia used by Grimme - is this optimal for every mode??
   Bav = 10.0e-44
   
   entropy = []
   frequency = [entry * SPEED_OF_LIGHT for entry in frequency_wn]
   
   for entry in frequency:
      mu = PLANCK_CONSTANT/(8*math.pi**2*entry*freq_scale_factor)
      muprime = mu*Bav/(mu +Bav)
      factor = (8*math.pi**3*muprime*BOLTZMANN_CONSTANT*temperature)/(PLANCK_CONSTANT**2)
      temp = 0.5 + math.log(factor**0.5)
      temp = temp*GAS_CONSTANT/4.184
      entropy.append(temp)
   return entropy

# A damping function to interpolate between RRHO and free rotor vibrational entropy values
def calc_damp(frequency_wn, FREQ_CUTOFF):
   """
   Calculates the Head-Gordon damping function with alpha=4
   """
   alpha = 4
   damp = []
   for entry in frequency_wn:
      omega = 1/(1+(FREQ_CUTOFF/entry)**alpha)
      damp.append(omega)
   return damp

# The funtion to compute the "black box" entropy values (and all other thermochemical quantities)
class calc_bbe:
   def __init__(self, file, QH, FREQ_CUTOFF, temperature, conc, freq_scale_factor,solv,spc):
      # Frequencies in waveunmbers
      frequency_wn = []
      # Read commandline arguments
      g09_output = open(file, 'r')

      linear_mol = 0
      roconst = [0.0,0.0,0.0]
      symmno = 1
      linkmax = 0
      freqloc = 0
      link = 0

      #count number of links
      for line in g09_output:
         # only read first link + freq not other link jobs
         if line.find("Normal termination") != -1:
            linkmax += 1
         if line.find('Frequencies --') != -1:
            freqloc = linkmax

      g09_output.seek(0)

      # Iterate over output
      for line in g09_output:
         # link counter
         if line.find("Normal termination")!= -1:
            link += 1
            # reset frequencies if in final freq link
            if link == freqloc: frequency_wn = []
         # if spc specified will take last Energy from file, otherwise will break after freq calc
         if link > freqloc and spc == 0: break

      	# Iterate over output
      	#for line in g09_output:
         # look for low frequencies
         #if line.find("Proceeding to internal job step")!= -1: frequency_wn = [] #resets the array if frequencies have been calculated more than once
         if line.strip().startswith('Frequencies --'):
            for i in range(2,5):
               try:
                  x = float(line.strip().split()[i])
                  #  only deal with real frequencies
                  if x > 0.00: frequency_wn.append(x)
               except IndexError:pass
 
         # For QM calculations look for SCF energies, last one will be the optimized energy
         if line.strip().startswith('SCF Done:'): self.scf_energy = float(line.strip().split()[4])
         # For ONIOM calculations use the extrapolated value rather than SCF value
         if line.strip().find("ONIOM: extrapolated energy") > -1: self.scf_energy = (float(line.strip().split()[4]))
         # For Semi-empirical or Molecular Mechanics calculations
         if line.strip().find("Energy= ") > -1 and line.strip().find("Predicted")==-1 and line.strip().find("Thermal")==-1: self.scf_energy = (float(line.strip().split()[1]))

         # look for thermal corrections, paying attention to point group symmetry
         if line.strip().startswith('Zero-point correction='): self.zero_point_corr = float(line.strip().split()[2])
         if line.strip().find('Multiplicity') > -1: mult = float(line.strip().split()[5])
         if line.strip().startswith('Molecular mass:'): molecular_mass = float(line.strip().split()[2])
         if line.strip().startswith('Rotational symmetry number'): symmno = int((line.strip().split()[3]).split(".")[0])
         if line.strip().startswith('Full point group'): 
            if line.strip().split()[3] == 'D*H' or line.strip().split()[3] == 'C*V': linear_mol = 1
         if line.strip().startswith('Rotational constants'): roconst = [float(line.strip().split()[3]), float(line.strip().split()[4]), float(line.strip().split()[5])]

      # skip the next steps if unable to parse the frequencies or zpe from the output file
      if hasattr(self, "zero_point_corr"):
      
         # create an array of frequencies equal to cut-off value
         cutoffs = []
         for j in range(0,len(frequency_wn)): cutoffs.append(FREQ_CUTOFF)

         # Calculate Translational, Rotational and Vibrational contributions to the energy
         Utrans = calc_translational_energy(temperature)
         Urot = calc_rotational_energy(self.zero_point_corr, symmno, temperature,linear_mol)
         Uvib = calc_vibrational_energy(frequency_wn, temperature,freq_scale_factor)
         ZPE = calc_zeropoint_energy(frequency_wn, freq_scale_factor)

         # Calculate Translational, Rotational and Vibrational contributions to the entropy
         Strans = calc_translational_entropy(molecular_mass, conc, temperature, solv)
         Selec = calc_electronic_entropy(mult)
         Srot = calc_rotational_entropy(self.zero_point_corr, linear_mol, symmno, roconst, temperature)

         # Calculate harmonic entropy, free-rotor entropy and damping function for each frequency
         Svib_rrho = calc_rrho_entropy(frequency_wn, temperature,freq_scale_factor)
         if FREQ_CUTOFF > 0.0: Svib_rrqho = calc_rrho_entropy(cutoffs, temperature,1.0)
         Svib_free_rot = calc_freerot_entropy(frequency_wn, temperature,freq_scale_factor)
         damp = calc_damp(frequency_wn, FREQ_CUTOFF)

         # Compute entropy (cal/mol/K) using the two values and damping function
         vib_entropy = []
         for j in range(0,len(frequency_wn)):
            if QH == "grimme": vib_entropy.append(Svib_rrho[j] * damp[j] + (1-damp[j]) * Svib_free_rot[j])
            elif QH == "truhlar" and FREQ_CUTOFF > 0.0:
               if frequency_wn[j] > FREQ_CUTOFF: vib_entropy.append(Svib_rrho[j])
               else: vib_entropy.append(Svib_rrqho[j])

         # Add all terms to get Free energy - perform separately for harmonic and quasi-harmonic values out of interest
         qh_Svib = sum(vib_entropy)
         h_Svib = sum(Svib_rrho)
         self.enthalpy = self.scf_energy + (Utrans + Urot + Uvib + GAS_CONSTANT*temperature/kjtokcal/1000.0)/autokcal
         self.zpe = ZPE/autokcal
         self.entropy = (Strans + Srot + h_Svib + Selec)/autokcal/1000.0
         self.qh_entropy = (Strans + Srot + qh_Svib + Selec)/autokcal/1000.0
         self.gibbs_free_energy = self.enthalpy - temperature * self.entropy
         self.qh_gibbs_free_energy = self.enthalpy - temperature * self.qh_entropy

         #Uncomment to compute the magnitude of the quasi-harmonic correction to the RRHO entropy
         #QH_correction = h_Svib - qh_Svib
         # self.QH_correction = -QH_correction * temperature/1000.0

         #Uncomment to compute the magnitude of the haptic (i.e. concentration-dependent) correction to the RRHO entropy
         #Strans1atm = calc_translational_entropy(molecular_mass, atmos/(GAS_CONSTANT*temperature), temperature, solv)
         #conc_correction = Strans - Strans1atm
         #self.conc_correction = -conc_correction * temperature/1000.0

if __name__ == "__main__":
   # Takes arguments: cutoff_freq g09_output_files
   files = []
   log = Logger("Goodvibes","dat", "output")
   QH = "grimme"; spc = "none"; FREQ_CUTOFF = "none"; temperature = "none"; conc = "none"; freq_scale_factor = "none"; solv = "none"; temperature_interval = []; conc_interval = []
   if len(sys.argv) > 1:
      for i in range(1,len(sys.argv)):
         if sys.argv[i] == "-f": FREQ_CUTOFF = float(sys.argv[i+1])
         elif sys.argv[i] == "-t": temperature = float(sys.argv[i+1])
         elif sys.argv[i] == "-qh": QH = (sys.argv[i+1]).lower()
         elif sys.argv[i] == "-c": conc = float(sys.argv[i+1])
         elif sys.argv[i] == "-v": freq_scale_factor = float(sys.argv[i+1])
         elif sys.argv[i] == "-s": solv = (sys.argv[i+1])
         elif sys.argv[i] == "-ti": temperature_interval = list(eval(sys.argv[i+1]))
         elif sys.argv[i] == "-ci": conc = list(sys.argv[i+1])
         elif sys.argv[i] == "-spc": spc = 1

         else:
            if len(sys.argv[i].split(".")) > 1:
               if sys.argv[i].split(".")[1] == "out" or sys.argv[i].split(".")[1] == "log":
                  files.append(sys.argv[i])
      freespace = get_free_space(solv)

      start = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
      log.Writeonlyfile("   GoodVibes analysis performed on "+start)
      log.Writeonlyfile("   Python written by Rob Paton and Ignacio Funes-Ardoiz")

      if temperature != "none": log.Write("\n   Temperature = "+str(temperature)+" Kelvin")
      else: log.Write("\n   Temperature (default) = 298.15K",); temperature = 298.15

      if conc != "none": log.Write("   Concn = "+str(conc)+" mol/l")
      else: log.Write("   Concn (default) = 1 atmosphere"); conc = 12.187274/temperature; conc_ini="None"

      if freq_scale_factor != "none": log.Write("   Frequency scale factor = "+str(freq_scale_factor))
      else: log.Write("   Frequency scale factor (default) = 1.0"); freq_scale_factor = 1.0

      if spc == "none": spc = 0
      else: log.Write("\n   Link job: combining final single point energy with thermal corrections")

      if freespace != 1000.0: log.Write("   Specified solvent "+solv+": free volume"+str("%.1f" % (freespace/10.0))+"(mol/l) corrects the translational entropy")

      if FREQ_CUTOFF == 0.0:
         log.Write("\n   Quasi-harmonic cut-off value = "+str(FREQ_CUTOFF)+" wavenumbers (no corrections applied!)")
         if QH == "truhlar": log.Fatal("\n   FATAL ERROR: The defined quasi-harmonic model is incompatible with a cut-off value of zero wavenumbers")

      elif FREQ_CUTOFF == "none":
         FREQ_CUTOFF = def_cut
      log.Write("\n   Quasi-harmonic treatment: frequency cut-off value of "+str(FREQ_CUTOFF)+" wavenumbers will be applied")

      if QH == "grimme": log.Write("\n   QH = Grimme: Using a mixture of RRHO and Free-rotor vibrational entropies, as proposed by Grimme")
      elif QH == "truhlar": log.Write("\n   QH = Truhlar: Using an RRHO treatment where low frequencies are adjusted to the cut-off value, as proposed by Truhlar")
      else: log.Fatal("\n   FATAL ERROR: Unknown quasi-harmonic model "+QH+" specified (QH must = grimme or truhlar)")

   else: log.Fatal("\n   FATAL ERROR: Wrong number of arguments used.\n   Correct format: GoodVibes.py (-qh grimme/truhlar) (-f cutoff_freq) (-t temp) (-c concn) (-v scalefactor) g09_output_file(s)\n")

   # Standard mode: tabulate thermochemistry ouput from file(s) at a single temperature and concentration
   if len(temperature_interval) == 0 and len(conc_interval) == 0:
      log.Write("\n\n  "+"".ljust(30)+"E/au".rjust(12)+ "  ZPE/au".rjust(12)+ "   H/au".rjust(12)+ "   T.S/au".rjust(12)+ "   T.qh-S/au".rjust(12)+ "   G(T)/au".rjust(12)+ "   qh-G(T)/au".rjust(12))
      log.Write("\n"+stars)
      for file in files:
         bbe = calc_bbe(file, QH, FREQ_CUTOFF, temperature, conc, freq_scale_factor, solv, spc)
         log.Write("\no ")
         log.Write((file.split(".")[0]).ljust(30))
         if hasattr(bbe, "scf_energy"): log.Write("%.6f" % bbe.scf_energy)
         else: log.Write("N/A   ")
         if not hasattr(bbe,"gibbs_free_energy"): log.Write("   Warning! Couldn't find frequency information ...\n")
         else:
            if hasattr(bbe, "zero_point_corr"): log.Write("   %.6f" % (bbe.zpe))
            else: log.Write("N/A   ")
            if hasattr(bbe, "enthalpy"): log.Write("   %.6f" % (bbe.enthalpy))
            else: log.Write("N/A   ")
            if hasattr(bbe, "entropy"): log.Write("   %.6f" % (temperature * bbe.entropy))
            else: log.Write("N/A   ")
            if hasattr(bbe, "qh_entropy"): log.Write("   %.6f" % (temperature * bbe.qh_entropy))
            else: log.Write("N/A   ")
            if hasattr(bbe, "gibbs_free_energy"): log.Write("   %.6f" % (bbe.gibbs_free_energy))
            else: log.Write("N/A   ")
            if hasattr(bbe, "qh_gibbs_free_energy"): log.Write("   %.6f" % (bbe.qh_gibbs_free_energy))
            else: log.Write("N/A")
      log.Write("\n")

   #Running a variable temperature analysis of the enthalpy, entropy and the free energy
   if len(temperature_interval) != 0:
      # If no temperature step was defined, divide the region into 10
      if len(temperature_interval) == 2: temperature_interval.append((temperature_interval[1]-temperature_interval[0])/10.0)
      log.Write("\n\n   Running a temperature analysis of the enthalpy, entropy and the entropy between")
      log.Write("\n   T_init:  %.1f,  T_final:  %.1f,  T_interval: %.1f" % (temperature_interval[0], temperature_interval[1], temperature_interval[2]))
      temperature = float(temperature_interval[0])

      log.Write("\n\n   "+"Temp/K".ljust(25)+ "  H/au".rjust(12)+ "   T.S/au".rjust(12)+ "   T.qh-S/au".rjust(12)+ "   G(T)/au".rjust(12)+ "   qh-G(T)/au".rjust(12))
      for file in files:
         #output_file = file.split(".")[0] + "_temperature.txt"
         #temperature_txt = open(output_file,"w")
         log.Write("\n"+stars+"\n")

         for i in range(int(temperature_interval[0]), int(temperature_interval[1]+1), int(temperature_interval[2])):
            temperature = float(i)
            log.Write("o  "+file+" @"+" %.1f   " % (temperature))
            if conc_ini == "None": conc =  atmos/(GAS_CONSTANT*temperature)
            bbe = calc_bbe(file, QH, FREQ_CUTOFF, temperature, conc, freq_scale_factor, solv, spc)
            if not hasattr(bbe,"gibbs_free_energy"): log.Write("Warning! Couldn't find frequency information ...\n")
            else:
               if hasattr(bbe, "enthalpy"): log.Write("   %.6f" % (bbe.enthalpy))
               else: log.Write("N/A   ")
               if hasattr(bbe, "entropy"): log.Write("   %.6f" % (temperature * bbe.entropy))
               else: log.Write("N/A   ")
               if hasattr(bbe, "qh_entropy"): log.Write("   %.6f" % (temperature * bbe.qh_entropy))
               else: log.Write("N/A   ")
               if hasattr(bbe, "gibbs_free_energy"): log.Write("   %.6f" % (bbe.gibbs_free_energy))
               else: log.Write("N/A   ")
               if hasattr(bbe, "qh_gibbs_free_energy"): log.Write("   %.6f" % (bbe.qh_gibbs_free_energy))
               else: log.Write("N/A")
            log.Write("\n")

   log.Finalize()
