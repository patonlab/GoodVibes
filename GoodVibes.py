#!/usr/bin/python

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
#  A program to recompute the vibrational entropy from a standard     #
#  output file as produced by a frequency calculation in Gaussian 09  #
#  The rigid-rotor harmonic approximation is used for frequencies     #
#  above a specified cut-off frequency, while the free-rotor          #
#  approximation is applied below this value. Rather than a hard cut- #
#  off, a damping function interpolates between these two             #
#  expressions for the Svib. rather than a hard cut-off. This avoids  #
#  values of Svib. that tend to infinite values as the frequency      #
#  tends to zero. This approach was described in: Grimme, S. Chem.    #
#  Eur. J. 2012, 18, 9955                                             #
#  The free energy is then reevaluated at a specified temperature,    #
#  with a specified frequency cut-off. An optional vibrational        #
#  scaling factor (defaults to 1.0) and concentration (defaults to    #
#  1 atmos, although values specified are assumed to be in mol/l)     #
#  may be specified. This latter term is factored into calculation of #
#  the translational entropy. A Shakhnovich/Whitesides model of       #
#  the restructed free space available for translation can be included#
#  for a handful of solvents, although this is under development.     #
#  Note that we have only tested for systems with C1-point group...   #
#######################################################################
#######  Written by:  Rob Paton #######################################
#######  Modified by:  Ignacio Funes-Ardoiz ###########################
#######  Last modified:  Mar 17, 2016 #################################
#######################################################################

import sys, math

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

# translational energy evaluation (depends on temp.)
def calc_translational_energy(temperature):
   """
   Calculates the translational energy (kcal/mol) of an ideal gas - i.e. 
   non-interactiing molecules so molar energy = Na * atomic energy
   This approximxation applies to all energies and entropies computed within
   Etrans = 3/2 RT!
   """
   energy = 1.5 * GAS_CONSTANT * temperature
   energy = energy/kjtokcal/1000.0
   return energy


# rotational energy evaluation (depends on molecular shape and temp.)
def calc_rotational_energy(zpe, symmno, temperature, linear):
   """
           Calculates the rotaional energy (kcal/mol)
           Etrans = 0 (atomic) ; RT (linear); 3/2 RT (non-linear)
           """
   if zpe == 0.0: energy = 0.0
   #if symmno == 0: energy = 0.0
   elif linear == 1: energy = GAS_CONSTANT * temperature 
   else: energy = 1.5 * GAS_CONSTANT * temperature
   energy = energy/kjtokcal/1000.0
   return energy


# vibrational energy evaluation (depends on frequencies, temp and scaling factor)
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
   return energy


# vibrational Zero point energy evaluation (depends on frequencies and scaling factor)
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
   return energy


def get_free_space(solv):
   """
           Calculates the free space in a litre of bulk solvent,
           based on Shakhnovich and Whitesides (J. Org. Chem. 1998, 63, 3821-3830)
           """

   # solvent densities are taken from the literature
   # solvent volumes are taken from a B3LYP/6-31G(d) calculation with g09
   if solv == "H2O":
           solv_molarity = 55.6 # mol/l
           solv_volume = 27.944 # Ang^3
           
   elif solv == "Toluene":
           solv_molarity = 9.4 # mol/l
           solv_volume = 149.070 # Ang^3

   elif solv == "DMF":
           solv_molarity = 12.9 # mol/l
           solv_volume = 77.442 # Ang^3
           
   elif solv == "AcOH":
           solv_molarity = 17.4 # mol/l
           solv_volume = 86.1 # Ang^3

   elif solv == "CHCl3":
           solv_molarity = 12.5 # mol/l
           solv_volume = 97 # Ang^3
   else: solv = "unk"

   if solv != "unk":
           V_free = 8 * ((1E27/(solv_molarity*AVOGADRO_CONSTANT)) ** 0.333333 - solv_volume ** 0.333333) ** 3
           freespace = V_free * solv_molarity * AVOGADRO_CONSTANT * 1E-24
   else: freespace = 1000.0

   #e.g. chloroform Vol free is 7.5mL
   #e.g. DMF molarity is 12.9 and 77.442 Ang^3 molecular volume, so Vfree = 8((10^27/12.9*Na)^1/3 -
   #4.26)^3 = 3.94 Ang^3 per molecule = 30.6 mL per L
   # M.H. Abraham, J. Liszi, J. Chem. Soc. Faraday Trans. 74 (1978) 1604.
   # e.g. Acetic acid molarity is 17.4 and 86.1 Ang^3 molecuar volume, so Vfree = 8((10^27/17.4*Na)^1/3 - 4.42)^3 = 0.026 Ang^3 per molecule = 0.272 mL per L
   #freespace = 0.272

   return freespace

# translational entropy evaluation (depends on mass, concentration, temp, solvent)
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
   return entropy


# electronic entropy evaluation (depends on multiplicity)
def calc_electronic_entropy(multiplicity):
   """
   Calculates the electronic entropic contribution (cal/(mol*K)) of the molecule
   Selec = R(Ln(multiplicity)
   """
   entropy = GAS_CONSTANT*(math.log(multiplicity))/4.184
   return entropy


# rotational entropy evaluation (depends on molecular shape and temp.)
def calc_rotational_entropy(zpe, linear, symmno, roconst, temperature):
   """
           Calculates the rotational entropy (cal/(mol*K))
           Strans = 0 (atomic) ; R(Ln(q)+1) (linear); R(Ln(q)+3/2) (non-linear)
           """
   if roconst == [0.0,0.0,0.0]:
           return 0.0
   rotemp = [roconst[0]*PLANCK_CONSTANT*1000000000/BOLTZMANN_CONSTANT,roconst[1]*PLANCK_CONSTANT*1000000000/BOLTZMANN_CONSTANT,roconst[2]*PLANCK_CONSTANT*1000000000/BOLTZMANN_CONSTANT]
   # diatomic
   if 0.0 in rotemp:
           rotemp.remove(0.0)
           qrot = temperature/rotemp[0]
   else:
           qrot = math.pi*temperature**3/(rotemp[0]*rotemp[1]*rotemp[2])
           qrot = qrot ** 0.5
   qrot = qrot/symmno
   if zpe == 0.0: entropy = 0.0
   if linear == 1: entropy = GAS_CONSTANT * (math.log(qrot) + 1) 
   else: entropy = GAS_CONSTANT * (math.log(qrot) + 1.5)
   entropy = entropy/kjtokcal
   return entropy


# rigid rotor harmonic oscillator (RRHO) entropy evaluation
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


# free rotor entropy evaluation
def calc_freerot_entropy(frequency_wn, temperature,freq_scale_factor):
   """
   Calculates the entropic contribution (cal/(mol*K)) of a rigid-rotor harmonic oscillator for
   a list of frequencies of vibrational modes
   Sr = R(1/2 + 1/2ln((8pi^3u'kT/h^2))
   """
   Bav = 10.0e-44
   entropy = []
   frequency = [entry * SPEED_OF_LIGHT for entry in frequency_wn]
   
   for entry in frequency:
           mu = PLANCK_CONSTANT/(8*math.pi**2*entry*freq_scale_factor)
           muprime = mu*Bav/(mu +Bav)
           #print entry, mu,muprime
           
           factor = (8*math.pi**3*muprime*BOLTZMANN_CONSTANT*temperature)/(PLANCK_CONSTANT**2)
           temp = 0.5 + math.log(factor**0.5)
           temp = temp*GAS_CONSTANT/4.184
           #print temp
           entropy.append(temp)
   return entropy


# damping function
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


class calc_bbe: 
   def __init__(self, file, FREQ_CUTOFF, temperature, conc, freq_scale_factor,solv):

           # Frequencies in waveunmbers
           frequency_wn = []
           
           # Read commandline arguments
           g09_output = open(file, 'r')
           
           linear_mol = 0
           roconst = [0.0,0.0,0.0]
           symmno = 1         

           # Iterate over output
           for line in g09_output:
              # look for low frequencies  
              if line.strip().startswith('Frequencies --'):
                      for i in range(2,5):
                         try:
                            x = float(line.strip().split()[i])
                            #  only deal with real frequencies
                            if x > 0.00: frequency_wn.append(x)
                         except IndexError:
                            pass
              # look for SCF energies, last one will be correct
              if line.strip().startswith('SCF Done:'): self.scf_energy = float(line.strip().split()[4])
              if line.strip().find("ONIOM: extrapolated energy") > -1: # Get energy from ONIOM calculation
                      elf.scf_energy = (float(line.strip().split()[4]))
              if line.strip().find("Energy= ") > -1 and line.strip().find("Predicted")==-1 and line.strip().find("Thermal")==-1: # Get energy from Semi-empirical or Molecular Mechanics calculation
                      self.scf_energy = (float(line.strip().split()[1]))

              # look for thermal corrections
              if line.strip().startswith('Zero-point correction='): self.zero_point_corr = float(line.strip().split()[2])
              if line.strip().find('Multiplicity') > -1: mult = float(line.strip().split()[5])
              if line.strip().startswith('Thermal correction to Energy='): self.energy_corr = float(line.strip().split()[4])
              if line.strip().startswith('Thermal correction to Enthalpy='): enthalpy_corr = float(line.strip().split()[4])
              if line.strip().startswith('Thermal correction to Gibbs Free Energy='): gibbs_corr = float(line.strip().split()[6])
              if line.strip().startswith('Molecular mass:'): molecular_mass = float(line.strip().split()[2])
              if line.strip().startswith('Rotational symmetry number'): symmno = int((line.strip().split()[3]).split(".")[0])
              if line.strip().startswith('Full point group'): 
                      if line.strip().split()[3] == 'D*H' or line.strip().split()[3] == 'C*V': linear_mol = 1
              if line.strip().startswith('Rotational constants'): roconst = [float(line.strip().split()[3]), float(line.strip().split()[4]), float(line.strip().split()[5])]
            
           
           # Calculate Translational, Rotational and Vibrational contributions to the energy
           Utrans = calc_translational_energy(temperature)
           Urot = calc_rotational_energy(self.zero_point_corr, symmno, temperature,linear_mol)
           Uvib = calc_vibrational_energy(frequency_wn, temperature,freq_scale_factor)
           ZPE = calc_zeropoint_energy(frequency_wn, freq_scale_factor)
           
           # Calculate Translational, Rotational and Vibrational contributions to the entropy
           Strans1atm = calc_translational_entropy(molecular_mass, atmos/(GAS_CONSTANT*temperature), temperature, solv)
           Strans = calc_translational_entropy(molecular_mass, conc, temperature, solv)
           conc_correction = Strans - Strans1atm
           Selec = calc_electronic_entropy(mult)
           Srot = calc_rotational_entropy(self.zero_point_corr, linear_mol, symmno, roconst, temperature)
                      
           # Calculate harmonic entropy, free-rotor entropy and damping function for each frequency - functions defined above
           Svibh = calc_rrho_entropy(frequency_wn, temperature,freq_scale_factor)
           Svibfree = calc_freerot_entropy(frequency_wn, temperature,freq_scale_factor)
           damp = calc_damp(frequency_wn, FREQ_CUTOFF)
   
           # Compute entropy (cal/mol/K) using the two values and damping function
           vib_entropy = []
           for j in range(0,len(frequency_wn)):
              vib_entropy.append(Svibh[j] * damp[j] + (1-damp[j]) * Svibfree[j])

           qh_Svib = sum(vib_entropy)
           h_Svib = sum(Svibh)
           QH_correction = h_Svib - qh_Svib
           
           # Add all terms to get Free energy
           #print Utrans, Urot, Uvib
           #print Strans, Srot, h_Svib, Selec
           self.enthalpy = self.scf_energy + (Utrans + Urot + Uvib + GAS_CONSTANT*temperature/kjtokcal/1000.0)/autokcal
           self.zpe = ZPE/autokcal
           self.entropy = (Strans + Srot + h_Svib + Selec)/autokcal/1000.0
           self.qh_entropy = (Strans + Srot + qh_Svib + Selec)/autokcal/1000.0
           self.gibbs_free_energy = self.enthalpy - temperature * self.entropy
           self.qh_gibbs_free_energy = self.enthalpy - temperature * self.qh_entropy
           self.QH_correction = -QH_correction * temperature/1000.0
           self.conc_correction = -conc_correction * temperature/1000.0
           
if __name__ == "__main__":
   
   # Takes arguments: cutoff_freq g09_output_files
   files = []
   FREQ_CUTOFF = "none"; temperature = "none"; conc = "none"; freq_scale_factor = "none"; solv = "none"
   if len(sys.argv) > 1:
           for i in range(1,len(sys.argv)):
              if sys.argv[i] == "-f": FREQ_CUTOFF = float(sys.argv[i+1])
              elif sys.argv[i] == "-t": temperature = float(sys.argv[i+1])
              elif sys.argv[i] == "-c": conc = float(sys.argv[i+1])
              elif sys.argv[i] == "-v": freq_scale_factor = float(sys.argv[i+1])
              elif sys.argv[i] == "-s": solv = (sys.argv[i+1])
              
              else:
                      if len(sys.argv[i].split(".")) > 1:
                         if sys.argv[i].split(".")[1] == "out" or sys.argv[i].split(".")[1] == "log": 
                            files.append(sys.argv[i])
                            name_of_file=str(sys.argv[i].split(".")[0])
                            #print(name_of_file, type(name_of_file))
           print ""   
           if temperature != "none": print "   Temperature =", temperature, "Kelvin",
           else: print "   Temperature (default) = 298.15K",; temperature = 298.15
           if conc != "none": print "   Concn =", conc, "mol/l",
           else: print "   Concn (default) = 1 atmosphere",; conc = 12.187274/temperature
           if freq_scale_factor != "none": print "   Frequency scal factor =", freq_scale_factor,
           else: print "   Frequency scale factor (default) = 1.0"; freq_scale_factor = 1.0
           if FREQ_CUTOFF != "none": 
              if FREQ_CUTOFF == 0.0: print "   Frequency cut-off value =", FREQ_CUTOFF, "wavenumbers - no corrections will be applied"
              else: print "   Frequency cut-off value =", FREQ_CUTOFF, "wavenumbers - quasiharmonic corrections will be applied"
           else: print "   Frequency cut-off value not defined!"; sys.exit()
   
           freespace = get_free_space(solv)
           if freespace != 1000.0: print "   Solvent =", solv+": % free volume (Strans)","%.1f" % (freespace/10.0)

   else:
           print "\nWrong number of arguments used. Correct format: GoodVibes.py -f cutoff_freq (-t temp) (-c concn) (-s scalefactor) g09_output_files\n"
           sys.exit()
   
   print "\n  ",
   print "  ".ljust(30), "    Energy    ZPE   Enthalpy    T.S       T.qh-S    G(T)      qh-G(T)"
   for file in files:
           bbe = calc_bbe(file, FREQ_CUTOFF, temperature, conc, freq_scale_factor, solv)
           print "o ",
           print (file.split(".")[0]).ljust(30),
           if not hasattr(bbe,"gibbs_free_energy"): print "Warning! Job did not finish normally!"
           if hasattr(bbe, "scf_energy"): print "%.6f" % bbe.scf_energy,
           else: print "N/A",
           if hasattr(bbe, "zero_point_corr"): print "   %.6f" % (bbe.zpe),
           else: print "N/A",
           if hasattr(bbe, "enthalpy"): print "   %.6f" % (bbe.enthalpy),
           else: print "N/A",
           if hasattr(bbe, "entropy"): print "   %.6f" % (temperature * bbe.entropy),
           else: print "N/A",
           if hasattr(bbe, "qh_entropy"): print "   %.6f" % (temperature * bbe.qh_entropy),
           else: print "N/A",
           if hasattr(bbe, "gibbs_free_energy"): print "   %.6f" % (bbe.gibbs_free_energy),
           else: print "N/A",
           if hasattr(bbe, "qh_gibbs_free_energy"): print "   %.6f" % (bbe.qh_gibbs_free_energy)
           else: print "N/A"
