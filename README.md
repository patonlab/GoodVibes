Goodvibes
=====

A Python program to compute quasi-harmonic thermochemical data from Gaussian frequency calculations at a given temperature/concentration, corrected for the effects of vibrational scaling-factors and available free space in solvent.

All (electronic, translational, rotational and vibrational) partition functions are recomputed and will be adjusted to any temperature or concentration. These default to 298.15 K and 1 atmosphere.

The quasi-harmonic approximation is applied to the vibrational entropy: below a given cut-off value vibrational normal modes are not well described by the rigid-rotor-harmonic-oscillator (RRHO) approximation and an alternative expression is instead used to compute the associated entropy. The quasi-harmonic vibrational entropy is always less than or equal to the standard (RRHO) value obtained using Gaussian. Two literature approaches have been implemented. In the simplest approach, from [Cramer and Truhlar](link)<sup>ref</sup>, all frequencies below the cut-off are uniformly shifted up to the cut-off value before entropy calculation in the RRHO approximation. Alternatively, as proposed by [Grimme](link)<sup>ref</sup>, entropic terms for frequencies below the cut-off are obtained from the free-rotor approximation; for those above the RRHO expression is retained. A damping function is used to interpolate between these two expressions close to the cut-off frequency. 

**Installation**
	1- Download the script from the wiki (ICIQ) or from the webpage (Add by Paton)
	2- Add the directory of the scripts to the PATH environmental variable. (optional).
	3- Run the script with your Gaussian outputs.

**Correct Usage**

```python
Goodvibes.py (-qh grimme/truhlar) (-f cutoff_freq) (-t temperature) (-c concentration) (-v scalefactor) (-ti temperature interval (initial, final, step(optional))) (-s solv) g09_output_file(s)
```
*	The `-qh` option selects the approximation for the quasiharmonic entropic correction: `-qh truhlar` or `-qh grimme` request the options explained above. Both avoid the tendency of RRHO vibrational entropies towards infinite values for low frequecies. If not specified this defaults to Grimme's expression.                                                      
*	The `-f` option specifies the frequency cut-off (in wavenumbers) i.e. `-qh 50` would use 50 cm<sup>-1</sup>. The default value is 100 cm<sup>-1</sup>. N.B. when set to zero all thermochemical values match standard (i.e. harmonic) Gaussian quantities.
*	The `-t` option specifies temperature (in Kelvin). N.B. This does not have to correspond to the temperature used in the Gaussian calculation since all thermal quantities are reevalulated by GoodVibes at the requested temperature. The default value is 298.15 K.
*	The `-c` option specifies concentration (in mol/l).  It is important to notice that the ideal gas approximation is used to relate the concentration with the pressure, so this option is the same as the Gaussian Pressure route line specification. The correction is applied to the Sackur-Tetrode equation of the translational entropy e.g. `-c 1` corrects to a solution-phase standard state of 1 mol/l. The default is 1 atmosphere.
*	The `-v` option is a scaling factor for vibrational frequencies. DFT-computed harmonic frequencies tend to overestimate experimentally measured IR and Raman absorptions. Empirical scaling factors have been determined for several functional/basis set combinations (e.g. by Radom and Truhlar groups). This correction scales the ZPE by the same factor, and also affects vibrational entropies. The default value is 1 (no scale factor).
*	The `-ti` option specifies a temperature interval (for example to see how a free energy barrier changes with the temperature). Usage is `-ti initial_temperature, final_temperature, step_size`. The step_size is optional, the default is set by the relationship (final_temp-initial_temp) /10
*	The `-s` option specifies the solvent. The amount of free space accessible to the solute is computed based on the solvent's molecular and bulk densities, which is used to correct the volume available to each molecule from the ideal gas approximation used in the Sackur-Tetrode calculation of translational entropy, as proposed by [Shakhnovich and Whitesides](link)<sup>ref</sup>. Based on the molecular Currently H2O, Toluene, DMF, AcOH and Chloroform are recognized. " the user can apply translational entropy solvent corrections… (Complete, I don’t know what it is exactly).


Example 1: python GoodVibes.py example01.out -f 100
------
With a freq. cut-off set to 0, the results will be identical to the standard values output by the Gaussian program.                             
This will calculate the quasi-harmonic corrected free energy (qh-G) using a frequency cut-off of 100 cm-1. 
In this case the numerical results are:
   G(T)           qh-G(T)
-1024.814592	-1024.809764

Note that the quasi-harmonic value will always be greater than or equal to the uncorrected free energy since the vibrational entropy associated with the low frequencies is reduced in this approach.  


Example 2: python GoodVibes.py example01.out –t 343 –c 1.0 –f 0
------
The code enables the free energy to be evaluated at any temperature (irrespective of the temperature used in the Gaussian calculation) and at any concentration. In this example no quasi-harmonic approximation would be used (-f 0), but the temperature is specified as 343K and the concentration as 1.0 mol/l. This latter term affects the translation entropy, and in this way can be used to correct the default Gaussian values at 1 atmosphere to a solution standard state of 1 mol/l.


Example 3: python GoodVibes.py example01.out –t 343 –c 1.0 –f 100 –s 0.995
------
It is also possible to apply a scaling factor to all frequencies since the harmonic approximation leads to an overestimate of experimentally measured IR and Raman absorptions. It is possible to find a scaling factor for most levels of theory (e.g. from Radom or Truhlar groups). This would reduce the ZPE by the same factor, and would also affect vibrational entropies. 

Tips:
*	The python file doesn’t need to be in the same folder as the Gaussian files. Just set the location of GoodVibes.py in the $PATH variable.
*	It is possible to run on any number of files at once, for example using wildcards to specify all of the Gaussian files in a directory (*.out)
*	The script will not work if terse output was requested in the Gaussian job.

[![DOI](https://zenodo.org/badge/16266/bobbypaton/GoodVibes.svg)](https://zenodo.org/badge/latestdoi/16266/bobbypaton/GoodVibes)
---
License: [CC-BY](https://creativecommons.org/licenses/by/3.0/)


