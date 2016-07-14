Goodvibes
=====

A Python program to compute quasi-harmonic thermochemical data from Gaussian frequency calculations at a given temperature/concentration, corrected for the effects of vibrational scaling-factors and available free space in solvent.

All (electronic, translational, rotational and vibrational) partition functions are recomputed and will be adjusted to any temperature or concentration. These default to 298.15 K and 1 atmosphere.

The quasi-harmonic approximation is applied to the vibrational entropy: below a given cut-off value vibrational normal modes are not well described by the rigid-rotor-harmonic-oscillator (RRHO) approximation and an alternative expression is instead used to compute the associated entropy. The quasi-harmonic vibrational entropy is always less than or equal to the standard (RRHO) value obtained using Gaussian. Two literature approaches have been implemented. In the simplest approach, from [Cramer and Truhlar](link)ref, all frequencies below the cut-off are uniformly shifted up to the cut-off value before entropy calculation in the RRHO approximation. Alternatively, as proposed by [Grimme](link)ref, entropic terms for frequencies below the cut-off are obtained from the free-rotor approximation; for those above the RRHO expression is retained. A damping function is used to interpolate between these two expressions close to the cut-off frequency. 

**Installation**
	1- Download the script from the wiki (ICIQ) or from the webpage (Add by Paton)
	2- Add the directory of the scripts to the PATH environmental variable. (optional).
	3- Run the script with your Gaussian outputs.

**Correct Usage**

```python
Goodvibes.py (-qh grimme/truhlar) (-f cutoff_freq) (-t temperature) (-c concentration) (-v scalefactor) (-ti temperature interval (initial, final, step(optional))) (-s solv) g09_output_file(s)
```
*	The `-qh` option is used to  select the approximation for the quasiharmonic entropic correction: `-qh truhlar` or `-qh grimme` request either option explained above. Both avoid the tendency of RRHO vibrational entropies towards infinite values as frequecies become smaller. If not specified this defaults to Grimme's expression.                                                      
*	The `-f` option is used to specify the frequency cut-off value (in wavenumbers) i.e. `-qh 50` would use 50 cm-1. The default value is 100 cm-1. N.B. when set to zero, all thermochemical values will correspond directly to those obtained with Gaussian in the standard harmonic approximation.
*	-t option: the user can select the temperature as in G09 with the option Temperature in the command line. The default value is 298.15 K.
*	-c option: the concentration is changed to a value in mol/l units. The default is 1 atmosphere. It is important to notice that the ideal gas approximation is used to relate the concentration with the pressure, so this option is the same that the g09 Pressure option in the command line of a calculation. It is useful to apply standard state corrections in the calculations.
*	-v option: This option allows the user to apply scale factor corrections to the vibrational frequencies. The default value is 1 (no scale factor).
*	-ti option: This option allows computing the free energy in an interval of temperature (for example to see how the barrier changes with the temperature). The interval should be written as initial_temperature, final_temperature, step_size. The step_size is optional, the default is set by the relationship (final_temp-initial_temp) /10
*	-s option: the user can apply translational entropy solvent corrections… (Complete, I don’t know what it is exactly).


Example 1: python GoodVibes.py example01.out -f 100
With a freq. cut-off set to 0, the results will be identical to the standard values output by the Gaussian program.                             
------
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
•	The python file doesn’t need to be in the same folder as the Gaussian files. Just set the location of GoodVibes.py in the $PATH variable.
•	It is possible to run on any number of files at once, for example using wildcards to specify all of the Gaussian files in a directory (*.out)
•	The script will not work if terse output was requested in the Gaussian job.

[![DOI](https://zenodo.org/badge/16266/bobbypaton/GoodVibes.svg)](https://zenodo.org/badge/latestdoi/16266/bobbypaton/GoodVibes)
---
License: [CC-BY](https://creativecommons.org/licenses/by/3.0/)


