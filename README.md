Goodvibes
=====

A Python program to compute quasi-harmonic thermochemical data from Gaussian frequency calculations at a given temperature/concentration, corrected for the effects of vibrational scaling-factors and available free space in solvent. Developed by [Dr Robert Paton](http://paton.chem.ox.ac.uk) (Oxford) and [Ignacio Funes-Ardois](http://www.iciq.org/staff/funes-ignacio/) (ICIQ).

All (electronic, translational, rotational and vibrational) partition functions are recomputed and will be adjusted to any temperature or concentration. These default to 298.15 K and 1 atmosphere.

The quasi-harmonic approximation is applied to the vibrational entropy: below a given cut-off value vibrational normal modes are not well described by the rigid-rotor-harmonic-oscillator (RRHO) approximation and an alternative expression is instead used to compute the associated entropy. The quasi-harmonic vibrational entropy is always less than or equal to the standard (RRHO) value obtained using Gaussian. Two literature approaches have been implemented. In the simplest approach, from [Cramer and Truhlar](http://pubs.acs.org/doi/abs/10.1021/jp205508z)<sup>1</sup>, all frequencies below the cut-off are uniformly shifted up to the cut-off value before entropy calculation in the RRHO approximation. Alternatively, as proposed by [Grimme](http://onlinelibrary.wiley.com/doi/10.1002/chem.201200497/full)<sup>2</sup>, entropic terms for frequencies below the cut-off are obtained from the free-rotor approximation; for those above the RRHO expression is retained. A damping function is used to interpolate between these two expressions close to the cut-off frequency. 

##Installation
1. Download the script from https://github.com/bobbypaton/GoodVibes or from the wiki (ICIQ)  
2. Add the directory of the scripts to the PATH environmental variable (optional).  
3.	Run the script with your Gaussian output files.  

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
*	The `-s` option specifies the solvent. The amount of free space accessible to the solute is computed based on the solvent's molecular and bulk densities. This is then used to correct the volume available to each molecule from the ideal gas approximation used in the Sackur-Tetrode calculation of translational entropy, as proposed by [Shakhnovich and Whitesides](http://pubs.acs.org/doi/abs/10.1021/jo970944f)<sup>3</sup>. Currently H2O, Toluene, DMF, AcOH and Chloroform are recognized.


Example 1: a Grimme-type quasi-harmonic correction with cut-off = 100 wavenumbers
------
```python
python GoodVibes.py examples/methylaniline.out -f 100 

                                         E/au      ZPE/au           H/au      T.S/au   T.qh-S/au        G(T)/au     qh-G(T)/au 
   *************************************************************************************************************************** 
o  examples/methylaniline         -326.664901    0.142118    -326.514489    0.039668    0.039535    -326.554157    -326.554024 

```

The output shows both standard harmonic and quasi-harmonic corrected thermochemical data (in Hartree). The corrected entropy is always less than or equal to the harmonic value, and the corrected Gibbs energy is greater than or equal to the uncorrected value.

**Tips and Troubleshooting**
*	The python file doesn’t need to be in the same folder as the Gaussian files. Just set the location of GoodVibes.py in the $PATH variable.
*	It is possible to run on any number of files at once, for example using wildcards to specify all of the Gaussian files in a directory (*.out)
*	The script will not work if terse output was requested in the Gaussian job.

##References
1. Ribeiro, R. F.; Marenich, A. V.; Cramer, C. J.; Truhlar, D. G. *J. Phys. Chem. B* **2011**, *115*, 14556-14562 [**DOI:** 10.1021/jp205508z](http://pubs.acs.org/doi/abs/10.1021/jp205508z)  
2. Grimme, S. *Chem. Eur. J.* **2012**, *18*, 9955–9964 [**DOI:** 10.1002/chem.201200497](http://onlinelibrary.wiley.com/doi/10.1002/chem.201200497/full)  
3. Mammen, M.; Shakhnovich, E. I.; Deutch, J. M.; Whitesides, G. M. *J. Org. Chem.* **1998**, *63*, 3821-3830 [**DOI:** 10.1021/jo970944f](http://pubs.acs.org/doi/abs/10.1021/jo970944f)  

[![DOI](https://zenodo.org/badge/16266/bobbypaton/GoodVibes.svg)](https://zenodo.org/badge/latestdoi/16266/bobbypaton/GoodVibes)
---
License: [CC-BY](https://creativecommons.org/licenses/by/3.0/)


