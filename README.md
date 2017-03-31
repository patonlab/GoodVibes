Goodvibes
=====

A Python program to compute quasi-harmonic thermochemical data from Gaussian frequency calculations at a given temperature/concentration, corrected for the effects of vibrational scaling-factors and available free space in solvent. Developed by [Dr Robert Paton](http://paton.chem.ox.ac.uk) (Oxford) and [Ignacio Funes-Ardoiz](http://www.iciq.org/staff/funes-ignacio/) (ICIQ).

All (electronic, translational, rotational and vibrational) partition functions are recomputed and will be adjusted to any temperature or concentration. These default to 298.15 K and 1 atmosphere.

The quasi-harmonic approximation is applied to the vibrational entropy: below a given cut-off value vibrational normal modes are not well described by the rigid-rotor-harmonic-oscillator (RRHO) approximation and an alternative expression is instead used to compute the associated entropy. The quasi-harmonic vibrational entropy is always less than or equal to the standard (RRHO) value obtained using Gaussian. Two literature approaches have been implemented. In the simplest approach, from [Cramer and Truhlar](http://pubs.acs.org/doi/abs/10.1021/jp205508z)<sup>1</sup>, all frequencies below the cut-off are uniformly shifted up to the cut-off value before entropy calculation in the RRHO approximation. Alternatively, as proposed by [Grimme](http://onlinelibrary.wiley.com/doi/10.1002/chem.201200497/full)<sup>2</sup>, entropic terms for frequencies below the cut-off are obtained from the free-rotor approximation; for those above the RRHO expression is retained. A damping function is used to interpolate between these two expressions close to the cut-off frequency. 

#### Installation
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
* the `-spc` option can be used for multi-step jobs in which a frequency calculation is followed by an additional (e.g. single point energy) calculation. The energy is taken from the final job and all thermal corrections are taken from the frequency calculation. The Gibbs energy is thus the single-point corrected value
*	The `-s` option specifies the solvent. The amount of free space accessible to the solute is computed based on the solvent's molecular and bulk densities. This is then used to correct the volume available to each molecule from the ideal gas approximation used in the Sackur-Tetrode calculation of translational entropy, as proposed by [Shakhnovich and Whitesides](http://pubs.acs.org/doi/abs/10.1021/jo970944f)<sup>3</sup>. The keywords H2O, Toluene, DMF (N,N-dimethylformamide), AcOH (acetic acid) and Chloroform are recognized.


#### Example 1: a Grimme-type quasi-harmonic correction with a (Grimme type) cut-off of 100 cm<sup>-1</sup>
```python
python GoodVibes.py examples/methylaniline.out -f 100 

                                         E/au      ZPE/au           H/au      T.S/au   T.qh-S/au        G(T)/au     qh-G(T)/au 
   *************************************************************************************************************************** 
o  examples/methylaniline         -326.664901    0.142118    -326.514489    0.039668    0.039535    -326.554157    -326.554024 

```

The output shows both standard harmonic and quasi-harmonic corrected thermochemical data (in Hartree). The corrected entropy is always less than or equal to the harmonic value, and the corrected Gibbs energy is greater than or equal to the uncorrected value.

#### Example 2: Quasi-harmonic thermochemistry with a larger basis set single point energy correction
```python
python GoodVibes.py examples/ethane_spc.out 

                                         E/au      ZPE/au           H/au      T.S/au   T.qh-S/au        G(T)/au     qh-G(T)/au 
   *************************************************************************************************************************** 
o  examples/ethane_spc             -79.830417    0.075236     -79.750766    0.025837    0.025839     -79.776603     -79.776605 

```

The calculation is a multi-step job: an optimization and frequency calculation with a small basis set followed by (--Link1--) a larger basis set single point energy. The standard harmonic and quasi-harmonic corrected thermochemical data are obtained from the small basis set partition function combined with the larger basis set single point electronic energy. 

#### Example 3: Changing the temperature (from standard 298.15 K to 1000 K) and concentration (from standard state in gas phase, 1 atm, to standard state in solution, 1 mol/l)
```python
python GoodVibes.py examples/methylaniline.out –t 1000 –c 1.0 

                                         E/au      ZPE/au           H/au      T.S/au   T.qh-S/au        G(T)/au     qh-G(T)/au 
   *************************************************************************************************************************** 
o  examples/methylaniline         -326.664901    0.142118    -326.452307    0.218212    0.216560    -326.670519    -326.668866

```

This correction from 1 atm to 1 mol/l is responsible for the addition 1.89 kcal/mol to the Gibbs energy of each species (at 298K). It affects the translational entropy, which is the only component of the molecular partition function to show concentration depdendence. In the example above the correction is larger due to the increase in temperature.

#### Example 4: Analyzing the Gibbs energy across an interval of temperatures 300-1000 K with a stepsize of 100 K, applying a (Truhlar type) cut-off of 100 cm<sup>-1</sup>
```python
python GoodVibes.py examples/methylaniline.out –ti 300,1000,100 –qh truhlar –f 100 
       
                                                   H/au      T.S/au   T.qh-S/au        G(T)/au     qh-G(T)/au 
   *************************************************************************************************************************** 
o  examples/methylaniline.out @ 300.0       -326.514399    0.040005    0.040005    -326.554404    -326.554404 
o  examples/methylaniline.out @ 400.0       -326.508735    0.059816    0.059816    -326.568551    -326.568551 
o  examples/methylaniline.out @ 500.0       -326.501670    0.082625    0.082625    -326.584296    -326.584296 
o  examples/methylaniline.out @ 600.0       -326.493429    0.108148    0.108148    -326.601577    -326.601577 
o  examples/methylaniline.out @ 700.0       -326.484222    0.136095    0.136095    -326.620317    -326.620317 
o  examples/methylaniline.out @ 800.0       -326.474218    0.166216    0.166216    -326.640434    -326.640434 
o  examples/methylaniline.out @ 900.0       -326.463545    0.198300    0.198300    -326.661845    -326.661845 
o  examples/methylaniline.out @ 1000.0      -326.452307    0.232169    0.232169    -326.684476    -326.684476

```

Note that the energy and ZPE are not printed in this instance since they are temperature-independent. The Truhlar-type quasiharmonic correction sets all frequencies below than 100 cm<sup>-1</sup> to a value of 100.

#### Example 5: Analyzing the Gibbs Energy using scaled vibrational frequencies
```python
python GoodVibes.py examples/methylaniline.out -v 0.95 
       
                                         E/au      ZPE/au           H/au      T.S/au   T.qh-S/au        G(T)/au     qh-G(T)/au 
   *************************************************************************************************************************** 
o  examples/methylaniline         -326.664901    0.135012    -326.521265    0.040238    0.040091    -326.561503    -326.561356 

```

The frequencies are scaled by a factor of 0.95 before they are used in the computation of the vibrational energies (including ZPE) and entropies. 

#### Example 6: Analyzing multiple files at once
```python
python GoodVibes.py examples/*.out 
       
                                         E/au      ZPE/au           H/au      T.S/au   T.qh-S/au        G(T)/au     qh-G(T)/au 
   *************************************************************************************************************************** 
o  examples/H2O                    -76.368128    0.020772     -76.343577    0.021458    0.021458     -76.365035     -76.365035 
o  examples/HCN_singlet            -93.358851    0.015978     -93.339373    0.022896    0.022896     -93.362269     -93.362269 
o  examples/HCN_triplet            -93.153787    0.012567     -93.137780    0.024070    0.024070     -93.161850     -93.161850 
o  examples/allene                -116.569605    0.053913    -116.510916    0.027618    0.027621    -116.538534    -116.538537 
o  examples/ethane                 -79.770819    0.073070     -79.693288    0.025918    0.025920     -79.719206     -79.719208 
o  examples/ethane_spc             -79.830417    0.075236     -79.750766    0.025837    0.025839     -79.776603     -79.776605 
o  examples/example01            -1025.266528    0.506850   -1024.734173    0.080419    0.075591   -1024.814592   -1024.809764 
o  examples/methylaniline         -326.664901    0.142118    -326.514489    0.039668    0.039535    -326.554157    -326.554024 

``` 

**Tips and Troubleshooting**
*	The python file doesn’t need to be in the same folder as the Gaussian files. Just set the location of GoodVibes.py in the $PATH variable.
*	It is possible to run on any number of files at once, for example using wildcards to specify all of the Gaussian files in a directory (*.out)
*	The script will not work if terse output was requested in the Gaussian job.

#### Papers citing GoodVibes
1. Li, Y.; Du, S. *RSC Adv.* **2016**, *6*, 84177-84186 [**DOI:** 10.1039/C6RA16321A](http://dx.doi.org/10.1039/C6RA16321A)
2. Myllys, N.; Elm, J.; Kurtén, T. *Comp. Theor. Chem.* **2016**, *1098*, 1–12 [**DOI:** 10.1016/j.comptc.2016.10.015](http://dx.doi.org/10.1016/j.comptc.2016.10.015)
3. Kiss, E.; Campbell, C. D.; Driver, R. W.; Jolliffe, J. D.; Lang, R.; Sergeieva, T.; Okovytyy, S.; Paton, R. S.; Smith, M. D. *Angew. Chem. Int. Ed.* **2016**, *128* 14017-14021 [**DOI:** 10.1002/ange.201608534](http://dx.doi.org/10.1002/ange.201608534)
4. Deb, A.; Hazra, A.; Peng, Q.; Paton, R. S.; Maiti, D. *J. Am. Chem. Soc.* **2017**, *139*, 763–775 [**DOI:** 10.1021/jacs.6b10309](http://dx.doi.org/10.1021/jacs.6b10309)
5. Gorobets, E.; Wong, N. E.; Paton, R. S.; Derksen, D. J. *Org. Lett.* **2017**, *19*, 484-487 [**DOI:** 10.1021/acs.orglett.6b03635](http://dx.doi.org/10.1021/acs.orglett.6b03635)
5. Grayson, M. N. *J. Org. Chem.* **2017** [**DOI:** 10.1021/acs.joc.7b00521](http://dx.doi.org/10.1021/acs.joc.7b00521) 
6. Simón, L.; Paton, R. S. *J. Org. Chem.* **2017** [**DOI:** 10.1021/acs.joc.7b00540](http://dx.doi.org/10.1021/acs.joc.7b00540)

#### References for the underlying theory
1. Ribeiro, R. F.; Marenich, A. V.; Cramer, C. J.; Truhlar, D. G. *J. Phys. Chem. B* **2011**, *115*, 14556-14562 [**DOI:** 10.1021/jp205508z](http://dx.doi.org/10.1021/jp205508z)  
2. Grimme, S. *Chem. Eur. J.* **2012**, *18*, 9955–9964 [**DOI:** 10.1002/chem.201200497](http://dx.doi.org/10.1002/chem.201200497/full)  
3. Mammen, M.; Shakhnovich, E. I.; Deutch, J. M.; Whitesides, G. M. *J. Org. Chem.* **1998**, *63*, 3821-3830 [**DOI:** 10.1021/jo970944f](http://dx.doi.org/10.1021/jo970944f)  

[![DOI](https://zenodo.org/badge/16266/bobbypaton/GoodVibes.svg)](https://zenodo.org/badge/latestdoi/16266/bobbypaton/GoodVibes)
---
License: [CC-BY](https://creativecommons.org/licenses/by/3.0/)


