Goodvibes
=========

[![Build Status](https://travis-ci.org/bobbypaton/GoodVibes.svg?branch=master)](https://travis-ci.org/bobbypaton/GoodVibes)
[![PyPI version](https://badge.fury.io/py/goodvibes.svg)](https://badge.fury.io/py/goodvibes)
[![Anaconda-Server Badge](https://anaconda.org/patonlab/goodvibes/badges/installer/conda.svg)](https://conda.anaconda.org/patonlab)
[![Anaconda-Server Badge](https://anaconda.org/patonlab/goodvibes/badges/downloads.svg)](https://anaconda.org/patonlab/goodvibes)
[![DOI](https://zenodo.org/badge/54848929.svg)](https://zenodo.org/badge/latestdoi/54848929)

A Python program to compute quasi-harmonic thermochemical data from Gaussian frequency calculations at a given temperature/concentration, corrected for the effects of vibrational scaling-factors and available free space in solvent. Developed by [Robert Paton](https://www.chem.colostate.edu/person/?id=234D5F1C7E4CBA9E192AB2B2837D6360) (Colorado State & Oxford) and [Ignacio Funes-Ardoiz](http://www.iciq.org/staff/funes-ignacio/) (ICIQ). Integration with Travis CI testing by [Jaime Rodríguez-Guerra](https://github.com/jaimergp).

All (electronic, translational, rotational and vibrational) partition functions are recomputed and will be adjusted to any temperature or concentration. These default to 298.15 K and 1 atmosphere.

The quasi-harmonic approximation is applied to the vibrational entropy: below a given cut-off value vibrational normal modes are not well described by the rigid-rotor-harmonic-oscillator (RRHO) approximation and an alternative expression is instead used to compute the associated entropy. The quasi-harmonic vibrational entropy is always less than or equal to the standard (RRHO) value obtained using Gaussian. Two literature approaches have been implemented. In the simplest approach, from [Cramer and Truhlar](http://pubs.acs.org/doi/abs/10.1021/jp205508z),<sup>1</sup> all frequencies below the cut-off are uniformly shifted up to the cut-off value before entropy calculation in the RRHO approximation. Alternatively, as proposed by [Grimme](http://onlinelibrary.wiley.com/doi/10.1002/chem.201200497/full),<sup>2</sup> entropic terms for frequencies below the cut-off are obtained from the free-rotor approximation; for those above the RRHO expression is retained. A damping function is used to interpolate between these two expressions close to the cut-off frequency.

The program will attempt to parse the level of theory and basis set used in the calculations and then try to apply the appropriate vibrational (zpe) scaling factor. Scaling factors are taken from the [Truhlar group database](https://t1.chem.umn.edu/freqscale/index.html).

#### Installation
*	With pypi: `pip install goodvibes`
*  With conda: `conda install -c patonlab goodvibes`
*  Manually Cloning the repository https://github.com/bobbypaton/GoodVibes.git and then adding the location of the GoodVibes directory to the PYTHONPATH environment variable.
3. Run the script with your Gaussian output files (the program expects log or out extensions). It has been tested with Python 2 and 3 on Linux, OSX and Windows


**Correct Usage**

```python
python -m goodvibes [-q grimme/truhlar] [-f cutoff_freq] [-t temperature] [-c concentration] [-v scalefactor] [-s solvent name] [--spc link/filename] [--xyz] [--imag] [--cpu] [--ti 't_initial, t_final, step'] [--ci 'c_initial, c_final, step'] <gaussian_output_file(s)>
```
*	The `-h` option gives help by listing all available options, default values and units, and proper usage.
*	The `-q` option selects the approximation for the quasiharmonic entropic correction: `-q truhlar` or `-q grimme` request the options explained above. Both avoid the tendency of RRHO vibrational entropies towards infinite values for low frequecies. If not specified this defaults to Grimme's expression.
*	The `-f` option specifies the frequency cut-off (in wavenumbers) i.e. `-f 50` would use 50 cm<sup>-1</sup>. The default value is 100 cm<sup>-1</sup>. N.B. when set to zero all thermochemical values match standard (i.e. harmonic) Gaussian quantities.
*	The `-t` option specifies temperature (in Kelvin). N.B. This does not have to correspond to the temperature used in the Gaussian calculation since all thermal quantities are reevalulated by GoodVibes at the requested temperature. The default value is 298.15 K.
*	The `-c` option specifies concentration (in mol/l).  It is important to notice that the ideal gas approximation is used to relate the concentration with the pressure, so this option is the same as the Gaussian Pressure route line specification. The correction is applied to the Sackur-Tetrode equation of the translational entropy e.g. `-c 1` corrects to a solution-phase standard state of 1 mol/l. The default is 1 atmosphere.
*	The `-v` option is a scaling factor for vibrational frequencies. DFT-computed harmonic frequencies tend to overestimate experimentally measured IR and Raman absorptions. Empirical scaling factors have been determined for several functional/basis set combinations, and these are applied automatically using values from the Truhlar group<sup>3</sup> based on detection of the level of theory and basis set in the output files. This correction scales the ZPE by the same factor, and also affects vibrational entropies. The default value when no scaling factor is available is 1 (no scale factor). The automated scaling can also be surpressed by `-v 1.0`
*	The `--ti` option specifies a temperature interval (for example to see how a free energy barrier changes with the temperature). Usage is `--ti 'initial_temperature, final_temperature, step_size'`. The step_size is optional, the default is set by the relationship (final_temp-initial_temp) /10
*	The `-s` option specifies the solvent. The amount of free space accessible to the solute is computed based on the solvent's molecular and bulk densities. This is then used to correct the volume available to each molecule from the ideal gas approximation used in the Sackur-Tetrode calculation of translational entropy, as proposed by [Shakhnovich and Whitesides](http://pubs.acs.org/doi/abs/10.1021/jo970944f).<sup>4</sup> The keywords H2O, toluene, DMF (N,N-dimethylformamide), AcOH (acetic acid) and chloroform are recognized.
* the `--spc` option can be used to obtain single point energy corrected values. For multi-step jobs in which a frequency calculation is followed by an additional (e.g. single point energy) calculation, the energy is taken from the final job and all thermal corrections are taken from the frequency calculation. Alternatively, the energy can be taken from an additional file.
*	The `--xyz` option will write all Cartesian coordinates to an xyz file.
* the `--imag` option will print any imaginary frequencies (in wavenumbers) for each structure. Presently, all are reported. The hard-coded variable im_freq_cutoff can be edited to change this. To generate new input files (i.e. if this is an undesirable imaginary frequency) see [pyQRC](https://github.com/bobbypaton/pyQRC)
* the `--imag` option will print any imaginary frequencies (in wavenumbers) for each structure. Presently, all are reported. The hard-coded variable im_freq_cutoff can be edited to change this.
* the `--cpu` option will add up all of the CPU time across all files (including single point calculations if requested).

#### Example 1: a Grimme-type quasi-harmonic correction with a (Grimme type) cut-off of 100 cm<sup>-1</sup>
```python
python -m goodvibes examples/methylaniline.out -f 100

                                         E/au      ZPE/au           H/au      T.S/au   T.qh-S/au        G(T)/au     qh-G(T)/au
   ***************************************************************************************************************************
o  examples/methylaniline         -326.664901    0.142118    -326.514489    0.039668    0.039535    -326.554157    -326.554024

```

The output shows both standard harmonic and quasi-harmonic corrected thermochemical data (in Hartree). The corrected entropy is always less than or equal to the harmonic value, and the corrected Gibbs energy is greater than or equal to the uncorrected value.

#### Example 2: Quasi-harmonic thermochemistry with a larger basis set single point energy correction
```python
python -m goodvibes examples/ethane_spc.out --spc link

   Structure                     E_link             E        ZPE        H_link        T.S     T.qh-S     G(T)_link  qh-G(T)_link
   *****************************************************************************************************************************
o  examples/ethane_spc       -79.858399    -79.830421   0.073508    -79.780448   0.027569   0.027570    -79.808017    -79.808019
   *****************************************************************************************************************************

```

This calculation contains a multi-step job: an optimization and frequency calculation with a small basis set followed by (--Link1--) a larger basis set single point energy. Note the use of the `--spc link` option. The standard harmonic and quasi-harmonic corrected thermochemical data are obtained from the small basis set partition function combined with the larger basis set single point electronic energy. In this example, GoodVibes automatically recognizes the level of theory used in the frequency calculation, B3LYP/6-31G(d), and applies the appropriate scaling factor of 0.977 (this can be surpressed to apply no scaling with -v 1.0)

Alternatively, if a single point energy calculation has been performed separately, provided both file names share a common root e.g. `ethane.out` and `ethane_TZ.out` then use of the `--spc TZ` option is appropriate. This will give identical results as above.

```python
python -m goodvibes examples/ethane.out --spc TZ

   Structure                       E_TZ             E        ZPE          H_TZ        T.S     T.qh-S       G(T)_TZ    qh-G(T)_TZ
   *****************************************************************************************************************************
o  examples/ethane           -79.858399    -79.830421   0.073508    -79.780448   0.027569   0.027570    -79.808017    -79.808019
   *****************************************************************************************************************************

```


#### Example 3: Changing the temperature (from standard 298.15 K to 1000 K) and concentration (from standard state in gas phase, 1 atm, to standard state in solution, 1 mol/l)
```python
python -m goodvibes examples/methylaniline.out –t 1000 –c 1.0

                                         E/au      ZPE/au           H/au      T.S/au   T.qh-S/au        G(T)/au     qh-G(T)/au
   ***************************************************************************************************************************
o  examples/methylaniline         -326.664901    0.142118    -326.452307    0.218212    0.216560    -326.670519    -326.668866

```

This correction from 1 atm to 1 mol/l is responsible for the addition 1.89 kcal/mol to the Gibbs energy of each species (at 298K). It affects the translational entropy, which is the only component of the molecular partition function to show concentration depdendence. In the example above the correction is larger due to the increase in temperature.

#### Example 4: Analyzing the Gibbs energy across an interval of temperatures 300-1000 K with a stepsize of 100 K, applying a (Truhlar type) cut-off of 100 cm<sup>-1</sup>
```python
python -m goodvibes examples/methylaniline.out –-ti '300,1000,100' –q truhlar –f 100

                                                   H/au      T.S/au   T.qh-S/au        G(T)/au     qh-G(T)/au
   **********************************************************************************************************
o  examples/methylaniline.out @ 300.0       -326.514399    0.040005    0.040005    -326.554404    -326.554404
o  examples/methylaniline.out @ 400.0       -326.508735    0.059816    0.059816    -326.568551    -326.568551
o  examples/methylaniline.out @ 500.0       -326.501670    0.082625    0.082625    -326.584296    -326.584296
o  examples/methylaniline.out @ 600.0       -326.493429    0.108148    0.108148    -326.601577    -326.601577
o  examples/methylaniline.out @ 700.0       -326.484222    0.136095    0.136095    -326.620317    -326.620317
o  examples/methylaniline.out @ 800.0       -326.474218    0.166216    0.166216    -326.640434    -326.640434
o  examples/methylaniline.out @ 900.0       -326.463545    0.198300    0.198300    -326.661845    -326.661845
o  examples/methylaniline.out @ 1000.0      -326.452307    0.232169    0.232169    -326.684476    -326.684476

```

Note that the energy and ZPE are not printed in this instance since they are temperature-independent. The Truhlar-type quasiharmonic correction sets all frequencies below than 100 cm<sup>-1</sup> to a value of 100. Constant pressure is assumed, so that the concentration is recomputed at each temperature.

#### Example 5: Analyzing the Gibbs Energy using scaled vibrational frequencies
```python
python -m goodvibes examples/methylaniline.out -v 0.95

                                         E/au      ZPE/au           H/au      T.S/au   T.qh-S/au        G(T)/au     qh-G(T)/au
   ***************************************************************************************************************************
o  examples/methylaniline         -326.664901    0.135012    -326.521265    0.040238    0.040091    -326.561503    -326.561356

```

The frequencies are scaled by a factor of 0.95 before they are used in the computation of the vibrational energies (including ZPE) and entropies.

#### Example 6: Writing Cartesian coordinates
```python
python -m goodvibes examples/HCN*.out --xyz

```

All optimized coordinates are written to Goodvibes_output.xyz

#### Example 7: Analyzing multiple files at once
```python
python -m goodvibes examples/*.out --cpu

                                         E/au      ZPE/au           H/au      T.S/au   T.qh-S/au        G(T)/au     qh-G(T)/au
   ***************************************************************************************************************************
o  examples/Al_298K               -242.328708    0.000000    -242.326347    0.017670    0.017670    -242.344018    -242.344018
o  examples/Al_400K               -242.328708    0.000000    -242.326347    0.017670    0.017670    -242.344018    -242.344018
o  examples/CuCN                  -289.005463    0.006594    -288.994307    0.025953    0.025956    -289.020260    -289.020264
o  examples/H2O                    -76.368128    0.020772     -76.343577    0.021458    0.021458     -76.365035     -76.365035
o  examples/HCN_singlet            -93.358851    0.015978     -93.339373    0.022896    0.022896     -93.362269     -93.362269
o  examples/HCN_triplet            -93.153787    0.012567     -93.137780    0.024070    0.024070     -93.161850     -93.161850
o  examples/allene                -116.569605    0.053913    -116.510916    0.027618    0.027621    -116.538534    -116.538537
o  examples/ethane                 -79.830421    0.075238     -79.750770    0.027523    0.027525     -79.778293     -79.778295
o  examples/ethane_spc             -79.830421    0.075238     -79.750770    0.027523    0.027525     -79.778293     -79.778295
o  examples/methylaniline         -326.664901    0.142118    -326.514489    0.039668    0.039535    -326.554157    -326.554024
   ***************************************************************************************************************************
   TOTAL CPU      0 days  0 hrs 30 mins 54 secs

```

The program will detect several different levels of theory and give a warning that any vibrational scaling factor other than 1 would be inappropriate in this case.

**Tips and Troubleshooting**
*	The python file doesn’t need to be in the same folder as the Gaussian files. Just set the location of GoodVibes.py in the $PATH variable
*	It is possible to run on any number of files at once, for example using wildcards to specify all of the Gaussian files in a directory (*.out)
*  File names not in the form of filename.log or filename.out are not read
*	The script will not work if terse output was requested in the Gaussian job

#### Papers citing GoodVibes
1. Li, Y.; Du, S. *RSC Adv.* **2016**, *6*, 84177-84186 [**DOI:** 10.1039/C6RA16321A](http://dx.doi.org/10.1039/C6RA16321A)
2. Myllys, N.; Elm, J.; Kurtén, T. *Comp. Theor. Chem.* **2016**, *1098*, 1–12 [**DOI:** 10.1016/j.comptc.2016.10.015](http://dx.doi.org/10.1016/j.comptc.2016.10.015)
3. Kiss, E.; Campbell, C. D.; Driver, R. W.; Jolliffe, J. D.; Lang, R.; Sergeieva, T.; Okovytyy, S.; Paton, R. S.; Smith, M. D. *Angew. Chem. Int. Ed.* **2016**, *128* 14017-14021 [**DOI:** 10.1002/ange.201608534](http://dx.doi.org/10.1002/ange.201608534)
4. Mohamed, S.; Krenske, E. H.; Ferro, V. *Org. Biomol. Chem.* **2016**, *14*, 2950-2960 [**DOI:** 10.1039/c6ob00283h](http://dx.doi.org/10.1039/c6ob00283h)
5. Deb, A.; Hazra, A.; Peng, Q.; Paton, R. S.; Maiti, D. *J. Am. Chem. Soc.* **2017**, *139*, 763–775 [**DOI:** 10.1021/jacs.6b10309](http://dx.doi.org/10.1021/jacs.6b10309)
6. Simón, L.; Paton, R. S. *J. Org. Chem.* **2017**, *82*, 3855-3863 [**DOI:** 10.1021/acs.joc.7b00540](http://dx.doi.org/10.1021/acs.joc.7b00540)
7. Grayson, M. N. *J. Org. Chem.* **2017**, *82*, 4396–4401 [**DOI:** 10.1021/acs.joc.7b00521](http://dx.doi.org/10.1021/acs.joc.7b00521)
8. Duarte, F.; Paton, R. S. *J. Am. Chem. Soc.* **2017**, *139*, 8886-8896 [**DOI:** 10.1021/jacs.7b02468](http://dx.doi.org/10.1021/jacs.7b02468)
9. Elm, J. *J. Phys. Chem. A* **2017**, *121*, 8288−8295 [**DOI:** 10.1021/acs.jpca.7b08962](http://dx.doi.org/10.1021/acs.jpca.7b08962)
10. Münster, N.; Parker, N. A.; van Dijk, L.; Paton, R. S.; Smith, M. D. *Angew. Chem. Int. Ed.* **2017**, *56*, 9468-9472 [**DOI:** 10.1002/anie.201705333](http://dx.doi.org/10.1002/anie.201705333)
11. Mekareeya, A.; Walker, P. R.; Couce-Rios, A.; Campbell, C. D.; Steven, A.; Paton, R. S.; Anderson, E. A. *J. Am. Chem. Soc.* **2017**, *139*, 10104–10114 [**DOI** 10.1021/jacs.7b05436](http://dx.doi.org/10.1021/jacs.7b05436)
12. Alegre-Requena, J. V.; Marqués-López, E.; Herrera, R. P. *ACS Catal.* **2017**, *7*, 6430–6439 [**DOI:** 10.1021/acscatal.7b02446](http://dx.doi.org/10.1021/acscatal.7b02446)
13. Elm, J. *J. Phys. Chem. A* **2017**, *121*, 8288–8295 [**DOI:** 10.1021/acs.jpca.7b08962](http://dx.doi.org/10.1021/acs.jpca.7b08962)
14. Li, Y.; Jackson, K. E.; Charlton, A.; Le Neve-Foster, B.; Khurshid, A.; Rudy, H.-K. A.; Thompson, A. L.; Paton, R. S.; Hodgson, D. M. *J. Org. Chem.* **2017**, *82*, 10479-10488 [**DOI:** 10.1021/acs.joc.7b01954](http://dx.doi.org/10.1021/acs.joc.7b01954)
14. Alegre-Requena, J. V.; Marqués-López, E.; Herrera, R. P. *Chem. Eur. J.* **2017**, *23*, 15336–15347[**DOI:** 10.1002/chem.201702841](http://dx.doi.org/10.1002/chem.201702841)
15. Funes‐Ardoiz, I.; Nelson, D. J.; Maseras, F. *Chem. Eur. J.* **2017**, *23*, 16728–16733 [**DOI:** 10.1002/chem.201702331](http://dx.doi.org/10.1002/chem.201702331)
16. Morris, D. S.; van Rees, K.; Curcio, M.; Cokoja, M.; Kühn, F. E.; Duarte, F.; Love, J. B. *Catal. Sci. Technol.* **2017**, 5644–5649 [**DOI:** 10.1039/C7CY01728F ](http://dx.doi.org/10.1039/C7CY01728F)
17. Besora, M.; Vidossich, P.; Lledos, A.; Ujaque, G.; Maseras, F. *J. Phys. Chem. A* **2018**, *122*, 1392–1399 [**DOI:** 10.1021/acs.jpca.7b11580 ](http://dx.doi.org/10.1021/acs.jpca.7b11580)
18. Harada, T. *J. Org. Chem.* **2018**, *83*, 7825–7835 [**DOI:** 10.1021/acs.joc.8b00712](http://dx.doi.org/10.1021/acs.joc.8b00712)
19. Lewis, R. D.; Garcia-Borràs, M.;  Chalkley, M. J.; Buller, A. R.; Houk, K. N.; Kan, S. B. J.; Arnold, F. H. *Proc. Natl. Acad. Sci.* **2018**, *115*, 7308-7313 [**DOI:** 10.1073/pnas.1807027115](http://dx.doi.org/10.1073/pnas.1807027115)

#### References for the underlying theory
1. Ribeiro, R. F.; Marenich, A. V.; Cramer, C. J.; Truhlar, D. G. *J. Phys. Chem. B* **2011**, *115*, 14556-14562 [**DOI:** 10.1021/jp205508z](http://dx.doi.org/10.1021/jp205508z)
2. Grimme, S. *Chem. Eur. J.* **2012**, *18*, 9955–9964 [**DOI:** 10.1002/chem.201200497](http://dx.doi.org/10.1002/chem.201200497/full)
3. Alecu, I. M.; Zheng, J.; Zhao, Y.; Truhlar, D. G.; *J. Chem. Theory Comput.* **2010**, *6*, 2872-2887 [**DOI:** 10.1021/ct100326h](http://dx.doi.org/10.1021/ct100326h)
4. Mammen, M.; Shakhnovich, E. I.; Deutch, J. M.; Whitesides, G. M. *J. Org. Chem.* **1998**, *63*, 3821-3830 [**DOI:** 10.1021/jo970944f](http://dx.doi.org/10.1021/jo970944f)

---
License: [CC-BY](https://creativecommons.org/licenses/by/3.0/)
