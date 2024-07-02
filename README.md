![GoodVibes](https://github.com/patonlab/GoodVibes/blob/master/GoodVibes.png)


[![Build Status](https://app.travis-ci.com/patonlab/GoodVibes.svg?branch=master)](https://app.travis-ci.com/github/patonlab/GoodVibes)
[![PyPI version](https://badge.fury.io/py/goodvibes.svg)](https://badge.fury.io/py/goodvibes)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/goodvibes/badges/downloads.svg)](https://anaconda.org/conda-forge/goodvibes)
[![Documentation Status](https://readthedocs.org/projects/goodvibespy/badge/?version=stable)](https://goodvibespy.readthedocs.io/en/stable/?badge=stable)
[![DOI](https://zenodo.org/badge/54848929.svg)](https://zenodo.org/badge/latestdoi/54848929)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/goodvibes/badges/license.svg)](https://anaconda.org/conda-forge/goodvibes)

GoodVibes is a Python package to compute thermochemical data from one or a series of electronic structure calculations. It has been used since 2015 by several groups, primarily to correct the poor description of low frequency vibrations by the rigid-rotor harmonic oscillator treatment. The current version includes thermochemistry at variable temperature/concentration, various quasi-harmonic entropy and enthalpy schemes, automated detection of frequency scaling factors, Boltzmann averaging, duplicate conformer filtering, automated tabulation and plotting of energy profiles, and error checking.

All (electronic, translational, rotational and vibrational) partition functions are recomputed and will be adjusted to any temperature or concentration. These default to 298.15 Kelvin and 1 atmosphere.

The program will attempt to parse the level of theory and basis set used in the calculations and then try to apply the appropriate vibrational (zpe) scaling factor. Scaling factors are taken from the [Truhlar group database](https://t1.chem.umn.edu/freqscale/index.html).

#### Documentation
GoodVibes documentation can be found on our [read-the-docs page](https://goodvibespy.readthedocs.io/en/latest/).

#### Quasi-Harmonic Approximation
Two types of quasi-harmonic approximation are readily applied. The first is vibrational entropy: below a given cut-off value vibrational normal modes are not well described by the rigid-rotor-harmonic-oscillator (RRHO) approximation and an alternative expression is instead used to compute the associated entropy. The quasi-harmonic vibrational entropy is always less than or equal to the standard (RRHO) value obtained using Gaussian. Two literature approaches have been implemented. In the simplest approach, from [Cramer and Truhlar](http://pubs.acs.org/doi/abs/10.1021/jp205508z),<sup>1</sup> all frequencies below the cut-off are uniformly shifted up to the cut-off value before entropy calculation in the RRHO approximation. Alternatively, as proposed by [Grimme](http://onlinelibrary.wiley.com/doi/10.1002/chem.201200497/full),<sup>2</sup> entropic terms for frequencies below the cut-off are obtained from the free-rotor approximation; for those above the RRHO expression is retained. A damping function is used to interpolate between these two expressions close to the cut-off frequency.

The second type of quasi-harmonic approximation available is applied to the vibrational energy used in enthalpy calculations. Similar to the entropy corrections, the enthalpy correction implements a quasi-harmonic correction to the RRHO vibrational energy computed in DFT methods. The quasi-harmonic enthalpy value as specified by [Head-Gordon](https://pubs.acs.org/doi/10.1021/jp509921r)<sup>3</sup> will be less than or equal to the uncorrected value using the RRHO approach, as the quasi-RRHO value of the vibrational energy used to compute the enthalpy is damped to approach a value of 0.5RT, opposed to the RRHO value of RT. Because of this, the quasi-harmonic enthalpy correction is appropriate for use in systems and reactions resulting in a loss of a rotational or translational degree of freedom.

#### Installation
*  With pypi: `pip install goodvibes`
*  With conda: `conda install -c conda-forge goodvibes`

#### Citing GoodVibes
Luchini, G.; Alegre-Requena, J. V.; Funes-Ardoiz, I.; Paton, R. S. GoodVibes: Automated Thermochemistry for Heterogeneous Computational Chemistry Data. *F1000Research*, **2020**, *9*, 291 [**DOI:** 10.12688/f1000research.22758.1](https://doi.org/10.12688/f1000research.22758.1)

#### Usage

```python
python -m goodvibes [-q] [--qs grimme/truhlar] [--qh] [-f cutoff_freq] [--fs S_cutoff_freq] [--fh H_cutoff_freq]
[--check] [-t temperature] [-c concentration] [--ti 't_initial, t_final, step'] [--ee] [--bav "global" or "conf"]
[--cosmo cosmo_filename] [--cosmoint cosmo_filename,initial_temp,final_temp] [-v frequency_scale_factor]
[--nosymm] [--spc link/filename] [--boltz] [--dup][--pes pes_yaml] [--nogconf]
[--graph graph_yaml] [--cpu] [--imag] [--invertifreq] [--freespace solvent_name] [--output output_name]
[--media solvent_name] [--xyz] [--csv] [--custom_ext file_extension] <output_file(s)>
```

[**Usage**](https://goodvibespy.readthedocs.io/en/latest/source/README.html#using-goodvibes) for an explanation of these arguments

[**Examples**](https://goodvibespy.readthedocs.io/en/latest/source/README.html#examples) for example usage.

[**Checks**](https://goodvibespy.readthedocs.io/en/latest/source/README.html#checks) for information on automated job checking

#### Symmetry
GoodVibes is able to detect a probable point group for each species and apply a symmetry correction to the entropy. As of version 4 this uses the python interface to the [pymsym](https://github.com/corinwagen/pymsym) package.


#### Tips and Troubleshooting
*	The python file doesn’t need to be in the same folder as the Gaussian files. Just set the location of GoodVibes.py in the `$PATH` variable of your system (this is not necessary if installed with pip or conda)
*	It is possible to run on any number of files at once using wildcards to specify all of the Gaussian files in a directory (specify `*.out` or `*.log`)
*   File names not in the form of filename.log or filename.out are not read, however more file extensions can be added with the option `--custom_ext`
*	The script will not work if terse output was requested in the Gaussian job
*  Problems may occur with Restart Gaussian jobs due to missing information in the output file.
*  HF, DFT, MP2, semi-empirical, time dependent (TD) DFT and HF, ONIOM, and G4 calculations from Gaussian are also supported.


#### Contributors

- [Robert Paton](https://orcid.org/0000-0002-0104-4166)
- [Ignacio Funes-Ardoiz](https://orcid.org/0000-0002-5843-9660)
- [Guilian Luchini](https://orcid.org/0000-0003-0135-9624)
- [Juan V. Alegre-Requena](https://orcid.org/0000-0002-0769-7168)
- [Yanfei Guan](https://orcid.org/0000-0003-1817-0190)
- [Jaime Rodríguez-Guerra](https://orcid.org/0000-0001-8974-1566)
- [Eric Berquist](https://github.com/berquist)
- [JingTao Chen](https://github.com/NKUCodingCat)
- [Julia Velmiskina](https://github.com/Margoju)
- [Shree Sowndarya](https://orcid.org/0000-0002-4568-5854)
- [Heather Mayes](https://github.com/hmayes)
- [Sibo Wang](https://github.com/sibo)
- [F Roessler](https://github.com/fdroessler)

---
#### License:

GoodVibes is freely available under an [MIT](https://opensource.org/licenses/MIT) License
