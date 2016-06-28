GoodVibes.py reads vibrational data from a Gaussian frequency calculation and recomputes the partition function using a combination of the rigid-rotor harmonic oscillator (RRHO) and free rotor models. The resulting vibrational entropy is interpolated between these two limiting values: the user specifies a cut-off frequency (typically 50-100 cm-1) to set the value at which the switching takes place. Frequencies higher than this cut-off will continue to be described as RRHOs, while those below will be treated as free rotors although a damping function ensure that the switching between the two different expressions is not abrupt.

The reason for correcting vibrational entropies of low frequencies has been discussed in the literature, by Cramer, Truhlar and Grimme. The present implementation in GoodVibes remains faithful to recommendations of Grimme in the computation of binding thermodynamics (S. Grimme Chem. Eur. J. 2012 18, 9955). 

The output of this program will show the total energy, ZPE, H, TS  and G terms which are already listed in the Gaussian output file. It will also show the corrected values of TS and G which result from this so-called quasiharmonic approximation, which are listed as T.qh-S and qh-G.


Example A: python GoodVibes.py example01.out -f 100

This will calculate the quasi-harmonic corrected free energy (qh-G) using a frequency cut-off of 100 cm-1. 
In this case the numerical results are:
   G(T)           qh-G(T)
-1024.814592	-1024.809764

Note that the quasi-harmonic value will always be greater than or equal to the uncorrected free energy since the vibrational entropy associated with the low frequencies is reduced in this approach.  


Example B: python GoodVibes.py example01.out –t 343 –c 1.0 –f 0

The code enables the free energy to be evaluated at any temperature (irrespective of the temperature used in the Gaussian calculation) and at any concentration. In this example no quasi-harmonic approximation would be used (-f 0), but the temperature is specified as 343K and the concentration as 1.0 mol/l. This latter term affects the translation entropy, and in this way can be used to correct the default Gaussian values at 1 atmosphere to a solution standard state of 1 mol/l.


Example C: python GoodVibes.py example01.out –t 343 –c 1.0 –f 100 –s 0.995

It is also possible to apply a scaling factor to all frequencies since the harmonic approximation leads to an overestimate of experimentally measured IR and Raman absorptions. It is possible to find a scaling factor for most levels of theory (e.g. from Radom or Truhlar groups). This would reduce the ZPE by the same factor, and would also affect vibrational entropies. 

Tips:
•	The python file doesn’t need to be in the same folder as the Gaussian files. Just set the location of GoodVibes.py in the $PATH variable.
•	It is possible to run on any number of files at once, for example using wildcards to specify all of the Gaussian files in a directory (*.out)
•	The script will not work if terse output was requested in the Gaussian job.

[![DOI](https://zenodo.org/badge/16266/bobbypaton/GoodVibes.svg)](https://zenodo.org/badge/latestdoi/16266/bobbypaton/GoodVibes)
	

