%chk=50_ts_cyclopropane_opening.chk
%mem=96GB
%nproc=16
# CASSCF(2,2)/6-31G(d) Opt=(TS,CalcFC) Freq

TS10: Thermal ring opening of cyclopropane -> propylene (1,3-H shift pathway)
Uses CASSCF(2,2) for correct description of partial bond breaking
Active space: 2 electrons in 2 orbitals (breaking C-C sigma bond)
Expect: 1 imaginary frequency corresponding to C-C bond cleavage + H migration

0 1
C   0.000000   0.000000   0.740000
C   0.740000   0.000000  -0.370000
C  -0.740000   0.000000  -0.370000
H   0.000000   0.930000   1.300000
H   0.000000  -0.930000   1.300000
H   1.380000   0.890000  -0.610000
H   1.380000  -0.890000  -0.610000
H  -1.380000   0.890000  -0.610000
H  -1.700000  -0.700000  -1.200000

