%chk=44_ts_sn2_identity.chk
%mem=96GB
%nproc=16
# B3LYP/6-311+G(d,p) Opt=(TS,CalcFC,NoEigenTest) Freq

TS01: SN2 identity reaction Cl- + CH3Cl -> ClCH3 + Cl-
Classic textbook SN2 TS: C3v symmetry, collinear Cl-C-Cl arrangement
Expect: 1 imaginary frequency (~400i cm-1), antisymmetric Cl-C-Cl stretch

-1 1
Cl  0.000000   0.000000  -2.400000
C   0.000000   0.000000   0.000000
Cl  0.000000   0.000000   2.400000
H   1.026719   0.000000   0.000000
H  -0.513360   0.889165   0.000000
H  -0.513360  -0.889165   0.000000

