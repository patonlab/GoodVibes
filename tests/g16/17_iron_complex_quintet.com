%chk=fe_co5_complex.chk
%mem=96GB
%nprocshared=16
# PBE1PBE/def2SVP SCF(xqc,maxcycles=500) Opt Freq Guess=Mix

Iron pentacarbonyl Fe(CO)5: transition metal, high-spin quintet (multiplicity=5), def2-TZVP

0 5
Fe  0.000000   0.000000   0.000000
C   0.000000   0.000000   1.810000
C   0.000000   0.000000  -1.810000
C   1.810000   0.000000   0.000000
C  -1.810000   0.000000   0.000000
C   0.000000   1.810000   0.000000
O   0.000000   0.000000   2.990000
O   0.000000   0.000000  -2.990000
O   2.990000   0.000000   0.000000
O  -2.990000   0.000000   0.000000
O   0.000000   2.990000   0.000000

