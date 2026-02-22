%chk=furan_mn15.chk
%mem=8GB
%nprocshared=8
# MN15/def2TZVP Opt Freq

Furan with MN15 functional (Minnesota 2016, excellent broad thermochemistry)
MN15: local hybrid meta-GGA, good for diverse chemical environments

0 1
O   0.000000   0.000000   1.392000
C   1.192000   0.000000   0.697000
C   0.724000   0.000000  -0.600000
C  -0.724000   0.000000  -0.600000
C  -1.192000   0.000000   0.697000
H   2.148000   0.000000   1.219000
H   1.286000   0.000000  -1.485000
H  -1.286000   0.000000  -1.485000
H  -2.148000   0.000000   1.219000

