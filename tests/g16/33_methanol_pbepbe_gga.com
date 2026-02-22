%chk=methanol_pbe.chk
%mem=8GB
%nprocshared=4
# PBEPBE/6-311G(d,p) Opt Freq=NoRaman

Methanol with PBE/PBE GGA functional (popular in solid-state / periodic DFT)
PBEPBE: PBE exchange + PBE correlation, non-empirical GGA

0 1
C   0.000000   0.000000   0.000000
O   1.430000   0.000000   0.000000
H  -0.363000   1.026000   0.000000
H  -0.363000  -0.513000   0.889000
H  -0.363000  -0.513000  -0.889000
H   1.830000   0.890000   0.000000

