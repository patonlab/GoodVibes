%chk=ethane_opt_freq.chk
%mem=8GB
%nprocshared=8
#p B3LYP/6-311+G(d,p) Opt Freq=(Temperature=398.15,Pressure=2.0)

Ethane B3LYP opt+freq at 398.15 K and 2.0 atm (industrial conditions)

0 1
C   0.000000   0.000000   0.000000
C   0.000000   0.000000   1.540000
H   1.026719   0.000000  -0.363000
H  -0.513360   0.889165  -0.363000
H  -0.513360  -0.889165  -0.363000
H  -1.026719   0.000000   1.903000
H   0.513360   0.889165   1.903000
H   0.513360  -0.889165   1.903000

