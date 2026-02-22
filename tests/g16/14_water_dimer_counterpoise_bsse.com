%chk=water_dimer_cp.chk
%mem=8GB
%nprocshared=8
# B3LYP/6-311+G(d,p) emp=GD3 Counterpoise=2

Water dimer BSSE-corrected interaction energy (Boys-Bernardi counterpoise correction)

0 1
O(Fragment=1)   0.000000   0.000000   0.119262
H(Fragment=1)   0.000000   0.756950  -0.477047
H(Fragment=1)   0.000000  -0.756950  -0.477047
O(Fragment=2)   0.000000   0.000000   2.836000
H(Fragment=2)   0.000000   0.756950   3.432047
H(Fragment=2)   0.000000  -0.756950   3.432047

