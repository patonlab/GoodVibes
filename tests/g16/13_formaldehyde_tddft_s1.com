%chk=formaldehyde_s1.chk
%mem=16GB
%nprocshared=8
#p B97D/def2SVP TD=(NStates=5,Root=1) Opt Freq

Formaldehyde S1 excited state optimization and frequency (TD-DFT, n->pi* transition)

0 1
C   0.000000   0.000000  -0.523841
O   0.000000   0.000000   0.676159
H   0.000000   0.939547  -1.109428
H   0.000000  -0.939547  -1.109428

