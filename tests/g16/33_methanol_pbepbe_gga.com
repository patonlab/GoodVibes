%chk=methanol_pbe.chk
%mem=8GB
%nprocshared=4
# opt freq=noraman pbepbe/6-311g(d,p)

Methanol with PBE/PBE GGA functional (popular in solid-state / periodic
DFT) PBEPBE: PBE exchange + PBE correlation, non-empirical GGA

0 1
 C                 -0.04922700    0.66511200    0.00000000
 O                 -0.04922700   -0.76543700    0.00000000
 H                 -0.04922700    1.09498000   -1.01645200
 H                  0.84411300    1.04476600    0.52685100
 H                 -0.94256700    1.04476600    0.52685100
 H                 -0.04922700   -1.05168600    0.92333800

