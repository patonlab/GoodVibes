%chk=acetic_acid_smd.chk
%mem=8GB
%nprocshared=8
# opt freq b3lyp/6-311+g(d,p) scrf=(smd,solvent=dmso) emp=gd3bj

Acetic acid optimization+freq with SMD implicit solvation model in DMSO

0 1
 C                  0.00000000    0.17558300    0.00000000
 O                  0.57301804    0.29014442    1.05640152
 O                 -1.32820900   -0.08996100    0.00000000
 H                 -1.65784902   -0.15586473    0.90897738
 C                  0.67132811    0.30979919   -1.34264378
 H                  1.47201700   -0.43062668   -1.40472198
 H                  1.12578676    1.30116215   -1.40472198
 H                 -0.00653771    0.17427594   -2.18655398

