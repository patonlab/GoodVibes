%chk=formaldehyde_s1.chk
%mem=16GB
%nprocshared=8
#p opt freq td=(nstates=5,root=1) b97d def2svp

Formaldehyde S1 excited state optimization and frequency (TD-DFT, n->pi*
transition)

0 1
 C                  0.00000000    0.00000000   -0.52384100
 O                  0.30085383   -0.00000000    0.69806633
 H                  0.40450492    0.87365134   -0.99075545
 H                  0.40450492   -0.87365134   -0.99075545

