%chk=hf_ccsdt.chk
%mem=96GB
%nprocshared=16
# CCSD(T)/cc-pVTZ

HF molecule CCSD(T)/cc-pVTZ single point energy 

0 1
H   0.000000   0.000000   0.000000
F   0.000000   0.000000   0.916800

