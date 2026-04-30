%chk=n2o_highT_highP.chk
%mem=96GB
%nproc=16
# CCSD/cc-pVTZ Opt Freq=(Temperature=1000.0,Pressure=100.0)

N2O linear triatomic (C*v symmetry) at high temperature (1000 K) and pressure (100 atm)
CCSD/cc-pVTZ for accurate thermochemistry; industrial/combustion conditions

0 1
N   0.000000   0.000000  -1.128000
N   0.000000   0.000000   0.000000
O   0.000000   0.000000   1.185000

