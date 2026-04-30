%chk=pyridine_smd_mecn.chk
%mem=8GB
%nprocshared=8
# wB97XD/6-311+G(d,p) Opt Freq SCRF=(SMD,Solvent=Acetonitrile)

Pyridine optimization+freq with SMD solvation in acetonitrile
Functional: wB97X-D (range-separated hybrid with dispersion)

0 1
N   0.000000   0.000000   1.417226
C   1.143193   0.000000   0.739193
C   1.199400   0.000000  -0.646540
C   0.000000   0.000000  -1.349880
C  -1.199400   0.000000  -0.646540
C  -1.143193   0.000000   0.739193
H   2.037264   0.000000   1.352020
H   2.147310   0.000000  -1.162730
H   0.000000   0.000000  -2.433660
H  -2.147310   0.000000  -1.162730
H  -2.037264   0.000000   1.352020

