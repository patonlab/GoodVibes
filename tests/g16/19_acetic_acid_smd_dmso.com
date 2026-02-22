%chk=acetic_acid_smd.chk
%mem=8GB
%nprocshared=8
# B3LYP/6-311+G(d,p) emp=GD3BJ Opt Freq SCRF=(SMD,Solvent=DMSO)

Acetic acid optimization+freq with SMD implicit solvation model in DMSO

0 1
C   0.000000   0.000000   0.000000
O   1.208000   0.000000   0.000000
O  -0.540000   1.140000   0.000000
H  -1.480000   1.030000   0.000000
C  -0.770000  -1.250000   0.000000
H  -0.430000  -1.850000   0.890000
H  -0.430000  -1.850000  -0.890000
H  -1.860000  -1.100000   0.000000

