%chk=methylammonium_cpcm.chk
%mem=8GB
%nprocshared=8
# M06/6-311+G(d,p) Opt Freq SCRF=(CPCM,Solvent=Water)

Methylammonium cation (charge=+1) in CPCM water - pKa relevant calculation
CPCM particularly suitable for charged species in polar solvents

1 1
N   0.000000   0.000000   0.000000
C   0.000000   0.000000   1.490000
H   1.013000   0.000000  -0.342000
H  -0.507000   0.878000  -0.342000
H  -0.507000  -0.878000  -0.342000
H  -1.025000   0.000000   1.832000
H   0.513000   0.889000   1.832000
H   0.513000  -0.889000   1.832000

