%chk=dmso_solvation.chk
%mem=8GB
%nprocshared=8
# B3LYP/6-311+G(d,p) Opt SCRF=(CPCM,Solvent=Water)

DMSO solvent molecule: CPCM geometry optimization in water 

0 1
S   0.000000   0.000000   0.000000
O   0.000000   0.000000   1.530000
C   1.765000   0.000000  -0.490000
C  -1.765000   0.000000  -0.490000
H   1.900000   0.890000  -1.110000
H   1.900000  -0.890000  -1.110000
H   2.580000   0.000000   0.230000
H  -1.900000   0.890000  -1.110000
H  -1.900000  -0.890000  -1.110000
H  -2.580000   0.000000   0.230000

--Link1--
%chk=dmso_solvation.chk
%mem=8GB
%nprocshared=8
# B3LYP/6-311+G(d,p) Freq=NoRaman Geom=AllCheck Guess=Read SCRF=(CPCM,Solvent=Water)

DMSO frequency on CPCM geometry 


