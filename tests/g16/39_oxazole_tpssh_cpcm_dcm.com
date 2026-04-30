%chk=oxazole_tpssh_cpcm.chk
%mem=8GB
%nprocshared=8
# TPSSh/6-311+G(d,p) Opt Freq SCRF=(CPCM,Solvent=DiChloroMethane)

Oxazole with TPSSh meta-hybrid functional (10% HF exchange) in CPCM DCM
TPSSh: non-empirical meta-hybrid; good for transition metal compounds and organics

0 1
O   0.000000   0.000000   1.355000
C   1.161000   0.000000   0.726000
N   0.705000   0.000000  -0.538000
C  -0.590000   0.000000  -0.538000
C  -1.149000   0.000000   0.742000
H   2.150000   0.000000   1.145000
H  -1.200000   0.000000  -1.431000
H  -2.205000   0.000000   0.996000

