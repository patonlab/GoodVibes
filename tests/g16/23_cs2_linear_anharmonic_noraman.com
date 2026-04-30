%chk=cs2_anharmonic.chk
%mem=16GB
%nprocshared=8
# B3LYP/6-311+G(2df,p) Opt Freq=(Anharmonic,NoRaman)

CS2 linear triatomic (D*h) - anharmonic VPT2 frequencies without Raman
Combines Freq=Anharmonic with NoRaman for efficiency

0 1
S   0.000000   0.000000  -1.554000
C   0.000000   0.000000   0.000000
S   0.000000   0.000000   1.554000

