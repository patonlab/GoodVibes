%chk=phenol_smd_thf.chk
%mem=8GB
%nprocshared=8
# opt freq 6-311+g(d,p) scrf=(smd,solvent=thf) empiricaldispersion=gd3bj
pbe1pbe

Phenol opt+freq with SMD solvation in THF and D3BJ dispersion correction
PBE0 (25% HF exchange) with Grimme D3BJ empirical dispersion

0 1
 C                  0.00217600    0.93360600    0.00000000
 C                  0.86366311    0.20560185    0.81482440
 C                  0.84409980   -1.18436743    0.77209258
 C                 -0.02693980   -1.84906552   -0.08562923
 C                 -0.88019139   -1.11476259   -0.90354193
 C                 -0.86574641    0.27541331   -0.86578401
 O                  0.05773700    2.30999500    0.00000000
 H                 -0.56136540    2.65536238   -0.65142611
 H                  1.54271278    0.73561047    1.47523594
 H                  1.51666263   -1.74888532    1.41027035
 H                 -0.03819757   -2.93352678   -0.11909243
 H                 -1.55966112   -1.62470268   -1.57924458
 H                 -1.52219295    0.85933220   -1.50318315

