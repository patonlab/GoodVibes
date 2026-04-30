%chk=acetone_linked.chk
%mem=8GB
%nprocshared=8
# B3LYP/6-311G(d,p) Opt

Acetone geometry optimization (step 1 of linked job)

0 1
 C                 -2.84627743    0.19759832   -0.22954858
 O                 -1.68585831    0.22549950   -0.71557281
 C                 -3.59471343    1.51356653    0.05271229
 H                 -3.27752469    2.26085416   -0.64431892
 H                 -4.64776491    1.35407030   -0.04996933
 H                 -3.37886789    1.84011669    1.04854144
 C                 -3.51793476   -1.15251471    0.08297543
 H                 -4.18916722   -1.03447725    0.90784763
 H                 -4.06267637   -1.48697286   -0.77510051
 H                 -2.76863170   -1.87415955    0.33332248

--Link1--
%chk=acetone_linked.chk
%mem=8GB
%nprocshared=8
# B3LYP/6-311G(d,p) Freq=(Temperature=298.15,Pressure=1.0) Geom=AllCheck Guess=Read

Acetone frequency calculation (step 2 of linked job - reads optimized geometry)


