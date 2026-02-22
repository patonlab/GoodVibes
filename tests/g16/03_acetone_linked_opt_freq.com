%chk=acetone_linked.chk
%mem=8GB
%nprocshared=8
# B3LYP/6-311G(d,p) Opt

Acetone geometry optimization (step 1 of linked job)

0 1
C  -1.237058  -0.072095  0.008552
C  -0.045874  -0.232754  0.027849
C  -2.681953  0.120061  -0.014637
O  3.20096  -0.01995  0.006461
H  1.013463  -0.374456  0.044905
H  3.180992  0.678428  0.669082
H  3.203343  0.450236  -0.833844
H  -3.205729  -0.839911  0.007269
H  -3.015812  0.702912  0.848694
H  -2.994627  0.651115  -0.918376

--Link1--
%chk=acetone_linked.chk
%mem=8GB
%nprocshared=8
# B3LYP/6-311G(d,p) Freq=(Temperature=298.15,Pressure=1.0) Geom=AllCheck Guess=Read

Acetone frequency calculation (step 2 of linked job - reads optimized geometry)


