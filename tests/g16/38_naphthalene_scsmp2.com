%chk=naphthalene_scsmp2.chk
%mem=96GB
%nprocshared=16
# MP2/cc-pVTZ IOp(3/125=0500060506) Freq=NoRaman

Naphthalene with SCS-MP2 (Spin-Component-Scaled MP2) via IOp
IOp(3/125=0500060506): sets SCS scaling factors (0.5 for same-spin, 1.2 for opposite-spin)
SCS-MP2 (Grimme 2003): improved accuracy over regular MP2 for non-covalent interactions

0 1
C   0.000000   1.406100   0.718400
C   0.000000   1.406100  -0.718400
C   0.000000   0.000000   1.406100
C   0.000000   0.000000  -1.406100
C   0.000000  -1.406100   0.718400
C   0.000000  -1.406100  -0.718400
C   0.000000   2.440200   1.397200
C   0.000000   2.440200  -1.397200
C   0.000000  -2.440200   1.397200
C   0.000000  -2.440200  -1.397200
H   0.000000   0.000000   2.490000
H   0.000000   0.000000  -2.490000
H   0.000000   3.374000   0.860000
H   0.000000   3.374000  -0.860000
H   0.000000  -3.374000   0.860000
H   0.000000  -3.374000  -0.860000

