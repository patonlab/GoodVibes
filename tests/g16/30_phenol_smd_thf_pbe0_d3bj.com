%chk=phenol_smd_thf.chk
%mem=8GB
%nprocshared=8
# PBE1PBE/6-311+G(d,p) EmpiricalDispersion=GD3BJ Opt Freq SCRF=(SMD,Solvent=THF)

Phenol opt+freq with SMD solvation in THF and D3BJ dispersion correction
PBE0 (25% HF exchange) with Grimme D3BJ empirical dispersion

0 1
C   0.000000   1.396792   0.000000
C   1.209613   0.698396   0.000000
C   1.209613  -0.698396   0.000000
C   0.000000  -1.396792   0.000000
C  -1.209613  -0.698396   0.000000
C  -1.209613   0.698396   0.000000
O   0.000000   2.720000   0.000000
H   0.000000   3.100000   0.880000
H   2.150590   1.242106   0.000000
H   2.150590  -1.242106   0.000000
H   0.000000  -2.484212   0.000000
H  -2.150590  -1.242106   0.000000
H  -2.150590   1.242106   0.000000

