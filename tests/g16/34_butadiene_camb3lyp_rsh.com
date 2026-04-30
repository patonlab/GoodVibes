%chk=butadiene_camb3lyp.chk
%mem=8GB
%nprocshared=8
# CAM-B3LYP/6-311+G(d,p) Opt Freq emp=GD3BJ

1,3-Butadiene with CAM-B3LYP range-separated hybrid functional
CAM-B3LYP: Coulomb-attenuating method; excellent for charge-transfer excitations

0 1
C   0.000000   0.000000   0.000000
C   0.000000   0.000000   1.340000
C   0.000000   1.215000   2.040000
C   0.000000   1.215000   3.380000
H   0.000000  -0.921000  -0.547000
H   0.000000   0.921000  -0.547000
H   0.000000  -0.921000   1.887000
H   0.000000   2.136000   1.493000
H   0.000000   0.294000   3.927000
H   0.000000   2.136000   3.927000

