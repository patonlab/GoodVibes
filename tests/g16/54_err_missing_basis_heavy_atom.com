%chk=54_err_missing_basis.chk
%mem=96GB
%nproc=16
# B3LYP/6-311+G(d,p) Opt Freq

ERROR EXAMPLE: Basis set not available for heavy element
PbCl2 with Pople basis set — 6-311+G(d,p) does not include parameters for Pb (Z=82)
Pople basis sets only cover elements up to Kr (Z=36), some extend to I (Z=53)
Gaussian error: "Atomic number out of range" / basis function error in L301
Fix: Use GenECP with SDD or def2-TZVP for Pb, or use an all-electron basis like def2-TZVP

0 1
Pb     0.000000    0.000000    0.000000
Cl     2.320000    0.700000    0.000000
Cl    -2.320000    0.700000    0.000000

