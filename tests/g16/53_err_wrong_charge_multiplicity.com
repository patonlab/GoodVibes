%chk=53_err_wrong_charge_mult.chk
%mem=96GB
%nproc=16
# B3LYP/6-31G(d) Opt Freq

ERROR EXAMPLE: Impossible charge/multiplicity combination
Water (10 electrons) specified as neutral doublet (0,2)
Even electron count is incompatible with doublet (which requires odd electrons)
Gaussian error: "The combination of multiplicity 2 and N electrons is impossible"
Fix: Use (0,1) for neutral singlet water, or (-1,2) / (+1,2) for a radical ion

0 2
O      0.000000    0.000000    0.117300
H      0.000000    0.756950   -0.469200
H      0.000000   -0.756950   -0.469200

