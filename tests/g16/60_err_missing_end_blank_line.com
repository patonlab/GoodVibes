%chk=60_err_missing_blank_line.chk
%mem=96GB
%nproc=16
# B3LYP/6-31G(d) Opt Freq

ERROR EXAMPLE: Missing trailing blank line after coordinates
Gaussian requires at least one blank line after the last atom coordinate
Without it, the input parser encounters an unexpected end-of-file
Gaussian error: "End of file in ZMat" or "End of file reading connectivity" / L101
Fix: Add a blank line after the last atom; most text editors strip trailing newlines

0 1
O      0.000000    0.000000    0.117300
H      0.000000    0.756950   -0.469200
H      0.000000   -0.756950   -0.469200