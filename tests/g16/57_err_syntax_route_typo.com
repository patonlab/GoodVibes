%chk=57_err_syntax_route.chk
%mem=96GB
%nproc=16
# B3LPY/6-31G(d) Opt Feq

ERROR EXAMPLE: Typos on the route line
Two deliberate misspellings: "B3LPY" (should be B3LYP) and "Feq" (should be Freq)
Gaussian's parser cannot recognise unknown method or keyword strings
Gaussian error: "Unknown method" or "Illegal keyword" in L1 / L101
Fix: Correct to B3LYP and Freq; always double-check route line spelling

0 1
O      0.000000    0.000000    0.117300
H      0.000000    0.756950   -0.469200
H      0.000000   -0.756950   -0.469200

