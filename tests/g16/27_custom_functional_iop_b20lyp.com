%chk=custom_functional.chk
%mem=8GB
%nprocshared=8
# BLYP/6-311+G(d,p) IOp(3/76=1000002000) Opt Freq

Custom hybrid functional defined via IOp(3/76):
IOp(3/76=1000002000) modifies HF exchange mixing in BLYP -> creates a 20% HF exchange hybrid
Format: IOp(3/76=XXXXX YYYYY) where XXXXX = (1-c_HF)*1000 and YYYYY = c_HF*1000
Here: 10000 - 2000 = 8000 (80% DFT) + 2000 (20% HF) = custom B20LYP

0 1
C   0.000000   0.000000   0.000000
C   0.000000   0.000000   1.540000
H   1.026719   0.000000  -0.363000
H  -0.513360   0.889165  -0.363000
H  -0.513360  -0.889165  -0.363000
H  -1.026719   0.000000   1.903000
H   0.513360   0.889165   1.903000
H   0.513360  -0.889165   1.903000

