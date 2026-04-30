%chk=imidazole_apfd.chk
%mem=8GB
%nprocshared=8
# APFD/6-311+G(d,p) Opt Freq=NoRaman

Imidazole with APFD functional (Austin-Petersson-Frisch with dispersion)
APFD: hybrid meta-GGA with built-in dispersion; Freq=NoRaman skips Raman intensities

0 1
H  0.000000  2.119822  0.714354
H  0.000000  1.202262 -1.904898
H  0.000000 -2.104815  0.663782
H  0.000000 -0.010302  2.116597
C  0.000000  1.120107  0.305897
C  0.000000  0.635508 -0.983749
C  0.000000 -1.091835  0.283881
N  0.000000 -0.741378 -0.994001
N  0.000000  0.000000  1.104571

