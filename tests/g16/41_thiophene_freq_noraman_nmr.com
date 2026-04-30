%chk=thiophene_nmr_freq.chk
%mem=8GB
%nprocshared=8
# mPW1PW91/6-311+G(2d,p) Opt Freq=NoRaman NMR=GIAO

Thiophene: combined opt, freq (no Raman), and NMR in one job
Efficient workflow: single pass for IR frequencies and NMR chemical shifts

0 1
S   0.000000   0.000000   1.718000
C   1.153000   0.000000   0.607000
C   0.716000   0.000000  -0.686000
C  -0.716000   0.000000  -0.686000
C  -1.153000   0.000000   0.607000
H   2.148000   0.000000   0.992000
H   1.284000   0.000000  -1.568000
H  -1.284000   0.000000  -1.568000
H  -2.148000   0.000000   0.992000

