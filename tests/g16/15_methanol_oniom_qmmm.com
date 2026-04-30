%chk=oniom_methanol.chk
%mem=16GB
%nprocshared=8
# ONIOM(B3LYP/6-31G(d):PM6) Opt Freq

ONIOM 2-layer QM/MM: methanol high layer (B3LYP/6-31G(d)), full system low layer (PM6)
H = high layer (QM), L = low layer (MM/semi-empirical)

0 1 0 1 0 1
C  -0.748000   0.000000   0.000000 H
H  -1.090000   1.026000   0.000000 H
H  -1.090000  -0.513000  -0.889000 H
H  -1.090000  -0.513000   0.889000 H
O   0.677000   0.000000   0.000000 L
H   1.044000   0.000000  -0.891000 L

