%chk=49_ts_oh_methane.chk
%mem=96GB
%nproc=16
# wB97XD/6-311+G(d,p) Opt=(noeigen,TS,CalcFC) Freq

TS08: H-atom abstraction from methane by OH radical
CH4 + OH -> CH3 + H2O; doublet surface
Atmospherically and combustion-relevant reaction; near-linear C-H-O arrangement
Expect: 1 imaginary frequency (~1500i cm-1), H-transfer mode

0 2
 O                 -1.37566100   -0.04253200    0.00000000
 H                 -0.45812500    0.04762400    0.00000000
 C                  1.04300000   -0.03375000    0.00000000
 H                  1.42405097    0.03367320    0.99757419
 H                  1.33783075    0.83303747   -0.55376378
 H                  1.43512772   -0.90969036   -0.47314304
 H                 -1.61197695    0.77623803    0.44200726

