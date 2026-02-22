%chk=47_ts_e2_elimination.chk
%mem=96GB
%nproc=16
# B3LYP/6-311+G(d,p) Opt=(loose,noeigen,TS,CalcFC) Freq

TS06: E2 elimination - OH- + CH3CH2Cl -> CH2=CH2 + H2O + Cl-
Anti-periplanar TS: H, C, C, Cl dihedral ~180 degrees
Concerted base-induced elimination; anti-periplanar geometry required
Expect: 1 imaginary frequency (~600i cm-1), H transfer + C-Cl breaking mode

-1 1
 O                  2.15519445   -2.31822413   -0.52655144
 H                  3.11400172   -2.18048550   -0.54245027
 C                  1.08332329   -0.12899668    0.01950054
 C                 -0.35274571   -0.41387368    0.01190954
 Cl                -2.14012929    0.21680968    0.00561546
 H                  1.59316229   -1.06496868   -0.18520646
 H                  1.42498929    0.25360132    0.98469754
 H                  1.35721229    0.59217132   -0.75267246
 H                 -0.67916171   -1.23410268    0.65747554
 H                 -0.59547471   -0.83358568   -0.97583846

