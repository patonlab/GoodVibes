%chk=46_ts_h3_abstraction.chk
%mem=96GB
%nproc=16
# MP2/aug-cc-pVTZ Opt=(TS,CalcFC) Freq

TS04: H + H2 -> H2 + H identity hydrogen atom abstraction
The simplest possible TS: linear H-H-H (D*h), doublet
Textbook benchmark for ab initio methods; well-characterised potential surface
Expect: 1 imaginary frequency (~1500i cm-1), antisymmetric H-H-H stretch

0 2
H   0.000000   0.000000  -0.930000
H   0.000000   0.000000   0.000000
H   0.000000   0.000000   0.930000

