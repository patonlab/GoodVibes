%chk=48_ts_nh3_inversion.chk
%mem=96GB
%nproc=16
# MP2/6-311+G(d,p) Opt=(TS,CalcFC) Freq

TS07: Ammonia umbrella inversion TS (D3h planar structure)
Classic low-barrier TS: planar NH3 connecting two C3v minima
Barrier is only ~5.8 kcal/mol; tunnelling is dominant at room temperature
Expect: 1 imaginary frequency (~1000i cm-1), out-of-plane N inversion mode

0 1
N   0.000000   0.000000   0.000000
H   0.000000   0.947200   0.000000
H   0.820100  -0.473600   0.000000
H  -0.820100  -0.473600   0.000000

