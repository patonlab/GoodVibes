%chk=51_err_scf_convergence.chk
%mem=96GB
%nproc=16
# UB3LYP/6-31G(d) Opt Freq

ERROR EXAMPLE: SCF convergence failure
Fe(II) porphine model — open-shell transition metal with near-degenerate states
Default SCF settings will fail to converge (oscillating energy, no convergence)
Gaussian error: "Convergence criterion not met" / L502 termination
Fix: Add SCF=(xqc,maxcycles=500) and/or Guess=Mix to route line

0 5
Fe     0.000000    0.000000    0.000000
N      2.020000    0.000000    0.000000
N      0.000000    2.020000    0.000000
N     -2.020000    0.000000    0.000000
N      0.000000   -2.020000    0.000000
C      2.870000    1.100000    0.000000
C      2.870000   -1.100000    0.000000
C     -2.870000    1.100000    0.000000
C     -2.870000   -1.100000    0.000000
C      1.100000    2.870000    0.000000
C     -1.100000    2.870000    0.000000
C      1.100000   -2.870000    0.000000
C     -1.100000   -2.870000    0.000000
C      4.200000    0.680000    0.000000
C      4.200000   -0.680000    0.000000
C     -4.200000    0.680000    0.000000
C     -4.200000   -0.680000    0.000000
C      0.680000    4.200000    0.000000
C     -0.680000    4.200000    0.000000
C      0.680000   -4.200000    0.000000
C     -0.680000   -4.200000    0.000000
H      5.080000    1.310000    0.000000
H      5.080000   -1.310000    0.000000
H     -5.080000    1.310000    0.000000
H     -5.080000   -1.310000    0.000000
H      1.310000    5.080000    0.000000
H     -1.310000    5.080000    0.000000
H      1.310000   -5.080000    0.000000
H     -1.310000   -5.080000    0.000000

