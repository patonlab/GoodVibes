%chk=58_err_linear_bend.chk
%mem=96GB
%nproc=16
# HF/6-31G(d) Opt Freq

ERROR EXAMPLE: Internal coordinate failure (FormBX / linear bend problem)
Allene (H2C=C=CH2) with near-linear CCC arrangement and slight asymmetric distortion
During optimization the CCC angle approaches 180 degrees, causing redundant internal
coordinates to become singular (linear bend undefined in standard Z-matrix internals)
Gaussian error: "FormBX had a problem" or "Error in internal coordinate system" / L103
Fix: Add Opt=Cartesian to use Cartesian coordinates, or add nosymm to route line

0 1
C      0.000000    0.000000    0.000000
C      1.310000    0.005000    0.000000
C      2.620000   -0.003000    0.000000
H     -0.540000    0.930000    0.000000
H     -0.540000   -0.930000    0.000000
H      3.160000    0.000000    0.930000
H      3.160000    0.000000   -0.930000

