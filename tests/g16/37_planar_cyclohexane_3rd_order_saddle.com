%chk=37_planar_cyclohexane_sosp.chk
%mem=96GB
%nproc=16
# symm=loose opt=(NoEigenTest) b3lyp/6-31G* freq

TS11: Planar cyclohexane - SECOND ORDER SADDLE POINT (2 imaginary frequencies)
This is NOT a true TS but a higher-order critical point on the PES
Planar D6h cyclohexane lies above both chair and boat conformations
The two imaginary frequencies correspond to out-of-plane ring distortions leading
toward chair (lower barrier pathway) and boat conformations respectively
NoEigenTest is essential: without it, Gaussian will reject the structure
as a TS because it has more than 1 negative Hessian eigenvalue
Expect: 2 imaginary frequencies (~100-200i cm-1), both out-of-plane ring modes
Real chair minimum: ~-26 kcal/mol relative to planar
Real boat TS: ~-23 kcal/mol relative to planar (boat is a TRUE TS with 1 imag freq)

0 1
C                    -2.68575   0.01827   0.00000
C                    -1.29059   0.01827   0.00000
C                    -0.59305   1.22602   0.00000
C                    -1.29071   2.43453   0.00000
C                    -2.68553   2.43445   0.00000
C                    -3.38313   1.22625   0.00000
H                    -3.00194  -0.52938  -0.86283
H                    -3.00194  -0.52938   0.86283
H                    -0.97444  -0.52942   0.86313
H                    -0.97444  -0.52942  -0.86313
H                     0.03932   1.22596   0.86313
H                     0.03932   1.22596  -0.86313
H                    -0.97453   2.98218   0.86313
H                    -0.97453   2.98218  -0.86313
H                    -3.00199   2.98211   0.86283
H                    -3.00199   2.98211  -0.86283
H                    -4.01551   1.22626   0.86313
H                    -4.01551   1.22626  -0.86313 

