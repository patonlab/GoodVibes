%chk=37_planar_cyclohexane_sosp.chk
%mem=96GB
%nproc=16
# symm=loose opt b3lyp/6-31G* freq

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
C                    -2.68575   0.01827   0. 
 C                    -1.29059   0.01827   0. 
 C                    -0.59305   1.22602   0. 
 C                    -1.29071   2.43453  -0.0012 
 C                    -2.68553   2.43445  -0.00168 
 C                    -3.38313   1.22625  -0.00068 
 H                    -3.00194  -0.52987  -0.86283 
 H                    -0.97444  -0.52942   0.86313 
 H                     0.03858   1.22639   0.86368 
 H                    -0.97482   2.98286   0.86161 
 H                    -3.00203   2.98299   0.86078 
 H                    -4.0157    1.22566  -0.86367 
 H                    -4.01531   1.22686   0.86259 
 H                    -3.00143   2.98122  -0.86548 
 H                    -3.00194  -0.52889   0.86344 
 H                    -0.97444  -0.52942  -0.86313 
 H                     0.04006   1.22553  -0.8626 
 H                    -0.97423   2.98149  -0.86466 

