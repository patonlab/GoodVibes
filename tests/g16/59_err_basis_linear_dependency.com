%chk=59_err_basis_lindep.chk
%mem=96GB
%nproc=16
# HF/Aug-cc-pV5Z Opt Freq

ERROR EXAMPLE: Basis set linear dependency
Lithium hydride with heavily augmented quintuple-zeta basis set
The diffuse functions on Li and H overlap so strongly that the overlap matrix
becomes near-singular, causing numerical instability in the SCF procedure
Gaussian error: "Basis set is linearly dependent" / overlap eigenvalue below threshold
Fix: Use a smaller basis (cc-pVTZ, aug-cc-pVTZ), or add IOp(3/32=2) to drop
near-dependent functions, or increase linear dependency threshold

0 1
Li     0.000000    0.000000    0.000000
H      1.595000    0.000000    0.000000

