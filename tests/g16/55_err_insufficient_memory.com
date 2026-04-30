%chk=55_err_insufficient_memory.chk
%mem=100MB
%nproc=16
# CCSD(T)/cc-pVTZ

ERROR EXAMPLE: Insufficient memory allocation
CCSD(T)/cc-pVTZ single point on naphthalene requires many GB of RAM
%mem=100MB is far too small for 398 basis functions with coupled-cluster
Gaussian error: "galloc: could not allocate memory" or "Out-of-memory error in routine"
Fix: Increase %mem to at least 16GB (or more); check available RAM on compute node

0 1
C      1.245500    0.717100    0.000000
C      1.245500   -0.717100    0.000000
C      0.000000   -1.403500    0.000000
C     -1.245500   -0.717100    0.000000
C     -1.245500    0.717100    0.000000
C      0.000000    1.403500    0.000000
C      2.437400    1.398300    0.000000
C      2.437400   -1.398300    0.000000
C     -2.437400   -1.398300    0.000000
C     -2.437400    1.398300    0.000000
H      0.000000   -2.490700    0.000000
H      0.000000    2.490700    0.000000
H      2.437700    2.485400    0.000000
H      2.437700   -2.485400    0.000000
H     -2.437700   -2.485400    0.000000
H     -2.437700    2.485400    0.000000
H      3.381100    0.858700    0.000000
H      3.381100   -0.858700    0.000000

