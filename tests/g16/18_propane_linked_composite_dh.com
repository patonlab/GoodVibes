%chk=propane_composite.chk
%mem=8GB
%nprocshared=8
# B3LYP/6-31G(d) Opt Freq

Propane: DFT geometry optimization (composite approach step 1)

0 1
 C                  0.00000000    0.57561600    0.00000000
 C                 -1.01250451   -0.40324259    0.64863533
 C                  1.20713345   -0.09740048   -0.70310249
 H                 -0.53851009   -1.02270156    1.41807583
 H                 -1.84256310    0.12585091    1.12957782
 H                 -1.45062807   -1.08609472   -0.08757361
 H                  1.79739281   -0.70627476   -0.00938638
 H                  1.88807925    0.63999348   -1.14235074
 H                 -0.52732151    1.20819085   -0.72301425
 H                  0.37188923    1.26502805    0.76640586
 H                  0.88383375   -0.75869721   -1.51478000

--Link1--
%chk=propane_composite.chk
%mem=96GB
%nproc=16
# B2PLYP/6-311+G(d,p) Geom=AllCheck Guess=Read

Propane: double hybrid energy on DFT geometry (composite approach step 2)


