%chk=propane_composite.chk
%mem=8GB
%nprocshared=8
# B3LYP/6-31G(d) Opt Freq

Propane: DFT geometry optimization (composite approach step 1)

0 1
 C                  0.00000000    0.00000000    0.00000000
 C                  1.54000000    0.00000000    0.00000000
 C                 -0.77000000    1.33398700    0.00000000
 H                  1.90300000    1.02671900    0.00000000
 H                  1.90300000   -0.51336000    0.88916500
 H                  1.90300000   -0.51336000   -0.88916500
 H                 -1.38515334    1.39072946    0.87365134
 H                 -1.38515334    1.39072946   -0.87365134
 H                 -0.31621226   -0.54763251   -0.86313871
 H                 -0.31621226   -0.54763251    0.86313871
 H                 -0.07460071    2.14720273    0.00000000

--Link1--
%chk=propane_composite.chk
%mem=96GB
%nproc=16
# B2PLYP/6-311+G(d,p) Geom=AllCheck Guess=Read

Propane: double hybrid energy on DFT geometry (composite approach step 2)


