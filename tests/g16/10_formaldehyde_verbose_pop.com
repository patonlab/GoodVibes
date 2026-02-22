%chk=formaldehyde_verbose.chk
%mem=8GB
%nprocshared=4
#p wB97XD/6-31G**  Opt Freq Pop=(Full,NPA,MK) 
IOP(6/7=3)

Formaldehyde with verbose printing (#p), NPA charges, MK ESP charges, WFN output
IOP(6/7=3) requests extra SCF convergence printing; Pop=Full prints all MOs

0 1
C   0.000000   0.000000  -0.523841
O   0.000000   0.000000   0.676159
H   0.000000   0.939547  -1.109428
H   0.000000  -0.939547  -1.109428


