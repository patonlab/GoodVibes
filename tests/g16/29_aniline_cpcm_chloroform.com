%chk=aniline_cpcm_chcl3.chk
%mem=8GB
%nprocshared=8
# M062X/6-311+G(d,p) Opt Freq SCRF=(CPCM,Solvent=Chloroform)

Aniline optimization+freq with CPCM solvation model in chloroform
CPCM (conductor-like PCM) is faster than IEFPCM, good for geometry optimizations

0 1
N   0.000000   0.000000   2.420000
C   1.208000   0.000000   1.785000
C   1.211000   0.000000   0.392000
C   0.000000   0.000000  -0.307000
C  -1.211000   0.000000   0.392000
C  -1.208000   0.000000   1.785000
H   2.147000   0.000000   2.325000
H   2.155000   0.000000  -0.148000
H   0.000000   0.000000  -1.393000
H  -2.155000   0.000000  -0.148000
H  -2.147000   0.000000   2.325000
H   0.876000   0.000000   2.975000
H  -0.876000   0.000000   2.975000

