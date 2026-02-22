%chk=hcn_noraman.chk
%mem=96GB
%nproc=16
# nosymm M062X/6-311+G(d,p) Opt Freq=NoRaman

HCN linear triatomic (C*v symmetry) - freq without Raman intensities (faster)
NoRaman skips calculation of Raman activities; useful when only IR is needed

0 1
H   0.000000   0.000000  -1.064700
C   0.000000   0.000000   0.000000
N   0.000000   0.000000   1.155600

