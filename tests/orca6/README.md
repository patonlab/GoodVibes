# ORCA 6 Example Input Files

## Overview

60 example `.inp` input files covering a wide range of ORCA 6 job types and features.
Files 01–43 are general calculations; files 44–50 are transition state examples;
files 51–60 are deliberate error examples illustrating common ORCA failures.

---

## File Index

| # | File | Method | Job Type | Key Feature |
|---|------|--------|----------|-------------|
| 01a | `01a_water_hf_freq.inp` | HF/6-31G(d) | Freq | Symmetric molecule (C2v), default thermochemistry |
| 01b | `01b_water_hf_freq_scaled.inp` | HF/6-31G(d) | Freq | Scaled frequencies (SCALFREQ 1.035) |
| 01c | `01c_water_hf_freq_harmonic.inp` | HF/6-31G(d) | Freq | Pure harmonic RRHO (QuasiRRHO false) |
| 01d | `01d_water_hf_freq_qhcutoff.inp` | HF/6-31G(d) | Freq | Quasi-RRHO with CutOffFreq 200 cm-1 |
| 02 | `02_ethane_opt_freq_thermo.inp` | B3LYP/6-311+G(d,p) | Opt+Freq | Multi-temperature thermochemistry (77, 298, 330, 450 K) |
| 03 | `03_acetone_linked_opt_freq.inp` | B3LYP/6-311G(d,p) | Compound: Opt→Freq | $new_job, xyzfile |
| 04 | `04_benzene_radical_cation.inp` | TPSS/6-311+G(d,p) | Opt+Freq | Charge=+1, Multiplicity=2, UKS |
| 05 | `05_methylene_triplet_carbene.inp` | UMP2/cc-pVDZ | Opt+Freq | Charge=0, Multiplicity=3 |
| 06 | `06_carbon_atom_single_point.inp` | ROHF/cc-pVQZ | SP | Single atom, no freq |
| 07 | `07_neon_atom_with_freq.inp` | HF/aug-cc-pVTZ | Freq | Single atom with Freq (0 modes) |
| 08 | `08_alanine_C1_pcm_water.inp` | M062X/def2-SVP | Opt+Freq | Non-symmetric C1, CPCM water |
| 09 | `09_caffeine_nmr_giao.inp` | mPW1PW/6-311+G(2d,p) | NMR | GIAO chemical shifts (mPW1PW = ORCA name for mPW1PW91) |
| 10 | `10_formaldehyde_verbose_pop.inp` | wB97X-D3/6-31G(d,p) | Opt+Freq | Print orbitals + Mulliken (NPA N/A) |
| 11 | `11_hf_molecule_dlpno_ccsdt_gold_standard.inp` | DLPNO-CCSD(T)/cc-pVTZ | SP | Gold standard energy (local approximation) |
| 12 | `12_water_anharmonic_vpt2.inp` | RHF/def2-SVP | VPT2 | VPT2 with custom %vpt2 block |
| 13 | `13_formaldehyde_tddft_s1.inp` | BP86/def2-SVP | TD-DFT SP | Excited state (S1), %tddft NRoots 3 |
| 14 | `14_water_dimer_gcp.inp` | B3LYP/D3BJ/def2-SV(P) | Opt+Freq | GCP(DFT/SV(P)) geometrical counterpoise |
| 15 | `15_methanol_qmqm2_xtb.inp` | wB97X-D3/def2-TZVP | Opt+Freq | QM/QM2 with xTB lower level (ONIOM substitute) |
| 16 | `16_o2_superoxide_anion.inp` | UB3LYP/aug-cc-pVTZ | Opt+Freq | Charge=-1, Multiplicity=2, diffuse |
| 17 | `17_iron_complex_quintet.inp` | PBE0/def2-SVP | Opt+Freq | Transition metal, Mult=5, UKS, SlowConv |
| 18 | `18_propane_linked_composite_dh.inp` | B3LYP→RI-B2PLYP | Compound: Opt+Freq→SP | Composite double-hybrid approach, D3BJ |
| 19 | `19_acetic_acid_smd_dmso.inp` | B3LYP/6-311+G(d,p) | Opt+Freq | SMD DMSO, D3BJ |
| 20 | `20_benzene_singlepoint.inp` | B3LYP/6-311+G(d,p) | SP | D3BJ dispersion, D6h symmetric |
| 21 | `21_naphthalene_xtb2_semiempirical.inp` | XTB2 (GFN2-xTB) | Opt+Freq | Semi-empirical substitute (PM7 N/A) |
| 22 | `22_hcn_linear_freq_noraman.inp` | M062X/6-311+G(d,p) | Opt+Freq | Linear molecule |
| 23 | `23_cs2_linear_anharmonic_noraman.inp` | B3LYP/6-311+G(2df,p) | Opt+AnFreq | Linear, anharmonic |
| 24 | `24_iodobenzene_genecp_sdd.inp` | B3LYP/D3BJ | Opt+Freq | %basis NewGTO/NewECP for I (SDD) |
| 26 | `26_pt_complex_genecp_3zone.inp` | B3LYP | Opt+Freq | %basis 3-zone: def2-TZVP/def2-ECP |
| 28 | `28_pyridine_smd_acetonitrile_wb97xd3.inp` | wB97X-D3/6-311+G(d,p) | Opt+Freq | SMD Acetonitrile |
| 29 | `29_aniline_cpcm_chloroform.inp` | M062X/6-311+G(d,p) | Opt+Freq | CPCM Chloroform |
| 30 | `30_phenol_smd_thf_pbe0_d3bj.inp` | PBE0/6-311+G(d,p) | Opt+Freq | SMD THF, D3BJ |
| 31 | `31_methylammonium_cpcm_water.inp` | M06/6-311+G(d,p) | Opt+Freq | CPCM water, cation |
| 32 | `32_cyclohexane_tpss_meta_gga.inp` | TPSS/def2-TZVP | Opt+Freq | Meta-GGA, D3BJ |
| 33 | `33_methanol_pbe_gga.inp` | PBE/6-311G(d,p) | Opt+Freq | Pure GGA (PBE) |
| 34 | `34_butadiene_camb3lyp_rsh.inp` | CAM-B3LYP/6-311+G(d,p) | Opt+Freq | Range-separated hybrid, D3BJ |
| 35 | `35_furan_wb97xv_functional.inp` | wB97X-V/def2-TZVP | Opt+NumFreq | MN15 substitute (N/A in ORCA) |
| 36 | `36_imidazole_pbe0d3bj_noraman.inp` | PBE0-D3BJ/6-311+G(d,p) | Opt+Freq | APFD substitute (N/A in ORCA) |
| 37 | `37_planar_cyclohexane_2nd_order_saddle.inp` | B3LYP/6-31G(d) | Opt+Freq | **2nd order saddle point** |
| 38 | `38_naphthalene_scsmp2.inp` | RI-SCS-MP2/cc-pVDZ | NumFreq | Native SCS-MP2 (no IOP needed) |
| 39 | `39_oxazole_tpssh_cpcm_dcm.inp` | TPSSh/6-311+G(d,p) | Opt+Freq | CPCM DCM |
| 40 | `40_n2o_linear_highT.inp` | HF/cc-pVTZ (RIJCOSX) | SP | Linear, %freq temp=1000 K (no Freq keyword) |
| 41 | `41_thiophene_nmr_giao.inp` | mPW1PW/6-311+G(2d,p) | NMR | GIAO NMR chemical shifts |
| 42 | `42_dmso_linked_cpcm_gasfreq.inp` | B3LYP/6-311+G(d,p) | Compound: CPCM Opt→Freq | Thermodynamic cycle |
| 43 | `43_dmabn_bhandhlyp_chargetransfer.inp` | BHandHLYP/6-311+G(d,p) | Opt+Freq | Charge transfer |
| 44 | `44_ts_sn2_identity_chloride.inp` | B3LYP/6-311+G(d,p) | OptTS+Freq | Classic SN2 TS, C3v, collinear |
| 45 | `45_ts_diels_alder_butadiene_ethylene.inp` | M062X/6-31G(d) | OptTS+Freq | Concerted [4+2] TS, C2v |
| 46 | `46_ts_neb_cope_rearrangement.inp` | PBEh-3c | NEB-TS+Freq | Fast-NEB-TS, Cope [3,3] rearrangement |
| 47 | `47_ts_e2_elimination_ethylchloride.inp` | B3LYP/6-311+G(d,p) | OptTS+Freq | E2 anti-periplanar TS, anion |
| 48 | `48_ts_nh3_umbrella_inversion.inp` | wB97M-V/6-311+G(d,p) | OptTS+NumFreq | Low barrier, D3h planar TS, NumHess |
| 49 | `49_ts_oh_abstraction_methane.inp` | wB97X-D3/6-311+G(d,p) | OptTS+Freq | Radical H-abstraction TS, doublet |
| 50 | `50_ts_scants_oh_ch4_abstraction.inp` | B3LYP/SV(P) | ScanTS | Relaxed scan + auto TS optimization, doublet |
| 51 | `51_err_scf_convergence_fe_complex.inp` | UB3LYP/6-31G(d) | Opt+Freq | **Error:** SCF convergence failure |
| 52 | `52_err_opt_not_converged_maxcycles.inp` | B3LYP/6-31G(d) | Opt+Freq | **Error:** Opt convergence (MaxIter=2) |
| 53 | `53_err_wrong_charge_multiplicity.inp` | B3LYP/6-31G(d) | Opt+Freq | **Error:** Impossible charge/multiplicity |
| 54 | `54_err_missing_basis_heavy_atom.inp` | B3LYP/6-311+G(d,p) | Opt+Freq | **Error:** Basis set missing for Pb |
| 55 | `55_err_insufficient_memory.inp` | CCSD(T)/cc-pVTZ | SP | **Error:** Insufficient memory (%maxcore=6) |
| 56 | `56_err_timed_out.inp` | B3LYP/6-31G(d) | Opt+Freq | **Error:** Job timed out (no termination) |
| 57 | `57_err_syntax_route_typo.inp` | B3LPY/6-31G(d) | Opt+Feq | **Error:** Typos on simple input line |
| 58 | `58_err_linear_bend_formBX.inp` | HF/6-31G(d) | Opt+Freq | **Error:** Internal coordinate failure |
| 59 | `59_err_basis_linear_dependency.inp` | HF/aug-cc-pV5Z | Opt+Freq | **Error:** Basis set linear dependency |
| 60 | `60_err_missing_end_blank_line.inp` | B3LYP/6-31G(d) | Opt+Freq | **Error:** Missing closing `*` |