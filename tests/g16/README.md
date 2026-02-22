# Gaussian 16 Example Input Files

## Overview

Example `.com` input files covering a wide range of Gaussian 16 job types and features.
Files 01–43 are general calculations; files 44–50 are transition state examples;
files 51–60 are deliberate error examples illustrating common Gaussian failures;
file 61 is a blank log file for edge-case testing.

---

## File Index

| # | File | Method | Job Type | Key Feature |
|---|------|--------|----------|-------------|
| 01a | `01a_water_hf_freq.com` | HF/6-31G(d) | Freq | Symmetric molecule (C2v) |
| 01b | `01b_water_hf_freq_scaled.com` | HF/6-31G(d) | Freq=(Scale=0.95) | Frequency scaling |
| 01c | `01c_water_hf_freq_isotopes.com` | HF/6-31G(d) | Freq=(Scale=0.95) | Deuterium isotope substitution |
| 02 | `02_ethane_opt_freq_T398_P2.com` | B3LYP/6-311+G(d,p) | Opt+Freq | Temperature=398.15, Pressure=2.0 |
| 03 | `03_acetone_linked_opt_freq.com` | B3LYP/6-311G(d,p) | Linked: Opt→Freq | --Link1--, Geom=AllCheck |
| 04 | `04_benzene_radical_cation.com` | TPSSTPSS/6-311+G(d,p) | Opt+Freq | Charge=+1, Multiplicity=2 |
| 05 | `05_methylene_triplet_carbene.com` | UMP2/cc-pVDZ | Opt+Freq | Charge=0, Multiplicity=3 |
| 06 | `06_carbon_atom_single_point.com` | ROHF/cc-pVQZ | SP | Single atom, no freq |
| 07 | `07_neon_atom_with_freq.com` | HF/aug-cc-pVTZ | Freq | Single atom with Freq (0 modes) |
| 08 | `08_alanine_C1_pcm_water.com` | M062X/def2SVP | Opt+Freq | Non-symmetric C1, CPCM water |
| 09 | `09_caffeine_nmr_giao.com` | mPW1PW91/6-311+G(2d,p) | NMR | GIAO chemical shifts |
| 10 | `10_formaldehyde_verbose_pop.com` | wB97XD/6-31G** | Opt+Freq | Verbose #p, IOP, NPA/MK charges |
| 11 | `11_hf_molecule_ccsdt_gold_standard.com` | CCSD(T)/cc-pVTZ | SP | Gold standard energy |
| 12 | `12_water_anharmonic_vpt2.com` | B3LYP/6-31G(d) | Freq=Anharmonic | VPT2, overtones, Fermi resonance |
| 13 | `13_formaldehyde_tddft_s1.com` | B97D/def2SVP | TD-DFT Opt+Freq | Excited state (S1), TD=(NStates=5,Root=1) |
| 14 | `14_water_dimer_counterpoise_bsse.com` | B3LYP/6-311+G(d,p) | SP | Counterpoise=2, BSSE, emp=GD3 |
| 15 | `15_methanol_oniom_qmmm.com` | ONIOM(B3LYP/6-31G(d):PM6) | Opt+Freq | 2-layer QM/MM |
| 16 | `16_o2_superoxide_anion.com` | UB3LYP/aug-cc-pVTZ | Opt+Freq | Charge=-1, Multiplicity=2, diffuse |
| 17 | `17_iron_complex_quintet.com` | PBE1PBE/def2SVP | Opt+Freq | Transition metal, Mult=5, SCF(xqc), Guess=Mix |
| 18 | `18_propane_linked_composite_dh.com` | B3LYP→B2PLYP | Linked: Opt+Freq→SP | Composite double-hybrid approach |
| 19 | `19_acetic_acid_smd_dmso.com` | B3LYP/6-311+G(d,p) | Opt+Freq | SMD DMSO, GD3BJ |
| 20 | `20_benzene_singlepoint.com` | B3LYP/6-311+G(d,p) | SP | GD3BJ dispersion, D6h symmetric |
| 21 | `21_naphthalene_pm7_semiempirical.com` | PM7 | Opt+Freq | Semi-empirical, D2h symmetric |
| 22 | `22_hcn_linear_freq_noraman.com` | M062X/6-311+G(d,p) | Opt+Freq | Linear molecule, nosymm, NoRaman |
| 23 | `23_cs2_linear_anharmonic_noraman.com` | B3LYP/6-311+G(2df,p) | Freq=Anharmonic | Linear, anharmonic, NoRaman |
| 24 | `24_iodobenzene_genecp_sdd.com` | B3LYP/GenECP | Opt+Freq | GenECP with SDD, GD3BJ |
| 25 | `25_pd_complex_genecp_def2.com` | PBE1PBE/GenECP | Opt+Freq | Pd complex, def2 basis, GD3BJ |
| 26 | `26_pt_complex_genecp_3zone.com` | B3LYP/GenECP | Opt+Freq | 3-zone GenECP, NoRaman |
| 27 | `27_custom_functional_iop_b20lyp.com` | BLYP/6-311+G(d,p)+IOP | Opt+Freq | Custom functional via IOP |
| 28 | `28_pyridine_smd_acetonitrile_wb97xd.com` | wB97XD/6-311+G(d,p) | Opt+Freq | SMD Acetonitrile |
| 29 | `29_aniline_cpcm_chloroform.com` | M062X/6-311+G(d,p) | Opt+Freq | CPCM Chloroform |
| 30 | `30_phenol_smd_thf_pbe0_d3bj.com` | PBE1PBE/6-311+G(d,p) | Opt+Freq | SMD THF, GD3BJ |
| 31 | `31_methylammonium_cpcm_water.com` | M06/6-311+G(d,p) | Opt+Freq | CPCM water, cation |
| 32 | `32_cyclohexane_tpss_meta_gga.com` | TPSSTPSS/def2TZVP | Opt+Freq | Meta-GGA, GD3BJ |
| 33 | `33_methanol_pbepbe_gga.com` | PBEPBE/6-311G(d,p) | Opt+Freq | Pure GGA (PBE), NoRaman |
| 34 | `34_butadiene_camb3lyp_rsh.com` | CAM-B3LYP/6-311+G(d,p) | Opt+Freq | Range-separated hybrid, GD3BJ |
| 35 | `35_furan_mn15_functional.com` | MN15/def2TZVP | Opt+Freq | Minnesota functional |
| 36 | `36_imidazole_apfd_noraman.com` | APFD/6-311+G(d,p) | Opt+Freq | APFD functional, NoRaman |
| 37 | `37_planar_cyclohexane_2nd_order_saddle.com` | B3LYP/6-31G* | Opt+Freq | **2nd order saddle point** (see below) |
| 38 | `38_naphthalene_scsmp2.com` | MP2/cc-pVTZ+IOP | Freq | SCS-MP2 via IOP, NoRaman |
| 39 | `39_oxazole_tpssh_cpcm_dcm.com` | TPSSh/6-311+G(d,p) | Opt+Freq | CPCM DCM |
| 40 | `40_n2o_linear_highT_highP.com` | CCSD/cc-pVTZ | Opt+Freq | Linear, T=1000 K, P=100 atm |
| 41 | `41_thiophene_freq_noraman_nmr.com` | mPW1PW91/6-311+G(2d,p) | Opt+Freq+NMR | Freq NoRaman + GIAO NMR |
| 42 | `42_dmso_linked_cpcm_gasfreq.com` | B3LYP/6-311+G(d,p) | Linked: CPCM Opt→Freq | Thermodynamic cycle, NoRaman |
| 43 | `43_dmabn_bhandhlyp_chargetransfer.com` | BHandHLYP/6-311+G(d,p) | Opt+Freq | Charge transfer, NoRaman |
| 44 | `44_ts_sn2_identity_chloride.com` | B3LYP/6-311+G(d,p) | Opt(TS)+Freq | Classic SN2 TS, C3v, collinear |
| 45 | `45_ts_diels_alder_butadiene_ethylene.com` | M062X/6-31G(d) | Opt(TS)+Freq | Concerted [4+2] TS, C2v |
| 46 | `46_ts_h3_hydrogen_abstraction.com` | MP2/aug-cc-pVTZ | Opt(TS)+Freq | Simplest possible TS, linear, doublet |
| 47 | `47_ts_e2_elimination_ethylchloride.com` | B3LYP/6-311+G(d,p) | Opt(TS)+Freq | E2 anti-periplanar TS, anion |
| 48 | `48_ts_nh3_umbrella_inversion.com` | MP2/6-311+G(d,p) | Opt(TS)+Freq | Low barrier, D3h planar TS |
| 49 | `49_ts_oh_abstraction_methane.com` | wB97XD/6-311+G(d,p) | Opt(TS)+Freq | Radical H-abstraction TS, doublet |
| 50 | `50_ts_cyclopropane_ring_opening.com` | CASSCF(2,2)/6-31G(d) | Opt(TS)+Freq | Bond breaking TS, multireference |
| 51 | `51_err_scf_convergence_fe_complex.com` | UB3LYP/6-31G(d) | Opt+Freq | **Error:** SCF convergence failure |
| 52 | `52_err_opt_not_converged_maxcycles.com` | B3LYP/6-31G(d) | Opt+Freq | **Error:** Opt convergence (MaxCycles=2) |
| 53 | `53_err_wrong_charge_multiplicity.com` | B3LYP/6-31G(d) | Opt+Freq | **Error:** Impossible charge/multiplicity |
| 54 | `54_err_missing_basis_heavy_atom.com` | B3LYP/6-311+G(d,p) | Opt+Freq | **Error:** Basis set missing for Pb |
| 55 | `55_err_insufficient_memory.com` | CCSD(T)/cc-pVTZ | SP | **Error:** Insufficient memory (%mem=100MB) |
| 56 | `56_err_timed_out.com` | B3LYP/6-31G(d) | Opt+Freq | **Error:** Job timed out (no termination) |
| 57 | `57_err_syntax_route_typo.com` | B3LPY/6-31G(d) | Opt+Feq | **Error:** Typos on route line |
| 58 | `58_err_linear_bend_formBX.com` | HF/6-31G(d) | Opt+Freq | **Error:** FormBX linear bend failure |
| 59 | `59_err_basis_linear_dependency.com` | HF/Aug-cc-pV5Z | Opt+Freq | **Error:** Basis set linear dependency |
| 60 | `60_err_missing_end_blank_line.com` | B3LYP/6-31G(d) | Opt+Freq | **Error:** Missing trailing blank line |
| 61 | `61_empty.log` | — | — | Blank log file (edge case) |
