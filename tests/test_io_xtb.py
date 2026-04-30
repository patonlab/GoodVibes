#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for parsing xtb output files using goodvibes.io.

Ground-truth values are extracted from the xtb output blocks themselves:
``:: total energy`` for the electronic SCF, ``:: zero point energy`` for
the ZPE, and the ``# frequencies`` / ``# imaginary freq.`` SETUP entries
for vibrational counts.
"""

import pytest
from goodvibes.io import (parse_data, level_of_theory, read_initial,
                          parse_xtb_thermo, parse_qcdata)
from conftest import (xtb_path, XTB_FREQ_FILES, XTB_SP_ONLY_FILES,
                      XTB_SOLVATION_FILES, XTB_LINEAR_FILES,
                      XTB_GFN1_FILES, XTB_SADDLE_FILES, XTB_ERROR_FILES)


# ---------------------------------------------------------------------------
# Program detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename",
                         XTB_FREQ_FILES + XTB_SP_ONLY_FILES + XTB_ERROR_FILES)
def test_parse_qcdata_xtb_program(filename):
    qcdata = parse_qcdata(xtb_path(filename))
    assert qcdata.program == 'xtb'


@pytest.mark.parametrize("filename",
                         XTB_FREQ_FILES + XTB_SP_ONLY_FILES + XTB_ERROR_FILES)
def test_parse_xtb_thermo_version(filename):
    qcdata = parse_xtb_thermo(xtb_path(filename))
    assert qcdata.version_program.startswith('xtb version')


# ---------------------------------------------------------------------------
# Electronic SCF energy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_energy", [
    ('01_water.out', -5.0705443747),
    ('02_ethane.out', -7.3363705748),
    ('03_acetone.out', -13.4556615623),
    ('04_benzene_radical_cation.out', -15.3582082534),
    ('05_methylene_triplet_carbene.out', -2.9383026834),
    ('06_carbon_atom_single_point.out', -1.7951105180),
    ('07_neon_atom_with_freq.out', -5.9322150528),
    ('08_alanine.out', -21.0428661781),
    ('09_caffeine.out', -42.1544484990),
    ('10_formaldehyde.out', -7.1756479573),
    ('11_hf_molecule.out', -5.2228739774),
    ('12_water_dimer.out', -10.1415322437),
    ('13_methanol.out', -8.2261174493),
    ('14_o2_superoxide_anion.out', -8.1677069117),
    ('15_iron_complex_quintet.out', -33.8617986242),
    ('16_propane.out', -10.4915976940),
    ('17_acetic_acid_dmso.out', -14.4721096226),
    ('18_benzene_singlepoint.out', -15.8785689333),
    ('19_naphthalene.out', -25.4743860015),
    ('20_hcn_linear.out', -5.5040661864),
    ('21_cs2_linear.out', -8.6108421719),
    ('22_iodobenzene.out', -19.1880360641),
    ('23_pd_complex.out', -24.2404402344),
    ('24_pt_complex.out', -58.7703125534),
    ('25_ethane.out', -7.3363705748),
    ('26_pyridine_acetonitrile.out', -16.1580537714),
    ('27_aniline_chloroform.out', -19.3242635691),
    ('28_phenol_thf.out', -19.9667891693),
    ('29_methylammonium_water.out', -7.8424912291),
    ('30_cyclohexane.out', -18.9867158351),
    ('31_methanol.out', -8.2235632035),
    ('32_butadiene.out', -11.6273694031),
    ('33_furan.out', -14.6450308500),
    ('34_imidazole.out', -14.2976648267),
    ('35_planar_cyclohexane_3rd_order_saddle.out', -18.8377692779),
    ('36_naphthalene.out', -24.0651591103),
    ('37_oxazole_dcm.out', -14.9317020219),
    ('38_n2o_linear.out', -9.8449340473),
    ('39_thiophene.out', -13.8182741204),
    ('40_dmso.out', -14.6211793922),
    ('41_dmabn.out', -30.1711772955),
])
def test_parse_xtb_thermo_scf_energy(filename, expected_energy):
    qcdata = parse_xtb_thermo(xtb_path(filename))
    assert qcdata.scf_energy is not None
    assert abs(qcdata.scf_energy - expected_energy) < 1e-6


# ---------------------------------------------------------------------------
# Zero-point energy (only for files that have a Hessian)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_zpe", [
    ('01_water.out', 0.0201204784),
    ('02_ethane.out', 0.0742253332),
    ('03_acetone.out', 0.0757347239),
    ('04_benzene_radical_cation.out', 0.0843738367),
    ('05_methylene_triplet_carbene.out', 0.0167695906),
    ('07_neon_atom_with_freq.out', 0.0000000000),
    ('08_alanine.out', 0.1043099980),
    ('09_caffeine.out', 0.1823072207),
    ('10_formaldehyde.out', 0.0255497227),
    ('11_hf_molecule.out', 0.0085682856),
    ('12_water_dimer.out', 0.0406617290),
    ('13_methanol.out', 0.0496839865),
    ('14_o2_superoxide_anion.out', 0.0029901203),
    ('15_iron_complex_quintet.out', 0.0295756394),
    ('16_propane.out', 0.1022556493),
    ('19_naphthalene.out', 0.1428435702),
    ('20_hcn_linear.out', 0.0162568536),
    ('21_cs2_linear.out', 0.0067455916),
    ('22_iodobenzene.out', 0.0874793958),
    ('23_pd_complex.out', 0.1483341468),
    ('24_pt_complex.out', 0.4444931199),
    ('25_ethane.out', 0.0742253332),
    ('26_pyridine_acetonitrile.out', 0.0853885570),
    ('27_aniline_chloroform.out', 0.1121881949),
    ('28_phenol_thf.out', 0.1009462785),
    ('30_cyclohexane.out', 0.1684503348),
    ('31_methanol.out', 0.0489457264),
    ('32_butadiene.out', 0.0826806899),
    ('33_furan.out', 0.0682640341),
    ('34_imidazole.out', 0.0691131507),
    ('35_planar_cyclohexane_3rd_order_saddle.out', 0.1830640936),
    ('36_naphthalene.out', 0.1005205161),
    ('37_oxazole_dcm.out', 0.0567164912),
    ('38_n2o_linear.out', 0.0118395918),
    ('39_thiophene.out', 0.0649455297),
    ('40_dmso.out', 0.0773974442),
    ('41_dmabn.out', 0.1670355305),
])
def test_parse_xtb_thermo_zpe(filename, expected_zpe):
    qcdata = parse_xtb_thermo(xtb_path(filename))
    assert qcdata.zero_point_corr is not None
    assert abs(qcdata.zero_point_corr - expected_zpe) < 1e-6


@pytest.mark.parametrize("filename", XTB_SP_ONLY_FILES)
def test_parse_xtb_thermo_sp_only_no_zpe(filename):
    """SP-only runs produce no Hessian, so no ZPE."""
    qcdata = parse_xtb_thermo(xtb_path(filename))
    assert qcdata.zero_point_corr is None
    assert qcdata.frequency_wn == []
    assert qcdata.im_frequency_wn == []


# ---------------------------------------------------------------------------
# Frequency counts (real + imaginary)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, n_real, n_imag", [
    ('01_water.out', 3, 0),
    ('02_ethane.out', 18, 0),
    ('03_acetone.out', 23, 1),
    ('04_benzene_radical_cation.out', 27, 3),
    ('05_methylene_triplet_carbene.out', 3, 0),
    ('07_neon_atom_with_freq.out', 0, 0),  # single atom, --ohess
    ('08_alanine.out', 33, 0),
    ('10_formaldehyde.out', 6, 0),
    ('11_hf_molecule.out', 1, 0),
    ('12_water_dimer.out', 8, 4),
    ('13_methanol.out', 12, 0),
    ('14_o2_superoxide_anion.out', 1, 0),
    ('15_iron_complex_quintet.out', 22, 5),
    ('16_propane.out', 25, 2),
    ('19_naphthalene.out', 48, 0),
    ('20_hcn_linear.out', 4, 0),
    ('21_cs2_linear.out', 4, 0),
    ('22_iodobenzene.out', 30, 0),
    ('25_ethane.out', 18, 0),
    ('30_cyclohexane.out', 48, 0),
    ('31_methanol.out', 11, 1),
    ('32_butadiene.out', 24, 0),
    ('35_planar_cyclohexane_3rd_order_saddle.out', 45, 3),
    ('36_naphthalene.out', 38, 4),
    ('38_n2o_linear.out', 4, 0),
    ('41_dmabn.out', 57, 0),
])
def test_parse_xtb_thermo_frequencies(filename, n_real, n_imag):
    qcdata = parse_xtb_thermo(xtb_path(filename))
    assert len(qcdata.frequency_wn) == n_real
    assert len(qcdata.im_frequency_wn) == n_imag
    # Real freqs are positive, imaginary are negative
    assert all(v > 0 for v in qcdata.frequency_wn)
    assert all(v < 0 for v in qcdata.im_frequency_wn)


# ---------------------------------------------------------------------------
# Charge / multiplicity (special cases)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_charge, expected_mult", [
    ('01_water.out', 0, 1),
    ('04_benzene_radical_cation.out', 1, 2),
    ('05_methylene_triplet_carbene.out', 0, 3),
    ('14_o2_superoxide_anion.out', -1, 2),
    ('15_iron_complex_quintet.out', 0, 5),
])
def test_parse_xtb_thermo_charge_multiplicity(filename, expected_charge,
                                              expected_mult):
    qcdata = parse_xtb_thermo(xtb_path(filename))
    assert qcdata.charge == expected_charge
    assert qcdata.multiplicity == expected_mult


# ---------------------------------------------------------------------------
# Atom counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_natoms", [
    ('01_water.out', 3),
    ('02_ethane.out', 8),
    ('06_carbon_atom_single_point.out', 1),  # SP-only → falls back to .xyz
    ('07_neon_atom_with_freq.out', 1),
    ('09_caffeine.out', 24),
    ('17_acetic_acid_dmso.out', 8),  # SP+solv → .xyz fallback
    ('18_benzene_singlepoint.out', 12),  # SP-only → .xyz fallback
    ('24_pt_complex.out', 53),
    ('29_methylammonium_water.out', 8),  # SP+solv → .xyz fallback
    ('35_planar_cyclohexane_3rd_order_saddle.out', 18),  # --hess only → .xyz fallback
])
def test_parse_xtb_thermo_atoms(filename, expected_natoms):
    qcdata = parse_xtb_thermo(xtb_path(filename))
    assert len(qcdata.atom_nums) == expected_natoms
    assert len(qcdata.atom_types) == expected_natoms
    assert len(qcdata.cartesians) == expected_natoms


# ---------------------------------------------------------------------------
# Linear molecules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", XTB_LINEAR_FILES)
def test_parse_xtb_thermo_linear(filename):
    qcdata = parse_xtb_thermo(xtb_path(filename))
    assert qcdata.linear_mol is True
    # Linear filter keeps only the physical (degenerate) rotational temperatures
    assert len(qcdata.rotemp) == 2
    assert all(0 < t < 1e6 for t in qcdata.rotemp)


@pytest.mark.parametrize("filename", [
    '01_water.out', '03_acetone.out', '08_alanine.out', '30_cyclohexane.out',
])
def test_parse_xtb_thermo_nonlinear(filename):
    qcdata = parse_xtb_thermo(xtb_path(filename))
    assert qcdata.linear_mol is False
    assert len(qcdata.rotemp) == 3


# ---------------------------------------------------------------------------
# Solvation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_solv", [
    ('17_acetic_acid_dmso.out', 'CPCM-X,dimethylsulfoxide'),
    ('26_pyridine_acetonitrile.out', 'GBSA,acetonitrile'),
    ('27_aniline_chloroform.out', 'GBSA,chloroform'),
    ('28_phenol_thf.out', 'GBSA,thf'),
    ('29_methylammonium_water.out', 'GBSA,water'),
    ('37_oxazole_dcm.out', 'GBSA,CH2Cl2'),
])
def test_parse_xtb_thermo_solvation(filename, expected_solv):
    qcdata = parse_xtb_thermo(xtb_path(filename))
    assert qcdata.solvation_model == expected_solv


@pytest.mark.parametrize("filename", [
    '01_water.out', '02_ethane.out', '03_acetone.out', '20_hcn_linear.out',
])
def test_parse_xtb_thermo_gas_phase(filename):
    qcdata = parse_xtb_thermo(xtb_path(filename))
    assert qcdata.solvation_model == 'gas phase'


# ---------------------------------------------------------------------------
# Job type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_jobtype", [
    ('01_water.out', 'GSFreq'),         # --ohess
    ('06_carbon_atom_single_point.out', 'SP'),  # no flag
    ('17_acetic_acid_dmso.out', 'SP'),  # --cpcmx only
    ('18_benzene_singlepoint.out', 'SP'),
    ('29_methylammonium_water.out', 'SP'),  # -g water only, no opt/hess
    ('35_planar_cyclohexane_3rd_order_saddle.out', 'Freq'),  # --hess only
])
def test_parse_xtb_thermo_job_type(filename, expected_jobtype):
    qcdata = parse_xtb_thermo(xtb_path(filename))
    assert qcdata.job_type == expected_jobtype


# ---------------------------------------------------------------------------
# parse_data legacy helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", XTB_FREQ_FILES[:5] + XTB_SP_ONLY_FILES)
def test_parse_data_xtb_program(filename):
    spe, program, *_ = parse_data(xtb_path(filename))
    assert program == 'xtb'


@pytest.mark.parametrize("filename, expected_charge, expected_mult", [
    ('04_benzene_radical_cation.out', 1, 2),
    ('14_o2_superoxide_anion.out', -1, 2),
    ('15_iron_complex_quintet.out', 0, 5),
])
def test_parse_data_xtb_charge_mult(filename, expected_charge, expected_mult):
    (_, _, _, _, _, charge, _, multiplicity) = parse_data(xtb_path(filename))
    assert charge == expected_charge
    assert multiplicity == expected_mult


# ---------------------------------------------------------------------------
# level_of_theory: GFN1 vs GFN2 detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected_substr", [
    ('01_water.out', 'GFN2-xTB'),
    ('30_cyclohexane.out', 'GFN2-xTB'),
    ('32_butadiene.out', 'GFN1-xTB'),
])
def test_level_of_theory_xtb(filename, expected_substr):
    lot = level_of_theory(xtb_path(filename))
    assert expected_substr in lot


# ---------------------------------------------------------------------------
# read_initial: progress / termination + solvation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", XTB_FREQ_FILES[:5])
def test_read_initial_xtb_normal_termination(filename):
    _, _, progress, _, _ = read_initial(xtb_path(filename))
    assert progress == 'Normal'


def test_read_initial_xtb_error_empty():
    """The empty-input file aborts with a fatal error."""
    _, _, progress, _, _ = read_initial(xtb_path('42_empty.out'))
    assert progress == 'Error'


# ---------------------------------------------------------------------------
# Empty / aborted file
# ---------------------------------------------------------------------------

def test_parse_xtb_thermo_empty_file():
    qcdata = parse_xtb_thermo(xtb_path('42_empty.out'))
    assert qcdata.program == 'xtb'
    assert qcdata.scf_energy is None
    assert qcdata.zero_point_corr is None
    assert qcdata.frequency_wn == []
    assert qcdata.atom_nums == []


# ---------------------------------------------------------------------------
# CPU time
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", ['01_water.out', '09_caffeine.out',
                                      '24_pt_complex.out'])
def test_parse_xtb_thermo_cpu(filename):
    qcdata = parse_xtb_thermo(xtb_path(filename))
    # Some non-zero element should be set once the parser hits the "total:" block
    assert any(qcdata.cpu)
