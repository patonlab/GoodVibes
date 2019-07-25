#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pytest
import math
from goodvibes import GoodVibes as GV
from conftest import datapath
from goodvibes.media import solvents

@pytest.mark.parametrize("path, QS, temp, E, ZPE, H, TS, TqhS, G, qhG", [
    # Grimme, 298.15K
    ('Al_298K.out', 'grimme',  298.15, -242.328708, 0.000000, -242.326347, 0.017670, 0.017670, -242.344018, -242.344018),
    ('Al_400K.out', 'grimme',  298.15, -242.328708, 0.000000, -242.326347, 0.017670, 0.017670, -242.344018, -242.344018),
    ('allene.out', 'grimme', 298.15, -116.569605, 0.053913, -116.510916, 0.027618, 0.027621, -116.538534, -116.538537),
    ('CuCN.out', 'grimme', 298.15, -289.005463, 0.006594, -288.994307, 0.025953, 0.025956, -289.020260, -289.020264),
    ('ethane.out', 'grimme', 298.15, -79.830421, 0.075238, -79.750770, 0.027523, 0.027525, -79.778293, -79.778295),
    ('ethane_spc.out', 'grimme', 298.15, -79.830421, 0.075238, -79.750770, 0.027523, 0.027525, -79.778293, -79.778295),
    ('ethane_TZ.out', 'grimme', 298.15, -79.858399, None, None, None, None, None, None),
    ('H2O.out', 'grimme', 298.15, -76.368128, 0.020772, -76.343577, 0.021458, 0.021458, -76.365035, -76.365035),
    ('HCN_singlet.out', 'grimme', 298.15, -93.358851, 0.015978, -93.339373, 0.022896, 0.022896, -93.362269, -93.362269),
    ('HCN_triplet.out', 'grimme', 298.15, -93.153787, 0.012567, -93.137780, 0.024070, 0.024070, -93.161850, -93.161850),
    ('methylaniline.out', 'grimme', 298.15, -326.664901, 0.142118, -326.514489, 0.039668, 0.039535, -326.554157, -326.554024),
    # Grimme, 100.0K
    ('Al_298K.out', 'grimme', 100.0, -242.328708, 0.000000, -242.327916, 0.005062, 0.005062, -242.332978, -242.332978),
    ('Al_400K.out', 'grimme', 100.0, -242.328708, 0.000000, -242.327916, 0.005062, 0.005062, -242.332978, -242.332978),
    ('allene.out', 'grimme', 100.0, -116.569605, 0.053913, -116.514408, 0.007423, 0.007423, -116.521831, -116.521831),
    ('CuCN.out', 'grimme', 100.0, -289.005463, 0.006594, -288.997568, 0.006944, 0.006946, -289.004512, -289.004514),
    ('ethane.out', 'grimme', 100.0, -79.830421, 0.075238, -79.753900, 0.007558, 0.007559, -79.761458, -79.761459),
    ('ethane_spc.out', 'grimme', 100.0, -79.830421, 0.075238, -79.753900, 0.007558, 0.007559, -79.761458, -79.761459),
    ('ethane_TZ.out', 'grimme', 100.0, -79.858399, None, None, None, None, None, None),
    ('H2O.out', 'grimme', 100.0, -76.368128, 0.020772, -76.346089, 0.005812, 0.005812, -76.351901, -76.351901),
    ('HCN_singlet.out', 'grimme', 100.0, -93.358851, 0.015978, -93.341765, 0.006385, 0.006385, -93.348150, -93.348150),
    ('HCN_triplet.out', 'grimme', 100.0, -93.153787, 0.012567, -93.140111, 0.006803, 0.006803, -93.146915, -93.146915),
    ('methylaniline.out', 'grimme', 100.0, -326.664901, 0.142118, -326.521226, 0.009864, 0.009905, -326.531090, -326.531131),
    # Truhlar, 298.15K
    ('Al_298K.out', 'truhlar', 298.15, -242.328708, 0.000000, -242.326347, 0.017670, 0.017670, -242.344018, -242.344018),
    ('Al_400K.out', 'truhlar', 298.15, -242.328708, 0.000000, -242.326347, 0.017670, 0.017670, -242.344018, -242.344018),
    ('allene.out', 'truhlar', 298.15, -116.569605, 0.053913, -116.510916, 0.027618, 0.027618, -116.538534, -116.538534),
    ('CuCN.out', 'truhlar', 298.15, -289.005463, 0.006594, -288.994307, 0.025953, 0.025953, -289.020260, -289.020260),
    ('ethane.out', 'truhlar', 298.15, -79.830421, 0.075238, -79.750770, 0.027523, 0.027523, -79.778293, -79.778293),
    ('ethane_spc.out', 'truhlar', 298.15, -79.830421, 0.075238, -79.750770, 0.027523, 0.027523, -79.778293, -79.778293),
    ('ethane_TZ.out',  'truhlar', 298.15, -79.858399, None, None, None, None, None, None),
    ('H2O.out', 'truhlar', 298.15, -76.368128, 0.020772, -76.343577, 0.021458, 0.021458, -76.365035, -76.365035),
    ('HCN_singlet.out', 'truhlar', 298.15, -93.358851, 0.015978, -93.339373, 0.022896, 0.022896, -93.362269, -93.362269),
    ('HCN_triplet.out', 'truhlar', 298.15, -93.153787, 0.012567, -93.137780, 0.024070, 0.024070, -93.161850, -93.161850),
    ('methylaniline.out', 'truhlar', 298.15, -326.664901, 0.142118, -326.514489, 0.039668, 0.039668, -326.554157, -326.554157),
    # Truhlar, 100.0K
    ('Al_298K.out', 'truhlar', 100.0, -242.328708, 0.000000, -242.327916, 0.005062, 0.005062, -242.332978, -242.332978),
    ('Al_400K.out', 'truhlar', 100.0, -242.328708, 0.000000, -242.327916, 0.005062, 0.005062, -242.332978, -242.332978),
    ('allene.out', 'truhlar', 100.0, -116.569605, 0.053913, -116.514408, 0.007423, 0.007423, -116.521831, -116.521831),
    ('CuCN.out', 'truhlar', 100.0, -289.005463, 0.006594, -288.997568, 0.006944, 0.006944, -289.004512, -289.004512),
    ('ethane.out', 'truhlar', 100.0, -79.830421, 0.075238, -79.753900, 0.007558, 0.007558, -79.761458, -79.761458),
    ('ethane_spc.out', 'truhlar', 100.0, -79.830421, 0.075238, -79.753900, 0.007558, 0.007558, -79.761458, -79.761458),
    ('ethane_TZ.out', 'truhlar', 100.0, -79.858399, None, None, None, None, None, None),
    ('H2O.out', 'truhlar', 100.0, -76.368128, 0.020772, -76.346089, 0.005812, 0.005812, -76.351901, -76.351901),
    ('HCN_singlet.out', 'truhlar', 100.0, -93.358851, 0.015978, -93.341765, 0.006385, 0.006385, -93.348150, -93.348150),
    ('HCN_triplet.out', 'truhlar', 100.0, -93.153787, 0.012567, -93.140111, 0.006803, 0.006803, -93.146915, -93.146915),
    ('methylaniline.out', 'truhlar', 100.0, -326.664901, 0.142118, -326.521226, 0.009864, 0.009864, -326.531090, -326.531090),
])
def test_QS(path, QS, temp, E, ZPE, H, TS, TqhS, G, qhG):
    # Defaults, no temp interval, no conc interval
    path = datapath(path)
    conc = GV.ATMOS / (GV.GAS_CONSTANT * temp)
    QH, s_freq_cutoff, h_freq_cutoff, freq_scale_factor, solv, spc, invert, d3 = False, 100.0, 100.0, 1.0, 'none', False, False, 0
    bbe = GV.calc_bbe(path, QS, QH, s_freq_cutoff, h_freq_cutoff, temp, conc, freq_scale_factor, solv, spc, invert, d3)
    precision = 6 # if temp == 298.15 else 4e-4
    assert E == round(bbe.scf_energy, precision)
    if hasattr(bbe, "gibbs_free_energy"):
        assert ZPE == round(bbe.zpe, precision)
        assert H == round(bbe.enthalpy, precision)
        assert TS == round(temp * bbe.entropy, precision)
        assert TqhS == round(temp * bbe.qh_entropy, precision)
        assert G == round(bbe.gibbs_free_energy, precision)
        assert qhG == round(bbe.qh_gibbs_free_energy, precision)

@pytest.mark.parametrize("path, QS, temp, E, ZPE, H, qhH, TS, TqhS, G, qhG", [
    # Grimme, Head-Gordon, 298.15K
    ('Al_298K.out', 'grimme', 298.15, -242.328708, 0.000000, -242.326347, -242.326347, 0.017670, 0.017670, -242.344018, -242.344018),
    ('Al_400K.out', 'grimme', 298.15, -242.328708, 0.000000, -242.326347, -242.326347, 0.017670, 0.017670, -242.344018, -242.344018),
    ('allene.out', 'grimme', 298.15, -116.569605, 0.053913, -116.510916, -116.510925, 0.027618, 0.027621, -116.538534, -116.538546),
    ('CuCN.out', 'grimme', 298.15, -289.005463, 0.006594, -288.994307, -288.994323, 0.025953, 0.025956, -289.020260, -289.020279),
    ('ethane.out', 'grimme', 298.15, -79.830421, 0.075238, -79.750770, -79.750778, 0.027523, 0.027525, -79.778293, -79.778303),
    ('ethane_spc.out', 'grimme', 298.15, -79.830421, 0.075238, -79.750770, -79.750778, 0.027523, 0.027525, -79.778293, -79.778303),
    ('ethane_TZ.out', 'grimme', 298.15, -79.858399, None, None, None, None, None, None, None),
    ('H2O.out', 'grimme', 298.15, -76.368128, 0.020772, -76.343577, -76.343577, 0.021458, 0.021458, -76.365035, -76.365035),
    ('HCN_singlet.out', 'grimme', 298.15, -93.358851, 0.015978, -93.339373, -93.339374, 0.022896, 0.022896, -93.362269, -93.362270),
    ('HCN_triplet.out', 'grimme', 298.15, -93.153787, 0.012567, -93.137780, -93.137780, 0.024070, 0.024070, -93.161850, -93.161851),
    ('methylaniline.out', 'grimme', 298.15, -326.664901, 0.142118, -326.514489, -326.514824, 0.039668, 0.039535, -326.554157, -326.554359),
    # Grimme, Head-Gordon, 100.0K
    ('Al_298K.out', 'grimme', 100.0, -242.328708,0.000000,-242.327916,-242.327916,0.005062,0.005062,-242.332978,-242.332978),
    ('Al_400K.out', 'grimme', 100.0, -242.328708,0.000000,-242.327916,-242.327916,0.005062,0.005062,-242.332978,-242.332978),
    ('allene.out', 'grimme', 100.0, -116.569605,0.053913,-116.514408,-116.514418,0.007423,0.007423,-116.521831,-116.521841),
    ('CuCN.out', 'grimme', 100.0, -289.005463,0.006594,-288.997568,-288.997581,0.006944,0.006946,-289.004512,-289.004527),
    ('ethane.out', 'grimme', 100.0, -79.830421,0.075238,-79.753900,-79.753908,0.007558,0.007559,-79.761458,-79.761466),
    ('ethane_spc.out', 'grimme', 100.0, -79.830421,0.075238,-79.753900,-79.753908,0.007558,0.007559,-79.761458,-79.761466),
    ('ethane_TZ.out', 'grimme', 100.0, -79.858399, None, None, None, None, None, None, None),
    ('H2O.out', 'grimme', 100.0, -76.368128,0.020772,-76.346089,-76.346089,0.005812,0.005812,-76.351901,-76.351901),
    ('HCN_singlet.out', 'grimme', 100.0, -93.358851,0.015978,-93.341765,-93.341766,0.006385,0.006385,-93.348150,-93.348151),
    ('HCN_triplet.out', 'grimme', 100.0, -93.153787,0.012567,-93.140111,-93.140112,0.006803,0.006803,-93.146915,-93.146916),
    ('methylaniline.out', 'grimme', 100.0, -326.664901,0.142118,-326.521226,-326.521398,0.009864,0.009905,-326.531090,-326.531303),
    # Truhlar, Head-Gordon, 298.15K
    ('Al_298K.out', 'truhlar', 298.15, -242.328708,0.000000,-242.326347,-242.326347,0.017670,0.017670,-242.344018,-242.344018),
    ('Al_400K.out', 'truhlar', 298.15, -242.328708,0.000000,-242.326347,-242.326347,0.017670,0.017670,-242.344018,-242.344018),
    ('allene.out', 'truhlar', 298.15, -116.569605,0.053913,-116.510916,-116.510925,0.027618,0.027618,-116.538534,-116.538543),
    ('CuCN.out', 'truhlar', 298.15, -289.005463,0.006594,-288.994307,-288.994323,0.025953,0.025953,-289.020260,-289.020276),
    ('ethane.out', 'truhlar', 298.15, -79.830421,0.075238,-79.750770,-79.750778,0.027523,0.027523,-79.778293,-79.778301),
    ('ethane_spc.out', 'truhlar', 298.15, -79.830421,0.075238,-79.750770,-79.750778,0.027523,0.027523,-79.778293,-79.778301),
    ('ethane_TZ.out',  'truhlar', 298.15, -79.858399, None, None, None, None, None, None, None),
    ('H2O.out', 'truhlar', 298.15, -76.368128,0.020772,-76.343577,-76.343577,0.021458,0.021458,-76.365035,-76.365035),
    ('HCN_singlet.out', 'truhlar', 298.15, -93.358851,0.015978,-93.339373,-93.339374,0.022896,0.022896,-93.362269,-93.362270),
    ('HCN_triplet.out', 'truhlar', 298.15, -93.153787,0.012567,-93.137780,-93.137780,0.024070,0.024070,-93.161850,-93.161851),
    ('methylaniline.out', 'truhlar', 298.15, -326.664901,0.142118,-326.514489,-326.514824,0.039668,0.039668,-326.554157,-326.554492),
    # Truhlar, Head-Gordon, 100.0K
    ('Al_298K.out', 'truhlar', 100.0, -242.328708,0.000000,-242.327916,-242.327916,0.005062,0.005062,-242.332978,-242.332978),
    ('Al_400K.out', 'truhlar', 100.0, -242.328708,0.000000,-242.327916,-242.327916,0.005062,0.005062,-242.332978,-242.332978),
    ('allene.out', 'truhlar', 100.0, -116.569605,0.053913,-116.514408,-116.514418,0.007423,0.007423,-116.521831,-116.521840),
    ('CuCN.out', 'truhlar', 100.0, -289.005463,0.006594,-288.997568,-288.997581,0.006944,0.006944,-289.004512,-289.004525),
    ('ethane.out', 'truhlar', 100.0, -79.830421,0.075238,-79.753900,-79.753908,0.007558,0.007558,-79.761458,-79.761466),
    ('ethane_spc.out', 'truhlar', 100.0, -79.830421,0.075238,-79.753900,-79.753908,0.007558,0.007558,-79.761458,-79.761466),
    ('ethane_TZ.out', 'truhlar', 100.0, -79.858399, None, None, None, None, None, None, None),
    ('H2O.out', 'truhlar', 100.0, -76.368128,0.020772,-76.346089,-76.346089,0.005812,0.005812,-76.351901,-76.351901),
    ('HCN_singlet.out', 'truhlar', 100.0, -93.358851,0.015978,-93.341765,-93.341766,0.006385,0.006385,-93.348150,-93.348151),
    ('HCN_triplet.out', 'truhlar', 100.0,-93.153787,0.012567,-93.140111,-93.140112,0.006803,0.006803,-93.146915,-93.146916),
    ('methylaniline.out', 'truhlar', 100.0, -326.664901,0.142118,-326.521226,-326.521398,0.009864,0.009864,-326.531090,-326.531261)
])
def test_QH(path, QS, temp, E, ZPE, H, qhH, TS, TqhS, G, qhG):
    # Defaults, no temp interval, no conc interval
    path = datapath(path)
    conc = GV.ATMOS / (GV.GAS_CONSTANT * temp)
    QH, s_freq_cutoff, h_freq_cutoff, freq_scale_factor, solv, spc, invert, d3 = True, 100.0, 100.0, 1.0, 'none', False, False, 0
    bbe = GV.calc_bbe(path, QS, QH, s_freq_cutoff, h_freq_cutoff, temp, conc, freq_scale_factor, solv, spc, invert, d3)
    precision = 6 # if temp == 298.15 else 4e-4
    assert E == round(bbe.scf_energy, precision)
    if hasattr(bbe, "gibbs_free_energy"):
        assert ZPE == round(bbe.zpe, precision)
        assert H == round(bbe.enthalpy, precision)
        assert qhH == round(bbe.qh_enthalpy, precision)
        assert TS == round(temp * bbe.entropy, precision)
        assert TqhS == round(temp * bbe.qh_entropy, precision)
        assert G == round(bbe.gibbs_free_energy, precision)
        assert qhG == round(bbe.qh_gibbs_free_energy, precision)


@pytest.mark.parametrize("QS, E, ZPE, H, TS, TqhS, G, qhG", [
    #temperature correction w/o Head-Gordon
    ('grimme', -242.328708, 0.000000, -242.327125, 0.011221, 0.011221, -242.338346, -242.338346),
    ('truhlar', -242.328708, 0.000000, -242.327125, 0.011221, 0.011221, -242.338346, -242.338346),
])
def test_temperature_corrections_QS(QS, E, ZPE, H, TS, TqhS, G, qhG):
    temp = 200
    conc = GV.ATMOS / (GV.GAS_CONSTANT * temp)
    QH, s_freq_cutoff, h_freq_cutoff, freq_scale_factor, solv, spc, invert, d3 = False, 100.0, 100.0, 1.0, 'none', False, False, 0
    bbe298 = GV.calc_bbe(datapath('Al_298K.out'), QS, QH, s_freq_cutoff, h_freq_cutoff, temp, conc, freq_scale_factor, solv, spc, invert, d3)
    bbe400 = GV.calc_bbe(datapath('Al_400K.out'), QS, QH, s_freq_cutoff, h_freq_cutoff, temp, conc, freq_scale_factor, solv, spc, invert, d3)
    precision = 6
    assert E == round(bbe298.scf_energy, precision) == round(bbe400.scf_energy, precision)
    assert ZPE == round(bbe298.zpe, precision) == round(bbe400.zpe, precision)
    assert H == round(bbe298.enthalpy, precision) == round(bbe400.enthalpy, precision)
    assert TS == round(temp * bbe298.entropy, precision) == round(temp * bbe400.entropy, precision)
    assert TqhS == round(temp * bbe298.qh_entropy, precision) == round(temp * bbe400.qh_entropy, precision)
    assert G == round(bbe298.gibbs_free_energy, precision) == round(bbe400.gibbs_free_energy, precision)
    assert qhG == round(bbe298.qh_gibbs_free_energy, precision) == round(bbe400.qh_gibbs_free_energy, precision)

@pytest.mark.parametrize("QS, E, ZPE, H, qhH, TS, TqhS, G, qhG", [
    ('grimme', -242.328708,0.000000,-242.327125,-242.327125,0.011221,0.011221,-242.338346,-242.338346),
    ('truhlar', -242.328708,0.000000,-242.327125,-242.327125,0.011221,0.011221,-242.338346,-242.338346),
])
def test_temperature_corrections_QH(QS, E, ZPE, H, qhH, TS, TqhS, G, qhG):
    temp = 200
    conc = GV.ATMOS / (GV.GAS_CONSTANT * temp)
    QH, s_freq_cutoff, h_freq_cutoff, freq_scale_factor, solv, spc, invert, d3 = True, 100.0, 100.0, 1.0, 'none', False, False, 0
    bbe298 = GV.calc_bbe(datapath('Al_298K.out'), QS, QH, s_freq_cutoff, h_freq_cutoff, temp, conc, freq_scale_factor, solv, spc, invert, d3)
    bbe400 = GV.calc_bbe(datapath('Al_400K.out'), QS, QH, s_freq_cutoff, h_freq_cutoff, temp, conc, freq_scale_factor, solv, spc, invert, d3)
    precision = 6
    assert E == round(bbe298.scf_energy, precision) == round(bbe400.scf_energy, precision)
    assert ZPE == round(bbe298.zpe, precision) == round(bbe400.zpe, precision)
    assert H == round(bbe298.enthalpy, precision) == round(bbe400.enthalpy, precision)
    assert qhH == round(bbe298.qh_enthalpy, precision) == round(bbe400.qh_enthalpy, precision)
    assert TS == round(temp * bbe298.entropy, precision) == round(temp * bbe400.entropy, precision)
    assert TqhS == round(temp * bbe298.qh_entropy, precision) == round(temp * bbe400.qh_entropy, precision)
    assert G == round(bbe298.gibbs_free_energy, precision) == round(bbe400.gibbs_free_energy, precision)
    assert qhG == round(bbe298.qh_gibbs_free_energy, precision) == round(bbe400.qh_gibbs_free_energy, precision)

@pytest.mark.parametrize("spc, E_spc, E, ZPE, H, TS, TqhS, GT, qhGT", [
    (False,        None, -79.830421, 0.075238, -79.750770, 0.027523, 0.027525, -79.778293, -79.778295),
    ('link', -79.830421, -79.830421, 0.075238, -79.750770, 0.027523, 0.027525, -79.778293, -79.778295),
    ('spc',  -79.858399, -79.830421, 0.075238, -79.778748, 0.027523, 0.027525, -79.806271, -79.806273),
    ('TZ',   -79.858399, -79.830421, 0.075238, -79.778748, 0.027523, 0.027525, -79.806271, -79.806273)
])
def test_single_point_correction(spc, E_spc, E, ZPE, H, TS, TqhS, GT, qhGT):
    temp = 298.15
    conc = GV.ATMOS / (GV.GAS_CONSTANT * temp)
    QS, QH, s_freq_cutoff, h_freq_cutoff, freq_scale_factor, solv, invert, d3 = 'grimme', False, 100.0, 100.0, 1.0, 'none', False, 0
    precision = 6

    bbe = GV.calc_bbe(datapath('ethane.out'), QS, QH, s_freq_cutoff, h_freq_cutoff, temp, conc, freq_scale_factor, solv, spc, invert, d3)
    if E_spc:
        assert E_spc == round(bbe.sp_energy, precision)
    assert E == round(bbe.scf_energy, precision)
    assert ZPE == round(bbe.zpe, precision)
    assert H == round(bbe.enthalpy, precision)
    assert TS == round(temp * bbe.entropy, precision)
    assert TqhS == round(temp * bbe.qh_entropy, precision)
    assert GT == round(bbe.gibbs_free_energy, precision)
    assert qhGT == round(bbe.qh_gibbs_free_energy, precision)


@pytest.mark.parametrize("path, ti, H, TS, TqhS, GT, qhGT", [
    ('allene.out','200,300,40',[-116.512865,-116.512128,-116.511313],[0.016953,0.021149,0.025552],[0.016955,0.021151,0.025555],[-116.529818,-116.533277,-116.536865],[-116.529821,-116.533280,-116.536868]),
    ('ethane.out','200,300,40',[-79.752458,-79.751811,-79.751109],[0.017099,0.021225,0.025519],[0.017101,0.021227,0.025521],[-79.769556,-79.773036,-79.776628],[-79.769558,-79.773038,-79.776630]),
    ('methylaniline.out','200,300,40',[-326.518529,-326.51706,-326.515348],[0.023362,0.029637,0.036421],[0.023345,0.029579,0.036313],[-326.541891,-326.546698,-326.551769],[-326.541875,-326.546639,-326.551661]),
])
def test_temperature_interval(path, ti, H, TS, TqhS, GT, qhGT):
    
    QS, QH, s_freq_cutoff, h_freq_cutoff, freq_scale_factor, solv, spc, invert, d3 = 'grimme', False, 100.0, 100.0, 1.0, 'none', False, False, 0
    precision = 6
    temperature_interval = [float(temp) for temp in ti.split(',')]
    interval = range(int(temperature_interval[0]), int(temperature_interval[1]+1), int(temperature_interval[2]))
    for i in range(len(interval)):
        temp = float(interval[i])
        conc = GV.ATMOS / (GV.GAS_CONSTANT * temp)
        bbe = GV.calc_bbe(datapath(path), QS, QH, s_freq_cutoff, h_freq_cutoff, temp, conc, freq_scale_factor, solv, spc, invert, d3)

        assert H[i] == round(bbe.enthalpy, precision)
        assert TS[i] == round(temp * bbe.entropy, precision)
        assert TqhS[i] == round(temp * bbe.qh_entropy, precision)
        assert GT[i] == round(bbe.gibbs_free_energy, precision)
        assert qhGT[i] == round(bbe.qh_gibbs_free_energy, precision)


@pytest.mark.parametrize("filename, freq_scale_factor, zpe", [
    ('ethane.out', 0.977, 0.073508)
])
def test_scaling_factor_search(filename, freq_scale_factor, zpe):
    temp = 298.15
    conc = GV.ATMOS / (GV.GAS_CONSTANT * temp)
    QS, QH, s_freq_cutoff, h_freq_cutoff, solv, spc, invert, d3 = 'grimme',True, 100.0, 100.0, 'none', False, False, 0
    precision = 6
    bbe = GV.calc_bbe(datapath('ethane.out'), QS, QH, s_freq_cutoff, h_freq_cutoff, temp, conc, freq_scale_factor, solv, spc, invert, d3)
    assert zpe == round(bbe.zpe, precision)


@pytest.mark.parametrize("path, conc, QS, E, ZPE, H, TS, TqhS, G, qhG", [
    #no c correction applied
    ("media_conc/Benzene.log", 0, "grimme", -232.227201,0.101377,-232.120521,0.032742,0.032745,-232.153263,-232.153265),
    ("media_conc/H2O.log", 0, "grimme", -75.322774,0.021564,-75.297433,0.021627,0.021627,-75.319060,-75.319060),
    ("media_conc/MeOH.log", 0, "grimme", -114.179050,0.054749,-114.120139,0.026909,0.026910,-114.147048,-114.147049),
    ("media_conc/Benzene.log", 0, "truhlar", -232.227201,0.101377,-232.120521,0.032742,0.032742,-232.153263,-232.153263),
    ("media_conc/H2O.log", 0, "truhlar", -75.322774,0.021564,-75.297433,0.021627,0.021627,-75.319060,-75.319060),
    ("media_conc/MeOH.log", 0, "truhlar", -114.179050,0.054749,-114.120139,0.026909,0.026909,-114.147048,-114.147048),
    
    #with c correction = 1M
    ("media_conc/Benzene.log", 1, "grimme", -232.227201,0.101377,-232.120521,0.029723,0.029726,-232.150244,-232.150247),
    ("media_conc/H2O.log", 1, "grimme", -75.322774,0.021564,-75.297433,0.018608,0.018608,-75.316041,-75.316041),
    ("media_conc/MeOH.log", 1, "grimme", -114.179050,0.054749,-114.120139,0.023890,0.023891,-114.144029,-114.144030),
    ("media_conc/Benzene.log", 1, "truhlar", -232.227201,0.101377,-232.120521,0.029723,0.029723,-232.150244,-232.150244),
    ("media_conc/H2O.log", 1, "truhlar", -75.322774,0.021564,-75.297433,0.018608,0.018608,-75.316041,-75.316041),
    ("media_conc/MeOH.log", 1, "truhlar", -114.179050,0.054749,-114.120139,0.023890,0.023890,-114.144029,-114.144029)
])
def test_concentration_correction(path, conc, QS, E, ZPE, H, TS, TqhS, G, qhG):
        path = datapath(path)
        QH, s_freq_cutoff, h_freq_cutoff, freq_scale_factor, temp, solv, spc, invert, d3 = False, 100.0, 100.0,1.0, 298.15, 'none', False, False, 0
        if conc == False:
            conc = GV.ATMOS/(GV.GAS_CONSTANT*temp)
        bbe = GV.calc_bbe(path, QS, QH, s_freq_cutoff, h_freq_cutoff, temp, conc, freq_scale_factor, solv, spc, invert, d3)
        precision = 6 
        assert E == round(bbe.scf_energy, precision)
        if hasattr(bbe, "gibbs_free_energy"):
            assert ZPE == round(bbe.zpe, precision)
            assert H == round(bbe.enthalpy, precision)
            assert TS == round(temp * bbe.entropy, precision)
            assert TqhS == round(temp * bbe.qh_entropy, precision)
            assert G == round(bbe.gibbs_free_energy, precision)
            assert qhG == round(bbe.qh_gibbs_free_energy, precision)


@pytest.mark.parametrize("path, conc, media, E, ZPE, H, TS, TqhS, G, qhG", [
    #no media correction applied
    ("media_conc/Benzene.log", 1, False, -232.227201,0.101377,-232.120521,0.029723,0.029726,-232.150244,-232.150247),
    ("media_conc/H2O.log", 1, False, -75.322774,0.021564,-75.297433,0.018608,0.018608,-75.316041,-75.316041),
    ("media_conc/MeOH.log", 1, False, -114.179050,0.054749,-114.120139,0.023890,0.023891,-114.144029,-114.144030),
    
    #corresponding media correction applied
    ("media_conc/Benzene.log", 1, "benzene", -232.227201,0.101377,-232.120521,0.027440,0.027443,-232.147961,-232.147964),
    ("media_conc/H2O.log", 1, "h2o", -75.322774,0.021564,-75.297433,0.014818,0.014818,-75.312251,-75.312251),
    ("media_conc/MeOH.log", 1, "meoh", -114.179050,0.054749,-114.120139,0.020863,0.020864,-114.141002,-114.141003)
])
def test_media_correction(path,conc, media, E, ZPE, H, TS, TqhS, G, qhG):
        path = datapath(path)
        QH, QS, s_freq_cutoff, h_freq_cutoff, freq_scale_factor, temp, solv, spc, invert, d3 = False, "grimme", 100.0, 100.0, 1.0, 298.15, 'none', False, False, 0
        bbe = GV.calc_bbe(path, QS, QH, s_freq_cutoff, h_freq_cutoff, temp, conc, freq_scale_factor, solv, spc, invert, d3)
        precision = 6 
        
        media_correction = 0.0
        if media is not False:
            MW_solvent = solvents[media][0]
            density_solvent = solvents[media][1]
            concentration_solvent = (density_solvent*1000)/MW_solvent
            media_correction = -(GV.GAS_CONSTANT/GV.J_TO_AU)*math.log(concentration_solvent)
            
        assert E == round(bbe.scf_energy, precision)
        if hasattr(bbe, "gibbs_free_energy"):
            assert ZPE == round(bbe.zpe, precision)
            assert H == round(bbe.enthalpy, precision)
            assert TS == round(temp * (bbe.entropy+media_correction), precision)
            assert TqhS == round(temp * (bbe.qh_entropy+media_correction), precision)
            assert G == round(bbe.gibbs_free_energy+(temp * (-media_correction)), precision)
            assert qhG == round(bbe.qh_gibbs_free_energy+(temp * (-media_correction)), precision)
            

@pytest.mark.parametrize("E, ZPE, H, TS, TqhS, GT, qhGT", [
    ([0.0,-8.01,-50.34],[0.0,0.86,4.27],[0.0,-7.1,-45.99],[0.0,-14.54,-26.25],[0.0,-15.21,-29.6],[0.0,7.44,-19.74],[0.0,8.11,-16.39])
])
def test_pes(E, ZPE, H, TS, TqhS, GT, qhGT):
    temp = 298.15
    conc = GV.ATMOS / (GV.GAS_CONSTANT * temp)
    QS, QH, s_freq_cutoff, h_freq_cutoff, freq_scale_factor, solv, invert, d3 = 'grimme', False, 100.0, 100.0, 1.0, 'none', False, 0
    invert, spc, gconf = False, False, True
    precision = 2
    files = ['pes/Int-III_Oax_cis_a.log', 'pes/Int-II_Oax_cis_a.log', 'pes/Int-I_Oax.log', 'pes/TolS.log', 'pes/TolSH.log']
    files = [datapath(file) for file in files]
    log = GV.Logger("GoodVibes",'test',False)

    bbe_vals = []
    for file in files: # loop over all specified output files and compute thermochemistry
        bbe = GV.calc_bbe(file, QS, QH, s_freq_cutoff, h_freq_cutoff, temp,
                        conc, freq_scale_factor, solv, spc, invert, d3)
        bbe_vals.append(bbe)
    fileList = [file for file in files]
    thermo_data = dict(zip(fileList, bbe_vals)) # the collected thermochemical data for all files

    pes = GV.get_pes(datapath('pes/Cis_complete_pathway.yaml'),thermo_data,log,temp,gconf,QH)

    zero_vals = [pes.e_zero[0][0], pes.zpe_zero[0][0], pes.h_zero[0][0], temp * pes.ts_zero[0][0], temp * pes.qhts_zero[0][0], pes.g_zero[0][0], pes.qhg_zero[0][0]]

    for i, path in enumerate(pes.path):
        for j, e_abs in enumerate(pes.e_abs[i]):
            species = [pes.e_abs[i][j], pes.zpe_abs[i][j], pes.h_abs[i][j], temp * pes.s_abs[i][j], temp * pes.qs_abs[i][j], pes.g_abs[i][j], pes.qhg_abs[i][j]]
            relative = [species[x]-zero_vals[x] for x in range(len(zero_vals))]
            formatted_list = [GV.KCAL_TO_AU * x for x in relative]
            assert  E[j] == round(formatted_list[0], precision)
            assert  ZPE[j] == round(formatted_list[1], precision)
            assert  H[j] == round(formatted_list[2], precision)
            assert  TS[j] == round(formatted_list[3], precision)
            assert  TqhS[j] == round(formatted_list[4], precision)
            assert  GT[j] == round(formatted_list[5], precision)
            assert  qhGT[j] == round(formatted_list[6], precision)
    log.Finalize()
    
