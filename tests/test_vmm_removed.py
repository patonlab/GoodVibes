"""The ONIOM MM-region frequency scaling feature (``--vmm`` /
``mm_freq_scale_factor`` / ``QCData.fract_modelsys``) was removed in v4.5.
These tests pin the removal and the back-compat of existing JSON exports."""
import inspect

from conftest import g16path
from goodvibes import GoodVibes as GV
from goodvibes.api import compute_thermo
from goodvibes.io import QCData, dict_to_qcdata, parse_qcdata, qcdata_to_dict
from goodvibes.thermo import ThermoOptions, calc_bbe

ONIOM = g16path('15_methanol_oniom_qmmm.log')


def test_vmm_flag_is_gone():
    parser = GV.build_parser() if hasattr(GV, 'build_parser') else None
    src = inspect.getsource(GV.parse_arguments)
    assert '--vmm' not in src
    assert 'mm_freq_scale_factor' not in src
    if parser is not None:
        assert not any('--vmm' in a.option_strings for a in parser._actions)


def test_thermo_options_and_api_have_no_mm_field():
    assert 'mm_freq_scale_factor' not in ThermoOptions.__dataclass_fields__
    assert 'mm_freq_scale_factor' not in inspect.signature(compute_thermo).parameters
    assert 'mm_freq_scale_factor' not in inspect.signature(calc_bbe.__init__).parameters
    assert 'fract_modelsys' not in QCData.__dataclass_fields__


def test_oniom_output_still_computes_with_single_scale_factor():
    r = compute_thermo(ONIOM)
    assert r.qh_gibbs_free_energy is not None
    q = parse_qcdata(ONIOM)
    assert q.has_oniom is True


def test_dict_to_qcdata_ignores_retired_fract_modelsys_key():
    d = qcdata_to_dict(parse_qcdata(g16path('01a_water_hf_freq.log')))
    d['fract_modelsys'] = [1.0, 1.0, 1.0]
    q = dict_to_qcdata(d)
    assert q.program == 'Gaussian'
    assert not hasattr(q, 'fract_modelsys')
