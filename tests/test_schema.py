"""Tests for goodvibes.schema — version constants + payload validator.

Schema v1.0 is the first stable release of the JSON layout. These tests
pin the contract so accidental mid-version edits get caught.
"""
import json

import pytest

from goodvibes import schema


# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------

def test_schema_version_is_one_dot_zero():
    """Pinned: bump only on a documented breaking change. v1.x minor
    versions are reserved for backwards-compatible additions."""
    assert schema.SCHEMA_VERSION == "1.0"


def test_schema_min_compatible_at_least_one():
    """Readers in this build must accept any version ≥ this."""
    assert schema.SCHEMA_MIN_COMPATIBLE == "1.0"


def test_schema_re_exported_from_output_module():
    """`output.JSON_SCHEMA_VERSION` is kept as a back-compat alias of
    `schema.SCHEMA_VERSION`. Anything that imported the old name keeps
    working without code changes."""
    from goodvibes.output import JSON_SCHEMA_VERSION
    assert JSON_SCHEMA_VERSION == schema.SCHEMA_VERSION


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------

def _minimal_payload(version="1.0"):
    return {
        "schema_version": version,
        "goodvibes_version": "4.2.0",
        "generated_at": "2026-05-02T00:00:00+00:00",
        "options": {},
        "results": [],
    }


def test_validate_accepts_minimal_payload():
    schema.validate(_minimal_payload())


def test_validate_accepts_full_payload():
    payload = _minimal_payload()
    payload["selectivity"] = {"labels": ["R", "S"], "key": "gibbs", "results": []}
    payload["pes"] = {"pathways": []}
    schema.validate(payload)


def test_validate_rejects_non_dict():
    with pytest.raises(ValueError, match="must be a JSON object"):
        schema.validate(["not", "a", "dict"])


def test_validate_rejects_missing_schema_version():
    payload = {"goodvibes_version": "4.2.0", "results": []}
    with pytest.raises(ValueError, match="schema_version"):
        schema.validate(payload)


def test_validate_rejects_pre_v1_payload():
    """v0.x payloads (the v4.x preview schema) must be rejected with a
    pointer to regenerate."""
    payload = _minimal_payload(version="0.4")
    with pytest.raises(ValueError, match="0.4|0\\.x|regenerate"):
        schema.validate(payload)


def test_validate_rejects_future_major_version():
    """A v2.0 payload (future-breaking) must be rejected."""
    payload = _minimal_payload(version="2.0")
    with pytest.raises(ValueError, match="newer|2\\.0|Upgrade"):
        schema.validate(payload)


def test_validate_accepts_minor_bump():
    """A v1.5 payload (backwards-compatible additions) must validate
    against a v1.0 reader — minor bumps are required to be additive."""
    payload = _minimal_payload(version="1.5")
    schema.validate(payload)


def test_validate_accepts_patch_version_string():
    """'1.0.3' should parse and validate the same as '1.0'."""
    payload = _minimal_payload(version="1.0.3")
    schema.validate(payload)


def test_validate_rejects_missing_required_fields():
    for missing in ("goodvibes_version", "results"):
        payload = _minimal_payload()
        del payload[missing]
        with pytest.raises(ValueError, match=missing):
            schema.validate(payload)


def test_validate_rejects_results_not_a_list():
    payload = _minimal_payload()
    payload["results"] = {"not": "a list"}
    with pytest.raises(ValueError, match="must be a list"):
        schema.validate(payload)


# ---------------------------------------------------------------------------
# End-to-end: write_json_results + validate
# ---------------------------------------------------------------------------

def test_write_json_results_emits_v1_payload(tmp_path):
    """A real CLI run's JSON output must validate against the v1.0 schema."""
    from types import SimpleNamespace
    from goodvibes.output import write_json_results

    options = SimpleNamespace(
        temperature=298.15, temperature_interval=None, conc=None,
        QS="grimme", QH=False, freq_cutoff=100.0,
        S_freq_cutoff=100.0, H_freq_cutoff=100.0,
        freq_scale_factor=1.0, media=None, freespace="none", spc=None,
        invert=False, symm=False, duplicate=False, boltz=False,
        inertia="global", mm_freq_scale_factor=None,
    )
    out = tmp_path / "out.json"
    write_json_results({}, options, str(out))
    payload = json.loads(out.read_text())
    schema.validate(payload)
    assert payload["schema_version"] == "1.0"
