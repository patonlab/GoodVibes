"""Conformance kit for reaction-profile 1.0 (tests/profile_conformance).

The directory is the kit a third-party writer uses: every document under
``valid/`` must be accepted, every one under ``invalid-structural/`` must
be rejected by both the JSON Schema and the reference validator, and every
one under ``invalid-semantic/`` breaks a referential rule the JSON Schema
cannot express (it passes the schema) that the reference validator must
reject. The agreement between the two validators is what this module pins.
"""
import json
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
jsonschema = pytest.importorskip("jsonschema")

from goodvibes.profile import load_json_schema, schema_errors, validate_document  # noqa: E402

KIT = Path(__file__).resolve().parent / "profile_conformance"


def _load(path):
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)


def _cases(sub):
    files = sorted((KIT / sub).glob("*.*"))
    assert files, f"no fixtures in {sub}"
    return pytest.mark.parametrize("path", files, ids=[p.name for p in files])


def test_the_schema_is_a_valid_draft_2020_12_schema():
    jsonschema.Draft202012Validator.check_schema(load_json_schema())


@_cases("valid")
def test_valid_documents_pass_both_validators(path):
    doc = _load(path)
    assert schema_errors(doc) == []
    errors, _warnings = validate_document(doc, use_jsonschema=False)
    assert errors == []


@_cases("invalid-structural")
def test_structural_errors_fail_both_validators(path):
    doc = _load(path)
    assert schema_errors(doc), "the JSON Schema accepted a structurally invalid document"
    errors, _warnings = validate_document(doc, use_jsonschema=False)
    assert errors, "the reference validator accepted a structurally invalid document"


@_cases("invalid-semantic")
def test_referential_errors_pass_the_schema_and_fail_the_reference_validator(path):
    doc = _load(path)
    assert schema_errors(doc) == [], "fixture belongs in invalid-structural/"
    errors, _warnings = validate_document(doc, use_jsonschema=False)
    assert errors, "the reference validator accepted a referentially invalid document"


def test_validate_document_merges_schema_errors_with_a_prefix():
    doc = _load(KIT / "invalid-structural" / "bad_units.yaml")
    errors, _w = validate_document(doc)
    assert any(e.startswith("schema: ") for e in errors)
    assert any(e.startswith("units: ") for e in errors)


def test_every_written_document_passes_the_schema(tmp_path):
    """What GoodVibes writes (the explicit form) is always schema-valid."""
    from goodvibes.profile import load_profile
    for path in sorted((KIT / "valid").glob("*.*")):
        prof = load_profile(path)
        assert schema_errors(prof.to_dict()) == [], path.name
