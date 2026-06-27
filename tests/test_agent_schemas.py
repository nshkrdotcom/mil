from __future__ import annotations


def test_agent_schemas_emit_handle_ids_for_handle_parameters():
    from mil.agent import build_tool_schemas

    schemas = {s["name"]: s for s in build_tool_schemas()}
    patch_model = schemas["patch"]["input_schema"]["properties"]["model"]
    patch_source = schemas["patch"]["input_schema"]["properties"]["source"]

    assert patch_model["type"] == "string"
    assert "handle_id" in patch_model["description"]
    assert "anyOf" in patch_source

