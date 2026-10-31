from __future__ import annotations

from querity.cli.init import DatabaseProbe, _default_source_schemas


def _probe(schemas: list[str]) -> DatabaseProbe:
    return DatabaseProbe(
        database="analytics", user="user", server_version="PostgreSQL", schemas=schemas
    )


def test_default_source_schemas_prefers_public():
    assert _default_source_schemas(_probe(["sales", "public"]), None, "querity") == "public"


def test_default_source_schemas_skips_metadata_schema():
    probe = _probe(["sales", "querity"])
    assert _default_source_schemas(probe, None, "querity") == "sales"
