from __future__ import annotations

from querity.ingestion.hashing import (
    enrichment_fingerprint,
    profile_fingerprint,
    structural_fingerprint,
)
from querity.ingestion.models import ColumnProfile, TableSchema


def make_column(name: str, **overrides: object) -> ColumnProfile:
    values: dict[str, object] = {
        "name": name,
        "data_type": "character varying(50)",
        "value_kind": "string",
    }
    values.update(overrides)
    return ColumnProfile(**values)


def make_table() -> TableSchema:
    return TableSchema(
        schema_name="public",
        table_name="dim_customers",
        comment="Customer dimension",
        columns=[
            make_column("customer_id", primary_key=True),
            make_column("tier", allowed_values=["basic", "premium"]),
        ],
        primary_key=["customer_id"],
        unique_constraints=[["customer_id"]],
    )


def test_structural_hash_is_stable_across_column_order() -> None:
    first = make_table()
    second = make_table()
    second.columns = list(reversed(second.columns))
    assert structural_fingerprint(first) == structural_fingerprint(second)


def test_structural_hash_changes_with_comment() -> None:
    first = make_table()
    second = make_table()
    second.columns[1].comment = "Account tier"
    assert structural_fingerprint(first) != structural_fingerprint(second)


def test_profile_hash_ignores_volatile_sample_values() -> None:
    first = [make_column("tier", sample_values=["gold", "silver"])]
    second = [make_column("tier", sample_values=["bronze", "gold"])]
    assert profile_fingerprint(first) == profile_fingerprint(second)


def test_profile_hash_tracks_allowed_values_and_ranges() -> None:
    first = [make_column("tier", allowed_values=["gold", "silver"])]
    second = [make_column("tier", allowed_values=["gold", "bronze"])]
    assert profile_fingerprint(first) != profile_fingerprint(second)

    third = [
        make_column(
            "created_at",
            data_type="timestamp with time zone",
            value_kind="temporal",
            min_value="2023-01-01T00:00:00",
            max_value="2024-01-01T00:00:00",
        )
    ]
    fourth = [
        make_column(
            "created_at",
            data_type="timestamp with time zone",
            value_kind="temporal",
            min_value="2023-01-01T00:00:00",
            max_value="2025-01-01T00:00:00",
        )
    ]
    assert profile_fingerprint(third) != profile_fingerprint(fourth)


def test_enrichment_hash_tracks_model_and_prompt() -> None:
    structural = structural_fingerprint(make_table())
    profile = profile_fingerprint(make_table().columns)
    base = enrichment_fingerprint(
        structural, profile, llm_model="google:gemini", prompt_version="v1"
    )
    changed_model = enrichment_fingerprint(
        structural, profile, llm_model="google:other", prompt_version="v1"
    )
    changed_prompt = enrichment_fingerprint(
        structural, profile, llm_model="google:gemini", prompt_version="v2"
    )
    assert base != changed_model
    assert base != changed_prompt
