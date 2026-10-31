from __future__ import annotations

import yaml

from querity.ingestion.enrichment import (
    apply_column_descriptions,
    build_embedding_text,
    build_m_schema,
    heuristic_enrichment,
    sanitize_enrichment,
)
from querity.ingestion.models import (
    ColumnDescription,
    ColumnProfile,
    ForeignKeyRef,
    TableEnrichment,
    TableSchema,
)


def make_table() -> TableSchema:
    return TableSchema(
        schema_name="public",
        table_name="fact_orders",
        comment="One row per order",
        columns=[
            ColumnProfile(
                name="order_id",
                data_type="bigint",
                value_kind="numeric",
                primary_key=True,
                description="Order identifier.",
            ),
            ColumnProfile(
                name="status",
                data_type="text",
                value_kind="string",
                allowed_values=["new", "paid", "cancelled"],
            ),
            ColumnProfile(
                name="customer_id",
                data_type="bigint",
                value_kind="numeric",
                foreign_key="public.dim_customers.customer_id",
            ),
        ],
        primary_key=["order_id"],
        unique_constraints=[["order_id"]],
        foreign_keys=[
            ForeignKeyRef(
                constrained_schema="public",
                constrained_table="fact_orders",
                constrained_columns=["customer_id"],
                referred_schema="public",
                referred_table="dim_customers",
                referred_columns=["customer_id"],
            )
        ],
    )


def make_enrichment() -> TableEnrichment:
    return TableEnrichment(
        business_description="Stores one row per order and is used for revenue analysis.",
        grain="one row per order",
        column_descriptions=[
            ColumnDescription(name="order_id", description="Unique order identifier."),
            ColumnDescription(name="status", description="Current order lifecycle state."),
            ColumnDescription(name="customer_id", description="Customer who placed the order."),
        ],
        sample_questions=[
            "How many orders were paid last month?",
            "Which customers placed the most orders?",
        ],
        caveats=["Cancelled orders should usually be excluded from revenue."],
    )


def test_m_schema_contains_business_and_allowed_values() -> None:
    table = make_table()
    enrichment = make_enrichment()
    m_schema = build_m_schema(
        table,
        enrichment,
        [
            {
                "table": "public.dim_customers",
                "condition": "fact_orders.customer_id = dim_customers.customer_id",
                "kind": "foreign_key",
                "confidence": 1.0,
            }
        ],
    )
    parsed = yaml.safe_load(m_schema)
    assert parsed["table_name"] == "public.fact_orders"
    assert parsed["grain"] == "one row per order"
    status = next(column for column in parsed["columns"] if column["name"] == "status")
    assert status["allowed_values"] == ["new", "paid", "cancelled"]
    assert parsed["join_keys"][0]["kind"] == "foreign_key"


def test_embedding_text_is_semantic_not_ddl() -> None:
    from querity.ingestion.settings import IngestionSettings

    settings = IngestionSettings(
        _env_file=None,
        source_database_url="postgresql://u:p@h/db",
        metadata_database_url="postgresql://u:p@h/db",
    )
    content = build_embedding_text(make_table(), make_enrichment(), settings)
    assert "Business purpose:" in content
    assert "Allowed values: new, paid, cancelled" in content
    assert "CREATE TABLE" not in content.upper()


def test_sanitize_fills_missing_column_descriptions() -> None:
    table = make_table()
    enrichment = TableEnrichment(
        business_description="Orders fact table.",
        grain=None,
        column_descriptions=[
            ColumnDescription(name="order_id", description="Order id."),
        ],
        sample_questions=[],
        caveats=[],
    )
    sanitized = sanitize_enrichment(table, enrichment)
    descriptions = {item.name: item.description for item in sanitized.column_descriptions}
    assert descriptions["order_id"] == "Order id."
    assert "customer_id" in descriptions
    assert descriptions["status"]


def test_sanitize_rejects_empty_business_description() -> None:
    table = make_table()
    enrichment = TableEnrichment(
        business_description=" ",
        grain=None,
        column_descriptions=[],
        sample_questions=[],
        caveats=[],
    )
    try:
        sanitize_enrichment(table, enrichment)
    except ValueError as exc:
        assert "business description" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_apply_column_descriptions() -> None:
    columns = apply_column_descriptions(make_table(), make_enrichment())
    by_name = {column["name"]: column for column in columns}
    assert by_name["status"]["description"] == "Current order lifecycle state."
    assert by_name["status"]["allowed_values"] == ["new", "paid", "cancelled"]


def test_heuristic_enrichment_uses_table_comment() -> None:
    enrichment = heuristic_enrichment(make_table())
    assert enrichment.business_description == "One row per order"
    assert len(enrichment.column_descriptions) == 3
    assert enrichment.sample_questions == []
