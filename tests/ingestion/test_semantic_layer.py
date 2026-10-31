from __future__ import annotations

import json

import pytest

from querity.ingestion.semantic_layer import (
    GoldenSQLValidationError,
    load_golden_queries,
    load_semantic_layer,
    validate_read_only_sql,
)

SEMANTIC_YAML = """
metrics:
  - name: Net Retention Rate (NRR)
    synonyms: [NDR, NRR]
    description: Recurring revenue retained from existing customers.
    required_tables: [public.fact_mrr_monthly]
    sql_formula: |
      SUM(mrr_usd)
    notes: Excludes churned accounts.
filters:
  - name: Enterprise
    synonyms: [ENT]
    description: Enterprise accounts.
    table: public.dim_customers
    sql_condition: "public.dim_customers.tier = 'ENT'"
"""


def test_load_semantic_layer(tmp_path) -> None:
    path = tmp_path / "metrics.yaml"
    path.write_text(SEMANTIC_YAML)
    entries = load_semantic_layer(path)
    assert {entry.kind for entry in entries} == {"metric", "filter"}
    metric = next(entry for entry in entries if entry.kind == "metric")
    assert metric.name == "Net Retention Rate (NRR)"
    assert "NDR" in metric.embedding_text
    assert metric.sql_formula is not None
    assert "FORMULA" not in metric.embedding_text.upper()
    assert "churned" in metric.embedding_text


def test_semantic_layer_rejects_unknown_fields(tmp_path) -> None:
    path = tmp_path / "metrics.yaml"
    path.write_text(
        "metrics:\n  - name: X\n    required_tables: [t]\n    sql_formula: sum(x)\n    typo: 1\n"
    )
    with pytest.raises(ValueError):
        load_semantic_layer(path)


def test_duplicate_semantic_names_rejected(tmp_path) -> None:
    path = tmp_path / "metrics.yaml"
    path.write_text(
        "metrics:\n"
        "  - name: MRR\n    required_tables: [t]\n    sql_formula: sum(x)\n"
        "  - name: MRR\n    required_tables: [t]\n    sql_formula: sum(y)\n"
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_semantic_layer(path)


def test_load_golden_queries(tmp_path) -> None:
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question": "How many orders?",
                    "sql": "SELECT count(*) FROM public.orders",
                    "tables": ["public.orders"],
                    "tags": ["orders"],
                    "notes": "Exclude test orders.",
                }
            ]
        )
    )
    entries = load_golden_queries(path)
    assert len(entries) == 1
    assert entries[0].sql.endswith("public.orders")
    assert "How many orders?" in entries[0].embedding_text
    assert "test orders" in entries[0].embedding_text


def test_golden_queries_reject_duplicate_questions(tmp_path) -> None:
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            [
                {"question": "How many orders?", "sql": "SELECT 1"},
                {"question": "how   many orders?", "sql": "SELECT 2"},
            ]
        )
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_golden_queries(path)


def test_validate_read_only_sql_accepts_cte() -> None:
    normalized = validate_read_only_sql("WITH x AS (SELECT 1 AS n) SELECT * FROM x")
    assert "SELECT" in normalized.upper()


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE public.orders SET total = 1",
        "DELETE FROM public.orders",
        "DROP TABLE public.orders",
        "SELECT 1; DELETE FROM public.orders",
        "SELECT * INTO backup FROM public.orders",
        "TRUNCATE public.orders",
    ],
)
def test_validate_read_only_sql_rejects_writes(sql: str) -> None:
    with pytest.raises(GoldenSQLValidationError):
        validate_read_only_sql(sql)
