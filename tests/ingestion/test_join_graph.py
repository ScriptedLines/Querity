from __future__ import annotations

from collections import defaultdict

import sqlalchemy as sa

from querity.ingestion.join_graph import (
    _extract_query_log_edges,
    build_join_graph,
    graph_to_json,
    infer_join_edges,
    join_keys_for_table,
    load_join_graph,
    shortest_table_path,
)
from querity.ingestion.models import ColumnProfile, ForeignKeyRef, TableSchema

ENGINE = sa.create_engine("postgresql+psycopg://user:pass@localhost/db")


def column(name: str, *, primary_key: bool = False) -> ColumnProfile:
    return ColumnProfile(
        name=name,
        data_type="bigint",
        value_kind="numeric",
        primary_key=primary_key,
    )


def customer_table() -> TableSchema:
    return TableSchema(
        schema_name="public",
        table_name="dim_customers",
        columns=[column("customer_id", primary_key=True), column("tier")],
        primary_key=["customer_id"],
        unique_constraints=[["customer_id"]],
    )


def orders_table() -> TableSchema:
    return TableSchema(
        schema_name="public",
        table_name="fact_orders",
        columns=[column("order_id", primary_key=True), column("customer_id")],
        primary_key=["order_id"],
        unique_constraints=[["order_id"]],
    )


def test_infer_join_from_naming_convention() -> None:
    edges = infer_join_edges([orders_table(), customer_table()], ENGINE)
    assert len(edges) == 1
    edge = edges[0]
    assert edge.source_key == "public.fact_orders"
    assert edge.target_key == "public.dim_customers"
    assert edge.kind == "inferred"
    assert edge.target_columns == ["customer_id"]
    assert "customer_id" in edge.condition


def test_inference_skips_non_unique_targets() -> None:
    target = customer_table()
    target.unique_constraints = []
    target.primary_key = []
    edges = infer_join_edges([orders_table(), target], ENGINE)
    assert edges == []


def test_explicit_graph_and_bridge_path() -> None:
    a = TableSchema(
        schema_name="public",
        table_name="a",
        columns=[column("a_id", primary_key=True), column("b_id")],
        primary_key=["a_id"],
    )
    b = TableSchema(
        schema_name="public",
        table_name="b",
        columns=[column("b_id", primary_key=True), column("c_id")],
        primary_key=["b_id"],
        unique_constraints=[["b_id"]],
    )
    c = TableSchema(
        schema_name="public",
        table_name="c",
        columns=[column("c_id", primary_key=True)],
        primary_key=["c_id"],
        unique_constraints=[["c_id"]],
    )
    a.foreign_keys.append(
        ForeignKeyRef(
            name="a_b_fk",
            constrained_schema="public",
            constrained_table="a",
            constrained_columns=["b_id"],
            referred_schema="public",
            referred_table="b",
            referred_columns=["b_id"],
        )
    )
    b.foreign_keys.append(
        ForeignKeyRef(
            name="b_c_fk",
            constrained_schema="public",
            constrained_table="b",
            constrained_columns=["c_id"],
            referred_schema="public",
            referred_table="c",
            referred_columns=["c_id"],
        )
    )

    graph = build_join_graph([a, b, c], ENGINE)
    assert graph.number_of_nodes() == 3
    assert shortest_table_path(graph, "public.a", "public.c") == [
        "public.a",
        "public.b",
        "public.c",
    ]
    keys = join_keys_for_table(graph, "public.a")
    assert any(item["kind"] == "foreign_key" for item in keys)


def test_graph_serialization_roundtrip() -> None:
    graph = build_join_graph([orders_table(), customer_table()], ENGINE)
    restored = load_join_graph(graph_to_json(graph))
    assert restored.number_of_nodes() == graph.number_of_nodes()
    assert restored.number_of_edges() == graph.number_of_edges()
    assert shortest_table_path(restored, "public.fact_orders", "public.dim_customers") == [
        "public.fact_orders",
        "public.dim_customers",
    ]


def test_extract_query_log_join_edge() -> None:
    by_key = {"public.fact_orders", "public.dim_customers"}
    by_bare_name: dict[str, list[str]] = defaultdict(list)
    by_bare_name["fact_orders"] = ["public.fact_orders"]
    by_bare_name["dim_customers"] = ["public.dim_customers"]
    query = (
        "SELECT * FROM public.fact_orders AS o "
        "JOIN public.dim_customers AS c ON o.customer_id = c.customer_id"
    )
    edges = _extract_query_log_edges(query, 12, ENGINE, by_key, by_bare_name)
    assert len(edges) == 1
    assert edges[0].kind == "query_log"
    assert edges[0].source_key == "public.fact_orders"
    assert edges[0].target_key == "public.dim_customers"
    assert edges[0].condition == (
        "public.fact_orders.customer_id = public.dim_customers.customer_id"
    )


def test_extract_query_log_ignores_non_select() -> None:
    by_key = {"public.fact_orders", "public.dim_customers"}
    by_bare_name = {
        "fact_orders": ["public.fact_orders"],
        "dim_customers": ["public.dim_customers"],
    }
    query = "UPDATE public.fact_orders SET customer_id = 1"
    assert _extract_query_log_edges(query, 5, ENGINE, by_key, by_bare_name) == []


def test_extract_query_log_skips_cte_references() -> None:
    by_key = {"public.fact_orders", "public.dim_customers"}
    by_bare_name = {
        "fact_orders": ["public.fact_orders"],
        "dim_customers": ["public.dim_customers"],
    }
    query = (
        "WITH fact_orders AS (SELECT * FROM public.fact_orders) "
        "SELECT * FROM fact_orders JOIN public.dim_customers AS c "
        "ON fact_orders.customer_id = c.customer_id"
    )
    edges = _extract_query_log_edges(query, 5, ENGINE, by_key, by_bare_name)
    assert edges == []
