from __future__ import annotations

import sqlalchemy as sa

from querity.ingestion.introspection import _single_column_unique_columns, _value_kind


def test_composite_unique_constraints_do_not_mark_columns_unique() -> None:
    unique_columns = _single_column_unique_columns([["order_id"], ["customer_id", "order_month"]])
    assert unique_columns == {"order_id"}


def test_only_single_column_constraints_are_reported() -> None:
    assert _single_column_unique_columns([["a", "b"], ["c", "d"]]) == set()


def test_unusual_types_are_not_profiled_as_scalars() -> None:
    assert _value_kind(sa.ARRAY(sa.Integer)) == "other"
    assert _value_kind(sa.JSON()) == "other"
    assert _value_kind(sa.LargeBinary()) == "other"
    assert _value_kind(sa.String()) == "string"
    assert _value_kind(sa.Date()) == "temporal"
    assert _value_kind(sa.Boolean()) == "boolean"
    assert _value_kind(sa.Integer()) == "numeric"
    assert _value_kind(sa.Float()) == "numeric"
    assert _value_kind(sa.Numeric()) == "numeric"
    assert _value_kind(sa.Interval()) == "temporal"


def test_introspect_table_profiles_scalar_types_and_constraints(
    monkeypatch, settings_factory
) -> None:
    import querity.ingestion.introspection as introspection

    class FakeInspector:
        def get_columns(self, table, schema):
            return [
                {"name": "order_id", "type": sa.Integer(), "nullable": False, "default": None},
                {"name": "customer_id", "type": sa.Integer(), "nullable": False, "default": None},
                {"name": "order_month", "type": sa.Date(), "nullable": False, "default": None},
                {"name": "status", "type": sa.String(), "nullable": True, "default": None},
                {"name": "is_paid", "type": sa.Boolean(), "nullable": True, "default": None},
                {"name": "tags", "type": sa.ARRAY(sa.Integer()), "nullable": True, "default": None},
            ]

        def get_pk_constraint(self, table, schema):
            return {"constrained_columns": ["order_id"]}

        def get_foreign_keys(self, table, schema):
            return [
                {
                    "name": "fk_orders_customer",
                    "constrained_columns": ["customer_id"],
                    "referred_schema": "public",
                    "referred_table": "customers",
                    "referred_columns": ["customer_id"],
                }
            ]

        def get_unique_constraints(self, table, schema):
            return [{"column_names": ["customer_id", "order_month"]}]

        def get_indexes(self, table, schema):
            return [{"unique": True, "column_names": ["order_id"]}]

        def get_table_comment(self, table, schema):
            return {"text": "One row per order."}

    monkeypatch.setattr(introspection, "_row_count_estimate", lambda *args: 42)
    table = introspection._introspect_table(
        object(),  # type: ignore[arg-type]
        FakeInspector(),  # type: ignore[arg-type]
        "public",
        "orders",
        "table",
        settings_factory(),
    )

    by_name = {column.name: column for column in table.columns}
    assert by_name["order_id"].unique is True
    assert by_name["order_id"].value_kind == "numeric"
    assert by_name["order_id"].primary_key is True
    assert by_name["customer_id"].unique is False
    assert by_name["customer_id"].foreign_key == "public.customers.customer_id"
    assert by_name["order_month"].value_kind == "temporal"
    assert by_name["status"].value_kind == "string"
    assert by_name["is_paid"].value_kind == "boolean"
    assert by_name["tags"].value_kind == "other"
    assert table.comment == "One row per order."
    assert table.row_count_estimate == 42
