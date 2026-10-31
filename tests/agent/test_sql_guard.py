from __future__ import annotations

import networkx as nx

from querity.agent.catalog import CatalogSnapshot, ColumnInfo, TableInfo
from querity.agent.settings import AgentSettings
from querity.agent.sql_guard import SqlGuard


def _settings(**overrides: object) -> AgentSettings:
    values = {
        "_env_file": None,
        "source_database_url": "postgresql://user:pass@localhost:5432/analytics",
        "metadata_database_url": "postgresql://user:pass@localhost:5432/analytics",
        "enable_explain_guard": False,
        "max_result_rows": 500,
    }
    values.update(overrides)
    return AgentSettings(**values)


def _catalog() -> CatalogSnapshot:
    users = TableInfo(
        key="public.users",
        schema_name="public",
        table_name="users",
        business_description="Customers",
        m_schema="users(id, name, email)",
        columns={
            "id": ColumnInfo("id", "integer"),
            "name": ColumnInfo("name", "text"),
            "email": ColumnInfo("email", "text", is_sensitive=True),
        },
    )
    orders = TableInfo(
        key="public.orders",
        schema_name="public",
        table_name="orders",
        business_description="Orders",
        m_schema="orders(id, user_id, amount)",
        columns={
            "id": ColumnInfo("id", "integer"),
            "user_id": ColumnInfo("user_id", "integer"),
            "amount": ColumnInfo("amount", "numeric"),
        },
    )
    graph = nx.MultiGraph()
    graph.add_node(users.key)
    graph.add_node(orders.key)
    graph.add_edge(users.key, orders.key, condition="users.id = orders.user_id")
    return CatalogSnapshot(
        source_id="default",
        tables={users.key: users, orders.key: orders},
        graph=graph,
    )


def test_valid_query_is_limited():
    guard = SqlGuard(_settings(), _catalog(), None)  # type: ignore[arg-type]
    result = guard.validate("SELECT id, name FROM public.users")
    assert result.ok
    assert result.tables_used == ["public.users"]
    assert result.limit_applied == 500
    assert "LIMIT 500" in result.sql


def test_existing_small_limit_is_preserved():
    guard = SqlGuard(_settings(), _catalog(), None)  # type: ignore[arg-type]
    result = guard.validate("SELECT id FROM public.users LIMIT 10")
    assert result.ok
    assert result.limit_applied == 10
    assert "LIMIT 500" not in result.sql


def test_unknown_table_and_column_are_rejected():
    guard = SqlGuard(_settings(), _catalog(), None)  # type: ignore[arg-type]
    table = guard.validate("SELECT id FROM public.missing")
    assert not table.ok
    assert any(issue.code == "unknown_table" for issue in table.errors)

    three_part = guard.validate("SELECT id FROM other.public.users")
    assert not three_part.ok
    assert any(issue.code == "unknown_table" for issue in three_part.errors)

    column = guard.validate("SELECT missing FROM public.users")
    assert not column.ok
    assert any(issue.code == "unknown_column" for issue in column.errors)

    qualified = guard.validate("SELECT public.users.missing FROM public.users")
    assert not qualified.ok
    assert any(issue.code == "unknown_column" for issue in qualified.errors)


def test_mutations_and_blocked_functions_are_rejected():
    guard = SqlGuard(_settings(), _catalog(), None)  # type: ignore[arg-type]
    for sql in (
        "DELETE FROM public.users",
        "UPDATE public.users SET name = 'x'",
        "SELECT pg_sleep(1)",
        "SELECT pg_catalog.pg_read_file('/etc/passwd')",
        "SELECT nextval('public.users_id_seq')",
        "SELECT * INTO copied_users FROM public.users",
        "SELECT * FROM public.users FOR UPDATE",
        "SELECT * FROM public.users; SELECT 1",
        "SELECT * FROM public.users -- comment",
    ):
        result = guard.validate(sql)
        assert not result.ok, sql


def test_comment_markers_inside_string_literals_are_allowed():
    guard = SqlGuard(_settings(), _catalog(), None)  # type: ignore[arg-type]
    result = guard.validate("SELECT id FROM public.users WHERE name = 'a--b'")
    assert result.ok, result.errors
    result = guard.validate("SELECT id FROM public.users WHERE name = 'a/*b*/c'")
    assert result.ok, result.errors


def test_select_star_is_a_warning():
    guard = SqlGuard(_settings(), _catalog(), None)  # type: ignore[arg-type]
    result = guard.validate("SELECT * FROM public.users")
    assert result.ok
    assert any(issue.code == "select_star" for issue in result.warnings)


def test_cte_and_order_by_alias_are_allowed():
    guard = SqlGuard(_settings(), _catalog(), None)  # type: ignore[arg-type]
    result = guard.validate(
        "WITH totals AS ("
        "  SELECT user_id, sum(amount) AS total FROM public.orders GROUP BY user_id"
        ") "
        "SELECT users.name, totals.total FROM totals "
        "JOIN public.users ON public.users.id = totals.user_id "
        "ORDER BY totals.total DESC"
    )
    assert result.ok, result.errors


def test_guard_result_cache_avoids_revalidation(tmp_path, monkeypatch):
    from querity.common.cache import LocalCache

    cache = LocalCache(tmp_path / "cache.sqlite3")
    guard = SqlGuard(
        _settings(),
        _catalog(),
        None,  # type: ignore[arg-type]
        cache=cache,
        catalog_version="run-1",
    )
    sql = "SELECT id, name FROM public.users"
    first = guard.validate(sql)
    assert first.ok
    assert cache.stats()["entries"] == 1

    def fail_if_called(self, value):
        raise AssertionError("guard cache was not used")

    monkeypatch.setattr(SqlGuard, "_validate", fail_if_called)
    second = guard.validate(sql)
    assert second == first


def test_guard_failures_and_catalog_versions_are_not_reused(tmp_path):
    from querity.common.cache import LocalCache

    cache = LocalCache(tmp_path / "cache.sqlite3")
    guard = SqlGuard(
        _settings(),
        _catalog(),
        None,  # type: ignore[arg-type]
        cache=cache,
        catalog_version="run-1",
    )
    assert not guard.validate("SELECT missing FROM public.users").ok
    assert cache.stats()["entries"] == 0

    assert guard.validate("SELECT id FROM public.users").ok
    assert cache.stats()["entries"] == 1
    other_version = SqlGuard(
        _settings(),
        _catalog(),
        None,  # type: ignore[arg-type]
        cache=cache,
        catalog_version="run-2",
    )
    assert other_version.validate("SELECT id FROM public.users").ok
    assert cache.stats()["entries"] == 2
