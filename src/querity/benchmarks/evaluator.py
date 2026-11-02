from __future__ import annotations

import collections
import datetime
import decimal
import math
import time
from typing import Any

import sqlalchemy as sa

from querity.agent.service import AgentService
from querity.agent.state import HumanDecision
from querity.benchmarks.models import AccuracyMetrics, QueryBenchmarkResult, StageLatency
from querity.common.logging import get_logger
from querity.ingestion.db import readonly_connection

logger = get_logger(__name__)


def canonicalize_value(val: Any) -> Any:
    if val is None or isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val
    if isinstance(val, (float, decimal.Decimal)):
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 2)
    if isinstance(val, (datetime.datetime, datetime.date)):
        return val.isoformat()
    if isinstance(val, str):
        s = val.strip()
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            try:
                return int(s)
            except ValueError:
                pass
        try:
            f = float(s)
            if not (math.isnan(f) or math.isinf(f)):
                return round(f, 2)
        except ValueError:
            pass
        return s
    if isinstance(val, (bytes, bytearray, memoryview)):
        return bytes(val)
    return val


def canonicalize_row(row: Any) -> tuple[Any, ...]:
    if hasattr(row, "_mapping"):
        items = list(row._mapping.values())
    elif hasattr(row, "_asdict"):
        items = list(row._asdict().values())
    elif isinstance(row, dict):
        items = list(row.values())
    elif isinstance(row, tuple | list):
        items = list(row)
    else:
        items = [row]
    return tuple(canonicalize_value(v) for v in items)


class ComparisonResult(tuple):
    row_count_match: bool
    exact_tuple_match: bool
    row_count_diff: int
    entity_recall: float
    entity_precision: float
    soft_f1: float
    value_parity: float

    def __new__(
        cls,
        row_count_match: bool,
        exact_tuple_match: bool,
        row_count_diff: int,
        entity_recall: float = 0.0,
        entity_precision: float = 0.0,
        soft_f1: float = 0.0,
        value_parity: float = 0.0,
    ) -> ComparisonResult:
        obj = super().__new__(cls, (row_count_match, exact_tuple_match, row_count_diff))
        obj.row_count_match = bool(row_count_match)
        obj.exact_tuple_match = bool(exact_tuple_match)
        obj.row_count_diff = int(row_count_diff)
        obj.entity_recall = round(float(entity_recall), 4)
        obj.entity_precision = round(float(entity_precision), 4)
        obj.soft_f1 = round(float(soft_f1), 4)
        obj.value_parity = round(float(value_parity), 4)
        return obj


def is_numeric_value(val: Any) -> bool:
    if val is None or isinstance(val, bool):
        return False
    if isinstance(val, (int, float, decimal.Decimal)):
        return True
    if isinstance(val, str):
        try:
            float(val.strip())
            return True
        except ValueError:
            return False
    return False


def values_close(v1: Any, v2: Any, rel_tol: float = 0.02, abs_tol: float = 1.0) -> bool:
    if v1 is None and v2 is None:
        return True
    if v1 is None or v2 is None:
        return False
    if not is_numeric_value(v1) or not is_numeric_value(v2):
        return str(v1).strip().lower() == str(v2).strip().lower()
    try:
        f1, f2 = float(v1), float(v2)
    except (ValueError, TypeError):
        return str(v1).strip().lower() == str(v2).strip().lower()
    if math.isnan(f1) and math.isnan(f2):
        return True
    return abs(f1 - f2) <= max(rel_tol * max(abs(f1), abs(f2)), abs_tol)


def normalize_metric_name(col: str) -> str:
    s = col.lower().strip()
    for suffix in [
        "_usd",
        "_pct",
        "_percentage",
        "_days",
        "_count",
        "_total",
        "_after_refunds",
        "_amounts",
        "_amount",
    ]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    for prefix in ["total_", "active_"]:
        if s.startswith(prefix):
            s = s[len(prefix) :]
    s = s.replace("sales", "revenue")
    return s


def row_to_dict(row: Any, columns: list[str]) -> dict[str, Any]:
    if hasattr(row, "_mapping"):
        return {str(k).lower(): canonicalize_value(v) for k, v in row._mapping.items()}
    if hasattr(row, "_asdict"):
        return {str(k).lower(): canonicalize_value(v) for k, v in row._asdict().items()}
    if isinstance(row, dict):
        return {str(k).lower(): canonicalize_value(v) for k, v in row.items()}
    if isinstance(row, (tuple, list)):
        return {str(col).lower(): canonicalize_value(val) for col, val in zip(columns, row)}
    return {}


def find_entity_key_columns(gen_cols: list[str], gold_cols: list[str]) -> list[str]:
    gen_lower = [c.lower() for c in gen_cols]
    gold_lower = [c.lower() for c in gold_cols]
    shared = set(gen_lower).intersection(set(gold_lower))

    # Priority 1: Primary ID columns
    id_cols = [c for c in gen_lower if c in shared and (c.endswith("_id") or c == "id")]
    if id_cols:
        return id_cols

    # Priority 2: Codes / natural identifiers (sku, promo_code, etc.)
    code_cols = [
        c
        for c in gen_lower
        if c in shared and any(k in c for k in ["sku", "code", "number", "email"])
    ]
    if code_cols:
        return code_cols

    # Priority 3: Categorical / dimension names (carrier_name, tier, category, etc.)
    dim_cols = [
        c
        for c in gen_lower
        if c in shared
        and any(
            k in c
            for k in [
                "name",
                "tier",
                "quartile",
                "category",
                "status",
                "carrier",
                "gateway",
                "method",
                "priority",
                "region",
            ]
        )
    ]
    if dim_cols:
        return dim_cols

    # Priority 4: Any column in shared
    text_cols = [c for c in gen_lower if c in shared]
    if text_cols:
        return [text_cols[0]]

    # Priority 5: Fallback to first column if both non-empty
    if gen_cols and gold_cols:
        return [gen_lower[0]]
    return []


def compare_result_sets(
    generated_rows: list[Any],
    golden_rows: list[Any],
    gen_columns: list[str] | None = None,
    gold_columns: list[str] | None = None,
) -> ComparisonResult:
    gen_count = len(generated_rows)
    gold_count = len(golden_rows)
    row_count_match = gen_count == gold_count
    row_count_diff = gen_count - gold_count

    if not generated_rows and not golden_rows:
        return ComparisonResult(
            row_count_match=True,
            exact_tuple_match=True,
            row_count_diff=0,
            entity_recall=1.0,
            entity_precision=1.0,
            soft_f1=1.0,
            value_parity=1.0,
        )

    if gen_count == 0 or gold_count == 0:
        return ComparisonResult(
            row_count_match=row_count_match,
            exact_tuple_match=False,
            row_count_diff=row_count_diff,
            entity_recall=0.0,
            entity_precision=0.0,
            soft_f1=0.0,
            value_parity=0.0,
        )

    gen_canon = [canonicalize_row(r) for r in generated_rows]
    gold_canon = [canonicalize_row(r) for r in golden_rows]

    # Multiset comparison: compare counts of canonical tuples
    gen_counter = collections.Counter(gen_canon)
    gold_counter = collections.Counter(gold_canon)
    exact_match = gen_counter == gold_counter

    # If column counts differ or direct counter doesn't match, check set of values per row
    if not exact_match and row_count_match:
        gen_sets = [frozenset(r) for r in gen_canon]
        gold_sets = [frozenset(r) for r in gold_canon]
        if collections.Counter(gen_sets) == collections.Counter(gold_sets):
            exact_match = True

    if exact_match:
        return ComparisonResult(
            row_count_match=row_count_match,
            exact_tuple_match=True,
            row_count_diff=row_count_diff,
            entity_recall=1.0,
            entity_precision=1.0,
            soft_f1=1.0,
            value_parity=1.0,
        )

    # Infer column names if not provided
    g_cols = gen_columns or [f"col_{i}" for i in range(len(gen_canon[0]))]
    d_cols = gold_columns or [f"col_{i}" for i in range(len(gold_canon[0]))]

    key_cols = find_entity_key_columns(g_cols, d_cols)

    gold_by_key: dict[Any, dict[str, Any]] = {}
    gold_keys: set[Any] = set()
    for r in golden_rows:
        d = row_to_dict(r, d_cols)
        k = tuple(d.get(col) for col in key_cols) if len(key_cols) > 1 else (d.get(key_cols[0]) if key_cols else canonicalize_row(r))
        gold_by_key[k] = d
        gold_keys.add(k)

    gen_by_key: dict[Any, dict[str, Any]] = {}
    gen_keys: set[Any] = set()
    for r in generated_rows:
        d = row_to_dict(r, g_cols)
        k = tuple(d.get(col) for col in key_cols) if len(key_cols) > 1 else (d.get(key_cols[0]) if key_cols else canonicalize_row(r))
        gen_by_key[k] = d
        gen_keys.add(k)

    matched_keys = gold_keys.intersection(gen_keys)
    entity_recall = (len(matched_keys) / len(gold_keys)) if gold_keys else 1.0
    entity_precision = (len(matched_keys) / len(gen_keys)) if gen_keys else 0.0
    soft_f1 = (
        (2 * entity_recall * entity_precision) / (entity_recall + entity_precision)
        if (entity_recall + entity_precision) > 0
        else 0.0
    )

    # Comparable metric columns
    pairs: list[tuple[str, str]] = []
    g_lower = [c.lower() for c in g_cols]
    d_lower = [c.lower() for c in d_cols]
    d_norm_map = {normalize_metric_name(c): c for c in d_lower if c not in key_cols}

    for gc in g_lower:
        if gc in key_cols:
            continue
        if gc in d_lower and gc not in key_cols:
            pairs.append((gc, gc))
        else:
            gn = normalize_metric_name(gc)
            if gn in d_norm_map:
                pairs.append((gc, d_norm_map[gn]))

    matching_values = 0
    total_comparable = 0
    for k in matched_keys:
        g_row = gen_by_key.get(k, {})
        d_row = gold_by_key.get(k, {})
        for g_c, d_c in pairs:
            v_gen = g_row.get(g_c)
            v_gold = d_row.get(d_c)
            if is_numeric_value(v_gen) and is_numeric_value(v_gold):
                total_comparable += 1
                if values_close(v_gen, v_gold):
                    matching_values += 1

    if total_comparable > 0:
        value_parity = matching_values / total_comparable
    elif matched_keys:
        value_parity = 1.0 if entity_recall >= 0.8 else soft_f1
    else:
        value_parity = 0.0

    return ComparisonResult(
        row_count_match=row_count_match,
        exact_tuple_match=exact_match,
        row_count_diff=row_count_diff,
        entity_recall=entity_recall,
        entity_precision=entity_precision,
        soft_f1=soft_f1,
        value_parity=value_parity,
    )


def serialize_preview(rows: list[Any], max_rows: int = 5) -> list[list[Any]]:
    preview: list[list[Any]] = []
    for row in rows[:max_rows]:
        canon = canonicalize_row(row)
        preview.append([str(v) if v is not None else "" for v in canon])
    return preview


def normalize_table_name(name: str) -> str:
    cleaned = name.lower().strip()
    if cleaned.startswith("public."):
        cleaned = cleaned[len("public.") :]
    return cleaned


class BenchmarkEvaluator:
    def __init__(self, service: AgentService) -> None:
        self.service = service

    def evaluate_query(
        self,
        index: int,
        golden_item: dict[str, Any],
    ) -> QueryBenchmarkResult:
        question = str(golden_item.get("question") or "").strip()
        golden_sql = str(golden_item.get("sql") or "").strip()
        expected_raw_tables = list(golden_item.get("tables") or [])
        expected_tables = [normalize_table_name(t) for t in expected_raw_tables]
        notes = str(golden_item.get("notes") or "")

        logger.info("Evaluating query [%d]: %s", index, question)

        stage_timings: dict[str, float] = {}
        last_stamp = time.perf_counter()

        def on_event(node_name: str, payload: Any) -> None:
            nonlocal last_stamp
            now = time.perf_counter()
            duration_ms = (now - last_stamp) * 1000.0
            last_stamp = now
            stage_timings[node_name] = duration_ms
            logger.debug("Node [%s] executed in %.2f ms", node_name, duration_ms)

        t_turn_start = time.perf_counter()
        try:
            turn = self.service.start_turn(question, on_event=on_event)
        except Exception as exc:
            total_ms = (time.perf_counter() - t_turn_start) * 1000.0
            logger.error("start_turn failed on query [%d]: %s", index, exc)
            return QueryBenchmarkResult(
                index=index,
                question=question,
                golden_sql=golden_sql,
                tables_expected=expected_tables,
                execution_error=f"start_turn error: {exc}",
                latencies=StageLatency(total_turn_ms=total_ms),
                notes=notes,
            )

        # Extract retrieval metrics
        retrieval = turn.state.get("retrieval")
        retrieved_tables: list[str] = []
        if retrieval and hasattr(retrieval, "tables"):
            for th in retrieval.tables:
                t_name = getattr(th, "table_name", None) or getattr(th, "key", "")
                retrieved_tables.append(normalize_table_name(str(t_name)))

        exp_set = set(expected_tables)
        ret_set = set(retrieved_tables)
        matched_tables = exp_set.intersection(ret_set)
        table_recall = (len(matched_tables) / len(exp_set)) if exp_set else 1.0
        table_precision = (len(matched_tables) / len(ret_set)) if ret_set else 0.0
        all_tables_retrieved = exp_set.issubset(ret_set) if exp_set else True

        # Extract guard metrics
        guard = turn.state.get("guard")
        guard_passed = bool(guard and getattr(guard, "ok", False))
        guard_errors = [e.message for e in getattr(guard, "errors", [])] if guard else []
        guard_warnings = [w.message for w in getattr(guard, "warnings", [])] if guard else []

        candidate_sql = turn.sql

        # If awaiting approval, automatically approve to measure execution
        gen_rows: list[Any] = []
        gen_columns: list[str] = []
        exec_error: str | None = None
        duration_postgres_ms = 0.0

        if turn.status == "awaiting_approval":
            sql_hash = turn.state.get("sql_hash") or ""
            decision = HumanDecision(kind="approve", sql_hash=sql_hash)
            t_resume_start = time.perf_counter()
            try:
                resumed = self.service.resume(turn.thread_id, decision, on_event=on_event)
                execution = resumed.execution
                if execution:
                    if getattr(execution, "ok", False):
                        gen_rows = list(getattr(execution, "rows", []) or [])
                        gen_columns = [str(c) for c in (getattr(execution, "columns", []) or [])]
                        duration_postgres_ms = float(getattr(execution, "duration_ms", 0.0) or 0.0)
                    else:
                        exec_error = str(getattr(execution, "error", "Execution failed"))
            except Exception as exc:
                exec_error = f"Resume error: {exc}"
        elif turn.error:
            exec_error = turn.error

        total_turn_ms = (time.perf_counter() - t_turn_start) * 1000.0

        # Execute Golden Query against the real PostgreSQL database for baseline
        gold_rows: list[Any] = []
        gold_columns: list[str] = []
        try:
            with readonly_connection(self.service.source_engine) as conn:
                res = conn.execute(sa.text(golden_sql))
                gold_columns = [str(k) for k in res.keys()]
                gold_rows = list(res.fetchall())
        except Exception as exc:
            logger.warning("Golden SQL execution failed on query [%d]: %s", index, exc)

        # If generated SQL wasn't executed during turn but candidate SQL is available and valid
        if candidate_sql and not gen_rows and not exec_error:
            try:
                with readonly_connection(self.service.source_engine) as conn:
                    t_pg = time.perf_counter()
                    res = conn.execute(sa.text(candidate_sql))
                    duration_postgres_ms = (time.perf_counter() - t_pg) * 1000.0
                    gen_columns = [str(k) for k in res.keys()]
                    gen_rows = list(res.fetchall())
            except Exception as exc:
                exec_error = f"Candidate execution failed: {exc}"

        # Evaluate comparison metrics
        comparison = compare_result_sets(
            gen_rows,
            gold_rows,
            gen_columns=gen_columns,
            gold_columns=gold_columns,
        )
        row_count_match = comparison.row_count_match
        exact_tuple_match = comparison.exact_tuple_match
        row_diff = comparison.row_count_diff
        execution_success = bool(candidate_sql and not exec_error)

        # Column overlap
        col_overlap = 0.0
        if gold_columns and gen_columns:
            g_cols = {c.lower() for c in gold_columns}
            c_cols = {c.lower() for c in gen_columns}
            col_overlap = len(g_cols.intersection(c_cols)) / max(len(g_cols.union(c_cols)), 1)

        latencies = StageLatency(
            ingest_ms=round(stage_timings.get("ingest", 0.0), 2),
            analyze_ms=round(stage_timings.get("analyze", 0.0), 2),
            retrieve_ms=round(stage_timings.get("retrieve", 0.0), 2),
            generate_ms=round(stage_timings.get("generate", 0.0), 2),
            guard_ms=round(stage_timings.get("guard", 0.0), 2),
            review_ms=round(stage_timings.get("review", 0.0), 2),
            execute_ms=round(stage_timings.get("execute", duration_postgres_ms), 2),
            persist_ms=round(stage_timings.get("persist", 0.0), 2),
            total_turn_ms=round(total_turn_ms, 2),
        )

        accuracy = AccuracyMetrics(
            table_recall=round(table_recall, 4),
            table_precision=round(table_precision, 4),
            all_tables_retrieved=all_tables_retrieved,
            sql_valid=bool(candidate_sql),
            guard_passed=guard_passed,
            execution_success=execution_success,
            row_count_match=row_count_match,
            exact_tuple_match=exact_tuple_match,
            column_overlap=round(col_overlap, 4),
            row_count_diff=row_diff,
            entity_recall=comparison.entity_recall,
            entity_precision=comparison.entity_precision,
            soft_f1=comparison.soft_f1,
            value_parity=comparison.value_parity,
        )

        return QueryBenchmarkResult(
            index=index,
            question=question,
            golden_sql=golden_sql,
            generated_sql=candidate_sql,
            tables_expected=expected_tables,
            tables_retrieved=retrieved_tables,
            latencies=latencies,
            accuracy=accuracy,
            guard_errors=guard_errors,
            guard_warnings=guard_warnings,
            execution_error=exec_error,
            generated_rows_count=len(gen_rows),
            golden_rows_count=len(gold_rows),
            duration_postgres_ms=round(duration_postgres_ms, 2),
            notes=notes,
            generated_columns=gen_columns,
            golden_columns=gold_columns,
            generated_rows_preview=serialize_preview(gen_rows),
            golden_rows_preview=serialize_preview(gold_rows),
        )

