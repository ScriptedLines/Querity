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
    if val is None:
        return None
    if isinstance(val, decimal.Decimal):
        return round(float(val), 2)
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return round(val, 2)
    if isinstance(val, datetime.datetime | datetime.date):
        return val.isoformat()
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, bytes | bytearray | memoryview):
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


def compare_result_sets(
    generated_rows: list[Any],
    golden_rows: list[Any],
) -> tuple[bool, bool, int]:
    gen_count = len(generated_rows)
    gold_count = len(golden_rows)
    row_count_match = gen_count == gold_count
    row_count_diff = gen_count - gold_count

    if not generated_rows and not golden_rows:
        return True, True, 0

    if gen_count == 0 or gold_count == 0:
        return row_count_match, False, row_count_diff

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

    return row_count_match, exact_match, row_count_diff


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
        row_count_match, exact_tuple_match, row_diff = compare_result_sets(gen_rows, gold_rows)
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
        )

