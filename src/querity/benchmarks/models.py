from __future__ import annotations

import statistics
from typing import Any
from pydantic import BaseModel, Field


class StageLatency(BaseModel):
    ingest_ms: float = 0.0
    analyze_ms: float = 0.0
    retrieve_ms: float = 0.0
    generate_ms: float = 0.0
    guard_ms: float = 0.0
    review_ms: float = 0.0
    execute_ms: float = 0.0
    persist_ms: float = 0.0
    total_turn_ms: float = 0.0


class AccuracyMetrics(BaseModel):
    table_recall: float = 0.0
    table_precision: float = 0.0
    all_tables_retrieved: bool = False
    sql_valid: bool = False
    guard_passed: bool = False
    execution_success: bool = False
    row_count_match: bool = False
    exact_tuple_match: bool = False
    column_overlap: float = 0.0
    row_count_diff: int = 0
    entity_recall: float = 0.0
    entity_precision: float = 0.0
    soft_f1: float = 0.0
    value_parity: float = 0.0


class QueryBenchmarkResult(BaseModel):
    index: int
    question: str
    golden_sql: str
    generated_sql: str | None = None
    tables_expected: list[str] = Field(default_factory=list)
    tables_retrieved: list[str] = Field(default_factory=list)
    latencies: StageLatency = Field(default_factory=StageLatency)
    accuracy: AccuracyMetrics = Field(default_factory=AccuracyMetrics)
    guard_errors: list[str] = Field(default_factory=list)
    guard_warnings: list[str] = Field(default_factory=list)
    execution_error: str | None = None
    generated_rows_count: int = 0
    golden_rows_count: int = 0
    duration_postgres_ms: float = 0.0
    notes: str = ""
    generated_columns: list[str] = Field(default_factory=list)
    golden_columns: list[str] = Field(default_factory=list)
    generated_rows_preview: list[list[Any]] = Field(default_factory=list)
    golden_rows_preview: list[list[Any]] = Field(default_factory=list)


class StageLatencySummary(BaseModel):
    stage: str
    mean_ms: float = 0.0
    median_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    stddev_ms: float = 0.0

    @classmethod
    def from_values(cls, stage: str, values: list[float]) -> StageLatencySummary:
        if not values:
            return cls(stage=stage)
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        p90_idx = min(int(n * 0.9), n - 1)
        p95_idx = min(int(n * 0.95), n - 1)
        std = statistics.stdev(values) if n > 1 else 0.0
        return cls(
            stage=stage,
            mean_ms=round(statistics.mean(values), 2),
            median_ms=round(statistics.median(values), 2),
            p90_ms=round(sorted_vals[p90_idx], 2),
            p95_ms=round(sorted_vals[p95_idx], 2),
            min_ms=round(min(values), 2),
            max_ms=round(max(values), 2),
            stddev_ms=round(std, 2),
        )


class BenchmarkRunSummary(BaseModel):
    run_id: str
    timestamp: str
    database_url_masked: str
    source_id: str
    model: str
    provider: str
    total_queries: int = 0
    successful_executions: int = 0
    execution_success_rate: float = 0.0
    exact_result_matches: int = 0
    exact_result_match_rate: float = 0.0
    row_count_matches: int = 0
    row_count_match_rate: float = 0.0
    avg_table_recall: float = 0.0
    avg_table_precision: float = 0.0
    guard_pass_rate: float = 0.0
    avg_entity_recall: float = 0.0
    avg_soft_f1: float = 0.0
    avg_value_parity: float = 0.0
    mean_e2e_latency_ms: float = 0.0
    p50_e2e_latency_ms: float = 0.0
    p95_e2e_latency_ms: float = 0.0
    stage_latencies: dict[str, StageLatencySummary] = Field(default_factory=dict)
    run_directory: str = ""

    def check_thresholds(
        self,
        *,
        min_success_rate: float | None = None,
        min_table_recall: float | None = None,
        min_guard_pass_rate: float | None = None,
        max_mean_latency_ms: float | None = None,
        min_exact_match_rate: float | None = None,
        min_entity_recall: float | None = None,
        min_soft_f1: float | None = None,
        min_value_parity: float | None = None,
    ) -> list[str]:
        violations: list[str] = []
        if min_success_rate is not None and self.execution_success_rate < min_success_rate:
            violations.append(
                f"Execution Success Rate {self.execution_success_rate}% is below threshold {min_success_rate}%"
            )
        if min_table_recall is not None and self.avg_table_recall < min_table_recall:
            violations.append(
                f"Average Table Recall {self.avg_table_recall}% is below threshold {min_table_recall}%"
            )
        if min_guard_pass_rate is not None and self.guard_pass_rate < min_guard_pass_rate:
            violations.append(
                f"SQL Guard Pass Rate {self.guard_pass_rate}% is below threshold {min_guard_pass_rate}%"
            )
        if max_mean_latency_ms is not None and self.mean_e2e_latency_ms > max_mean_latency_ms:
            violations.append(
                f"Mean End-to-End Latency {self.mean_e2e_latency_ms:.1f}ms exceeds threshold {max_mean_latency_ms:.1f}ms"
            )
        if min_exact_match_rate is not None and self.exact_result_match_rate < min_exact_match_rate:
            violations.append(
                f"Exact Tuple Match Rate {self.exact_result_match_rate}% is below threshold {min_exact_match_rate}%"
            )
        if min_entity_recall is not None and self.avg_entity_recall < min_entity_recall:
            violations.append(
                f"Entity Key Recall {self.avg_entity_recall}% is below threshold {min_entity_recall}%"
            )
        if min_soft_f1 is not None and self.avg_soft_f1 < min_soft_f1:
            violations.append(
                f"Soft F1 Match Rate {self.avg_soft_f1}% is below threshold {min_soft_f1}%"
            )
        if min_value_parity is not None and self.avg_value_parity < min_value_parity:
            violations.append(
                f"Value Parity Rate {self.avg_value_parity}% is below threshold {min_value_parity}%"
            )
        return violations

