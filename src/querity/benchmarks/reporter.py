from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from querity.benchmarks.models import (
    BenchmarkRunSummary,
    QueryBenchmarkResult,
    StageLatencySummary,
)
from querity.common.logging import get_logger

logger = get_logger(__name__)


def mask_database_url(url: str) -> str:
    if "@" not in url:
        return url
    prefix, suffix = url.split("@", 1)
    if "://" in prefix:
        scheme, creds = prefix.split("://", 1)
        user = creds.split(":", 1)[0] if ":" in creds else creds
        return f"{scheme}://{user}:***@{suffix}"
    return f"***@{suffix}"


def _format_preview_table(columns: list[str], rows: list[list[Any]]) -> list[str]:
    if not columns or not rows:
        return ["*(No rows returned)*"]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    table_lines = [header, separator]
    for row in rows:
        cells = [str(c).replace("|", "\\|").replace("\n", " ") for c in row]
        if len(cells) < len(columns):
            cells.extend([""] * (len(columns) - len(cells)))
        elif len(cells) > len(columns):
            cells = cells[:len(columns)]
        table_lines.append("| " + " | ".join(cells) + " |")
    return table_lines


class BenchmarkReporter:
    def __init__(
        self,
        base_dir: str | Path = "benchmarks/runs",
        run_id: str | None = None,
    ) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = run_id or uuid.uuid4().hex[:8]
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir_name = f"run_{timestamp_str}_{self.run_id}"
        self.run_dir = self.base_dir / self.run_dir_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.run_dir / "benchmark.log"
        self._setup_file_logger()

    def _setup_file_logger(self) -> None:
        self.file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        self.file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.file_handler.setFormatter(formatter)
        q_logger = logging.getLogger("querity")
        q_logger.setLevel(logging.DEBUG)
        q_logger.addHandler(self.file_handler)
        logging.getLogger().addHandler(self.file_handler)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.datetime.now().isoformat()}] Benchmark run {self.run_id} initialized.\n")
        logger.info("Initialized benchmark run log at %s", self.log_file)
        self.file_handler.flush()

    def log(self, message: str, level: int = logging.INFO) -> None:
        logger.log(level, message)
        if hasattr(self, "file_handler") and self.file_handler:
            self.file_handler.flush()

    def close(self) -> None:
        if hasattr(self, "file_handler") and self.file_handler:
            self.file_handler.flush()
            self.file_handler.close()
            logging.getLogger().removeHandler(self.file_handler)
            logging.getLogger("querity").removeHandler(self.file_handler)

    def generate_summary(
        self,
        results: list[QueryBenchmarkResult],
        *,
        database_url: str,
        source_id: str,
        model: str,
        provider: str,
    ) -> BenchmarkRunSummary:
        total = len(results)
        if total == 0:
            return BenchmarkRunSummary(
                run_id=self.run_id,
                timestamp=datetime.datetime.now().isoformat(),
                database_url_masked=mask_database_url(database_url),
                source_id=source_id,
                model=model,
                provider=provider,
                run_directory=str(self.run_dir),
            )

        successful = sum(1 for r in results if r.accuracy.execution_success)
        exact_matches = sum(1 for r in results if r.accuracy.exact_tuple_match)
        row_matches = sum(1 for r in results if r.accuracy.row_count_match)
        guard_passed = sum(1 for r in results if r.accuracy.guard_passed)

        table_recalls = [r.accuracy.table_recall for r in results]
        table_precisions = [r.accuracy.table_precision for r in results]
        entity_recalls = [r.accuracy.entity_recall for r in results]
        soft_f1s = [r.accuracy.soft_f1 for r in results]
        value_parities = [r.accuracy.value_parity for r in results]
        total_latencies = [r.latencies.total_turn_ms for r in results]

        # Stage latencies
        stage_map = {
            "analyze": [r.latencies.analyze_ms for r in results if r.latencies.analyze_ms > 0],
            "retrieve": [r.latencies.retrieve_ms for r in results if r.latencies.retrieve_ms > 0],
            "generate": [r.latencies.generate_ms for r in results if r.latencies.generate_ms > 0],
            "guard": [r.latencies.guard_ms for r in results if r.latencies.guard_ms > 0],
            "review": [r.latencies.review_ms for r in results if r.latencies.review_ms > 0],
            "execute": [r.latencies.execute_ms for r in results if r.latencies.execute_ms > 0],
            "total_turn": total_latencies,
        }

        stage_summaries = {
            stage: StageLatencySummary.from_values(stage, vals)
            for stage, vals in stage_map.items()
        }

        e2e_summary = stage_summaries["total_turn"]

        return BenchmarkRunSummary(
            run_id=self.run_id,
            timestamp=datetime.datetime.now().isoformat(),
            database_url_masked=mask_database_url(database_url),
            source_id=source_id,
            model=model,
            provider=provider,
            total_queries=total,
            successful_executions=successful,
            execution_success_rate=round((successful / total) * 100, 2),
            exact_result_matches=exact_matches,
            exact_result_match_rate=round((exact_matches / total) * 100, 2),
            row_count_matches=row_matches,
            row_count_match_rate=round((row_matches / total) * 100, 2),
            avg_table_recall=round((sum(table_recalls) / total) * 100, 2),
            avg_table_precision=round((sum(table_precisions) / total) * 100, 2),
            guard_pass_rate=round((guard_passed / total) * 100, 2),
            avg_entity_recall=round((sum(entity_recalls) / total) * 100, 2),
            avg_soft_f1=round((sum(soft_f1s) / total) * 100, 2),
            avg_value_parity=round((sum(value_parities) / total) * 100, 2),
            mean_e2e_latency_ms=e2e_summary.mean_ms,
            p50_e2e_latency_ms=e2e_summary.median_ms,
            p95_e2e_latency_ms=e2e_summary.p95_ms,
            stage_latencies=stage_summaries,
            run_directory=str(self.run_dir),
        )

    def save_artifacts(
        self,
        summary: BenchmarkRunSummary,
        results: list[QueryBenchmarkResult],
    ) -> Path:
        # 1. Save benchmark_metrics.json
        metrics_file = self.run_dir / "benchmark_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(summary.model_dump(mode="json"), f, indent=2)

        # 2. Save query_results.json
        results_file = self.run_dir / "query_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump([r.model_dump(mode="json") for r in results], f, indent=2)

        # 3. Save query_results.jsonl
        jsonl_file = self.run_dir / "query_results.jsonl"
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.model_dump(mode="json")) + "\n")

        # 4. Generate summary.md
        summary_md_file = self.run_dir / "summary.md"
        md_content = self._render_markdown(summary, results)
        with open(summary_md_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        # 5. Update latest symlink / copy
        latest_dir = self.base_dir / "latest"
        try:
            if latest_dir.is_symlink() or latest_dir.is_file():
                latest_dir.unlink()
            elif latest_dir.is_dir():
                shutil.rmtree(latest_dir)
            latest_dir.symlink_to(self.run_dir_name, target_is_directory=True)
        except OSError:
            # Fallback for systems without symlink support
            latest_dir.mkdir(parents=True, exist_ok=True)
            for f_name in ["benchmark_metrics.json", "query_results.json", "summary.md"]:
                src = self.run_dir / f_name
                if src.exists():
                    shutil.copy2(src, latest_dir / f_name)

        # 6. Append to history.jsonl
        if hasattr(self, "file_handler") and self.file_handler:
            self.file_handler.flush()
        history_file = self.base_dir / "history.jsonl"
        with open(history_file, "a", encoding="utf-8") as f:
            history_record = {
                "run_id": summary.run_id,
                "timestamp": summary.timestamp,
                "model": summary.model,
                "total_queries": summary.total_queries,
                "execution_success_rate": summary.execution_success_rate,
                "exact_match_rate": summary.exact_result_match_rate,
                "entity_recall": summary.avg_entity_recall,
                "soft_f1": summary.avg_soft_f1,
                "value_parity": summary.avg_value_parity,
                "table_recall": summary.avg_table_recall,
                "mean_e2e_ms": summary.mean_e2e_latency_ms,
                "run_directory": str(self.run_dir),
            }
            f.write(json.dumps(history_record) + "\n")

        logger.info("Saved all benchmark artifacts to %s", self.run_dir)
        return self.run_dir

    def _render_markdown(
        self,
        summary: BenchmarkRunSummary,
        results: list[QueryBenchmarkResult],
    ) -> str:
        lines: list[str] = [
            f"# Querity Benchmark Report: Run {summary.run_id}",
            "",
            f"**Execution Timestamp**: {summary.timestamp}  ",
            f"**Database**: `{summary.database_url_masked}`  ",
            f"**Catalog Source ID**: `{summary.source_id}`  ",
            f"**Primary LLM**: `{summary.provider}` / `{summary.model}`  ",
            f"**Total Queries Evaluated**: {summary.total_queries}  ",
            f"**Artifacts Directory**: `{summary.run_directory}`",
            "",
            "---",
            "",
            "## Executive Metrics Scorecard",
            "",
            "| Metric | Result | Target Benchmark | Status |",
            "| :--- | :--- | :--- | :--- |",
            f"| **Execution Success Rate** | **{summary.execution_success_rate}%** ({summary.successful_executions}/{summary.total_queries}) | >= 90.0% | {'PASS' if summary.execution_success_rate >= 90 else 'WARN'} |",
            f"| **Exact Tuple Match Rate** | **{summary.exact_result_match_rate}%** ({summary.exact_result_matches}/{summary.total_queries}) | >= 70.0% | {'PASS' if summary.exact_result_match_rate >= 70 else 'WARN'} |",
            f"| **Row Count Match Rate** | **{summary.row_count_match_rate}%** ({summary.row_count_matches}/{summary.total_queries}) | >= 80.0% | {'PASS' if summary.row_count_match_rate >= 80 else 'WARN'} |",
            f"| **Entity Key Recall** | **{summary.avg_entity_recall}%** | >= 80.0% | {'PASS' if summary.avg_entity_recall >= 80 else 'WARN'} |",
            f"| **Soft F1 Match Rate** | **{summary.avg_soft_f1}%** | >= 70.0% | {'PASS' if summary.avg_soft_f1 >= 70 else 'WARN'} |",
            f"| **Value Parity Rate** | **{summary.avg_value_parity}%** | >= 70.0% | {'PASS' if summary.avg_value_parity >= 70 else 'WARN'} |",
            f"| **Average Table Recall** | **{summary.avg_table_recall}%** | >= 85.0% | {'PASS' if summary.avg_table_recall >= 85 else 'WARN'} |",
            f"| **Average Table Precision** | **{summary.avg_table_precision}%** | >= 50.0% | {'PASS' if summary.avg_table_precision >= 50 else 'WARN'} |",
            f"| **SQL Guard Pass Rate** | **{summary.guard_pass_rate}%** | >= 95.0% | {'PASS' if summary.guard_pass_rate >= 95 else 'WARN'} |",
            f"| **Mean End-to-End Latency** | **{summary.mean_e2e_latency_ms:,.1f} ms** | < 35,000 ms | {'PASS' if summary.mean_e2e_latency_ms < 35000 else 'WARN'} |",
            f"| **P50 End-to-End Latency** | **{summary.p50_e2e_latency_ms:,.1f} ms** | < 30,000 ms | {'PASS' if summary.p50_e2e_latency_ms < 30000 else 'WARN'} |",
            f"| **P95 End-to-End Latency** | **{summary.p95_e2e_latency_ms:,.1f} ms** | < 50,000 ms | {'PASS' if summary.p95_e2e_latency_ms < 50000 else 'WARN'} |",
            "",
            "---",
            "",
            "## Per-Stage Latency Breakdown (Milliseconds)",
            "",
            "| Stage | Mean | P50 (Median) | P90 | P95 | Min | Max | StdDev |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]

        stage_order = ["analyze", "retrieve", "generate", "guard", "review", "execute", "total_turn"]
        for stage in stage_order:
            s = summary.stage_latencies.get(stage)
            if s:
                stage_display = "Total Turn (E2E)" if stage == "total_turn" else stage.capitalize()
                lines.append(
                    f"| **{stage_display}** | {s.mean_ms:,.1f} | {s.median_ms:,.1f} | {s.p90_ms:,.1f} | {s.p95_ms:,.1f} | {s.min_ms:,.1f} | {s.max_ms:,.1f} | {s.stddev_ms:,.1f} |"
                )

        lines.extend([
            "",
            "---",
            "",
            "## Per-Query Results Detail",
            "",
            "| # | Question | Status | Exact Match | Rows (Gen/Gold) | Entity Recall | Value Parity | Table Recall | Analyze | Retrieve | Generate | Guard | Exec | Total Turn |",
            "| :- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for r in results:
            short_q = (r.question[:45] + "...") if len(r.question) > 48 else r.question
            status = "SUCCESS" if r.accuracy.execution_success else "FAIL"
            exact = "YES" if r.accuracy.exact_tuple_match else "NO"
            rows = f"{r.generated_rows_count}/{r.golden_rows_count}"
            recall = f"{int(r.accuracy.table_recall * 100)}%"
            ent_recall = f"{int(r.accuracy.entity_recall * 100)}%"
            val_parity = f"{int(r.accuracy.value_parity * 100)}%"
            lines.append(
                f"| {r.index} | {short_q} | {status} | {exact} | {rows} | {ent_recall} | {val_parity} | {recall} | "
                f"{r.latencies.analyze_ms:,.0f}ms | {r.latencies.retrieve_ms:,.0f}ms | "
                f"{r.latencies.generate_ms:,.0f}ms | {r.latencies.guard_ms:,.0f}ms | "
                f"{r.latencies.execute_ms:,.0f}ms | {r.latencies.total_turn_ms:,.0f}ms |"
            )

        # Append failure or divergence details if any
        divergences = [r for r in results if not r.accuracy.exact_tuple_match or not r.accuracy.execution_success]
        if divergences:
            lines.extend([
                "",
                "---",
                "",
                "## Divergence and Error Diagnostics",
                "",
            ])
            for d in divergences:
                lines.extend([
                    f"### Query {d.index}: {d.question}",
                    "",
                    f"- **Execution Status**: {'Success' if d.accuracy.execution_success else 'Error'}",
                    f"- **Exact Match**: {'Yes' if d.accuracy.exact_tuple_match else 'No'}",
                    f"- **Row Counts**: Generated={d.generated_rows_count}, Golden={d.golden_rows_count} (Diff: {d.accuracy.row_count_diff})",
                    f"- **Entity Recall**: {int(d.accuracy.entity_recall * 100)}% | **Soft F1**: {int(d.accuracy.soft_f1 * 100)}% | **Value Parity**: {int(d.accuracy.value_parity * 100)}%",
                    f"- **Table Recall**: {int(d.accuracy.table_recall * 100)}% (Expected: {', '.join(d.tables_expected)}, Retrieved: {', '.join(d.tables_retrieved)})",
                ])
                if d.execution_error:
                    lines.append(f"- **Error**: `{d.execution_error}`")
                if d.guard_errors:
                    lines.append(f"- **Guard Errors**: `{'; '.join(d.guard_errors)}`")
                if d.guard_warnings:
                    lines.append(f"- **Guard Warnings**: `{'; '.join(d.guard_warnings)}`")
                lines.extend([
                    "",
                    "**Generated SQL**:",
                    "```sql",
                    d.generated_sql or "-- (No SQL generated)",
                    "```",
                    "",
                    "**Golden SQL**:",
                    "```sql",
                    d.golden_sql,
                    "```",
                    "",
                ])
                if d.generated_columns and d.generated_rows_preview:
                    lines.extend([
                        f"**Generated Result Preview** (first {len(d.generated_rows_preview)} of {d.generated_rows_count} rows):",
                        "",
                    ])
                    lines.extend(_format_preview_table(d.generated_columns, d.generated_rows_preview))
                    lines.append("")
                if d.golden_columns and d.golden_rows_preview:
                    lines.extend([
                        f"**Golden Result Preview** (first {len(d.golden_rows_preview)} of {d.golden_rows_count} rows):",
                        "",
                    ])
                    lines.extend(_format_preview_table(d.golden_columns, d.golden_rows_preview))
                    lines.append("")

        lines.extend([
            "---",
            "",
            "Report automatically generated by Querity Benchmark Suite.",
            "",
        ])
        return "\n".join(lines)

