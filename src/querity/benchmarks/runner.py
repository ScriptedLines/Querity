from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from querity.agent.service import AgentService
from querity.benchmarks.evaluator import BenchmarkEvaluator
from querity.benchmarks.models import BenchmarkRunSummary, QueryBenchmarkResult
from querity.benchmarks.reporter import BenchmarkReporter
from querity.cli.config import CliConfig, load_config
from querity.common.logging import get_logger

logger = get_logger(__name__)


def load_golden_queries(path: str | Path | None = None) -> list[dict[str, Any]]:
    candidates = [
        Path(path) if path else None,
        Path("data/golden_queries.json"),
        Path("examples/ingestion/golden_queries.json"),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            with open(candidate, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
    raise FileNotFoundError("Could not find golden queries file in data/ or specified path.")


def run_benchmark(
    *,
    config_path: str | Path | None = None,
    golden_path: str | Path | None = None,
    output_dir: str | Path = "benchmarks/runs",
    limit: int | None = None,
    min_success_rate: float | None = None,
    min_table_recall: float | None = None,
    min_guard_pass_rate: float | None = None,
    max_mean_latency_ms: float | None = None,
    min_exact_match_rate: float | None = None,
    fail_on_threshold: bool = False,
    verbose: bool = True,
) -> BenchmarkRunSummary:
    console = Console()
    console.print("[bold cyan]Querity Real-Database Benchmark Runner[/bold cyan]")

    # 1. Load active database configuration from .config
    cfg = CliConfig.load(Path(config_path)) if config_path else load_config()
    settings = cfg.to_settings()

    db_masked = cfg.database_url
    if "@" in db_masked:
        pref, suff = db_masked.split("@", 1)
        db_masked = f"{pref.split(':')[0]}://***@{suff}"

    console.print(f"Target Database: [green]{db_masked}[/green]")
    console.print(f"Metadata Schema: [green]{settings.metadata_schema}[/green]")
    console.print(f"LLM Model: [green]{settings.llm_provider} / {settings.llm_model}[/green]")

    # 2. Load golden queries
    golden_queries = load_golden_queries(golden_path)
    if limit and limit > 0:
        golden_queries = golden_queries[:limit]
    console.print(f"Evaluating [yellow]{len(golden_queries)}[/yellow] golden query benchmark cases...")

    # 3. Initialize Reporter & Evaluator
    reporter = BenchmarkReporter(base_dir=output_dir)
    console.print(f"Output Directory: [blue]{reporter.run_dir}[/blue]")
    console.print(f"Logging active at: [blue]{reporter.log_file}[/blue]")

    service = AgentService(settings)
    evaluator = BenchmarkEvaluator(service)

    results: list[QueryBenchmarkResult] = []
    try:
        for idx, item in enumerate(golden_queries, 1):
            q_text = item.get("question", "")
            console.print(f"\n[bold][{idx}/{len(golden_queries)}][/bold] {q_text[:70]}...")
            res = evaluator.evaluate_query(idx, item)
            results.append(res)

            status_str = "[green]SUCCESS[/green]" if res.accuracy.execution_success else "[red]FAIL[/red]"
            exact_str = "[green]MATCH[/green]" if res.accuracy.exact_tuple_match else "[yellow]DIFF[/yellow]"
            recall_str = f"{int(res.accuracy.table_recall * 100)}%"
            console.print(
                f"   Status: {status_str} | Exact Match: {exact_str} | "
                f"Recall: {recall_str} | E2E: [cyan]{res.latencies.total_turn_ms:,.0f}ms[/cyan] "
                f"(Gen: {res.latencies.generate_ms:,.0f}ms, Exec: {res.latencies.execute_ms:,.0f}ms)"
            )
    finally:
        service.close()

    # 4. Generate Summary & Save All Artifacts
    summary = reporter.generate_summary(
        results,
        database_url=str(cfg.database_url),
        source_id=settings.source_id,
        model=settings.llm_model,
        provider=settings.llm_provider,
    )
    reporter.save_artifacts(summary, results)
    reporter.close()

    # 5. Display Console Summary Table
    if verbose:
        _print_console_summary(console, summary, results)

    # 6. Evaluate Quality & Latency Thresholds
    violations = summary.check_thresholds(
        min_success_rate=min_success_rate,
        min_table_recall=min_table_recall,
        min_guard_pass_rate=min_guard_pass_rate,
        max_mean_latency_ms=max_mean_latency_ms,
        min_exact_match_rate=min_exact_match_rate,
    )
    if violations:
        console.print("\n[bold red]Benchmark Threshold Violations:[/bold red]")
        for v in violations:
            console.print(f"  [red]FAIL[/red] {v}")
        if fail_on_threshold:
            raise RuntimeError(
                f"Benchmark failed threshold gate with {len(violations)} violation(s)."
            )
    elif any(
        t is not None
        for t in [
            min_success_rate,
            min_table_recall,
            min_guard_pass_rate,
            max_mean_latency_ms,
            min_exact_match_rate,
        ]
    ):
        console.print("\n[bold green]All benchmark threshold criteria met successfully![/bold green]")

    return summary


def _print_console_summary(
    console: Console,
    summary: BenchmarkRunSummary,
    results: list[QueryBenchmarkResult],
) -> None:
    table = Table(title=f"Querity Benchmark Results (Run {summary.run_id})")
    table.add_column("Metric", style="bold white")
    table.add_column("Result", style="cyan")
    table.add_column("Benchmark Target", style="dim")

    table.add_row(
        "Execution Success Rate",
        f"{summary.execution_success_rate}% ({summary.successful_executions}/{summary.total_queries})",
        ">= 90%",
    )
    table.add_row(
        "Exact Tuple Match Rate",
        f"{summary.exact_result_match_rate}% ({summary.exact_result_matches}/{summary.total_queries})",
        ">= 70%",
    )
    table.add_row(
        "Row Count Match Rate",
        f"{summary.row_count_match_rate}% ({summary.row_count_matches}/{summary.total_queries})",
        ">= 80%",
    )
    table.add_row(
        "Average Table Recall",
        f"{summary.avg_table_recall}%",
        ">= 85%",
    )
    table.add_row(
        "SQL Guard Pass Rate",
        f"{summary.guard_pass_rate}%",
        ">= 95%",
    )
    table.add_row(
        "Mean End-to-End Latency",
        f"{summary.mean_e2e_latency_ms:,.1f} ms",
        "< 35,000 ms",
    )
    table.add_row(
        "P50 End-to-End Latency",
        f"{summary.p50_e2e_latency_ms:,.1f} ms",
        "< 30,000 ms",
    )
    table.add_row(
        "P95 End-to-End Latency",
        f"{summary.p95_e2e_latency_ms:,.1f} ms",
        "< 50,000 ms",
    )

    console.print("\n")
    console.print(table)
    console.print(f"\n[bold green]Report saved to:[/bold green] {summary.run_directory}/summary.md")
    console.print(f"[bold green]Log file:[/bold green] {summary.run_directory}/benchmark.log")

