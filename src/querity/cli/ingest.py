from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from querity.agent.settings import AgentSettings
from querity.ingestion.pipeline import PipelineOptions, RunSummary, ingest


def run_ingestion(
    settings: AgentSettings,
    *,
    console: Console | None = None,
    force: bool = False,
    table_patterns: list[str] | None = None,
    dry_run: bool = False,
    generate_semantic_layer: bool | None = None,
    enable_llm: bool | None = None,
    refresh_profiles: bool = False,
    golden_queries_path: Path | None = None,
) -> RunSummary:
    console = console or Console()
    options = PipelineOptions(
        force=force,
        table_patterns=list(table_patterns or []),
        generate_semantic_layer=generate_semantic_layer,
        dry_run=dry_run,
        enable_llm=enable_llm,
        refresh_profiles=refresh_profiles,
        golden_queries_path=golden_queries_path,
    )
    console.print(
        f"[cyan]Indexing[/cyan] source_id=[bold]{settings.source_id}[/bold] "
        f"schemas=[bold]{settings.source_schemas}[/bold] "
        f"metadata=[bold]{settings.metadata_schema}[/bold]"
    )
    try:
        with console.status("Profiling, enriching and embedding the catalog..."):
            summary = ingest(settings, options=options)
    except Exception as exc:
        console.print(f"[red]Ingestion failed:[/red] {exc}")
        raise
    _print_summary(console, summary)
    return summary


def _print_summary(console: Console, summary: RunSummary) -> None:
    style = {
        "succeeded": "green",
        "partial": "yellow",
        "failed": "red",
    }.get(summary.status, "white")
    console.print(
        Panel(
            Text(
                f"Status  : {summary.status}\n"
                f"Run id  : {summary.run_id or '-'}\n"
                f"Duration: {summary.duration_seconds:.1f}s",
                overflow="fold",
            ),
            title="Ingestion complete",
            border_style=style,
        )
    )
    stats = Table(title="Stats", show_header=True, header_style="bold cyan")
    stats.add_column("Metric")
    stats.add_column("Value", justify="right")
    for key in sorted(summary.stats):
        value: Any = summary.stats[key]
        if isinstance(value, dict):
            continue
        stats.add_row(str(key), str(value))
    console.print(stats)
    if summary.errors:
        errors = Table(title="Errors", show_header=True, header_style="bold red")
        errors.add_column("Table")
        errors.add_column("Error")
        for error in summary.errors[:50]:
            errors.add_row(
                str(error.get("table") or error.get("object") or "-"),
                str(error.get("error") or error.get("message") or error),
            )
        console.print(errors)
