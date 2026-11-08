from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from querity.agent.service import AgentService
from querity.cli.config import (
    ConfigError,
    ConfigNotFound,
    load_config,
)
from querity.common.cache import build_local_cache

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="querity",
        description=(
            "Chat with a PostgreSQL database using a guarded text-to-SQL agent. "
            "Run without a command to open the interactive UI."
        ),
    )
    sub = parser.add_subparsers(dest="command")
    parser.set_defaults(handler=_cmd_tui)

    status = sub.add_parser("status", help="show database, catalog and model status")
    status.add_argument("--json", action="store_true", help="print JSON")
    status.set_defaults(handler=_cmd_status)

    cache = sub.add_parser("cache", help="inspect or clear the local response cache")
    cache.add_argument("action", choices=("stats", "clear"))
    cache.set_defaults(handler=_cmd_cache)

    version = sub.add_parser("version", help="print the version")
    version.set_defaults(handler=_cmd_version)

    benchmark = sub.add_parser(
        "benchmark",
        help="evaluate latency, accuracy, and stage metrics on real database",
    )
    benchmark.add_argument("--limit", "-n", type=int, default=None, help="limit number of queries to evaluate")
    benchmark.add_argument("--golden-path", "-g", type=str, default="data/golden_queries.json", help="path to golden queries JSON")
    benchmark.add_argument("--output-dir", "-o", type=str, default="benchmarks/runs", help="directory to save run artifacts")
    benchmark.add_argument("--min-success-rate", type=float, default=None, help="minimum execution success rate percent (e.g. 90.0)")
    benchmark.add_argument("--min-table-recall", type=float, default=None, help="minimum average table recall percent (e.g. 80.0)")
    benchmark.add_argument("--min-guard-pass-rate", type=float, default=None, help="minimum SQL guard pass rate percent (e.g. 95.0)")
    benchmark.add_argument("--max-mean-latency-ms", type=float, default=None, help="maximum allowed mean latency in ms (e.g. 40000.0)")
    benchmark.add_argument("--min-exact-match-rate", type=float, default=None, help="minimum exact match rate percent (e.g. 70.0)")
    benchmark.add_argument("--min-entity-recall", type=float, default=None, help="minimum entity key recall percent (e.g. 80.0)")
    benchmark.add_argument("--min-soft-f1", type=float, default=None, help="minimum soft F1 match rate percent (e.g. 70.0)")
    benchmark.add_argument("--min-value-parity", type=float, default=None, help="minimum value parity rate percent (e.g. 70.0)")
    benchmark.add_argument("--fail-on-threshold", action="store_true", help="exit with error code 1 if any threshold is violated")
    benchmark.set_defaults(handler=_cmd_benchmark)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except ConfigNotFound as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/yellow]")
        return 130
    except Exception as exc:
        console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
        return 1


def _cmd_tui(_args: argparse.Namespace) -> int:
    from querity.cli.tui import run_tui

    try:
        config = load_config()
    except ConfigNotFound:
        config = None
    run_tui(config)
    return 0


# Alias for backward compatibility in existing tests
_cmd_chat = _cmd_tui


def _cmd_status(args: argparse.Namespace) -> int:
    config = load_config()
    service = AgentService(config.to_settings())
    try:
        report = service.status_report()
        if args.json:
            console.print_json(json.dumps(report, default=str))
            return 0
        _print_status(report)
        return 0
    finally:
        service.close()


def _print_status(report: dict[str, Any]) -> None:
    lines = [
        f"Database        : {report.get('database')}",
        f"Source id       : {report.get('source_id')}",
        f"Source schemas  : {report.get('source_schemas')}",
        f"Metadata schema : {report.get('metadata_schema')}",
        f"Catalog tables  : {report.get('catalog_tables')}",
        f"LLM             : {report.get('provider')}/{report.get('model')}",
        f"Fallbacks       : {', '.join(report.get('fallback_models') or []) or '(none)'}",
        f"Reviewer        : {report.get('review_model')}",
        f"Embeddings      : {report.get('embedding')}",
        f"History DB      : {report.get('history_db')}",
        f"Checkpoint DB   : {report.get('checkpoint_db')}",
    ]
    chat = report.get("chat") or {}
    lines.extend(
        [
            f"Messages        : {chat.get('messages', 0)}",
            f"Query runs      : {chat.get('query_runs', 0)}",
            f"Verified turns  : {chat.get('verified_turns', 0)}",
            f"Pending         : {report.get('pending') or 'none'}",
        ]
    )
    health = report.get("health") or {}
    counts = health.get("counts") if isinstance(health, dict) else None
    if counts:
        lines.append(f"Metadata counts : {counts}")
    if isinstance(health, dict) and health.get("error"):
        lines.append(f"Health error    : {health['error']}")
    console.print(
        Panel(Text("\n".join(lines), overflow="fold"), title="Status", border_style="blue")
    )


def _cmd_cache(args: argparse.Namespace) -> int:
    config = load_config()
    cache = build_local_cache(config.to_settings())
    if cache is None:
        console.print("Local cache is disabled (AGENT_CACHE_ENABLED=false).")
        return 0
    if args.action == "clear":
        cache.clear()
    stats = cache.stats()
    console.print(
        f"Cache file: {cache.path}\n"
        f"Entries  : {stats['entries']}\n"
        f"Size     : {stats['bytes']} bytes"
    )
    return 0


def _cmd_version(_args: argparse.Namespace) -> int:
    from importlib.metadata import PackageNotFoundError, version

    try:
        console.print(version("querity"))
    except PackageNotFoundError:
        console.print("unknown")
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from querity.benchmarks.runner import run_benchmark

    try:
        run_benchmark(
            golden_path=args.golden_path,
            output_dir=args.output_dir,
            limit=args.limit,
            min_success_rate=args.min_success_rate,
            min_table_recall=args.min_table_recall,
            min_guard_pass_rate=args.min_guard_pass_rate,
            max_mean_latency_ms=args.max_mean_latency_ms,
            min_exact_match_rate=args.min_exact_match_rate,
            min_entity_recall=args.min_entity_recall,
            min_soft_f1=args.min_soft_f1,
            min_value_parity=args.min_value_parity,
            fail_on_threshold=args.fail_on_threshold,
            verbose=True,
        )
        return 0
    except Exception as exc:
        console.print(f"[bold red]Benchmark run failed:[/bold red] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
