from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from querity.agent.service import AgentService, AgentTurnResult
from querity.agent.state import HumanDecision
from querity.cli.config import (
    CliConfig,
    ConfigError,
    ConfigNotFound,
    config_path,
    load_config,
)
from querity.cli.ingest import run_ingestion
from querity.cli.init import run_init
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
    parser.set_defaults(handler=_cmd_chat)

    init = sub.add_parser("init", help="configure the database, API keys and local storage")
    init.add_argument("--force", action="store_true", help="reconfigure without asking")
    init.set_defaults(handler=_cmd_init)

    chat = sub.add_parser("chat", help="open the interactive terminal UI")
    chat.set_defaults(handler=_cmd_chat)

    ask = sub.add_parser("ask", help="ask one question and print the result")
    ask.add_argument("question", nargs="+", help="natural-language question")
    ask.set_defaults(handler=_cmd_ask)

    ingest = sub.add_parser("ingest", help="refresh the catalog")
    ingest.add_argument("--force", action="store_true", help="re-profile every table")
    ingest.add_argument(
        "--table",
        action="append",
        default=[],
        help="limit ingestion to tables matching this glob (repeatable)",
    )
    ingest.add_argument("--dry-run", action="store_true", help="profile without writing")
    ingest.add_argument(
        "--semantic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="generate the semantic layer with the LLM",
    )
    ingest.add_argument(
        "--llm",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable or disable LLM enrichment",
    )
    ingest.add_argument("--refresh-profiles", action="store_true")
    ingest.add_argument(
        "--golden-queries",
        type=Path,
        default=None,
        help="path to golden queries JSON file",
    )
    ingest.set_defaults(handler=_cmd_ingest)

    status = sub.add_parser("status", help="show database, catalog and model status")
    status.add_argument("--json", action="store_true", help="print JSON")
    status.set_defaults(handler=_cmd_status)

    config = sub.add_parser("config", help="show the configuration location and settings")
    config.add_argument("--path", action="store_true", help="print the config file path")
    config.set_defaults(handler=_cmd_config)

    cache = sub.add_parser("cache", help="inspect or clear the local response cache")
    cache.add_argument("action", choices=("stats", "clear"))
    cache.set_defaults(handler=_cmd_cache)

    version = sub.add_parser("version", help="print the version")
    version.set_defaults(handler=_cmd_version)

    benchmark = sub.add_parser("benchmark", help="evaluate latency, accuracy, and stage metrics on real database")
    benchmark.add_argument("--limit", "-n", type=int, default=None, help="limit number of queries to evaluate")
    benchmark.add_argument("--golden-path", "-g", type=str, default="data/golden_queries.json", help="path to golden queries JSON")
    benchmark.add_argument("--output-dir", "-o", type=str, default="benchmarks/runs", help="directory to save run artifacts")
    benchmark.add_argument("--min-success-rate", type=float, default=None, help="minimum execution success rate % (e.g. 90.0)")
    benchmark.add_argument("--min-table-recall", type=float, default=None, help="minimum average table recall % (e.g. 80.0)")
    benchmark.add_argument("--min-guard-pass-rate", type=float, default=None, help="minimum SQL guard pass rate % (e.g. 95.0)")
    benchmark.add_argument("--max-mean-latency-ms", type=float, default=None, help="maximum allowed mean latency in ms (e.g. 40000.0)")
    benchmark.add_argument("--min-exact-match-rate", type=float, default=None, help="minimum exact match rate % (e.g. 70.0)")
    benchmark.add_argument("--min-entity-recall", type=float, default=None, help="minimum entity key recall % (e.g. 80.0)")
    benchmark.add_argument("--min-soft-f1", type=float, default=None, help="minimum soft F1 match rate % (e.g. 70.0)")
    benchmark.add_argument("--min-value-parity", type=float, default=None, help="minimum value parity rate % (e.g. 70.0)")
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


def _cmd_init(args: argparse.Namespace) -> int:
    run_init(force=args.force, console=console)
    return 0


def _cmd_chat(_args: argparse.Namespace) -> int:
    if sys.stdin.isatty() and sys.stdout.isatty():
        from querity.cli.tui import run_tui

        try:
            config = load_config()
        except ConfigNotFound:
            config = None
        run_tui(config)
        return 0
    return _run_line_chat(_load_or_init())


def _cmd_ask(args: argparse.Namespace) -> int:
    config = _load_or_init()
    question = " ".join(args.question).strip()
    service = AgentService(config.to_settings())
    try:
        with console.status("Thinking..."):
            result = service.start_turn(question)
        result = _resolve_interrupts(service, result)
        _print_result(result)
        return 0 if result.status == "completed" else 1
    finally:
        service.close()


def _cmd_ingest(args: argparse.Namespace) -> int:
    config = load_config()
    golden_path = args.golden_queries
    if golden_path is None:
        if config.golden_queries_path.strip():
            golden_path = Path(config.golden_queries_path.strip()).expanduser()
        elif Path("data/golden_queries.json").exists():
            golden_path = Path("data/golden_queries.json")
    summary = run_ingestion(
        config.to_ingestion_settings(),
        console=console,
        force=args.force,
        table_patterns=args.table,
        dry_run=args.dry_run,
        generate_semantic_layer=args.semantic,
        enable_llm=args.llm,
        refresh_profiles=args.refresh_profiles,
        golden_queries_path=golden_path,
    )
    return 0 if summary.status in {"succeeded", "partial"} else 1


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


def _fallback_label(config: CliConfig) -> str:
    models = [config.llm_secondary_model.strip(), config.llm_tertiary_model.strip()]
    unique = list(dict.fromkeys(model for model in models if model))
    return ", ".join(model for model in unique if model != config.llm_model) or "(none)"


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


def _cmd_config(args: argparse.Namespace) -> int:
    if args.path:
        console.print(str(config_path()))
        return 0
    config = load_config()
    saved_databases = [
        str(entry.get("label") or entry.get("url") or "")
        for entry in config.databases
        if isinstance(entry, dict) and (entry.get("label") or entry.get("url"))
    ]
    lines = [
        f"Config file     : {config_path()}",
        f"Database        : {config.masked_url()}",
        f"Saved databases : {len(saved_databases)}",
        *(f"  - {item}" for item in saved_databases),
        f"Source id       : {config.source_id}",
        f"Source schemas  : {config.source_schemas}",
        f"Metadata schema : {config.metadata_schema}",
        f"LLM             : {config.llm_provider}/{config.llm_model}",
        f"Fallbacks       : {_fallback_label(config)}",
        f"Reviewer        : {config.review_model or '(same as LLM)'}",
        f"Embeddings      : {config.embedding_provider}/{config.embedding_model} "
        f"({config.embedding_dimensions}d)",
        f"Semantic layer  : {'generated by LLM' if config.generate_semantic_layer else 'curated only'}",
        "API keys        : stored in the config file, never displayed",
    ]
    console.print(
        Panel(Text("\n".join(lines), overflow="fold"), title="Configuration", border_style="blue")
    )
    return 0


def _cmd_version(_args: argparse.Namespace) -> int:
    from importlib.metadata import PackageNotFoundError, version

    try:
        console.print(version("querity"))
    except PackageNotFoundError:
        console.print("unknown")
    return 0


def _load_or_init() -> CliConfig:
    try:
        return load_config()
    except ConfigNotFound:
        console.print("[yellow]First run: let's configure the API keys and models.[/yellow]")
        return run_init(console=console)


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


def _print_result(result: AgentTurnResult) -> None:
    state = result.state
    execution = state.get("execution")
    if execution is not None and getattr(execution, "ok", False):
        console.print(
            Panel(
                Text(
                    f"Rows: {execution.row_count}  ·  {execution.duration_ms} ms"
                    + ("  (truncated)" if execution.truncated else "")
                ),
                title="Result",
                border_style="green",
            )
        )
        if execution.columns:
            table = Table(show_header=True, header_style="bold cyan")
            for column in execution.columns:
                table.add_column(str(column))
            for row in execution.rows:
                table.add_row(*[Text(str(value)) for value in row])
            console.print(table)
        return
    if result.status == "failed":
        console.print(
            Panel(
                Text(result.error or str(state.get("final_message") or "Failed"), overflow="fold"),
                title="Failed",
                border_style="red",
            )
        )
        return
    if result.status == "rejected":
        console.print(
            Panel(
                Text(str(state.get("final_message") or "Rejected"), overflow="fold"),
                title="Rejected",
                border_style="yellow",
            )
        )
        return
    if result.status in {"awaiting_approval", "awaiting_clarification"}:
        return
    console.print(
        Panel(
            Text(str(state.get("final_message") or "Done."), overflow="fold"),
            title="Assistant",
            border_style="cyan",
        )
    )


def _resolve_interrupts(
    service: AgentService,
    result: AgentTurnResult,
) -> AgentTurnResult:
    while result.status in {"awaiting_clarification", "awaiting_approval"}:
        if not sys.stdin.isatty():
            _print_interrupt(result)
            console.print(
                "[yellow]A pending request must be resolved in an interactive terminal.[/yellow]"
            )
            return result
        if result.status == "awaiting_clarification":
            payload = result.interrupt or {}
            answer = Prompt.ask(str(payload.get("question") or "Clarification"))
            if not answer.strip():
                return result
            result = service.resume(
                result.thread_id,
                HumanDecision(kind="clarification", answer=answer),
            )
            continue
        result = _interactive_approval(service, result)
    return result


def _interactive_approval(
    service: AgentService,
    result: AgentTurnResult,
) -> AgentTurnResult:
    payload = result.interrupt or {}
    _print_interrupt(result)
    can_approve = bool(payload.get("can_approve"))
    action = Prompt.ask(
        "Action",
        choices=["a", "e", "r", "x"],
        default="a" if can_approve else "e",
    )
    if action == "a":
        if not can_approve:
            console.print("[red]This query cannot be approved.[/red]")
            return result
        return service.resume(
            result.thread_id,
            HumanDecision(kind="approve", sql_hash=payload.get("sql_hash")),
        )
    if action == "e":
        edited = Prompt.ask("SQL")
        if not edited.strip():
            return result
        return service.resume(
            result.thread_id,
            HumanDecision(kind="edit", sql=edited, pre_approve=True),
        )
    if action == "r":
        feedback = Prompt.ask("What should change?")
        if not feedback.strip():
            return result
        return service.resume(
            result.thread_id,
            HumanDecision(kind="revise", feedback=feedback),
        )
    return service.resume(
        result.thread_id,
        HumanDecision(kind="reject", reason="Rejected by the user."),
    )


def _print_interrupt(result: AgentTurnResult) -> None:
    payload = result.interrupt or {}
    if result.status == "awaiting_clarification":
        console.print(
            Panel(
                Text(str(payload.get("question") or "Please clarify."), overflow="fold"),
                title="Clarification needed",
                border_style="yellow",
            )
        )
        return
    console.print(
        Panel(
            Text(str(payload.get("question") or result.question), overflow="fold"),
            title="Question",
            border_style="blue",
        )
    )
    sql = str(payload.get("sql") or result.sql or "")
    console.print(
        Panel(Syntax(sql, "sql", theme="monokai", word_wrap=True), title="SQL", border_style="cyan")
    )
    review = payload.get("review") or {}
    if review:
        console.print(
            Panel(
                Text(
                    f"{'Approved' if review.get('approved') else 'Needs changes'}\n"
                    f"{review.get('summary') or ''}",
                    overflow="fold",
                ),
                title="Reviewer",
                border_style="green" if review.get("approved") else "yellow",
            )
        )
    guard = payload.get("guard") or {}
    for issue in guard.get("errors") or []:
        console.print(f"[red]blocking:[/red] {issue.get('message')}")
    for issue in guard.get("warnings") or []:
        console.print(f"[yellow]warning:[/yellow] {issue.get('message')}")
    for issue in review.get("issues") or []:
        console.print(f"[yellow]{issue.get('severity', 'info')}:[/yellow] {issue.get('message')}")


def _run_line_chat(config: CliConfig) -> int:
    service = AgentService(config.to_settings())
    pending: AgentTurnResult | None = None
    mode = "chat"
    console.print(
        Panel(
            Text(
                "Line mode (no TTY). /help for commands. Every query needs approval.",
                overflow="fold",
            ),
            title="Querity",
            border_style="cyan",
        )
    )
    try:
        recovered = service.pending_interrupt()
        if recovered is not None:
            pending = recovered
            mode = "approval" if recovered.status == "awaiting_approval" else "clarification"
            _print_interrupt(recovered)
        while True:
            try:
                prompt = (
                    "SQL> "
                    if mode == "clarification"
                    else ("[a/e/r/x] " if mode == "approval" else "> ")
                )
                value = input(prompt).strip()
            except EOFError:
                console.print()
                break
            if not value:
                continue
            if value.startswith("/"):
                command = value.lower()
                if command in {"/quit", "/exit", "/q"}:
                    break
                if command == "/help":
                    console.print("/help /status /ingest /history /config /clear /quit")
                    continue
                if command == "/status":
                    _print_status(service.status_report())
                    continue
                if command == "/config":
                    _cmd_config(argparse.Namespace(path=False))
                    continue
                if command == "/history":
                    for message in service.chat_store.recent_messages(
                        service.conversation_id, limit=10
                    ):
                        console.print(
                            f"[bold]{message.get('role')}[/bold]: "
                            f"{' '.join(str(message.get('content') or '').split())[:300]}"
                        )
                    continue
                if command == "/clear":
                    console.clear()
                    continue
                if command == "/ingest":
                    if pending is not None:
                        console.print("Resolve the pending request first.")
                        continue
                    run_ingestion(config.to_settings(), console=console)
                    continue
                console.print("Unknown command.")
                continue
            if pending is not None:
                if mode == "clarification":
                    pending = service.resume(
                        pending.thread_id,
                        HumanDecision(kind="clarification", answer=value),
                    )
                else:
                    pending = _apply_line_approval(service, pending, value)
                if pending.status in {"awaiting_approval", "awaiting_clarification"}:
                    _print_interrupt(pending)
                    mode = "approval" if pending.status == "awaiting_approval" else "clarification"
                else:
                    _print_result(pending)
                    pending = None
                    mode = "chat"
                continue
            with console.status("Thinking..."):
                result = service.start_turn(value)
            if result.status in {"awaiting_approval", "awaiting_clarification"}:
                pending = result
                mode = "approval" if result.status == "awaiting_approval" else "clarification"
                _print_interrupt(result)
            else:
                _print_result(result)
    finally:
        service.close()
    return 0


def _apply_line_approval(
    service: AgentService,
    pending: AgentTurnResult,
    value: str,
) -> AgentTurnResult:
    action = value.lower()
    payload = pending.interrupt or {}
    if action in {"a", "approve", "y"}:
        if not payload.get("can_approve"):
            console.print("[red]This query cannot be approved.[/red]")
            return pending
        return service.resume(
            pending.thread_id,
            HumanDecision(kind="approve", sql_hash=payload.get("sql_hash")),
        )
    if action in {"e", "edit"}:
        edited = Prompt.ask("SQL")
        return service.resume(
            pending.thread_id,
            HumanDecision(kind="edit", sql=edited, pre_approve=True),
        )
    if action in {"r", "revise"}:
        feedback = Prompt.ask("What should change?")
        return service.resume(
            pending.thread_id,
            HumanDecision(kind="revise", feedback=feedback),
        )
    if action in {"x", "reject", "n"}:
        return service.resume(
            pending.thread_id,
            HumanDecision(kind="reject", reason="Rejected by the user."),
        )
    console.print("Choose a/e/r/x.")
    return pending


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
