from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from rich.align import Align
from rich.color import Color
from rich.console import Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import (
    Button,
    Header,
    Input,
    OptionList,
    RichLog,
    Select,
    Static,
    TextArea,
)
from textual.widgets.option_list import Option

from querity.agent.service import AgentService, AgentTurnResult
from querity.agent.state import HumanDecision
from querity.cli.config import (
    CliConfig,
    ConfigError,
    config_path,
    database_label,
    source_id_from_url,
)
from querity.ingestion.settings import database_fingerprint, normalize_database_url

BRAND_ACCENT = "#e6b566"
BRAND_PRIMARY = "#8b8b93"
BRAND_SECONDARY = "#f4d79f"
BRAND_SUCCESS = "#7ccf9b"
BRAND_WARNING = "#e8c17a"
BRAND_ERROR = "#f08a8a"
BRAND_FOREGROUND = "#ffffff"
BRAND_MUTED = "#f3ecdf"
BRAND_BACKGROUND = "#000000"
BRAND_SURFACE = "#000000"
BRAND_PANEL = "#0b0b0b"

QUERITY_THEME = Theme(
    name="querity",
    primary=BRAND_PRIMARY,
    secondary=BRAND_SECONDARY,
    accent=BRAND_ACCENT,
    success=BRAND_SUCCESS,
    warning=BRAND_WARNING,
    error=BRAND_ERROR,
    foreground=BRAND_FOREGROUND,
    background=BRAND_BACKGROUND,
    surface=BRAND_SURFACE,
    panel=BRAND_PANEL,
    dark=True,
)

_DEFAULT_HINT = "enter send  ·  / commands  ·  ctrl+c quit"


def _blend_hex(start: str, end: str, ratio: float) -> str:
    first = Color.parse(start).get_truecolor()
    second = Color.parse(end).get_truecolor()
    channels = (
        round(first.red + (second.red - first.red) * ratio),
        round(first.green + (second.green - first.green) * ratio),
        round(first.blue + (second.blue - first.blue) * ratio),
    )
    return "#{:02x}{:02x}{:02x}".format(*channels)


def _brand_wordmark(value: str = "QUERITY") -> Text:
    wordmark = Text()
    steps = max(len(value) - 1, 1)
    for index, character in enumerate(value):
        color = _blend_hex(BRAND_ACCENT, BRAND_SECONDARY, index / steps)
        wordmark.append(character, style=f"bold {color}")
    return wordmark


def _brand_line() -> Text:
    line = Text()
    line.append("◈ ", style=f"bold {BRAND_ACCENT}")
    line.append_text(_brand_wordmark())
    return line


_LOGO_HEIGHT = 5

_LOGO_GLYPHS: dict[str, tuple[str, ...]] = {
    "Q": (
        ".#####.",
        "##...##",
        "##...##",
        "##..###",
        ".#####.",
    ),
    "U": (
        "##...##",
        "##...##",
        "##...##",
        "##...##",
        ".#####.",
    ),
    "E": (
        "#######",
        "##.....",
        "######.",
        "##.....",
        "#######",
    ),
    "R": (
        "######.",
        "##...##",
        "######.",
        "##..##.",
        "##...##",
    ),
    "I": (
        "#######",
        "...#...",
        "...#...",
        "...#...",
        "#######",
    ),
    "T": (
        "#######",
        "...#...",
        "...#...",
        "...#...",
        "...#...",
    ),
    "Y": (
        "##...##",
        "##...##",
        ".#####.",
        "...#...",
        "...#...",
    ),
}


def _logo_lines(value: str = "QUERITY") -> list[str]:
    rows = ["" for _ in range(_LOGO_HEIGHT)]
    for index, letter in enumerate(value):
        glyph = _LOGO_GLYPHS.get(letter)
        if glyph is None:
            continue
        if index:
            rows = [row + " " for row in rows]
        rows = [row + pattern for row, pattern in zip(rows, glyph, strict=True)]
    return rows


def _hero_logo(value: str = "QUERITY") -> Text:
    rows = _logo_lines(value)
    line_width = max((len(row) for row in rows), default=0)
    logo = Text()
    steps = max(line_width - 1, 1)
    for row_index, row in enumerate(rows):
        for column, cell in enumerate(row):
            if cell == "#":
                color = _blend_hex(BRAND_ACCENT, BRAND_FOREGROUND, column / steps)
                logo.append(" ", style=f"on {color}")
            else:
                logo.append(" ")
        logo.append("█", style=f"{BRAND_BACKGROUND} on {BRAND_BACKGROUND}")
        if row_index < len(rows) - 1:
            logo.append("\n")
    return logo


def _empty_state() -> Group:
    state = Text(justify="center")
    state.append("\n")
    state.append_text(_hero_logo())
    state.append("\n\n")
    state.append("guarded, human-in-the-loop text-to-SQL", style=f"italic {BRAND_MUTED}")
    state.append("\n\n")
    state.append(
        "Ask a question in plain language. Querity drafts SQL, validates it\n"
        "as read-only, and waits for your approval before anything runs.",
        style=BRAND_FOREGROUND,
    )
    state.append("\n\n")

    hints = Table.grid(padding=(0, 4))
    hints.add_column(style=f"bold {BRAND_FOREGROUND}", no_wrap=True)
    hints.add_column(style=BRAND_MUTED, no_wrap=True)
    hints.add_column(style=f"bold {BRAND_FOREGROUND}", no_wrap=True)
    hints.add_column(style=BRAND_MUTED, no_wrap=True)
    hints.add_row("/newchat", "new conversation", "/database", "switch or connect")
    hints.add_row("/help", "all commands", "/status", "catalog and model status")
    return Group(state, Align(hints, "center"))


def _welcome_panel(*, show_logo: bool = True) -> Panel | Group:
    if show_logo:
        return _empty_state()

    brand = Text()
    brand.append_text(_brand_line())
    brand.append("\n")
    brand.append("guarded, human-in-the-loop text-to-SQL", style=f"italic {BRAND_MUTED}")

    body = Text()
    body.append(
        "Ask a question in plain language. Querity maps your schema, drafts SQL,\n"
        "validates it as read-only, and waits for your approval before anything runs.\n",
        style=BRAND_FOREGROUND,
    )
    body.append("\n")
    body.append("/newchat", style=f"bold {BRAND_FOREGROUND}")
    body.append("  new conversation", style=BRAND_MUTED)
    body.append("        ")
    body.append("/database", style=f"bold {BRAND_FOREGROUND}")
    body.append("  switch or connect", style=BRAND_MUTED)
    body.append("\n")
    body.append("/help", style=f"bold {BRAND_FOREGROUND}")
    body.append("     all commands", style=BRAND_MUTED)
    body.append("            ")
    body.append("/status", style=f"bold {BRAND_FOREGROUND}")
    body.append("    catalog and model status", style=BRAND_MUTED)
    body.append("\n\n")
    body.append(
        "Each database is indexed once; chats and history stay scoped to that database.",
        style=f"italic {BRAND_MUTED}",
    )

    return Panel(
        Text.assemble(brand, "\n\n", body),
        border_style=BRAND_PRIMARY,
        padding=(0, 2),
    )


def _status_text(value: str) -> Text:
    lowered = value.lower()
    if lowered.startswith("ready"):
        color = BRAND_SUCCESS
    elif any(word in lowered for word in ("fail", "error", "invalid", "could not")):
        color = BRAND_ERROR
    elif lowered.startswith("working") or lowered.startswith("indexing"):
        color = BRAND_ACCENT
    elif any(
        word in lowered
        for word in ("awaiting", "wait", "choose", "no database", "resolve", "select")
    ):
        color = BRAND_WARNING
    else:
        color = BRAND_MUTED
    text = Text()
    text.append("● ", style=f"bold {color}")
    text.append(value, style=color)
    return text


class ConfirmScreen(ModalScreen[bool]):
    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(Text(self.message), id="dialog-title")
            with Horizontal(id="dialog-buttons"):
                yield Button("Yes", variant="error", id="yes")
                yield Button("Cancel", variant="default", id="no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_cancel(self) -> None:
        self.dismiss(False)


class FeedbackScreen(ModalScreen[str | None]):
    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Static(
                Text("What should the model change? The answer becomes revision feedback."),
                id="dialog-title",
            )
            yield Input(placeholder="e.g. use net revenue and limit to last 30 days", id="feedback")
            with Horizontal(id="dialog-buttons"):
                yield Button("Revise", variant="primary", id="save")
                yield Button("Cancel", variant="default", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#feedback", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        self.dismiss(value or None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            value = self.query_one("#feedback", Input).value.strip()
            self.dismiss(value or None)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SQLEditScreen(ModalScreen[str | None]):
    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, sql: str) -> None:
        super().__init__()
        self.sql = sql

    def compose(self) -> ComposeResult:
        with Vertical(id="sql-dialog"):
            yield Static(Text("Edit SQL"), id="dialog-title")
            yield TextArea(
                self.sql,
                language="sql",
                theme="monokai",
                show_line_numbers=True,
                soft_wrap=True,
                id="sql-editor",
            )
            with Horizontal(id="dialog-buttons"):
                yield Button("Run edited SQL", variant="primary", id="save")
                yield Button("Cancel", variant="default", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#sql-editor", TextArea).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save":
            value = self.query_one("#sql-editor", TextArea).text.strip()
            self.dismiss(value or None)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


_LLM_DEFAULTS = {
    "google": "gemini-3.5-flash-lite",
    "nvidia": "meta/llama-3.3-70b-instruct",
}
_SECONDARY_DEFAULTS = {
    "google": "gemini-3.1-flash-lite",
    "nvidia": "",
}
_TERTIARY_DEFAULTS = {
    "google": "gemini-2.5-flash-lite",
    "nvidia": "",
}


@dataclass(frozen=True)
class SlashCommand:
    name: str
    description: str
    aliases: tuple[str, ...] = ()


COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("/newchat", "start a fresh conversation on the current database", ("/new",)),
    SlashCommand("/database", "switch database or connect a new one", ("/db", "/switchdb")),
    SlashCommand("/chats", "switch between saved chats", ("/chat",)),
    SlashCommand("/settings", "edit API keys and models"),
    SlashCommand("/status", "show catalog, model and conversation status"),
    SlashCommand("/ingest", "re-index the current database catalog"),
    SlashCommand("/history", "show recent messages and query runs"),
    SlashCommand("/config", "show configuration and saved databases"),
    SlashCommand("/clear", "clear the visible transcript"),
    SlashCommand("/help", "show all commands", ("/h", "/?")),
    SlashCommand("/quit", "exit Querity", ("/exit", "/q")),
)


def _resolve_command(name: str) -> SlashCommand | None:
    for command in COMMANDS:
        if name == command.name or name in command.aliases:
            return command
    return None


def _matching_commands(value: str) -> list[SlashCommand]:
    if not value.startswith("/") or " " in value:
        return []
    prefix = value.lower()
    return [
        command
        for command in COMMANDS
        if command.name.startswith(prefix)
        or any(alias.startswith(prefix) for alias in command.aliases)
    ]


class PromptInput(Input):
    def on_key(self, event: events.Key) -> None:
        handler = getattr(self.app, "handle_prompt_key", None)
        if handler is not None and handler(event):
            event.stop()
            event.prevent_default()


class DatabaseScreen(ModalScreen[tuple[str, str, str, str] | None]):
    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(
        self,
        databases: list[dict[str, Any]],
        current_url: str = "",
        default_golden_path: str = "",
        default_source_schemas: str = "public",
        default_metadata_schema: str = "querity",
    ) -> None:
        super().__init__()
        self.databases = databases
        self.current_url = current_url
        self.default_golden_path = default_golden_path
        self.default_source_schemas = default_source_schemas or "public"
        self.default_metadata_schema = default_metadata_schema or "querity"

    def compose(self) -> ComposeResult:
        with Vertical(id="database-dialog"):
            yield Static(_brand_line(), classes="brand-mark")
            yield Static(
                Text(
                    "Connect or select a database. Ingestion and catalog schemas are "
                    "database-specific and shared across all chats for that database.",
                    overflow="fold",
                ),
                id="database-title",
            )
            with VerticalScroll(id="database-form"):
                options: list[Option] = []
                for entry in self.databases:
                    url = str(entry.get("url") or "")
                    if not url:
                        continue
                    label = str(entry.get("label") or database_label(url))
                    source_id = str(entry.get("source_id") or source_id_from_url(url))
                    marker = "●" if url == self.current_url else "○"
                    extras = []
                    if entry.get("ingested"):
                        tbls = entry.get("table_count")
                        extras.append(f"ingested ({tbls} tables)" if tbls else "ingested")
                    if entry.get("source_schemas"):
                        extras.append(f"schemas: {entry['source_schemas']}")
                    if entry.get("golden_queries_path"):
                        extras.append("goldens linked")
                    extra_tag = f"  [{', '.join(extras)}]" if extras else ""
                    options.append(
                        Option(f"{marker} {label}  ·  source {source_id}{extra_tag}", id=url)
                    )
                if not options:
                    options.append(Option("No saved databases yet.", disabled=True))
                yield Static("Saved databases", classes="field-label")
                yield OptionList(*options, id="database-list")
                yield Static("Or enter a new PostgreSQL URL", classes="field-label")
                yield Input(
                    placeholder="postgresql://user:password@host:5432/database",
                    password=True,
                    id="database-new-url",
                )
                yield Static("Source schemas (comma separated, e.g. public)", classes="field-label")
                initial_source_schemas = self.default_source_schemas
                initial_metadata_schema = self.default_metadata_schema
                initial_golden = self.default_golden_path
                if self.current_url:
                    for entry in self.databases:
                        if entry.get("url") == self.current_url:
                            if entry.get("source_schemas"):
                                initial_source_schemas = str(entry["source_schemas"])
                            if entry.get("metadata_schema"):
                                initial_metadata_schema = str(entry["metadata_schema"])
                            if entry.get("golden_queries_path"):
                                initial_golden = str(entry["golden_queries_path"])
                            break

                yield Input(
                    value=initial_source_schemas,
                    placeholder="public",
                    id="database-source-schemas",
                )
                yield Static("Metadata schema (catalog schema)", classes="field-label")
                yield Input(
                    value=initial_metadata_schema,
                    placeholder="querity",
                    id="database-metadata-schema",
                )
                yield Static(
                    "Golden queries file (optional JSON path for few-shot examples)",
                    classes="field-label",
                )
                yield Input(
                    value=initial_golden,
                    placeholder="e.g. data/golden_queries.json or /path/to/golden_queries.json",
                    id="database-golden-path",
                )
            yield Static("", id="database-error")
            with Horizontal(id="dialog-buttons"):
                yield Button("Connect", variant="primary", id="database-use")
                yield Button("Cancel", variant="default", id="database-cancel")

    def on_mount(self) -> None:
        if self.databases:
            self.query_one("#database-list", OptionList).focus()
        else:
            self.query_one("#database-new-url", Input).focus()

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option and event.option.id:
            url = str(event.option.id)
            for entry in self.databases:
                if entry.get("url") == url:
                    golden = str(
                        entry.get("golden_queries_path")
                        or (self.default_golden_path if url == self.current_url else "")
                    )
                    self.query_one("#database-golden-path", Input).value = golden
                    schemas = str(
                        entry.get("source_schemas")
                        or (self.default_source_schemas if url == self.current_url else "public")
                    )
                    self.query_one("#database-source-schemas", Input).value = schemas
                    meta = str(
                        entry.get("metadata_schema")
                        or (self.default_metadata_schema if url == self.current_url else "querity")
                    )
                    self.query_one("#database-metadata-schema", Input).value = meta
                    break

    def _submit(self, url: str | None = None) -> None:
        selected_url = url
        if not selected_url:
            typed = self.query_one("#database-new-url", Input).value.strip()
            if typed:
                selected_url = typed
            else:
                list_view = self.query_one("#database-list", OptionList)
                if list_view.highlighted is not None:
                    option = list_view.get_option_at_index(list_view.highlighted)
                    if option and option.id:
                        selected_url = str(option.id)
        if not selected_url:
            self.query_one("#database-error", Static).update(
                Text("Select a database from the list or enter a PostgreSQL URL.", style=BRAND_ERROR)
            )
            return
        try:
            normalized_url = normalize_database_url(selected_url)
        except Exception as exc:
            self.query_one("#database-error", Static).update(Text(str(exc), style=BRAND_ERROR))
            return

        source_schemas = (
            self.query_one("#database-source-schemas", Input).value.strip() or "public"
        )
        metadata_schema = (
            self.query_one("#database-metadata-schema", Input).value.strip() or "querity"
        )
        golden_path = self.query_one("#database-golden-path", Input).value.strip()
        if golden_path:
            p = Path(golden_path).expanduser()
            if not p.exists():
                self.query_one("#database-error", Static).update(
                    Text(f"Golden queries file not found: {golden_path}", style=BRAND_ERROR)
                )
                return
        self.dismiss((normalized_url, source_schemas, metadata_schema, golden_path))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id in {
            "database-new-url",
            "database-source-schemas",
            "database-metadata-schema",
            "database-golden-path",
        }:
            self._submit()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id:
            self._submit(str(event.option.id))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "database-use":
            self._submit()
        elif event.button.id == "database-cancel":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SetupHelpScreen(ModalScreen[None]):
    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="help-dialog"):
            yield Static(
                Text(
                    "Which options should I choose?\n\n"
                    "LLM provider: Google Gemini is recommended for SQL generation and "
                    "review. NVIDIA NIM is an alternative using hosted Llama models.\n\n"
                    "Models: keep the prefilled defaults unless you know you need another "
                    "model. Secondary/tertiary models are fallbacks; leave them blank to "
                    "disable fallbacks. The reviewer model defaults to the main LLM.\n\n"
                    "API keys: the Google key is used by Gemini; the NVIDIA key is required "
                    "for NVIDIA Nemotron embeddings (2048d). Keys are stored "
                    "only in your local config file with private permissions.\n\n"
                    "Databases & schemas: databases, source schemas, metadata catalog schemas, "
                    "and golden queries are configured per database when starting a chat with /newchat.\n\n"
                    "Semantic glossary: when enabled, the LLM analyzes table schemas during "
                    "ingestion to generate business metrics, friendly names, and filter definitions "
                    "(e.g., 'active customers', 'total revenue'). This improves SQL accuracy.\n\n"
                    "Index on first chat: when enabled, Querity automatically indexes any newly "
                    "selected database before your first query so it is immediately ready to answer.",
                    overflow="fold",
                ),
                id="help-text",
            )
            with Horizontal(id="dialog-buttons"):
                yield Button("Close", variant="primary", id="help-close")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "help-close":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SetupScreen(ModalScreen[tuple[CliConfig, bool] | None]):
    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, config: CliConfig | None, *, first_run: bool = False) -> None:
        super().__init__()
        self.config = config
        self.first_run = first_run

    def compose(self) -> ComposeResult:
        existing = self.config
        provider = existing.llm_provider if existing else "google"
        with Vertical(id="setup-dialog"):
            yield Static(_brand_line(), classes="brand-mark")
            yield Static(
                Text(
                    "Choose the models and API keys. Databases, schemas, and golden queries "
                    "are configured per chat with /newchat.",
                    overflow="fold",
                ),
                id="setup-title",
            )
            with VerticalScroll(id="setup-form"):
                yield Static(
                    "Google Gemini is recommended for SQL generation; NVIDIA is an "
                    "alternative. The NVIDIA API key is always required for embeddings. "
                    "Press Help for guidance.",
                    classes="field-hint",
                )
                yield Static("LLM provider", classes="field-label")
                yield Select(
                    [("Google Gemini", "google"), ("NVIDIA NIM", "nvidia")],
                    value=provider,
                    allow_blank=False,
                    id="setup-llm-provider",
                )
                yield Static("LLM model", classes="field-label")
                yield Input(
                    value=existing.llm_model if existing else _LLM_DEFAULTS[provider],
                    id="setup-llm-model",
                )
                yield Static("Secondary model (blank = no fallback)", classes="field-label")
                yield Input(
                    value=existing.llm_secondary_model
                    if existing
                    else _SECONDARY_DEFAULTS[provider],
                    id="setup-llm-secondary",
                )
                yield Static("Tertiary model (blank = no fallback)", classes="field-label")
                yield Input(
                    value=existing.llm_tertiary_model if existing else _TERTIARY_DEFAULTS[provider],
                    id="setup-llm-tertiary",
                )
                yield Static("Google API key", classes="field-label")
                yield Input(
                    value=existing.google_api_key if existing else "",
                    password=True,
                    placeholder="Required when using Google Gemini",
                    id="setup-google-key",
                )
                yield Static("NVIDIA API key", classes="field-label")
                yield Input(
                    value=existing.nvidia_api_key if existing else "",
                    password=True,
                    placeholder="Required for NVIDIA embeddings",
                    id="setup-nvidia-key",
                )
            yield Static("", id="setup-error")
            with Horizontal(id="dialog-buttons"):
                yield Button("Help", variant="default", id="setup-help")
                yield Button(
                    "Save",
                    variant="primary",
                    id="setup-save",
                )
                yield Button(
                    "Quit" if self.first_run or existing is None else "Cancel",
                    variant="default",
                    id="setup-cancel",
                )

    def on_mount(self) -> None:
        self.query_one("#setup-llm-provider", Select).focus()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "setup-llm-provider":
            return
        provider = str(event.value)
        model = self.query_one("#setup-llm-model", Input)
        current = model.value.strip()
        if current in {"", *_LLM_DEFAULTS.values()}:
            model.value = _LLM_DEFAULTS.get(provider, "")
        secondary = self.query_one("#setup-llm-secondary", Input)
        if secondary.value.strip() in {"", *_SECONDARY_DEFAULTS.values()}:
            secondary.value = _SECONDARY_DEFAULTS.get(provider, "")
        tertiary = self.query_one("#setup-llm-tertiary", Input)
        if tertiary.value.strip() in {"", *_TERTIARY_DEFAULTS.values()}:
            tertiary.value = _TERTIARY_DEFAULTS.get(provider, "")

    def _value(self, identifier: str) -> str:
        return self.query_one(f"#{identifier}", Input).value.strip()

    def _build_config(self) -> CliConfig:
        existing = self.config
        provider = str(self.query_one("#setup-llm-provider", Select).value or "google")
        source_schemas = existing.source_schemas if existing else "public"
        metadata_schema = existing.metadata_schema if existing else "querity"
        llm_model = self._value("setup-llm-model") or _LLM_DEFAULTS[provider]
        secondary = self._value("setup-llm-secondary")
        tertiary = self._value("setup-llm-tertiary")
        review_model = ""
        google_key = self._value("setup-google-key") or (
            existing.google_api_key if existing else ""
        )
        nvidia_key = self._value("setup-nvidia-key") or (
            existing.nvidia_api_key if existing else ""
        )
        if provider == "google" and not google_key:
            raise ConfigError("A Google API key is required for the selected LLM provider.")
        if not nvidia_key:
            raise ConfigError("An NVIDIA API key is required for embeddings.")

        embedding_model = "nvidia/nemotron-3-embed-1b"
        embedding_dimensions = 2048

        config = CliConfig(
            database_url=existing.database_url if existing else "",
            source_id=existing.source_id if existing else "default",
            databases=existing.databases if existing else [],
            source_schemas=source_schemas,
            metadata_schema=metadata_schema,
            llm_provider=provider,
            llm_model=llm_model,
            llm_secondary_model=secondary,
            llm_tertiary_model=tertiary,
            review_model=review_model,
            google_api_key=google_key,
            nvidia_api_key=nvidia_key,
            embedding_provider="nvidia",
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            generate_semantic_layer=True,
            golden_queries_path=existing.golden_queries_path if existing else "",
            statement_timeout_ms=existing.statement_timeout_ms if existing else 15_000,
            agent_statement_timeout_ms=(
                existing.agent_statement_timeout_ms if existing else 15_000
            ),
            max_result_rows=existing.max_result_rows if existing else 500,
            max_revisions=existing.max_revisions if existing else 2,
        )
        try:
            settings = config.to_settings(require_database=False)
            _ = settings.schemas
        except Exception as exc:
            raise ConfigError(f"Invalid configuration: {exc}") from exc
        return config

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "setup-help":
            self.app.push_screen(SetupHelpScreen())
            return
        if event.button.id == "setup-cancel":
            self.dismiss(None)
            return
        if event.button.id != "setup-save":
            return
        try:
            config = self._build_config()
        except Exception as exc:
            self.query_one("#setup-error", Static).update(Text(str(exc), style=BRAND_ERROR))
            return
        self.dismiss((config, True))

    def action_cancel(self) -> None:
        self.dismiss(None)


class ChatsScreen(ModalScreen[str | None]):
    BINDINGS: ClassVar[list[Binding]] = [Binding("escape", "cancel", "Cancel")]

    def __init__(
        self,
        conversations: list[dict[str, Any]],
        current_id: str,
    ) -> None:
        super().__init__()
        self.conversations = conversations
        self.current_id = current_id

    def compose(self) -> ComposeResult:
        with Vertical(id="chats-dialog"):
            yield Static(_brand_line(), classes="brand-mark")
            yield Static(
                Text("Choose a saved chat, or start a new one."),
                id="chats-title",
            )
            options: list[Option] = []
            for conversation in self.conversations:
                conversation_id = str(conversation["conversation_id"])
                title = str(conversation.get("title") or "Data chat")
                updated = str(conversation.get("updated_at") or "")[:16].replace("T", " ")
                marker = "●" if conversation_id == self.current_id else "○"
                database = database_label(str(conversation.get("database_url") or ""))
                options.append(
                    Option(
                        f"{marker} {title[:55]}   {database}   {updated}",
                        id=conversation_id,
                    )
                )
            if not options:
                options.append(Option("No saved chats yet.", disabled=True))
            yield OptionList(*options, id="chats-list")
            with Horizontal(id="dialog-buttons"):
                yield Button("New chat", variant="primary", id="chats-new")
                yield Button("Switch database", variant="default", id="chats-db")
                yield Button("Close", variant="default", id="chats-close")

    def on_mount(self) -> None:
        self.query_one("#chats-list", OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "chats-new":
            self.dismiss("__new__")
        elif event.button.id == "chats-db":
            self.dismiss("__switch_db__")
        elif event.button.id == "chats-close":
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class QuerityTUI(App[None]):
    TITLE = "Querity"
    CSS = """
    Screen {
        background: $background;
    }
    Header {
        background: $panel;
        color: $foreground;
    }
    HeaderIcon {
        color: $accent;
    }
    HeaderClock {
        color: $foreground 70%;
        background: transparent;
    }
    #body {
        height: 1fr;
        width: 100%;
        padding: 0 1;
    }
    #transcript {
        height: 1fr;
        width: 100%;
        border: round $primary 40%;
        border-title-color: $foreground;
        border-title-align: left;
        border-subtitle-color: $foreground 75%;
        border-subtitle-align: right;
        padding: 0 1;
        scrollbar-size: 1 1;
        scrollbar-background: $panel;
        scrollbar-color: $primary 45%;
        scrollbar-color-hover: $accent 70%;
        scrollbar-color-active: $accent;
    }
    #transcript:focus {
        border: round $primary 55%;
    }
    #status-bar {
        height: 1;
        width: 100%;
        padding: 0 1;
        background: $panel;
    }
    #status {
        width: 1fr;
        height: 1;
        content-align: left middle;
    }
    #hints {
        width: auto;
        height: 1;
        color: $foreground 75%;
        text-align: right;
        text-wrap: nowrap;
        text-overflow: ellipsis;
        content-align: right middle;
    }
    #command-menu {
        display: none;
        height: auto;
        max-height: 40%;
        min-height: 3;
        margin: 0 1;
        padding: 0 1;
        background: $panel;
        border: round $primary 50%;
        border-title-color: $foreground;
        border-title-align: left;
        border-subtitle-color: $foreground 75%;
        border-subtitle-align: right;
        scrollbar-background: $panel;
        scrollbar-color: $primary 45%;
    }
    #command-menu > .option-list--option-highlighted,
    #command-menu:focus > .option-list--option-highlighted {
        background: $accent 18%;
        color: $foreground;
    }
    #prompt {
        height: 3;
        width: 100%;
        background: $surface;
        border: round $accent;
        border-title-color: $accent;
        border-title-align: left;
    }
    #prompt:focus {
        border: round $accent;
        background-tint: $accent 4%;
    }
    #prompt:disabled {
        border: round $primary 20%;
        opacity: 70%;
    }
    Input, Select, TextArea {
        background: $panel;
        border: tall $primary 30%;
    }
    Input:focus, Select:focus, TextArea:focus {
        border: tall $accent;
    }
    Input > .input--placeholder {
        color: $foreground 55%;
    }
    Button {
        min-width: 12;
    }
    Button:focus {
        border: tall $accent;
    }
    #dialog, #sql-dialog {
        width: 86%;
        max-width: 120;
        background: $surface;
        border: thick $primary 80%;
        padding: 1 2;
    }
    #sql-dialog {
        width: 92%;
        max-width: 140;
        height: 82%;
    }
    .brand-mark {
        height: auto;
        padding: 0 0 1 0;
    }
    #setup-form, #database-form, #database-list, #chats-list {
        scrollbar-background: $surface;
        scrollbar-color: $primary 45%;
        scrollbar-color-hover: $accent 70%;
        scrollbar-color-active: $accent;
    }
    #setup-dialog {
        width: 90%;
        max-width: 110;
        height: 90%;
        background: $surface;
        border: thick $primary 80%;
        padding: 1 2;
    }
    #setup-title {
        padding: 0 0 1 0;
        text-style: bold;
    }
    #setup-form {
        height: 1fr;
    }
    .field-label {
        margin-top: 1;
        color: $foreground;
        text-style: bold;
    }
    .field-hint {
        margin-top: 1;
        color: $foreground;
        text-style: italic;
    }
    #setup-error, #database-error {
        height: auto;
        min-height: 1;
        padding-top: 1;
    }
    #database-dialog {
        width: 88%;
        max-width: 110;
        height: 88%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }
    #database-form {
        height: 1fr;
    }
    #help-dialog {
        width: 84%;
        max-width: 100;
        height: auto;
        max-height: 85%;
        background: $surface;
        border: thick $primary 80%;
        padding: 1 2;
    }
    #database-title {
        padding: 0 0 1 0;
        text-style: bold;
    }
    #database-list {
        height: auto;
        max-height: 7;
        border: round $primary 35%;
        background: $panel;
    }
    #help-text {
        height: auto;
        padding-bottom: 1;
    }
    #chats-dialog {
        width: 86%;
        max-width: 110;
        height: 80%;
        background: $surface;
        border: thick $primary 80%;
        padding: 1 2;
    }
    #chats-title {
        padding: 0 0 1 0;
        text-style: bold;
    }
    #chats-list {
        height: 1fr;
        border: round $primary 35%;
        background: $panel;
    }
    #dialog-title {
        padding: 0 0 1 0;
        text-style: bold;
    }
    #dialog-buttons {
        height: auto;
        align-horizontal: right;
        padding-top: 1;
    }
    #dialog-buttons Button {
        margin-left: 1;
    }
    #sql-editor {
        height: 1fr;
    }
    ConfirmScreen, FeedbackScreen, SQLEditScreen, SetupScreen, ChatsScreen,
    DatabaseScreen, SetupHelpScreen {
        align: center middle;
    }
    """
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    def __init__(self, config: CliConfig | None = None) -> None:
        super().__init__()
        self.register_theme(QUERITY_THEME)
        self.theme = QUERITY_THEME.name
        self.config = config
        self.service: AgentService | None = None
        self.pending: AgentTurnResult | None = None
        self.mode = "chat"
        self.busy = False
        self.fatal = False
        self._menu_suppressed_value: str | None = None
        self._pending_ingest = False
        self._redraw_scheduled = False
        if config is not None and config.database_url:
            self.sub_title = f"{config.source_id} · {database_label(config.database_url)}"
        else:
            self.sub_title = "Setup" if config is None else "No database selected"

    def format_title(self, title: str, sub_title: str) -> Text:
        line = Text()
        line.append("◈ ", style=f"bold {BRAND_ACCENT}")
        line.append_text(_brand_wordmark(title.upper() or "QUERITY"))
        if sub_title:
            line.append("   ")
            line.append(sub_title, style=BRAND_MUTED)
        return line

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True, icon="◆", time_format="%H:%M")
        with Vertical(id="body"):
            yield RichLog(
                id="transcript",
                wrap=True,
                highlight=True,
                markup=False,
                auto_scroll=True,
            )
            with Horizontal(id="status-bar"):
                yield Static("", id="status")
                yield Static(_DEFAULT_HINT, id="hints")
            yield OptionList(id="command-menu")
            yield PromptInput(
                placeholder="Ask a question about your data, or type / for commands",
                id="prompt",
            )

    def on_mount(self) -> None:
        with contextlib.suppress(Exception):
            transcript = self.query_one("#transcript", RichLog)
            transcript.border_title = "Conversation"
            transcript.border_subtitle = "read-only · approval required"
            commands = self.query_one("#command-menu", OptionList)
            commands.border_title = "Commands"
            commands.border_subtitle = "enter to run"
        self._sync_prompt_chrome()
        if self.config is None:
            self._write_welcome()
            self._set_status("Choose models and API keys to start")
            self._disable_input()
            self.push_screen(
                SetupScreen(None, first_run=True),
                self._setup_done,
            )
            return
        url = self.config.database_url or self._first_saved_database()
        if not url:
            self._write_welcome()
            self._set_status("No database selected · type /database")
            self._enable_input()
            return
        self._connect_database(self.config, url)

    def on_resize(self, event: events.Resize) -> None:
        self._sync_prompt_chrome(width=event.size.width)
        if self.busy or self._redraw_scheduled:
            return
        self._redraw_scheduled = True
        with contextlib.suppress(Exception):
            self.call_after_refresh(self._redraw_transcript)

    def _redraw_transcript(self) -> None:
        self._redraw_scheduled = False
        with contextlib.suppress(Exception):
            transcript = self.query_one("#transcript", RichLog)
            width = transcript.size.width or self.size.width
            transcript.clear()
            if self.service is None:
                self._write_welcome(show_logo=(width >= 70))
                return
            stats = self.service.chat_store.stats(self.service.conversation_id)
            messages_count = int(stats.get("messages") or 0)
            if messages_count == 0:
                self._write_welcome(show_logo=(width >= 70))
                title = str(self.service.conversation.get("title") or "Data chat")
                db_lbl = database_label(self.service.settings.source_database_url or "")
                self._write(
                    Text(
                        f"Connected to {db_lbl}  ·  {title}  ·  "
                        f"{stats.get('messages', 0)} messages · {stats.get('query_runs', 0)} queries"
                    )
                )
                self._show_database_status()
            else:
                self._render_conversation_history(self.service.conversation_id)

            if self.pending is not None:
                if self.pending.status == "awaiting_approval":
                    self._show_approval(self.pending)
                elif self.pending.status == "awaiting_clarification":
                    payload = self.pending.interrupt or {}
                    self._write(
                        Panel(
                            Text(
                                str(payload.get("question") or "Please clarify the request."),
                                overflow="fold",
                            ),
                            title="Clarification needed",
                            border_style=BRAND_WARNING,
                        )
                    )

    def _clear_transcript(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#transcript", RichLog).clear()

    def _write_welcome(self, *, show_logo: bool = True) -> None:
        if show_logo:
            with contextlib.suppress(Exception):
                width = self.query_one("#transcript", RichLog).size.width
                if width and width < 70:
                    show_logo = False
        self._write(_welcome_panel(show_logo=show_logo), expand=True)

    def _first_saved_database(self) -> str:
        if self.config is None:
            return ""
        for entry in self.config.databases:
            url = str(entry.get("url") or "") if isinstance(entry, dict) else ""
            if url:
                return url
        return ""

    def _database_entry(self, url: str) -> dict[str, str]:
        if self.config is None:
            return {}
        for entry in self.config.databases:
            entry_url = str(entry.get("url") or "") if isinstance(entry, dict) else ""
            if not entry_url:
                continue
            with contextlib.suppress(Exception):
                if normalize_database_url(entry_url) == normalize_database_url(url):
                    return entry
        return {}

    def _url_for_fingerprint(self, fingerprint: str) -> str:
        if (
            fingerprint
            and self.service is not None
            and self.service.settings.db_fingerprint == fingerprint
        ):
            return self.service.settings.source_url
        if self.config is None:
            return ""
        for entry in self.config.databases:
            url = str(entry.get("url") or "") if isinstance(entry, dict) else ""
            if not url:
                continue
            with contextlib.suppress(Exception):
                if database_fingerprint(url) == fingerprint:
                    return url
        return ""

    def _save_config(self) -> None:
        if self.config is None:
            return
        try:
            self.config.save()
        except Exception as exc:
            self._write(
                Text(f"Could not save configuration: {type(exc).__name__}: {exc}", style=BRAND_WARNING)
            )

    def _setup_done(self, result: tuple[CliConfig, bool] | None) -> None:
        if result is None:
            if self.service is None:
                self.exit()
            else:
                self._enable_input()
            return
        config, run_ingest = result
        try:
            saved = config.save()
        except Exception as exc:
            self._write(
                Panel(
                    Text(f"{type(exc).__name__}: {exc}", overflow="fold"),
                    title="Could not save settings",
                    border_style=BRAND_ERROR,
                )
            )
            self.push_screen(
                SetupScreen(config, first_run=self.config is None),
                self._setup_done,
            )
            return
        current_url = self.service.settings.source_url if self.service is not None else ""
        self.config = config
        self.fatal = False
        self._write(Text(f"Configuration saved to {saved}"))
        self._clear_transcript()
        self._pending_ingest = run_ingest
        url = current_url or config.database_url or self._first_saved_database()
        if url:
            self._connect_database(config, url, ingest=run_ingest)
            self._pending_ingest = False
            return
        self._write_welcome()
        self._set_status("Choose a database to start")
        self._enable_input()
        self.action_select_database()

    def _connect_database(
        self,
        config: CliConfig,
        url: str,
        *,
        ingest: bool = False,
        new_chat: bool = False,
        conversation_id: str | None = None,
        source_schemas: str | None = None,
        metadata_schema: str | None = None,
        golden_queries_path: str | None = None,
    ) -> bool:
        try:
            normalized = normalize_database_url(url)
        except Exception as exc:
            self._set_status("Invalid database URL")
            self._write(Text(str(exc), style=BRAND_ERROR))
            self._enable_input()
            return False
        entry = self._database_entry(normalized)
        source_id = str(entry.get("source_id") or "")
        current_url = ""
        if config.database_url:
            with contextlib.suppress(Exception):
                current_url = normalize_database_url(config.database_url)
        if not source_id:
            if normalized == current_url and config.source_id:
                source_id = config.source_id
            else:
                source_id = source_id_from_url(normalized)
        resolved_source_schemas = (
            source_schemas
            if source_schemas is not None
            else (entry.get("source_schemas") or config.source_schemas or "public")
        )
        resolved_metadata_schema = (
            metadata_schema
            if metadata_schema is not None
            else (entry.get("metadata_schema") or config.metadata_schema or "querity")
        )
        if golden_queries_path is not None:
            config.golden_queries_path = golden_queries_path
        elif entry.get("golden_queries_path"):
            config.golden_queries_path = str(entry["golden_queries_path"])
        config.source_schemas = resolved_source_schemas
        config.metadata_schema = resolved_metadata_schema
        try:
            settings = config.to_settings(
                database_url=normalized,
                source_id=source_id,
                source_schemas=resolved_source_schemas,
                metadata_schema=resolved_metadata_schema,
            )
            service = AgentService(settings)
            service.check_connection()
            existing_source = None
            with contextlib.suppress(Exception):
                existing_source = service.metadata_store.primary_source_id()
            if existing_source and existing_source != source_id:
                source_id = existing_source
                config.source_id = source_id
                settings = config.to_settings(
                    database_url=normalized,
                    source_id=source_id,
                    source_schemas=resolved_source_schemas,
                    metadata_schema=resolved_metadata_schema,
                )
                service = AgentService(settings)
        except Exception as exc:
            self.fatal = self.service is None
            self._set_status("Could not connect · type /newchat")
            self._write_welcome()
            self._write(
                Panel(
                    Text(f"{type(exc).__name__}: {exc}", overflow="fold"),
                    title="Could not connect",
                    border_style=BRAND_ERROR,
                )
            )
            self._enable_input()
            return False
        config.database_url = normalized
        config.source_id = source_id
        self.config = config
        self.sub_title = f"{source_id} · {database_label(normalized)}"
        self.pending = None
        self.mode = "chat"
        self.busy = False
        self.fatal = False
        self._close_service()
        self.service = service
        try:
            if new_chat:
                service.new_conversation()
            elif conversation_id:
                service.open_conversation(conversation_id)
        except Exception as exc:
            self._write(Text(f"{type(exc).__name__}: {exc}", style=BRAND_ERROR))
            self._enable_input()
            return False
        config.remember_database(
            normalized,
            source_id=source_id,
            source_schemas=resolved_source_schemas,
            metadata_schema=resolved_metadata_schema,
            golden_queries_path=config.golden_queries_path,
        )
        self._save_config()
        stats = service.chat_store.stats(service.conversation_id)
        messages = int(stats.get("messages") or 0)
        self._write_welcome(show_logo=messages == 0)
        title = str(service.conversation.get("title") or "Data chat")
        self._write(
            Text(
                f"Connected to {database_label(normalized)}  ·  {title}  ·  "
                f"{stats['messages']} messages · {stats['query_runs']} queries"
            )
        )
        self._show_database_status()
        if not new_chat and service.conversation_id:
            self._render_conversation_history(service.conversation_id)
        pending = service.pending_interrupt()
        if pending is not None:
            self._write(
                Panel(
                    Text(
                        "A previous request is still waiting for your decision.",
                        overflow="fold",
                    ),
                    title="Recovered session",
                    border_style=BRAND_WARNING,
                )
            )
            self._handle_turn_result(pending)
            return True
        status_info = service.ingestion_status()
        is_already_ingested = bool(status_info.get("ingested")) or bool(entry.get("ingested"))
        needs_ingest = not is_already_ingested

        if is_already_ingested and self.config is not None:
            self.config.mark_database_ingested(
                normalized,
                ingested=True,
                table_count=int(status_info.get("tables") or entry.get("table_count") or 0),
            )
            self._save_config()

        if needs_ingest:
            self._write(
                Text(
                    f"Database '{database_label(normalized)}' has not been indexed yet. Starting initial ingestion...",
                    style=BRAND_ACCENT,
                )
            )
            self._start_ingest()
        else:
            self._set_status("Ready")
            self._enable_input()
        return True

    def _show_database_status(self) -> None:
        if self.service is None:
            return
        status = self.service.ingestion_status()
        last = status.get("last_run") or {}
        lines = [
            f"Database : {database_label(self.service.settings.source_url)}",
            f"Source ID: {self.service.settings.source_id}",
            f"Schemas  : {self.service.settings.source_schemas}",
            f"Metadata : {self.service.settings.metadata_schema}",
        ]
        if self.config and self.config.golden_queries_path:
            lines.append(f"Goldens  : {self.config.golden_queries_path}")
        if status.get("error"):
            lines.append(f"Catalog  : unavailable ({status['error']})")
            lines.append("Hint     : run /ingest once the metadata schema is reachable.")
        elif status.get("ingested"):
            when = _format_timestamp(last.get("finished_at") or last.get("started_at"))
            run_status = str(last.get("status") or "unknown")
            lines.append(f"Catalog  : ingested · {status.get('tables', 0)} tables")
            lines.append(f"Last run : {run_status} at {when}")
        else:
            lines.append("Catalog  : NOT ingested yet")
            lines.append("Hint     : type /ingest to index this database before querying.")
        border = BRAND_SUCCESS if status.get("ingested") and not status.get("error") else BRAND_WARNING
        self._write(
            Panel(
                Text("\n".join(lines), overflow="fold"),
                title="Chat database",
                border_style=border,
            )
        )

    def _close_service(self) -> None:
        if self.service is not None:
            with contextlib.suppress(Exception):
                self.service.close()
            self.service = None

    def on_unmount(self) -> None:
        self._close_service()

    def _modal_open(self) -> bool:
        return isinstance(self.screen, ModalScreen)

    def action_edit_settings(self) -> None:
        if self._modal_open():
            return
        if self.busy:
            self._write(Text("Wait for the current operation to finish.", style=BRAND_WARNING))
            return
        if self.pending is not None:
            self._write(
                Text("Resolve the pending request before changing settings.", style=BRAND_WARNING)
            )
            return
        self.push_screen(
            SetupScreen(self.config, first_run=self.config is None),
            self._setup_done,
        )

    def action_new_chat(self) -> None:
        if self._modal_open():
            return
        if self.busy:
            self._write(Text("Wait for the current operation to finish.", style=BRAND_WARNING))
            return
        if self.pending is not None:
            self._write(
                Text("Resolve the pending request before starting a new chat.", style=BRAND_WARNING)
            )
            return
        if self.config is None:
            self._write(Text("Finish setup before starting a chat.", style=BRAND_WARNING))
            return
        if self.service is not None:
            try:
                self.service.new_conversation()
            except Exception as exc:
                self._write(Text(f"Failed to create new chat: {exc}", style=BRAND_ERROR))
                self._enable_input()
                return
            self._clear_transcript()
            self._write_welcome()
            self._show_database_status()
            self._write(
                Text(
                    f"Started new chat on {database_label(self.service.settings.source_url)}",
                    style=f"bold {BRAND_SUCCESS}",
                )
            )
            self._set_status("Ready")
            self._enable_input()
            return

        self.action_select_database()

    def action_select_database(self) -> None:
        if self._modal_open():
            return
        if self.busy:
            self._write(Text("Wait for the current operation to finish.", style=BRAND_WARNING))
            return
        if self.pending is not None:
            self._write(
                Text("Resolve the pending request before switching databases.", style=BRAND_WARNING)
            )
            return
        if self.config is None:
            self._write(Text("Finish setup before choosing a database.", style=BRAND_WARNING))
            return
        current_url = self.service.settings.source_url if self.service is not None else ""
        default_golden = ""
        default_source_schemas = "public"
        default_metadata_schema = "querity"
        if current_url:
            for entry in self.config.databases:
                if entry.get("url") == current_url:
                    default_golden = str(entry.get("golden_queries_path") or "")
                    default_source_schemas = str(entry.get("source_schemas") or "public")
                    default_metadata_schema = str(entry.get("metadata_schema") or "querity")
                    break
        if not default_golden:
            default_golden = self.config.golden_queries_path
        if not default_source_schemas:
            default_source_schemas = self.config.source_schemas or "public"
        if not default_metadata_schema:
            default_metadata_schema = self.config.metadata_schema or "querity"

        self.push_screen(
            DatabaseScreen(
                self.config.databases,
                current_url,
                default_golden_path=default_golden,
                default_source_schemas=default_source_schemas,
                default_metadata_schema=default_metadata_schema,
            ),
            self._database_selected,
        )

    def _database_selected(
        self,
        result: tuple[str, str, str, str] | tuple[str, str] | str | None,
    ) -> None:
        if result is None:
            if self.service is None:
                self._set_status("No database selected · type /database")
            self._enable_input()
            return
        if isinstance(result, tuple):
            if len(result) == 4:
                url, source_schemas, metadata_schema, golden_path = result
            elif len(result) == 2:
                url, golden_path = result
                source_schemas, metadata_schema = None, None
            else:
                url = result[0]
                source_schemas, metadata_schema, golden_path = None, None, ""
        else:
            url, source_schemas, metadata_schema, golden_path = result, None, None, ""
        ingest = self._pending_ingest
        self._pending_ingest = False
        self._clear_transcript()
        if self.config is None:
            self._write_welcome()
            return
        connected = self._connect_database(
            self.config,
            url,
            ingest=ingest,
            new_chat=True,
            source_schemas=source_schemas,
            metadata_schema=metadata_schema,
            golden_queries_path=golden_path,
        )
        if connected and not ingest:
            self._set_status("Ready")

    def action_show_chats(self) -> None:
        if self._modal_open():
            return
        if self.service is None:
            return
        if self.busy:
            self._write(Text("Wait for the current operation to finish.", style=BRAND_WARNING))
            return
        if self.pending is not None:
            self._write(Text("Resolve the pending request before switching chats.", style=BRAND_WARNING))
            return
        self.push_screen(
            ChatsScreen(self.service.conversations(), self.service.conversation_id),
            self._chat_selected,
        )

    def _chat_selected(self, conversation_id: str | None) -> None:
        if conversation_id is None:
            self._enable_input()
            return
        if conversation_id == "__new__":
            self.action_new_chat()
            return
        if conversation_id == "__switch_db__":
            self.action_select_database()
            return
        if self.service is None:
            self._enable_input()
            return
        conversation = self.service.chat_store.get_conversation(conversation_id)
        if conversation is None:
            self._write(Text("Conversation not found.", style=BRAND_ERROR))
            self._enable_input()
            return
        current_fingerprint = self.service.settings.db_fingerprint
        if conversation.get("db_fingerprint") != current_fingerprint:
            url = str(conversation.get("database_url") or "")
            if not url:
                url = self._url_for_fingerprint(str(conversation.get("db_fingerprint") or ""))
            if not url:
                self._write(
                    Text(
                        "This chat belongs to a database that is no longer configured. "
                        "Use /newchat to reconnect to it.",
                        style=BRAND_WARNING,
                    )
                )
                self._enable_input()
                return
            self._clear_transcript()
            if self.config is None:
                self._write_welcome()
                self._enable_input()
                return
            if not self._connect_database(
                self.config,
                url,
                conversation_id=conversation_id,
            ):
                self._enable_input()
            return
        try:
            conversation = self.service.open_conversation(conversation_id)
        except Exception as exc:
            self._write(Text(str(exc), style=BRAND_ERROR))
            self._enable_input()
            return
        self.pending = None
        self.mode = "chat"
        self._clear_transcript()
        stats = self.service.chat_store.stats(conversation_id)
        messages = int(stats.get("messages") or 0)
        self._write_welcome(show_logo=messages == 0)
        self._show_database_status()
        self._write(
            Text(
                f"Opened {conversation.get('title') or 'chat'}  ·  "
                f"{stats['messages']} messages · {stats['query_runs']} queries"
            )
        )
        self._render_conversation_history(conversation_id)
        pending = self.service.pending_interrupt()
        if pending is not None:
            self._handle_turn_result(pending)
        else:
            self._set_status("Ready")
            self._enable_input()

    def _render_conversation_history(self, conversation_id: str) -> None:
        if self.service is None:
            return
        messages = self.service.chat_store.get_messages(conversation_id)
        if not messages:
            return
        runs = self.service.chat_store.get_query_runs(conversation_id)
        self._write(Rule(f"Chat history · {len(messages)} messages", style=f"{BRAND_ACCENT} dim"))

        for message in messages:
            role = str(message.get("role") or "").lower()
            content = str(message.get("content") or "").strip()
            query_run_id = str(message.get("query_run_id") or "")
            run = runs.get(query_run_id) if query_run_id else None

            if role == "user":
                self._write(
                    Panel(Text(content, overflow="fold"), title="You", border_style=BRAND_PRIMARY)
                )
            elif role == "assistant":
                sql = str(message.get("sql") or (run.get("sql") if run else "") or "").strip()
                status = str(message.get("status") or (run.get("status") if run else "") or "")

                if status == "completed":
                    explanation = content
                    if "\n\nSQL:\n" in explanation:
                        explanation = explanation.split("\n\nSQL:\n")[0].strip()
                    if explanation and explanation not in {"Query completed.", "Done."}:
                        self._write(
                            Panel(
                                Text(explanation, overflow="fold"),
                                title="Assistant",
                                border_style=BRAND_ACCENT,
                            )
                        )

                    if sql:
                        self._write(
                            Panel(
                                Syntax(sql, "sql", theme="monokai", word_wrap=True),
                                title="Executed SQL",
                                border_style=BRAND_ACCENT,
                            )
                        )

                    row_count = run.get("row_count") if run else None
                    duration_ms = run.get("duration_ms") if run else None
                    raw_preview = run.get("preview") if run else None
                    raw_metadata = message.get("metadata")
                    metadata = {}
                    if isinstance(raw_metadata, str) and raw_metadata:
                        with contextlib.suppress(Exception):
                            metadata = json.loads(raw_metadata)
                    elif isinstance(raw_metadata, dict):
                        metadata = raw_metadata

                    if row_count is None and "row_count" in metadata:
                        row_count = metadata["row_count"]
                    if duration_ms is None and "duration_ms" in metadata:
                        duration_ms = metadata["duration_ms"]

                    preview = {}
                    if isinstance(raw_preview, str) and raw_preview:
                        with contextlib.suppress(Exception):
                            preview = json.loads(raw_preview)
                    elif isinstance(raw_preview, dict):
                        preview = raw_preview

                    columns = preview.get("columns") or []
                    rows = preview.get("rows") or []
                    truncated = preview.get("truncated", False)

                    body = f"Rows: {row_count if row_count is not None else len(rows)}"
                    if duration_ms is not None:
                        body += f"  ·  {duration_ms} ms"
                    if truncated:
                        body += "  (truncated)"
                    self._write(Panel(Text(body), title="Result", border_style=BRAND_SUCCESS))

                    if columns and rows:
                        self._write(_rows_table(columns, rows))
                    elif not columns:
                        self._write(Text("The query completed with no rows returned."))

                elif status == "failed":
                    if sql:
                        self._write(
                            Panel(
                                Syntax(sql, "sql", theme="monokai", word_wrap=True),
                                title="Failed SQL",
                                border_style=BRAND_ERROR,
                            )
                        )
                    error = (run.get("error") if run else "") or content or "The request failed."
                    self._write(
                        Panel(Text(error, overflow="fold"), title="Failed", border_style=BRAND_ERROR)
                    )

                elif status == "rejected":
                    if sql:
                        self._write(
                            Panel(
                                Syntax(sql, "sql", theme="monokai", word_wrap=True),
                                title="Rejected SQL",
                                border_style=BRAND_WARNING,
                            )
                        )
                    reason = (
                        (run.get("error") if run else "") or content or "The query was rejected."
                    )
                    self._write(
                        Panel(
                            Text(reason, overflow="fold"),
                            title="Rejected",
                            border_style=BRAND_WARNING,
                        )
                    )

                else:
                    if sql:
                        self._write(
                            Panel(
                                Syntax(sql, "sql", theme="monokai", word_wrap=True),
                                title="Proposed SQL",
                                border_style=BRAND_ACCENT,
                            )
                        )
                    if content:
                        self._write(
                            Panel(
                                Text(content, overflow="fold"),
                                title="Assistant",
                                border_style=BRAND_ACCENT,
                            )
                        )

        self._write(Rule(style="dim"))

    def _write(self, renderable: Any, *, expand: bool = False) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#transcript", RichLog).write(renderable, expand=expand)

    def _set_status(self, value: str) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#status", Static).update(_status_text(value))

    def _set_hints(self, value: str) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#hints", Static).update(Text(value))

    def _sync_prompt_chrome(self, *, width: int | None = None) -> None:
        titles = {
            "approval": "Decision",
            "clarification": "Clarification",
            "feedback": "Revision feedback",
        }
        if width is None:
            with contextlib.suppress(Exception):
                width = self.size.width
        cols = width or 80

        if cols < 65:
            hints = {
                "approval": "a:ok  e:edit  r:rev  x:rej  ?:help",
                "clarification": "enter answer",
                "feedback": "enter send  ·  empty cancels",
                "chat": "enter send  ·  / cmds",
            }
        elif cols < 85:
            hints = {
                "approval": "a approve · e edit · r revise · x reject",
                "clarification": "enter answer",
                "feedback": "enter send · empty enter cancels",
                "chat": "enter send · / commands",
            }
        else:
            hints = {
                "approval": "a approve  ·  e edit  ·  r revise  ·  x reject  ·  ? help",
                "clarification": "enter answer",
                "feedback": "enter send  ·  empty enter cancels",
                "chat": _DEFAULT_HINT,
            }

        if self.service is None and self.mode == "chat":
            hints["chat"] = "/newchat connect DB" if cols < 65 else "/newchat connect a database"

        with contextlib.suppress(Exception):
            self.query_one("#prompt", Input).border_title = titles.get(self.mode, "Ask Querity")
        self._set_hints(hints.get(self.mode, hints["chat"]))

    def _set_placeholder(self, value: str) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#prompt", Input).placeholder = value
        self._sync_prompt_chrome()

    def _enable_input(self) -> None:
        self.busy = False
        try:
            widget = self.query_one("#prompt", Input)
            widget.disabled = False
            widget.focus()
        except Exception:
            pass
        if self.mode == "approval":
            self._set_placeholder("a=approve  e=edit  r=revise  x=reject")
        elif self.mode == "clarification":
            self._set_placeholder("Answer the clarification and press Enter")
        elif self.mode == "feedback":
            self._set_placeholder("Describe the change you want")
        elif self.service is None:
            self._set_placeholder("Type /newchat to choose a PostgreSQL database")
        else:
            self._set_placeholder("Ask a question about your data, or type / for commands")

    def _disable_input(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#prompt", Input).disabled = True

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        if busy:
            self._disable_input()
        else:
            self._enable_input()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        event.input.value = ""
        if value.startswith("/"):
            self._handle_command(value)
            return
        if self.fatal or self.service is None:
            self._write(Text("Choose a database first with /newchat.", style=BRAND_WARNING))
            return
        if self.busy:
            self._write(Text("Still working on the previous request.", style=BRAND_WARNING))
            return
        if self.mode == "approval":
            self._handle_approval_input(value)
            return
        if self.mode == "clarification":
            if not value:
                return
            self._resume(HumanDecision(kind="clarification", answer=value))
            return
        if self.mode == "feedback":
            if not value:
                self.mode = "approval"
                self._write(Text("Revision cancelled; still awaiting approval.", style=BRAND_WARNING))
                self._enable_input()
                return
            self._resume(HumanDecision(kind="revise", feedback=value))
            return
        if not value:
            return
        self._write(Panel(Text(value), title="You", border_style=BRAND_PRIMARY))
        self._start_agent(value)

    def _handle_approval_input(self, value: str) -> None:
        action = value.lower()
        if action in {"a", "approve", "y", "yes"}:
            self._approve()
        elif action in {"e", "edit"}:
            self._open_sql_editor()
        elif action in {"r", "revise"}:
            self.mode = "feedback"
            self._set_placeholder("Describe the change you want")
        elif action in {"x", "reject", "n", "no"}:
            self._reject()
        elif action in {"?", "help"}:
            self._write(approval_help())
        else:
            self._write(
                Text(
                    "Choose a=approve, e=edit, r=revise, x=reject, or ? for help.",
                    style=BRAND_WARNING,
                )
            )

    def _handle_command(self, command: str) -> None:
        name, _, _argument = command.partition(" ")
        name = name.lower()
        resolved = _resolve_command(name)
        if resolved is None:
            self._write(Text(f"Unknown command {name}. Type /help.", style=BRAND_WARNING))
            return
        canonical = resolved.name
        if canonical in {"/status", "/history", "/chats", "/ingest"} and self.service is None:
            self._write(Text("Choose a database first with /database.", style=BRAND_WARNING))
            return
        if canonical == "/quit":
            self.exit()
        elif canonical == "/help":
            self._write(help_text())
        elif canonical == "/status":
            self._status_worker()
        elif canonical == "/history":
            self._show_history()
        elif canonical == "/config":
            self._show_config()
        elif canonical == "/settings":
            self.action_edit_settings()
        elif canonical == "/newchat":
            self.action_new_chat()
        elif canonical == "/database":
            self.action_select_database()
        elif canonical == "/chats":
            self.action_show_chats()
        elif canonical == "/clear":
            self._clear_transcript()
            self._write(Text("Transcript cleared. The conversation is still stored locally."))
        elif canonical == "/ingest":
            self._start_ingest()

    def _command_menu(self) -> OptionList | None:
        try:
            return self.query_one("#command-menu", OptionList)
        except Exception:
            return None

    def _command_menu_visible(self) -> bool:
        menu = self._command_menu()
        return menu is not None and menu.display

    def _hide_command_menu(self) -> None:
        menu = self._command_menu()
        if menu is not None:
            menu.display = False

    def _refresh_command_menu(self, value: str) -> None:
        menu = self._command_menu()
        if menu is None:
            return
        matches = _matching_commands(value)
        if not matches:
            self._hide_command_menu()
            return
        menu.clear_options()
        for command in matches:
            menu.add_option(
                Option(
                    Text.assemble(
                        (command.name, f"bold {BRAND_FOREGROUND}"),
                        "  ",
                        (command.description, BRAND_MUTED),
                    ),
                    id=command.name,
                )
            )
        menu.highlighted = 0
        menu.display = True

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "prompt":
            return
        if event.value == self._menu_suppressed_value:
            self._menu_suppressed_value = None
            self._hide_command_menu()
            return
        self._refresh_command_menu(event.value)

    def _apply_command_option(self, command: str, *, submit: bool) -> None:
        prompt = self.query_one("#prompt", PromptInput)
        self._menu_suppressed_value = command
        prompt.value = command
        prompt.cursor_position = len(command)
        self._hide_command_menu()
        if submit:
            prompt.post_message(Input.Submitted(prompt, command))

    def handle_prompt_key(self, event: events.Key) -> bool:
        menu = self._command_menu()
        if menu is None or not menu.display:
            return False
        if event.key == "down":
            menu.action_cursor_down()
            return True
        if event.key == "up":
            menu.action_cursor_up()
            return True
        if event.key == "escape":
            self._hide_command_menu()
            return True
        if event.key in {"tab", "enter"}:
            option = menu.highlighted_option
            if option is None or option.id is None:
                return False
            self._apply_command_option(str(option.id), submit=event.key == "enter")
            return True
        return False

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id != "command-menu":
            return
        if event.option.id is None:
            return
        self._apply_command_option(str(event.option.id), submit=True)

    def _approve(self) -> None:
        if self.pending is None:
            return
        payload = self.pending.interrupt or {}
        if not payload.get("can_approve"):
            self._write(
                Text(
                    "This query failed deterministic validation and cannot be approved. "
                    "Edit it or reject it.",
                    style=BRAND_ERROR,
                )
            )
            return
        sql_hash = payload.get("sql_hash")
        self._resume(HumanDecision(kind="approve", sql_hash=sql_hash))

    def _open_sql_editor(self) -> None:
        if self.pending is None:
            return
        payload = self.pending.interrupt or {}
        sql = str(payload.get("sql") or "")
        self.push_screen(SQLEditScreen(sql), self._editor_done)

    def _editor_done(self, value: str | None) -> None:
        if value:
            self._resume(HumanDecision(kind="edit", sql=value, pre_approve=True))
        else:
            self._write(Text("Edit cancelled.", style=BRAND_WARNING))
            self._enable_input()

    def _reject(self) -> None:
        self.push_screen(
            ConfirmScreen("Reject this query without running it?"),
            self._reject_done,
        )

    def _reject_done(self, confirmed: bool | None) -> None:
        if confirmed:
            self._resume(
                HumanDecision(kind="reject", reason="Rejected by the user in the terminal.")
            )
        else:
            self._enable_input()

    def _start_agent(self, question: str) -> None:
        assert self.service is not None
        self.mode = "chat"
        self.pending = None
        self._set_busy(True)
        self._run_agent("start", question)

    def _resume(self, decision: HumanDecision) -> None:
        if self.pending is None or self.service is None:
            return
        thread_id = self.pending.thread_id
        self._set_busy(True)
        self.pending = None
        self.mode = "chat"
        self._run_agent("resume", (thread_id, decision))

    @work(thread=True, exclusive=True, group="agent")
    def _run_agent(self, action: str, payload: Any) -> None:
        assert self.service is not None
        try:
            if action == "start":
                result = self.service.start_turn(
                    str(payload),
                    on_event=self._thread_event,
                )
            else:
                thread_id, decision = payload
                result = self.service.resume(
                    str(thread_id),
                    decision,
                    on_event=self._thread_event,
                )
        except Exception as exc:
            self.call_from_thread(self._finish_error, exc)
            return
        self.call_from_thread(self._finish_turn, result)

    def _thread_event(self, label: str, update: Any) -> None:
        friendly = {
            "ingest": "recording turn",
            "analyze": "planning",
            "clarify": "waiting for clarification",
            "retrieve": "retrieving catalog context",
            "generate": "writing SQL",
            "guard": "validating SQL",
            "review": "reviewing SQL",
            "revise": "revising SQL",
            "human_review": "waiting for approval",
            "execute": "executing query",
            "persist": "saving result",
            "finalize": "finishing",
            "interrupt": "waiting for input",
        }.get(label, label)
        with contextlib.suppress(Exception):
            self.call_from_thread(self._set_status, f"Working: {friendly}...")

    def _finish_error(self, exc: Exception) -> None:
        self._set_busy(False)
        self._set_status("Failed")
        self._write(
            Panel(
                Text(f"{type(exc).__name__}: {exc}", overflow="fold"),
                title="Agent error",
                border_style=BRAND_ERROR,
            )
        )

    def _finish_turn(self, result: AgentTurnResult) -> None:
        self._set_busy(False)
        self._set_status("Ready")
        self._handle_turn_result(result)

    def _handle_turn_result(self, result: AgentTurnResult) -> None:
        if result.status == "awaiting_clarification":
            self.pending = result
            self.mode = "clarification"
            self._set_status("Awaiting clarification")
            payload = result.interrupt or {}
            self._write(
                Panel(
                    Text(
                        str(payload.get("question") or "Please clarify the request."),
                        overflow="fold",
                    ),
                    title="Clarification needed",
                    border_style=BRAND_WARNING,
                )
            )
            self._enable_input()
            return
        if result.status == "awaiting_approval":
            self.pending = result
            self.mode = "approval"
            self._set_status("Awaiting approval")
            self._show_approval(result)
            self._enable_input()
            return
        self.pending = None
        self.mode = "chat"
        self._set_status("Failed" if result.status == "failed" else "Ready")
        self._show_final(result)
        self._enable_input()

    def _show_approval(self, result: AgentTurnResult) -> None:
        payload = result.interrupt or {}
        question = str(payload.get("question") or result.question)
        sql = str(payload.get("sql") or result.sql or "")
        explanation = str(payload.get("explanation") or "")
        assumptions = payload.get("assumptions") or []
        guard = payload.get("guard") or {}
        review = payload.get("review") or {}
        lines: list[str] = [question]
        if explanation:
            lines.extend(["", explanation])
        if assumptions:
            lines.append("")
            lines.append("Assumptions:")
            lines.extend(f"  - {item}" for item in assumptions)
        self._write(
            Panel(Text("\n".join(lines), overflow="fold"), title="Question", border_style=BRAND_PRIMARY)
        )
        self._write(
            Panel(
                Syntax(sql, "sql", theme="monokai", word_wrap=True),
                title="Proposed SQL",
                border_style=BRAND_ACCENT,
            )
        )
        issues: list[tuple[str, str, str]] = []
        for issue in guard.get("warnings") or []:
            issues.append(
                ("warning", str(issue.get("code") or "guard"), str(issue.get("message") or ""))
            )
        for issue in guard.get("errors") or []:
            issues.append(
                ("blocking", str(issue.get("code") or "guard"), str(issue.get("message") or ""))
            )
        for issue in review.get("issues") or []:
            issues.append(
                (
                    str(issue.get("severity") or "info"),
                    str(issue.get("code") or "reviewer"),
                    str(issue.get("message") or ""),
                )
            )
        if review:
            summary = str(review.get("summary") or "")
            verdict = "approved" if review.get("approved") else "needs changes"
            body = f"Verdict: {verdict}"
            if summary:
                body += f"\n{summary}"
            self._write(
                Panel(
                    Text(body, overflow="fold"),
                    title="Reviewer",
                    border_style=BRAND_SUCCESS if review.get("approved") else BRAND_WARNING,
                )
            )
        if issues:
            body = "\n".join(
                f"[{severity}] {code}: {message}" for severity, code, message in issues
            )
            self._write(
                Panel(
                    Text(body, overflow="fold"),
                    title="Warnings",
                    border_style=BRAND_WARNING,
                )
            )
        if payload.get("can_approve"):
            hint = "a=approve  e=edit  r=revise  x=reject  ?=help"
        else:
            hint = "Guard blocked approval: e=edit  r=revise  x=reject  ?=help"
        self._write(Text(hint, style=f"bold {BRAND_WARNING}"))

    def _show_final(self, result: AgentTurnResult) -> None:
        state = result.state
        execution = state.get("execution")
        if execution is not None and getattr(execution, "ok", False):
            body = f"Rows: {execution.row_count}  ·  {execution.duration_ms} ms" + (
                "  (truncated)" if execution.truncated else ""
            )
            self._write(Panel(Text(body), title="Result", border_style=BRAND_SUCCESS))
            if execution.columns:
                self._write(_rows_table(execution.columns, execution.rows))
            else:
                self._write(Text("The query completed (result already recorded)."))
            return
        if result.status == "failed":
            message = result.error or str(state.get("final_message") or "The request failed.")
            self._write(Panel(Text(message, overflow="fold"), title="Failed", border_style=BRAND_ERROR))
            return
        if result.status == "rejected":
            message = str(state.get("final_message") or result.error or "The query was rejected.")
            self._write(
                Panel(Text(message, overflow="fold"), title="Rejected", border_style=BRAND_WARNING)
            )
            return
        message = str(state.get("final_message") or "Done.")
        self._write(Panel(Text(message, overflow="fold"), title="Assistant", border_style=BRAND_ACCENT))

    def _show_history(self) -> None:
        if self.service is None:
            return
        messages = self.service.chat_store.recent_messages(self.service.conversation_id, limit=12)
        runs = self.service.chat_store.recent_runs(self.service.conversation_id, limit=10)
        blocks: list[str] = ["Recent messages:"]
        for message in messages:
            role = str(message.get("role") or "?").upper()
            content = " ".join(str(message.get("content") or "").split())
            blocks.append(f"  {role}: {content[:220]}")
        blocks.append("")
        blocks.append("Recent queries:")
        for run in runs:
            blocks.append(f"  {run.get('status')} · {str(run.get('question') or '')[:100]}")
        self._write(
            Panel(Text("\n".join(blocks), overflow="fold"), title="History", border_style=BRAND_PRIMARY)
        )

    def _show_config(self) -> None:
        if self.config is None:
            self._write(Text("No configuration yet. Type /settings to set it up.", style=BRAND_WARNING))
            return
        saved = []
        for entry in self.config.databases:
            url = str(entry.get("url") or "") if isinstance(entry, dict) else ""
            if url:
                source_id = str(entry.get("source_id") or source_id_from_url(url))
                status = " [ingested]" if entry.get("ingested") else " [not indexed]"
                saved.append(f"{database_label(url)} (source {source_id}){status}")
        lines = [
            f"Config : {config_path()}",
            f"Current database: {self.config.masked_url()}",
            f"Source id: {self.config.source_id}",
            f"Saved databases: {len(saved)}",
            *(f"  - {item}" for item in saved),
            f"Schemas: {self.config.source_schemas}",
            f"Metadata schema: {self.config.metadata_schema}",
            f"LLM: {self.config.llm_provider}/{self.config.llm_model}",
            f"Fallbacks: "
            f"{', '.join(model for model in (self.config.llm_secondary_model, self.config.llm_tertiary_model) if model and model != self.config.llm_model) or '(none)'}",
            f"Reviewer: {self.config.review_model or '(same as LLM)'}",
            f"Embeddings: {self.config.embedding_provider}/{self.config.embedding_model} "
            f"({self.config.embedding_dimensions}d)",
            "API keys are stored in the config file and are never shown here.",
        ]
        self._write(
            Panel(
                Text("\n".join(lines), overflow="fold"), title="Configuration", border_style=BRAND_PRIMARY
            )
        )

    def _start_ingest(self) -> None:
        if self.service is None or self.config is None:
            return
        if self.busy:
            self._write(Text("Wait for the current operation to finish.", style=BRAND_WARNING))
            return
        if self.pending is not None:
            self._write(Text("Resolve the pending request before re-indexing.", style=BRAND_WARNING))
            return
        self._set_busy(True)
        self._set_status("Indexing catalog...")
        self._write(Text("Ingestion started. This can take a while for a large database."))
        self._ingest_worker()

    @work(thread=True, exclusive=True, group="ingest")
    def _ingest_worker(self) -> None:
        try:
            from querity.ingestion.pipeline import PipelineOptions, ingest

            assert self.service is not None
            settings = self.service.settings
            golden_path = None
            if self.config and self.config.golden_queries_path:
                candidate = Path(self.config.golden_queries_path).expanduser()
                if candidate.exists():
                    golden_path = candidate
            options = PipelineOptions(
                force=False,
                generate_semantic_layer=settings.generate_semantic_layer,
                golden_queries_path=golden_path,
            )
            summary = ingest(settings, options=options)
            self.call_from_thread(self._ingest_done, summary)
        except Exception as exc:
            self.call_from_thread(self._finish_error, exc)

    def _ingest_done(self, summary: Any) -> None:
        self._set_busy(False)
        if self.service is not None:
            try:
                self.service.refresh()
            except Exception:
                self._write(
                    Text("Catalog refreshed, but the session could not reload it.", style=BRAND_WARNING)
                )
            if self.config is not None:
                url = self.service.settings.source_url
                status = self.service.ingestion_status()
                tables = int(status.get("tables") or 0)
                self.config.mark_database_ingested(url, ingested=True, table_count=tables)
                self._save_config()
        self._set_status("Ready")
        stats_lines = [
            f"Status  : {summary.status}",
            f"Duration: {summary.duration_seconds:.1f}s",
        ]
        for key in sorted(summary.stats):
            value = summary.stats[key]
            if not isinstance(value, dict):
                stats_lines.append(f"{key}: {value}")
        if summary.errors:
            stats_lines.append("")
            stats_lines.append(f"Errors ({len(summary.errors)}):")
            for error in summary.errors[:5]:
                label = error.get("table") or error.get("object") or "-"
                stats_lines.append(
                    f"  {label}: {error.get('error') or error.get('message') or error}"
                )
        self._write(
            Panel(
                Text("\n".join(stats_lines), overflow="fold"),
                title="Ingestion complete",
                border_style=BRAND_SUCCESS if summary.status == "succeeded" else BRAND_WARNING,
            )
        )
        self._show_database_status()

    @work(thread=True, group="meta")
    def _status_worker(self) -> None:
        try:
            assert self.service is not None
            report = self.service.status_report()
            ingestion = self.service.ingestion_status()
            self.call_from_thread(self._show_status, report, ingestion)
        except Exception as exc:
            self.call_from_thread(self._finish_error, exc)

    def _show_status(
        self,
        report: dict[str, Any],
        ingestion: dict[str, Any] | None = None,
    ) -> None:
        health = report.get("health") or {}
        counts = health.get("counts") if isinstance(health, dict) else {}
        ingestion = ingestion or {}
        last = ingestion.get("last_run") or {}
        if ingestion.get("error"):
            ingestion_line = f"unavailable ({ingestion['error']})"
        elif ingestion.get("ingested"):
            when = _format_timestamp(last.get("finished_at") or last.get("started_at"))
            ingestion_line = (
                f"ingested · {ingestion.get('tables', 0)} tables · "
                f"{last.get('status') or 'unknown'} at {when}"
            )
        else:
            ingestion_line = "NOT ingested yet · run /ingest"
        lines = [
            f"Database : {report.get('database')}",
            f"Source id: {report.get('source_id')}",
            f"Metadata : {report.get('metadata_schema')}",
            f"Tables   : {report.get('catalog_tables')}",
            f"Catalog  : {ingestion_line}",
            f"LLM      : {report.get('provider')}/{report.get('model')}",
            f"Fallbacks: {', '.join(report.get('fallback_models') or []) or '(none)'}",
            f"Reviewer : {report.get('review_model')}",
            f"Embedding: {report.get('embedding')}",
            f"Messages : {report.get('chat', {}).get('messages', 0)}",
            f"Queries  : {report.get('chat', {}).get('query_runs', 0)}",
            f"Pending  : {report.get('pending') or 'none'}",
        ]
        if counts:
            lines.append(f"Metadata counts: {counts}")
        if isinstance(health, dict) and health.get("error"):
            lines.append(f"Health error: {health['error']}")
        self._write(
            Panel(Text("\n".join(lines), overflow="fold"), title="Status", border_style=BRAND_PRIMARY)
        )

    def action_quit(self) -> None:
        self.exit()


def _format_timestamp(value: Any) -> str:
    if not value:
        return "unknown"
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return str(value)
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M %Z")


def _rows_table(columns: list[str], rows: list[list[str]]) -> Table:
    table = Table(show_header=True, header_style=f"bold {BRAND_ACCENT}", expand=True)
    for column in columns:
        table.add_column(str(column), overflow="ellipsis", max_width=48)
    for row in rows:
        table.add_row(*[Text(str(value)) for value in row])
    return table


def approval_help() -> Panel:
    return Panel(
        Text(
            "a / approve  execute the validated SQL\n"
            "e / edit     open the SQL editor; edited SQL is guarded and executed\n"
            "r / revise   send natural-language feedback back to the generator\n"
            "x / reject   discard the query without executing\n"
            "?            show this help",
            overflow="fold",
        ),
        title="Approval",
        border_style=BRAND_WARNING,
    )


def help_text() -> Panel:
    lines = ["Commands"]
    for command in COMMANDS:
        aliases = f" (aliases: {', '.join(command.aliases)})" if command.aliases else ""
        lines.append(f"  {command.name:<12}{command.description}{aliases}")
    lines.extend(
        [
            "",
            "Type / in the prompt to browse commands; use ↑/↓ and Enter to run one.",
            "/newchat starts a fresh chat on the active database.",
            "/database connects or switches between configured databases.",
            "Ingestion is database-specific: each database is indexed once and shared across all its chats.",
            "Every query is read-only, validated, and requires your approval before execution.",
        ]
    )
    return Panel(
        Text("\n".join(lines), overflow="fold"),
        title="Help",
        border_style=BRAND_ACCENT,
    )


def run_tui(config: CliConfig | None = None) -> None:
    app = QuerityTUI(config)
    app.run()
