from __future__ import annotations

import asyncio

from textual.widgets import Input

from querity.cli.main import _cmd_chat, build_parser
from querity.cli.tui import DatabaseScreen, QuerityTUI, SetupScreen


def test_cli_defaults_to_tui() -> None:
    args = build_parser().parse_args([])
    assert args.handler is _cmd_chat


def test_setup_screen_builds_config_without_database() -> None:
    async def run() -> None:
        app = QuerityTUI(None)
        async with app.run_test(size=(110, 55)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, SetupScreen)
            screen.query_one("#setup-google-key", Input).value = "google-key"
            screen.query_one("#setup-nvidia-key", Input).value = "nvidia-key"
            config = screen._build_config()
            assert config.database_url == ""
            assert config.databases == []
            assert config.llm_provider == "google"
            assert config.google_api_key == "google-key"
            assert config.nvidia_api_key == "nvidia-key"
            app.exit()

    asyncio.run(run())


def test_database_screen_configures_db_and_golden_queries(tmp_path) -> None:
    golden_file = tmp_path / "goldens.json"
    golden_file.write_text("[]", encoding="utf-8")

    async def run() -> None:
        screen = DatabaseScreen(
            databases=[{"url": "postgresql://user:pass@localhost:5432/db1", "label": "db1"}],
            current_url="postgresql://user:pass@localhost:5432/db1",
            default_golden_path=str(golden_file),
        )
        app = QuerityTUI(None)
        async with app.run_test(size=(110, 55)) as pilot:
            app.push_screen(screen)
            await pilot.pause()
            result = None

            def capture(val):
                nonlocal result
                result = val

            screen.dismiss = capture
            screen._submit("postgresql://user:pass@localhost:5432/db1")
            assert result is not None
            url, source_schemas, metadata_schema, golden_path = result
            assert "localhost:5432/db1" in url
            assert source_schemas == "public"
            assert metadata_schema == "querity"
            assert golden_path == str(golden_file)
            app.exit()

    asyncio.run(run())


def test_database_screen_custom_schemas() -> None:
    async def run() -> None:
        screen = DatabaseScreen(databases=[])
        app = QuerityTUI(None)
        async with app.run_test(size=(110, 55)) as pilot:
            app.push_screen(screen)
            await pilot.pause()
            result = None

            def capture(val):
                nonlocal result
                result = val

            screen.dismiss = capture
            screen.query_one("#database-new-url", Input).value = "postgresql://user:pass@localhost:5432/customdb"
            screen.query_one("#database-source-schemas", Input).value = "sales,inventory"
            screen.query_one("#database-metadata-schema", Input).value = "custom_meta"
            screen._submit()
            assert result is not None
            url, source_schemas, metadata_schema, golden_path = result
            assert "localhost:5432/customdb" in url
            assert source_schemas == "sales,inventory"
            assert metadata_schema == "custom_meta"
            assert golden_path == ""
            app.exit()

    asyncio.run(run())


def test_render_conversation_history_renders_user_sql_and_results(tmp_path) -> None:
    from unittest.mock import MagicMock

    from textual.widgets import RichLog

    from querity.agent.chat_store import ChatStore

    history_db = tmp_path / "history.sqlite3"
    chat_store = ChatStore(history_db)
    conv = chat_store.create_conversation("fp1", "src1", title="Test Chat")
    conv_id = conv["conversation_id"]

    run_id = "run-123"
    chat_store.create_query_run(
        run_id=run_id,
        conversation_id=conv_id,
        question="Show all products",
    )
    chat_store.update_query_run(
        run_id=run_id,
        sql="SELECT id, name FROM products;",
        status="completed",
        row_count=2,
        duration_ms=12,
        preview={"columns": ["id", "name"], "rows": [["1", "Widget"], ["2", "Gadget"]], "truncated": False},
    )

    chat_store.add_message(
        conversation_id=conv_id,
        role="user",
        content="Show all products",
        query_run_id=run_id,
    )
    chat_store.add_message(
        conversation_id=conv_id,
        role="assistant",
        content="Here are the products.\n\nSQL:\nSELECT id, name FROM products;\n\nRows returned: 2 (12 ms)",
        sql="SELECT id, name FROM products;",
        status="completed",
        query_run_id=run_id,
    )

    async def run() -> None:
        app = QuerityTUI(None)
        async with app.run_test(size=(110, 55)) as pilot:
            await pilot.pause()
            mock_service = MagicMock()
            mock_service.chat_store = chat_store
            app.service = mock_service

            log = app.query_one("#transcript", RichLog)
            initial_count = len(log.lines)

            app._render_conversation_history(conv_id)
            await pilot.pause()

            # Transcript must have gained rendered history lines
            assert len(log.lines) > initial_count
            app.exit()

    asyncio.run(run())


def test_connect_database_does_not_reingest_when_already_ingested(monkeypatch) -> None:
    from unittest.mock import MagicMock

    from querity.cli.config import CliConfig

    config = CliConfig(google_api_key="google-key", nvidia_api_key="nvidia-key")
    config.remember_database("postgresql://user:pass@localhost:5432/testdb", source_id="testdb")

    mock_service = MagicMock()
    mock_service.settings.source_url = "postgresql+psycopg://user:pass@localhost:5432/testdb"
    mock_service.settings.source_id = "testdb"
    mock_service.settings.source_schemas = "public"
    mock_service.settings.metadata_schema = "querity"
    mock_service.metadata_store.primary_source_id.return_value = "testdb"
    mock_service.ingestion_status.return_value = {"ingested": True, "tables": 10, "last_run": None, "error": None}
    mock_service.conversation = {"title": "Test Chat", "conversation_id": "c1"}
    mock_service.conversation_id = "c1"
    mock_service.chat_store.stats.return_value = {"messages": 0, "query_runs": 0, "verified_turns": 0}
    mock_service.chat_store.get_messages.return_value = []
    mock_service.pending_interrupt.return_value = None

    start_ingest_called = False

    def mock_start_ingest():
        nonlocal start_ingest_called
        start_ingest_called = True

    monkeypatch.setattr("querity.cli.tui.AgentService", lambda *args, **kwargs: mock_service)

    async def run() -> None:
        app = QuerityTUI(config)
        async with app.run_test(size=(110, 55)) as pilot:
            await pilot.pause()
            monkeypatch.setattr(app, "_start_ingest", mock_start_ingest)

            connected = app._connect_database(
                config,
                "postgresql://user:pass@localhost:5432/testdb",
                new_chat=True,
            )
            assert connected is True
            assert start_ingest_called is False
            app.exit()

    asyncio.run(run())


def test_action_new_chat_on_active_database_creates_new_conversation_without_modal(monkeypatch) -> None:
    from unittest.mock import MagicMock

    from querity.cli.config import CliConfig

    config = CliConfig(google_api_key="google-key", nvidia_api_key="nvidia-key")
    config.remember_database("postgresql://user:pass@localhost:5432/testdb", source_id="testdb")

    mock_service = MagicMock()
    mock_service.settings.source_url = "postgresql+psycopg://user:pass@localhost:5432/testdb"
    mock_service.settings.source_id = "testdb"
    mock_service.settings.source_schemas = "public"
    mock_service.settings.metadata_schema = "querity"
    mock_service.metadata_store.primary_source_id.return_value = "testdb"
    mock_service.ingestion_status.return_value = {"ingested": True, "tables": 10, "last_run": None, "error": None}
    mock_service.conversation = {"title": "Test Chat", "conversation_id": "c1"}
    mock_service.conversation_id = "c1"
    mock_service.chat_store.stats.return_value = {"messages": 0, "query_runs": 0, "verified_turns": 0}
    mock_service.chat_store.get_messages.return_value = []
    mock_service.pending_interrupt.return_value = None

    monkeypatch.setattr("querity.cli.tui.AgentService", lambda *args, **kwargs: mock_service)

    async def run() -> None:
        app = QuerityTUI(config)
        async with app.run_test(size=(110, 55)) as pilot:
            await pilot.pause()
            assert app.service is not None
            mock_service.new_conversation.reset_mock()

            # Calling action_new_chat on active database
            app.action_new_chat()
            await pilot.pause()

            # Must invoke new_conversation on the existing service directly
            assert mock_service.new_conversation.called
            # No modal screen should have been pushed
            assert not isinstance(app.screen, DatabaseScreen)
            app.exit()

    asyncio.run(run())


def test_hero_logo_renders_block_wordmark() -> None:
    from rich.console import Group
    from rich.text import Text

    from querity.cli.tui import _hero_logo, _logo_lines, _welcome_panel

    rows = _logo_lines("QUERITY")
    assert len(rows) == 5
    assert len({len(row) for row in rows}) == 1
    assert len(rows[0]) == 55

    logo = _hero_logo("QUERITY")
    assert isinstance(logo, Text)
    assert "█" in logo.plain
    assert "#" not in logo.plain
    assert len(logo.plain.splitlines()) == 5
    assert logo.spans

    welcome = _welcome_panel(show_logo=True)
    assert isinstance(welcome, Group)
    assert not isinstance(_welcome_panel(show_logo=False), Group)


def test_connect_database_shows_logo_only_for_empty_conversations(monkeypatch) -> None:
    from unittest.mock import MagicMock

    from querity.cli.config import CliConfig

    config = CliConfig(google_api_key="google-key", nvidia_api_key="nvidia-key")
    url = "postgresql://user:pass@localhost:5432/testdb"
    config.remember_database(url, source_id="testdb")

    mock_service = MagicMock()
    mock_service.settings.source_url = "postgresql+psycopg://user:pass@localhost:5432/testdb"
    mock_service.settings.source_id = "testdb"
    mock_service.settings.source_schemas = "public"
    mock_service.settings.metadata_schema = "querity"
    mock_service.metadata_store.primary_source_id.return_value = "testdb"
    mock_service.ingestion_status.return_value = {"ingested": True, "tables": 10, "last_run": None, "error": None}
    mock_service.conversation = {"title": "Test Chat", "conversation_id": "c1"}
    mock_service.conversation_id = "c1"
    mock_service.chat_store.stats.return_value = {"messages": 0, "query_runs": 0, "verified_turns": 0}
    mock_service.chat_store.get_messages.return_value = []
    mock_service.pending_interrupt.return_value = None

    monkeypatch.setattr("querity.cli.tui.AgentService", lambda *args, **kwargs: mock_service)

    async def run() -> None:
        app = QuerityTUI(config)
        async with app.run_test(size=(110, 55)) as pilot:
            await pilot.pause()
            shown: list[bool] = []
            monkeypatch.setattr(
                app,
                "_write_welcome",
                lambda *, show_logo=True: shown.append(show_logo),
            )

            assert app._connect_database(config, url, new_chat=True) is True
            mock_service.chat_store.stats.return_value = {
                "messages": 3,
                "query_runs": 1,
                "verified_turns": 1,
            }
            assert app._connect_database(config, url, new_chat=True) is True
            assert shown == [True, False]

            mock_service.check_connection.side_effect = RuntimeError("connection refused")
            assert app._connect_database(config, url, new_chat=True) is False
            assert shown == [True, False, True]
            app.exit()

    asyncio.run(run())


def test_config_tracks_database_ingested_status() -> None:
    from querity.cli.config import CliConfig

    config = CliConfig()
    url = "postgresql://user:pass@localhost:5432/analytics"
    config.remember_database(url, source_id="analytics")
    assert config.is_database_ingested(url) is False

    config.mark_database_ingested(url, ingested=True, table_count=25)
    assert config.is_database_ingested(url) is True

    # Re-remembering the database preserves its ingested state
    entry = config.remember_database(url, source_id="analytics")
    assert entry["ingested"] is True
    assert entry["table_count"] == 25
    assert config.is_database_ingested(url) is True


def test_tui_terminal_resize_scales_layout_and_hints(monkeypatch) -> None:
    import asyncio
    from unittest.mock import MagicMock
    from textual.widgets import Static
    from querity.cli.config import CliConfig

    config = CliConfig(google_api_key="google-key", nvidia_api_key="nvidia-key")
    url = "postgresql://user:pass@localhost:5432/testdb"
    config.remember_database(url, source_id="testdb")

    mock_service = MagicMock()
    mock_service.settings.source_url = "postgresql+psycopg://user:pass@localhost:5432/testdb"
    mock_service.settings.source_id = "testdb"
    mock_service.settings.source_schemas = "public"
    mock_service.settings.metadata_schema = "querity"
    mock_service.metadata_store.primary_source_id.return_value = "testdb"
    mock_service.ingestion_status.return_value = {"ingested": True, "tables": 5, "last_run": None, "error": None}
    mock_service.conversation = {"title": "Test Chat", "conversation_id": "c1"}
    mock_service.conversation_id = "c1"
    mock_service.chat_store.stats.return_value = {"messages": 2, "query_runs": 1, "verified_turns": 1}
    mock_service.chat_store.get_messages.return_value = [
        {"role": "user", "content": "How many users?"},
        {"role": "assistant", "content": "Found 42 users.", "status": "completed", "sql": "SELECT COUNT(*) FROM users;"},
    ]
    mock_service.chat_store.get_query_runs.return_value = {}
    mock_service.pending_interrupt.return_value = None

    monkeypatch.setattr("querity.cli.tui.AgentService", lambda *args, **kwargs: mock_service)

    async def run() -> None:
        app = QuerityTUI(config)
        async with app.run_test(size=(110, 50)) as pilot:
            await pilot.pause()
            hints_widget = app.query_one("#hints", Static)
            # Wide hints
            assert "ctrl+c quit" in str(hints_widget.render())

            # Resize to narrow viewport (< 65 cols)
            await pilot.resize_terminal(60, 30)
            await pilot.pause()
            assert "cmds" in str(hints_widget.render())

            # Resize to medium viewport (< 85 cols)
            await pilot.resize_terminal(75, 35)
            await pilot.pause()
            assert "/ commands" in str(hints_widget.render())

            # Resize back to wide viewport (>= 85 cols)
            await pilot.resize_terminal(100, 45)
            await pilot.pause()
            assert "ctrl+c quit" in str(hints_widget.render())

            app.exit()

    asyncio.run(run())

