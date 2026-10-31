from __future__ import annotations

import json
import stat

import pytest

from querity.agent.settings import AgentSettings
from querity.cli.config import CliConfig, ConfigNotFound


def _config(**overrides: object) -> CliConfig:
    values = {
        "database_url": "postgresql://user:secret@localhost:5432/analytics",
        "source_id": "analytics",
        "google_api_key": "google-key",
        "embedding_dimensions": 1536,
    }
    values.update(overrides)
    return CliConfig(**values)


def test_config_round_trip_uses_private_permissions(tmp_path):
    target = tmp_path / "config.json"
    config = _config()
    saved = config.save(target)
    assert saved == target
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["database_url"] == config.database_url

    loaded = CliConfig.load(target)
    assert loaded.database_url == config.database_url
    assert loaded.google_api_key == "google-key"
    assert loaded.to_settings().source_url.startswith("postgresql+psycopg://")
    assert loaded.to_settings().metadata_url == loaded.to_settings().source_url


def test_missing_config_raises(tmp_path):
    with pytest.raises(ConfigNotFound):
        CliConfig.load(tmp_path / "missing.json")


def test_config_can_start_without_a_database_and_remembers_urls():
    config = CliConfig(google_api_key="google-key", nvidia_api_key="nvidia-key")
    settings = config.to_settings(require_database=False)
    assert settings.source_database_url is None
    assert settings.metadata_database_url is None

    entry = config.remember_database(
        "postgresql://user:secret@localhost:5432/shop",
        source_id="shop",
    )
    assert entry["source_id"] == "shop"
    assert entry["label"] == "localhost:5432/shop"
    assert config.database_url.startswith("postgresql+psycopg://")
    assert config.to_settings().source_id == "shop"


def test_config_per_database_schemas_and_golden_queries():
    config = CliConfig(google_api_key="google-key", nvidia_api_key="nvidia-key")
    entry1 = config.remember_database(
        "postgresql://user:secret@localhost:5432/db1",
        source_id="db1",
        source_schemas="app,analytics",
        metadata_schema="querity_meta",
        golden_queries_path="/path/to/db1_goldens.json",
    )
    assert entry1["source_schemas"] == "app,analytics"
    assert entry1["metadata_schema"] == "querity_meta"
    assert entry1["golden_queries_path"] == "/path/to/db1_goldens.json"

    entry2 = config.remember_database(
        "postgresql://user:secret@localhost:5432/db2",
        source_id="db2",
        source_schemas="sales",
        metadata_schema="custom_querity",
        golden_queries_path="/path/to/db2_goldens.json",
    )
    assert entry2["source_schemas"] == "sales"
    assert entry2["metadata_schema"] == "custom_querity"

    # Settings for db1
    settings1 = config.to_settings(database_url="postgresql://user:secret@localhost:5432/db1")
    assert settings1.source_schemas == "app,analytics"
    assert settings1.metadata_schema == "querity_meta"

    # Settings for active database (db2)
    settings2 = config.to_settings()
    assert settings2.source_schemas == "sales"
    assert settings2.metadata_schema == "custom_querity"


def test_config_loads_without_database_url(tmp_path):
    target = tmp_path / "config.json"
    target.write_text(
        json.dumps({"google_api_key": "google-key", "nvidia_api_key": "nvidia-key"}),
        encoding="utf-8",
    )
    config = CliConfig.load(target)
    assert config.database_url == ""
    assert config.databases == []


def test_config_masks_password():
    masked = _config().masked_url()
    assert "secret" not in masked
    assert "localhost:5432/analytics" in masked


def test_db_fingerprint_ignores_password():
    first = AgentSettings(
        _env_file=None,
        source_database_url="postgresql://user:one@localhost:5432/analytics",
        metadata_database_url="postgresql://user:one@localhost:5432/analytics",
    )
    second = AgentSettings(
        _env_file=None,
        source_database_url="postgresql://user:two@localhost:5432/analytics",
        metadata_database_url="postgresql://user:two@localhost:5432/analytics",
    )
    third = AgentSettings(
        _env_file=None,
        source_database_url="postgresql://user:one@localhost:5432/other",
        metadata_database_url="postgresql://user:one@localhost:5432/other",
    )
    assert first.db_fingerprint == second.db_fingerprint
    assert first.db_fingerprint != third.db_fingerprint


def test_large_embedding_dimensions_disable_hnsw():
    settings = _config(embedding_dimensions=3072).to_settings()
    assert settings.embedding_index == "none"


def test_local_storage_directories_are_private(tmp_path):
    settings = AgentSettings(
        _env_file=None,
        source_database_url="postgresql://user:secret@localhost:5432/analytics",
        metadata_database_url="postgresql://user:secret@localhost:5432/analytics",
        data_dir=tmp_path / "private",
    )
    settings.ensure_local_dirs()
    assert stat.S_IMODE(settings.resolved_data_dir.stat().st_mode) == 0o700


def test_config_ignores_project_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "SOURCE_DATABASE_URL=postgresql://env:env@elsewhere:5432/envdb\n",
        encoding="utf-8",
    )
    settings = _config().to_settings()
    assert settings.source_url == "postgresql+psycopg://user:secret@localhost:5432/analytics"


def test_config_save_does_not_change_existing_parent_permissions(tmp_path):
    parent = tmp_path / "shared"
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)
    _config().save(parent / "config.json")
    assert stat.S_IMODE(parent.stat().st_mode) == 0o755


def test_config_load_repairs_file_permissions(tmp_path):
    target = tmp_path / "config.json"
    _config().save(target)
    target.chmod(0o644)
    CliConfig.load(target)
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_cache_path_defaults_to_private_data_dir(tmp_path):
    settings = AgentSettings(
        _env_file=None,
        source_database_url="postgresql://user:secret@localhost:5432/analytics",
        metadata_database_url="postgresql://user:secret@localhost:5432/analytics",
        data_dir=tmp_path / "private",
    )
    settings.ensure_local_dirs()
    assert settings.cache_enabled
    assert settings.resolved_cache_db_path == settings.resolved_data_dir / "cache.sqlite3"
    assert stat.S_IMODE(settings.resolved_cache_db_path.parent.stat().st_mode) == 0o700
