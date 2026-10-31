from __future__ import annotations

from types import SimpleNamespace

import networkx as nx

import querity.agent.catalog as catalog_module
from querity.agent.catalog import CatalogCache, CatalogSnapshot


def test_catalog_cache_reuses_snapshot_until_version_changes(monkeypatch):
    versions = {"default": "run-1"}
    store = SimpleNamespace(catalog_version=lambda source_id: versions[source_id])
    snapshot = CatalogSnapshot(source_id="default", tables={}, graph=nx.MultiGraph())
    calls = 0

    def fake_load_catalog(_store, _source_id):
        nonlocal calls
        calls += 1
        return snapshot

    monkeypatch.setattr(catalog_module, "load_catalog", fake_load_catalog)
    cache = CatalogCache(store, "default")  # type: ignore[arg-type]

    assert cache.get() is snapshot
    assert cache.get() is snapshot
    assert calls == 1

    versions["default"] = "run-2"
    assert cache.get() is snapshot
    assert calls == 2

    cache.invalidate()
    assert cache.get() is snapshot
    assert calls == 3


def test_catalog_cache_treats_missing_metadata_as_empty_version(monkeypatch):
    import sqlalchemy as sa

    class FakeStore:
        def catalog_version(self, source_id: str) -> str:
            raise sa.exc.ProgrammingError("SELECT", {}, Exception("missing"))

    snapshot = CatalogSnapshot(source_id="default", tables={}, graph=nx.MultiGraph())
    monkeypatch.setattr(catalog_module, "load_catalog", lambda _store, _source_id: snapshot)
    cache = CatalogCache(FakeStore(), "default")  # type: ignore[arg-type]
    assert cache.get() is snapshot
    assert cache.get() is snapshot
