from __future__ import annotations

import stat

from querity.common.cache import LocalCache, build_local_cache, hash_key


def test_local_cache_round_trip_and_permissions(tmp_path):
    cache = LocalCache(tmp_path / "cache.sqlite3")
    assert stat.S_IMODE(cache.path.stat().st_mode) == 0o600
    cache.set_json("test", "key", {"value": 42})
    assert cache.get_json("test", "key") == {"value": 42}
    assert cache.stats()["entries"] == 1
    cache.delete("test", "key")
    assert cache.get_json("test", "key") is None


def test_local_cache_ttl_and_max_age(tmp_path):
    cache = LocalCache(tmp_path / "cache.sqlite3")
    cache.set("test", "expired", b"value", ttl=10, now=100)
    assert cache.get("test", "expired", now=105) == b"value"
    assert cache.get("test", "expired", now=111) is None
    cache.set("test", "old", b"value", now=100)
    assert cache.get("test", "old", max_age=10, now=111) is None


def test_local_cache_vectors_round_trip(tmp_path):
    cache = LocalCache(tmp_path / "cache.sqlite3")
    vector = [0.25, -1.5, 3.0]
    cache.set_vector("key", vector)
    assert cache.get_vector("key") == vector


def test_local_cache_prunes_largest_old_entries(tmp_path):
    cache = LocalCache(tmp_path / "cache.sqlite3", max_bytes=1_048_576)
    cache.set("test", "first", b"a" * 700_000, now=100)
    cache.set("test", "second", b"b" * 700_000, now=101)
    assert cache.get("test", "first", now=102) is None
    assert cache.get("test", "second", now=102) == b"b" * 700_000


def test_build_local_cache_respects_settings(tmp_path):
    class Settings:
        cache_enabled = False
        resolved_cache_db_path = tmp_path / "cache.sqlite3"
        cache_max_bytes = 1_048_576

    assert build_local_cache(Settings()) is None

    Settings.cache_enabled = True  # type: ignore[misc]
    cache = build_local_cache(Settings())
    assert cache is not None
    assert cache.path == tmp_path / "cache.sqlite3"


def test_hash_key_is_stable_and_order_sensitive():
    assert hash_key("a", 1, {"b": 2}) == hash_key("a", 1, {"b": 2})
    assert hash_key("a", 1) != hash_key("a", 2)
