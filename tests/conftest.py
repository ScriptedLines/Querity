from __future__ import annotations

from collections.abc import Callable

import pytest

from querity.ingestion.settings import IngestionSettings


@pytest.fixture
def settings_factory() -> Callable[..., IngestionSettings]:
    def factory(**overrides: object) -> IngestionSettings:
        values: dict[str, object] = {
            "_env_file": None,
            "source_database_url": "postgresql://user:pass@localhost:5432/source",
            "metadata_database_url": "postgresql://user:pass@localhost:5432/meta",
        }
        values.update(overrides)
        return IngestionSettings(**values)

    return factory
