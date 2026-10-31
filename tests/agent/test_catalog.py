from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from querity.agent.catalog import CatalogEmptyError, load_catalog


def test_missing_metadata_raises_catalog_empty():
    engine = MagicMock()
    engine.dialect = postgresql.dialect()
    connection = MagicMock()
    connection.execute.side_effect = sa.exc.ProgrammingError(
        "SELECT",
        {},
        Exception("relation does not exist"),
    )
    engine.connect.return_value.__enter__.return_value = connection
    store = SimpleNamespace(engine=engine, schema="querity")

    with pytest.raises(CatalogEmptyError):
        load_catalog(store, "default")  # type: ignore[arg-type]
