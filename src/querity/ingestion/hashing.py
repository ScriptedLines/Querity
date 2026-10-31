from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from querity.ingestion.models import ColumnProfile, TableSchema

STRUCTURAL_HASH_VERSION = "structural-v2"
PROFILE_HASH_VERSION = "profile-v1"
ENRICHMENT_HASH_VERSION = "enrichment-v1"
CONTENT_HASH_VERSION = "content-v1"


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def structural_fingerprint(table: TableSchema) -> str:
    columns = [
        {
            "name": column.name,
            "type": column.data_type,
            "nullable": column.nullable,
            "default": column.default,
            "comment": column.comment,
            "primary_key": column.primary_key,
            "unique": column.unique,
            "foreign_key": column.foreign_key,
        }
        for column in sorted(table.columns, key=lambda item: item.name)
    ]
    payload = {
        "version": STRUCTURAL_HASH_VERSION,
        "table": table.key,
        "type": table.table_type,
        "comment": table.comment,
        "columns": columns,
        "primary_key": sorted(table.primary_key),
        "unique_constraints": sorted(sorted(item) for item in table.unique_constraints),
        "foreign_keys": sorted(
            (
                {
                    "name": fk.name,
                    "columns": sorted(fk.constrained_columns),
                    "referred": fk.referred_key,
                    "referred_columns": sorted(fk.referred_columns),
                }
                for fk in table.foreign_keys
            ),
            key=lambda item: stable_json(item),
        ),
    }
    return sha256_text(stable_json(payload))


def profile_fingerprint(columns: Iterable[ColumnProfile]) -> str:
    """Hash semantic profile facts, deliberately excluding volatile row samples.

    Allowed enum values, ranges and null fractions trigger re-enrichment because they
    change how an LLM should describe a column. Random high-cardinality format samples
    do not, otherwise every nightly run would pay for a new LLM call.
    """
    payload = []
    for column in sorted(columns, key=lambda item: item.name):
        payload.append(
            {
                "name": column.name,
                "type": column.data_type,
                "value_kind": column.value_kind,
                "is_sensitive": column.is_sensitive,
                "distinct_count_estimate": column.distinct_count_estimate,
                "null_fraction": (
                    round(column.null_fraction, 4) if column.null_fraction is not None else None
                ),
                "allowed_values": sorted(column.allowed_values),
                "min_value": column.min_value,
                "max_value": column.max_value,
            }
        )
    return sha256_text(stable_json({"version": PROFILE_HASH_VERSION, "columns": payload}))


def enrichment_fingerprint(
    structural_hash: str,
    profile_hash: str,
    *,
    llm_model: str,
    prompt_version: str,
) -> str:
    return sha256_text(
        stable_json(
            {
                "version": ENRICHMENT_HASH_VERSION,
                "structural_hash": structural_hash,
                "profile_hash": profile_hash,
                "llm_model": llm_model,
                "prompt_version": prompt_version,
            }
        )
    )


def content_hash(content: str) -> str:
    return sha256_text(stable_json({"version": CONTENT_HASH_VERSION, "content": content}))
