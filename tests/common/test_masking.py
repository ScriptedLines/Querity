from __future__ import annotations

from querity.common.masking import (
    is_sensitive_column,
    mask_value,
    truncate_value,
)


def test_detects_sensitive_columns() -> None:
    assert is_sensitive_column("customer_email")
    assert is_sensitive_column("phone_number")
    assert is_sensitive_column("emailAddress")
    assert is_sensitive_column("firstName")
    assert is_sensitive_column("date_of_birth")
    assert is_sensitive_column("first_name")
    assert is_sensitive_column("anything", comment="[PII] do not expose")
    assert is_sensitive_column("salary", extra_patterns=[r"(^|_)salary(_|$)"])
    assert not is_sensitive_column("product_name")
    assert not is_sensitive_column("total_amount")


def test_masks_email_and_phone() -> None:
    assert mask_value("alice@example.com", "email") == "***@example.com"
    masked = mask_value("+91 98765 43210", "mobile_number")
    assert masked is not None
    assert masked.endswith("10")
    assert "98765" not in masked


def test_masks_generic_values() -> None:
    assert mask_value("4111 1111 1111 1234", "card_number") == "****1234"
    assert mask_value("221B Baker Street", "street_address") == "***"
    assert mask_value("abc", "api_key") == "***"


def test_masks_uuid_values_for_sensitive_columns() -> None:
    value = "123e4567-e89b-12d3-a456-426614174000"
    assert mask_value(value, "national_id") == "***"


def test_truncate_value_removes_nul_bytes() -> None:
    assert truncate_value("abc\x00def", 100) == "abcdef"
    assert truncate_value("abcdef", 4) == "abc\u2026"
    assert truncate_value(None, 4) is None
