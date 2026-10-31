from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

DEFAULT_SENSITIVE_PATTERNS: tuple[str, ...] = (
    r"(^|_)(email|e_mail|email_address)(_|$)",
    r"(^|_)(phone|mobile|msisdn|telephone|contact_no|contact_number)(_|$)",
    r"(^|_)(ssn|social_security|aadhaar|aadhar|pan_number|pan_no|passport|national_id|tax_id|tin)(_|$)",
    r"(^|_)(credit_card|card_number|cc_num|iban|bank_account|account_number)(_|$)",
    r"(^|_)(password|passwd|secret|api_key|access_token|refresh_token|private_key)(_|$)",
    r"(^|_)(dob|date_of_birth|birth_date|birthdate)(_|$)",
    r"(^|_)(address|street_address|postal_code|zip_code|zip)(_|$)",
    r"(^|_)(ip_address|ip_addr)(_|$)",
    r"(^|_)(first_name|last_name|full_name|maiden_name|middle_name)(_|$)",
)

_COMMENT_MARKERS = ("pii", "sensitive", "confidential", "personal data")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{6,}\d")


def compile_sensitive_patterns(extra_patterns: Iterable[str] = ()) -> list[re.Pattern[str]]:
    patterns = list(DEFAULT_SENSITIVE_PATTERNS) + [pattern for pattern in extra_patterns if pattern]
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


def is_sensitive_column(
    column_name: str,
    *,
    comment: str | None = None,
    extra_patterns: Sequence[str] = (),
) -> bool:
    if comment:
        lowered_comment = comment.lower()
        if any(marker in lowered_comment for marker in _COMMENT_MARKERS):
            return True
    normalized_name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", column_name).lower()
    return any(
        pattern.search(normalized_name) for pattern in compile_sensitive_patterns(extra_patterns)
    )


def _redaction_for(column_name: str, value: str) -> str:
    lowered = column_name.lower()
    if "email" in lowered:
        return "***@example.com"
    if any(token in lowered for token in ("phone", "mobile", "msisdn", "telephone")):
        digits = re.sub(r"\D", "", value)
        return f"***-***-**{digits[-2:]}" if len(digits) >= 2 else "***"
    if any(token in lowered for token in ("name",)):
        initial = value.strip()[:1]
        return f"{initial}***" if initial else "***"
    if any(token in lowered for token in ("address", "street", "zip", "postal")):
        return "***"
    if "ip" in lowered and "." in value:
        return "***.***.***.***"
    if "card" in lowered or "account" in lowered or "iban" in lowered:
        digits = re.sub(r"\D", "", value)
        return f"****{digits[-4:]}" if len(digits) >= 4 else "***"
    return "***"


def mask_value(value: Any, column_name: str) -> str | None:
    if value is None:
        return None
    text = str(value)
    lowered = column_name.lower()

    if any(token in lowered for token in ("email", "e_mail")):
        return _EMAIL_RE.sub("***@example.com", text)
    if any(token in lowered for token in ("phone", "mobile", "msisdn", "telephone")):
        digits = re.sub(r"\D", "", text)
        return f"***-***-**{digits[-2:]}" if len(digits) >= 2 else "***"
    return _redaction_for(column_name, text)


def mask_sensitive_value(value: Any, column_name: str) -> str | None:
    return mask_value(value, column_name)


def truncate_value(value: Any, max_chars: int) -> str | None:
    if value is None:
        return None
    text = str(value)
    text = text.replace("\x00", "")
    if len(text) > max_chars:
        return text[: max_chars - 1] + "\u2026"
    return text
