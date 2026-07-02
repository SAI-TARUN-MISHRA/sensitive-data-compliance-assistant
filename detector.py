"""
detector.py
-----------
Rule-based sensitive data detection engine.

Approach
--------
Regex + light heuristics rather than a heavyweight NER model. This is a
deliberate design choice for a 24-hour assignment:

1. It's deterministic, explainable, and has zero inference cost/latency -
   important for anything touching compliance, where you must be able to
   say *why* something was flagged.
2. Most of the target entities (Aadhaar, PAN, card numbers, API keys) have
   a fixed structural format that regex handles well, arguably better than
   a generic NER model would out of the box.
3. It leaves room for a genuine NLP/LLM layer (see summarizer.py / qa_engine.py)
   to add the "understanding" on top of these structured findings, which is
   where an LLM actually adds value over pure pattern matching.

Each detector returns a list of Finding objects with the matched value
(masked for display), position, and a short reason string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Finding:
    category: str          # e.g. "Aadhaar Number"
    value: str              # masked value, safe to display
    raw_value: str = field(repr=False, default="")  # unmasked, used only internally
    count: int = 1


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

PATTERNS = {
    "Aadhaar Number": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "PAN Number": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    "Email Address": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "Phone Number": re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"),
    "Credit Card Number": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "IFSC Code": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    "Bank Account Number": re.compile(r"\b\d{9,18}\b"),
    "API Key / Secret": re.compile(
        r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,})\b"
        r"(?:\s*[:=]\s*['\"]?[A-Za-z0-9_\-/+]{8,}['\"]?)?"
    ),
    "Password": re.compile(r"(?i)\bpassword\s*[:=]\s*['\"]?\S+['\"]?"),
    "Employee ID": re.compile(r"\b(?:EMP|EID|E-)[- ]?\d{3,8}\b", re.IGNORECASE),
}

CONFIDENTIAL_KEYWORDS = [
    "confidential", "internal use only", "do not distribute", "trade secret",
    "proprietary", "not for external distribution", "strictly confidential",
    "nda", "non-disclosure",
]


def _luhn_valid(number: str) -> bool:
    """Standard Luhn checksum - filters out random 13-16 digit numbers that
    aren't actually plausible card numbers (e.g. long account/reference IDs)."""
    digits = [int(d) for d in number if d.isdigit()]
    if not (13 <= len(digits) <= 16):
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def _mask(category: str, value: str) -> str:
    digits_only = re.sub(r"\D", "", value)
    if category in ("Aadhaar Number", "Credit Card Number", "Bank Account Number"):
        if len(digits_only) >= 4:
            return "X" * (len(digits_only) - 4) + digits_only[-4:]
        return "X" * len(digits_only)
    if category == "Email Address":
        try:
            user, domain = value.split("@", 1)
            return f"{user[:2]}{'*' * max(len(user) - 2, 1)}@{domain}"
        except ValueError:
            return "***"
    if category == "Phone Number":
        return "X" * max(len(digits_only) - 4, 0) + digits_only[-4:]
    if category in ("Password", "API Key / Secret"):
        return "********"
    return value


BANK_CONTEXT_HINTS = re.compile(r"(?i)\b(account|a/c|acc\.?\s*no)\b")
AADHAAR_CONTEXT_HINTS = re.compile(r"(?i)\b(aadhaar|uidai)\b")
CONTEXT_WINDOW = 40  # chars to look back/around a match for disambiguation


def detect(text: str) -> List[Finding]:
    """Run all detectors over `text` and return a de-duplicated finding list
    with per-category counts."""
    findings_map = {}

    for category, pattern in PATTERNS.items():
        for m in pattern.finditer(text):
            candidate = m.group(0).strip()
            if not candidate:
                continue

            start = max(0, m.start() - CONTEXT_WINDOW)
            context = text[start:m.start()]

            # Extra validation to cut down false positives
            if category == "Credit Card Number":
                if not _luhn_valid(candidate):
                    continue
            if category == "Bank Account Number":
                # avoid re-flagging things that are actually Aadhaar/phone/card numbers
                digits = re.sub(r"\D", "", candidate)
                if len(digits) == 12 and not BANK_CONTEXT_HINTS.search(context):
                    continue
                if len(digits) == 10:
                    continue
                if _luhn_valid(candidate):
                    continue

            # A 12-digit number tagged as Aadhaar but sitting right after
            # "Account"/"A/C" context is almost certainly a bank account
            # number, not Aadhaar - re-route it.
            effective_category = category
            if category == "Aadhaar Number":
                digits = re.sub(r"\D", "", candidate)
                if len(digits) == 12 and BANK_CONTEXT_HINTS.search(context) \
                        and not AADHAAR_CONTEXT_HINTS.search(context):
                    effective_category = "Bank Account Number"

            masked = _mask(effective_category, candidate)
            key = (effective_category, masked)
            if key in findings_map:
                findings_map[key].count += 1
            else:
                findings_map[key] = Finding(category=effective_category, value=masked, raw_value=candidate)

    # Confidential business info - keyword based, not a per-item finding
    lowered = text.lower()
    hits = sum(lowered.count(kw) for kw in CONFIDENTIAL_KEYWORDS)
    if hits:
        findings_map[("Confidential Business Information", "keyword match")] = Finding(
            category="Confidential Business Information",
            value=f"{hits} keyword occurrence(s)",
            count=hits,
        )

    return list(findings_map.values())


def summarize_counts(findings: List[Finding]) -> dict:
    """Return {category: total_count} for quick display / charts."""
    out = {}
    for f in findings:
        out[f.category] = out.get(f.category, 0) + f.count
    return out
