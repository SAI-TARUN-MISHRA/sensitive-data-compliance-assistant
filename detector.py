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
2. Most of the target entities (Aadhaar, PAN, card numbers, API keys, passport,
   voter ID, driving licence, addresses) have fixed structural formats that
   regex handles well, arguably better than a generic NER model would out of the box.
3. It leaves room for a genuine NLP/LLM layer (see summarizer.py / qa_engine.py)
   to add the "understanding" on top of these structured findings, which is
   where an LLM actually adds value over pure pattern matching.

Each detector returns a list of Finding objects with the matched value
(masked for display), position, and a short reason string.

Detected PII categories
-----------------------
  HIGH  : Aadhaar, PAN, Passport, Voter ID, Driving Licence, Credit Card,
           Bank Account, API Key/Secret, Password, IFSC Code
  MEDIUM: Phone Number, Employee ID, GST Number, Vehicle Registration,
           Confidential Business Information
  LOW   : Email Address, Date of Birth, Physical Address
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
    # ── Identity documents ──────────────────────────────────────────────────
    "Aadhaar Number": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "PAN Number": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    # Indian passport: letter + 7 digits  (e.g. A1234567)
    "Passport Number": re.compile(r"\b[A-PR-WY][1-9]\d\s?\d{4}[1-9]\b"),
    # Indian Voter ID (EPIC): 3 letters + 7 digits  (e.g. ABC1234567)
    "Voter ID": re.compile(r"\b[A-Z]{3}[0-9]{7}\b"),
    # Indian Driving Licence: state code (2 letters) + 2 digits + 4 digits + 7 digits
    # formats vary by state, so we use a broad pattern and rely on context hints
    "Driving Licence Number": re.compile(
        r"\b[A-Z]{2}[0-9]{2}\s?[0-9]{4}\s?[0-9]{7}\b"
    ),

    # ── Financial ───────────────────────────────────────────────────────────
    "Email Address": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "Phone Number": re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"),
    "Credit Card Number": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "IFSC Code": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    "Bank Account Number": re.compile(r"\b\d{9,18}\b"),
    # GST: 2-digit state + PAN (10 chars) + 1-digit entity + Z + 1-char checksum
    "GST Number": re.compile(r"\b[0-3][0-9][A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b"),

    # ── Credentials ─────────────────────────────────────────────────────────
    "API Key / Secret": re.compile(
        r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,})\b"
        r"(?:\s*[:=]\s*['\"]?[A-Za-z0-9_\-/+]{8,}['\"]?)?"
    ),
    "Password": re.compile(r"(?i)\bpassword\s*[:=]\s*['\"]?\S+['\"]?"),

    # ── Employee / HR ────────────────────────────────────────────────────────
    "Employee ID": re.compile(r"\b(?:EMP|EID|E-)[- ]?\d{3,8}\b", re.IGNORECASE),
    # Indian vehicle registration: state (2L) + 2D + letter(s) + 4D  e.g. MH12AB1234
    "Vehicle Registration Number": re.compile(
        r"\b[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}\b"
    ),

    # ── Personal demographics ────────────────────────────────────────────────
    # Date of birth in common formats: DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD
    "Date of Birth": re.compile(
        r"\b(?:(?:0?[1-9]|[12]\d|3[01])[/\-.](?:0?[1-9]|1[0-2])[/\-.](19|20)\d{2}"
        r"|(19|20)\d{2}[/\-.](?:0?[1-9]|1[0-2])[/\-.](?:0?[1-9]|[12]\d|3[01]))\b"
    ),
    # Physical/house address: look for house/flat/plot/door + number combos or
    # street/road/lane/nagar/colony keywords that strongly suggest an address
    "Physical Address": re.compile(
        r"(?i)\b(?:(?:h\.?no|house\s*no|flat\s*no|plot\s*no|door\s*no|d\.?no|s\.?no)"
        r"[.:\s]*[A-Z0-9/\-]+[,\s]+[A-Za-z0-9\s,./\-]{5,80}?"
        r"(?:street|st|road|rd|lane|ln|nagar|colony|layout|enclave|apartment|apt|block|sector|phase|area))"
        r"|(?:[A-Za-z0-9\s,./\-]{5,60}?"
        r"(?:street|road|lane|nagar|colony|layout|enclave|apartments?|sector|phase|block)\b)"
    ),
}

CONFIDENTIAL_KEYWORDS = [
    "confidential", "internal use only", "do not distribute", "trade secret",
    "proprietary", "not for external distribution", "strictly confidential",
    "nda", "non-disclosure", "classified", "privileged", "sensitive",
    "for official use only", "restricted", "commercial in confidence",
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
    if category == "Passport Number":
        # mask middle digits: e.g. A1234567 -> A***4567
        return value[0] + "***" + value[-4:] if len(value) >= 5 else "*****"
    if category == "Voter ID":
        # show first 3 letters + mask digits: ABC****567
        return value[:3] + "****" + value[-3:] if len(value) >= 6 else "*******"
    if category == "Driving Licence Number":
        clean = value.replace(" ", "")
        return clean[:4] + "X" * max(len(clean) - 6, 0) + clean[-2:]
    if category == "GST Number":
        return value[:2] + "*" * 10 + value[-3:]
    if category in ("Date of Birth",):
        # show only year: DD/MM/YYYY -> **/**/YYYY
        return re.sub(r"(\d{1,2})[/\-.](\d{1,2})[/\.\-](\d{4})", r"**/***/\3", value)
    if category == "Physical Address":
        words = value.split()
        return " ".join(words[:2]) + " [ADDRESS REDACTED]" if words else "[ADDRESS REDACTED]"
    if category == "Vehicle Registration Number":
        return value[:4] + "****"
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
