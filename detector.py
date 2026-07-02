"""
detector.py
-----------
Rule-based + NER-based sensitive data detection engine.

Two-layer detection strategy
-----------------------------
Layer 1 – Structural regex (PATTERNS dict):
    Fixed-format fields — Aadhaar, PAN, Passport, phone, card numbers, etc.
    These have a deterministic structure so regex works perfectly.

Layer 2 – Labeled-field regex (LABELED_PATTERNS dict):
    Form/document fields where the label tells us what the value is.
    e.g. "Name: JOHN DOE", "Address: 12 MG Road", "Father's Name: RAMESH"
    Captures the value after the label, regardless of free-form content.

Layer 3 – spaCy NER (optional, graceful fallback):
    When spaCy is installed, runs Named Entity Recognition to catch
    PERSON names and GPE/LOC place names that don't have a label prefix.
    Falls back silently if spaCy / model is not available.

Detected PII categories
-----------------------
  HIGH  : Aadhaar, PAN, Passport Number, Voter ID, Driving Licence,
           Credit Card, Bank Account, API Key/Secret, Password, IFSC Code,
           Full Name, Father's / Mother's Name
  MEDIUM: Phone Number, Employee ID, GST Number, Vehicle Registration,
           Place of Birth, Nationality, Confidential Business Information
  LOW   : Email Address, Date of Birth, Physical Address, Gender
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Optional NER via spaCy
# ---------------------------------------------------------------------------
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    NER_AVAILABLE = True
except Exception:           # ImportError or OSError (model not found)
    NER_AVAILABLE = False


@dataclass
class Finding:
    category: str           # e.g. "Full Name"
    value: str              # masked value, safe to display
    raw_value: str = field(repr=False, default="")
    count: int = 1


# ---------------------------------------------------------------------------
# Layer 1 – Structural / format-based patterns
# ---------------------------------------------------------------------------

PATTERNS = {
    # ── Identity documents ───────────────────────────────────────────────────
    "Aadhaar Number": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "PAN Number": re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"),
    # Indian passport: letter + 7 digits  e.g. A1234567
    "Passport Number": re.compile(r"\b[A-PR-WY][1-9]\d\s?\d{4}[1-9]\b"),
    # Indian Voter ID (EPIC): 3 letters + 7 digits  e.g. ABC1234567
    "Voter ID": re.compile(r"\b[A-Z]{3}[0-9]{7}\b"),
    # Indian Driving Licence
    "Driving Licence Number": re.compile(r"\b[A-Z]{2}[0-9]{2}\s?[0-9]{4}\s?[0-9]{7}\b"),

    # ── Financial ────────────────────────────────────────────────────────────
    "Email Address": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "Phone Number": re.compile(r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)"),
    "Credit Card Number": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "IFSC Code": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    "Bank Account Number": re.compile(r"\b\d{9,18}\b"),
    # GST: 2-digit state + PAN (10 chars) + entity + Z + checksum
    "GST Number": re.compile(r"\b[0-3][0-9][A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b"),

    # ── Credentials ──────────────────────────────────────────────────────────
    "API Key / Secret": re.compile(
        r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9]{20,})\b"
        r"(?:\s*[:=]\s*['\"]?[A-Za-z0-9_\-/+]{8,}['\"])?"
    ),
    "Password": re.compile(r"(?i)\bpassword\s*[:=]\s*['\"]?\S+['\"]?"),

    # ── Employee / HR ─────────────────────────────────────────────────────────
    "Employee ID": re.compile(r"\b(?:EMP|EID|E-)[- ]?\d{3,8}\b", re.IGNORECASE),
    # Indian vehicle registration  e.g. MH12AB1234
    "Vehicle Registration Number": re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}\b"),

    # ── Personal demographics ─────────────────────────────────────────────────
    # Date of birth: DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD
    "Date of Birth": re.compile(
        r"\b(?:(?:0?[1-9]|[12]\d|3[01])[/\-.](?:0?[1-9]|1[0-2])[/\-.](19|20)\d{2}"
        r"|(19|20)\d{2}[/\-.](?:0?[1-9]|1[0-2])[/\-.](?:0?[1-9]|[12]\d|3[01]))\b"
    ),
    # Physical address via structural keywords
    "Physical Address": re.compile(
        r"(?i)\b(?:(?:h\.?no|house\s*no|flat\s*no|plot\s*no|door\s*no|d\.?no|s\.?no)"
        r"[.:\s]*[A-Z0-9/\-]+[,\s]+[A-Za-z0-9\s,./\-]{5,80}?"
        r"(?:street|st|road|rd|lane|ln|nagar|colony|layout|enclave|apartment|apt|block|sector|phase|area))"
        r"|(?:[A-Za-z0-9\s,./\-]{5,60}?"
        r"(?:street|road|lane|nagar|colony|layout|enclave|apartments?|sector|phase|block)\b)"
    ),
}

# ---------------------------------------------------------------------------
# Layer 2 – Labeled-field patterns (label: value)
# These catch PII in structured documents / forms / scanned IDs where the
# field name acts as a reliable anchor for the value that follows.
# ---------------------------------------------------------------------------

# Helper: build a labeled pattern that captures text after "Label : value"
def _lp(labels: str, value_re: str, flags=re.IGNORECASE) -> re.Pattern:
    return re.compile(
        rf"(?:{labels})\s*[:\-–]\s*({value_re})",
        flags,
    )

# A "name word" = one or more capitalized/uppercase words (2–25 chars each)
_NAME_VALUE = r"[A-Z][A-Za-z]{1,24}(?:\s+[A-Z][A-Za-z]{1,24}){0,4}"

LABELED_PATTERNS: dict[str, re.Pattern] = {

    # ── Names ────────────────────────────────────────────────────────────────
    "Full Name": _lp(
        r"(?:name|full\s*name|applicant'?s?\s*name|passenger'?s?\s*name|"
        r"account\s*holder'?s?\s*name|customer\s*name|holder'?s?\s*name|"
        r"nominee\s*name|beneficiary\s*name|claimant\s*name)",
        _NAME_VALUE,
    ),
    "Father's / Mother's Name": _lp(
        r"(?:father'?s?\s*name|mother'?s?\s*name|parent'?s?\s*name|"
        r"s/o|d/o|w/o|f/o|son\s*of|daughter\s*of|wife\s*of)",
        _NAME_VALUE,
    ),
    "Spouse Name": _lp(
        r"(?:spouse'?s?\s*name|husband'?s?\s*name|wife'?s?\s*name)",
        _NAME_VALUE,
    ),

    # ── Address (labeled) ─────────────────────────────────────────────────────
    "Physical Address": _lp(
        r"(?:address|residential\s*address|permanent\s*address|"
        r"correspondence\s*address|current\s*address|home\s*address|"
        r"mailing\s*address|addr\.?)",
        r"[A-Za-z0-9 ,./\-#]{10,200}",
    ),

    # ── Birth & demographics ──────────────────────────────────────────────────
    "Place of Birth": _lp(
        r"(?:place\s*of\s*birth|birth\s*place|pob|born\s*(?:at|in))",
        r"[A-Za-z\s,]{3,60}",
    ),
    "Nationality": _lp(
        r"nationality",
        r"[A-Za-z]{3,30}",
    ),
    "Gender": _lp(
        r"(?:gender|sex)",
        r"(?:male|female|transgender|other|m|f)",
    ),
    "Marital Status": _lp(
        r"(?:marital\s*status|civil\s*status)",
        r"(?:single|married|divorced|widowed|separated)",
    ),
    "Religion": _lp(
        r"religion",
        r"[A-Za-z]{3,20}",
    ),
    "Blood Group": _lp(
        r"(?:blood\s*group|blood\s*type)",
        r"(?:A|B|AB|O)[+-]",
    ),
    "Occupation": _lp(
        r"(?:occupation|profession|designation|job\s*title|employment)",
        r"[A-Za-z\s]{3,50}",
    ),

    # ── Identification numbers (labeled — catches cases where structural
    #    regex might miss due to formatting or OCR spacing issues) ────────────
    "Aadhaar Number": _lp(
        r"(?:aadhaar|aadhaar\s*no\.?|uid|uidai\s*no\.?)",
        r"\d[\d\s]{10,13}\d",
    ),
    "PAN Number": _lp(
        r"(?:pan|pan\s*no\.?|permanent\s*account\s*number)",
        r"[A-Z]{5}[0-9]{4}[A-Z]",
    ),
    "Passport Number": _lp(
        r"(?:passport\s*no\.?|passport\s*number)",
        r"[A-PR-WY][0-9]{7}",
    ),
    "Voter ID": _lp(
        r"(?:voter\s*id|epic\s*no\.?|election\s*card\s*no\.?)",
        r"[A-Z]{3}[0-9]{7}",
    ),
}

# ---------------------------------------------------------------------------
# Confidential keyword list
# ---------------------------------------------------------------------------

CONFIDENTIAL_KEYWORDS = [
    "confidential", "internal use only", "do not distribute", "trade secret",
    "proprietary", "not for external distribution", "strictly confidential",
    "nda", "non-disclosure", "classified", "privileged", "sensitive",
    "for official use only", "restricted", "commercial in confidence",
]


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------

def _mask(category: str, value: str) -> str:
    value = value.strip()
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
        return value[0] + "***" + value[-4:] if len(value) >= 5 else "*****"
    if category == "Voter ID":
        return value[:3] + "****" + value[-3:] if len(value) >= 6 else "*******"
    if category == "Driving Licence Number":
        clean = value.replace(" ", "")
        return clean[:4] + "X" * max(len(clean) - 6, 0) + clean[-2:]
    if category == "GST Number":
        return value[:2] + "*" * 10 + value[-3:]
    if category == "Date of Birth":
        return re.sub(r"(\d{1,2})[/\-.](\d{1,2})[/.\-](\d{4})", r"**/***/\3", value)
    if category == "Physical Address":
        words = value.split()
        return " ".join(words[:2]) + " [ADDRESS REDACTED]" if words else "[ADDRESS REDACTED]"
    if category == "Vehicle Registration Number":
        return value[:4] + "****"
    # Names — show first letter + asterisks
    if category in ("Full Name", "Father's / Mother's Name", "Spouse Name"):
        parts = value.split()
        if parts:
            return parts[0][0] + "*** " + ("*" * len(parts[-1]) if len(parts) > 1 else "")
        return "****"
    # Short free-text fields — partial mask
    if category in ("Place of Birth", "Nationality", "Occupation", "Religion"):
        return value[:3] + "***" if len(value) > 3 else "***"
    # Very sensitive short values — full mask
    if category in ("Gender", "Marital Status", "Blood Group"):
        return value   # not sensitive to display, but flag the presence
    return value


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

BANK_CONTEXT_HINTS    = re.compile(r"(?i)\b(account|a/c|acc\.?\s*no)\b")
AADHAAR_CONTEXT_HINTS = re.compile(r"(?i)\b(aadhaar|uidai)\b")
CONTEXT_WINDOW        = 40


def _luhn_valid(number: str) -> bool:
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


# ---------------------------------------------------------------------------
# Main detection entry point
# ---------------------------------------------------------------------------

def detect(text: str) -> List[Finding]:
    """Run all three detection layers and return a merged, de-duplicated list."""
    findings_map: dict[tuple, Finding] = {}

    # ── Layer 1: structural regex ──────────────────────────────────────────
    for category, pattern in PATTERNS.items():
        for m in pattern.finditer(text):
            candidate = m.group(0).strip()
            if not candidate:
                continue

            start   = max(0, m.start() - CONTEXT_WINDOW)
            context = text[start: m.start()]

            if category == "Credit Card Number":
                if not _luhn_valid(candidate):
                    continue
            if category == "Bank Account Number":
                digits = re.sub(r"\D", "", candidate)
                if len(digits) == 12 and not BANK_CONTEXT_HINTS.search(context):
                    continue
                if len(digits) == 10:
                    continue
                if _luhn_valid(candidate):
                    continue

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
                findings_map[key] = Finding(
                    category=effective_category, value=masked, raw_value=candidate
                )

    # ── Layer 2: labeled-field regex ───────────────────────────────────────
    for category, pattern in LABELED_PATTERNS.items():
        for m in pattern.finditer(text):
            # group(1) is the captured value after the label
            try:
                candidate = m.group(1).strip()
            except IndexError:
                candidate = m.group(0).strip()

            if not candidate or len(candidate) < 2:
                continue

            masked = _mask(category, candidate)
            key = (category, masked)
            if key in findings_map:
                findings_map[key].count += 1
            else:
                findings_map[key] = Finding(
                    category=category, value=masked, raw_value=candidate
                )

    # ── Layer 3: spaCy NER (optional) ─────────────────────────────────────
    if NER_AVAILABLE:
        try:
            doc = _nlp(text[:50_000])   # cap for performance
            for ent in doc.ents:
                ent_text = ent.text.strip()
                if len(ent_text) < 3:
                    continue

                if ent.label_ == "PERSON":
                    category = "Full Name"
                    masked   = _mask(category, ent_text)
                    key      = (category, masked)
                    if key in findings_map:
                        findings_map[key].count += 1
                    else:
                        findings_map[key] = Finding(
                            category=category, value=masked, raw_value=ent_text
                        )

                elif ent.label_ in ("GPE", "LOC"):
                    category = "Place / Location"
                    key      = (category, ent_text)
                    if key in findings_map:
                        findings_map[key].count += 1
                    else:
                        findings_map[key] = Finding(
                            category=category, value=ent_text, raw_value=ent_text
                        )
        except Exception:
            pass   # never crash the whole pipeline for NER errors

    # ── Confidential keyword check ─────────────────────────────────────────
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
