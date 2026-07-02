"""
risk.py
-------
Turns a list of detector.Finding objects into a Low / Medium / High risk
classification plus a human-readable rationale.

Scoring model
-------------
Each category has a severity weight. Risk score = sum(weight * count),
capped per-category so one repeated low-severity field (e.g. 50 emails)
doesn't drown out a single high-severity one (e.g. 1 Aadhaar number).

    HIGH_RISK_CATEGORIES   -> weight 10, cap 3 occurrences counted
    MEDIUM_RISK_CATEGORIES -> weight 4,  cap 5 occurrences counted
    LOW_RISK_CATEGORIES    -> weight 1,  cap 5 occurrences counted

Thresholds:
    score >= 20          -> High Risk
    5 <= score < 20       -> Medium Risk
    0 < score < 5         -> Low Risk
    score == 0            -> Low Risk (no sensitive data found)

High-risk categories
--------------------
  Aadhaar Number, PAN Number, Passport Number, Voter ID,
  Driving Licence Number, Credit Card Number, Bank Account Number,
  API Key / Secret, Password, IFSC Code

Medium-risk categories
----------------------
  Phone Number, Employee ID, GST Number,
  Vehicle Registration Number, Confidential Business Information

Low-risk categories
-------------------
  Email Address, Date of Birth, Physical Address
"""

from __future__ import annotations

from typing import List
from detector import Finding

HIGH_RISK_CATEGORIES = {
    # Government-issued identity documents
    "Aadhaar Number", "PAN Number", "Passport Number",
    "Voter ID", "Driving Licence Number",
    # Financial
    "Credit Card Number", "Bank Account Number", "IFSC Code",
    # Credentials
    "API Key / Secret", "Password",
}
MEDIUM_RISK_CATEGORIES = {
    "Phone Number", "Employee ID", "GST Number",
    "Vehicle Registration Number", "Confidential Business Information",
}
LOW_RISK_CATEGORIES = {
    "Email Address", "Date of Birth", "Physical Address",
}

WEIGHTS = {"high": 10, "medium": 4, "low": 1}
CAPS = {"high": 3, "medium": 5, "low": 5}


def _tier(category: str) -> str:
    if category in HIGH_RISK_CATEGORIES:
        return "high"
    if category in MEDIUM_RISK_CATEGORIES:
        return "medium"
    return "low"


def classify(findings: List[Finding]) -> dict:
    score = 0
    breakdown = []

    for f in findings:
        tier = _tier(f.category)
        counted = min(f.count, CAPS[tier])
        contribution = counted * WEIGHTS[tier]
        score += contribution
        breakdown.append({
            "category": f.category,
            "tier": tier,
            "count": f.count,
            "contribution": contribution,
        })

    if score >= 20:
        level = "High Risk"
    elif score >= 5:
        level = "Medium Risk"
    else:
        level = "Low Risk"

    return {
        "level": level,
        "score": score,
        "breakdown": sorted(breakdown, key=lambda b: -b["contribution"]),
    }
