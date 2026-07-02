"""
summarizer.py
-------------
Generates a compliance/security summary from detector findings + risk
classification.

Two modes:
1. Rule-based (default, always available, zero external dependency/cost).
   Produces observations, risks, and remediation steps from templates driven
   by which categories were found.
2. LLM-enhanced (optional). If an OPENAI_API_KEY is set, we ask an LLM to
   turn the structured findings into a more natural, context-aware summary.
   The rule-based output is always computed first and passed to the LLM as
   grounding context, so the LLM is elaborating on verified findings rather
   than inventing them - this keeps the "compliance" summary trustworthy.
"""

from __future__ import annotations

import os
from typing import List, Optional

from detector import Finding
from risk import classify

REMEDIATION_MAP = {
    "Aadhaar Number": "Mask or tokenize Aadhaar numbers before storage; restrict access under DPDP Act guidelines.",
    "PAN Number": "Redact PAN numbers in logs/exports; encrypt at rest if retention is required.",
    "Credit Card Number": "Never store raw PAN/card numbers - use a PCI-DSS compliant vault or tokenization service.",
    "Bank Account Number": "Encrypt bank account data at rest and in transit; limit field visibility by role.",
    "IFSC Code": "Low sensitivity alone, but combined with account numbers it enables fraud - review co-location.",
    "API Key / Secret": "Rotate the exposed key/secret immediately and move it to a secrets manager (e.g. Vault, AWS Secrets Manager).",
    "Password": "Rotate any exposed credentials immediately; never store plaintext passwords in documents.",
    "Email Address": "Low individual risk, but bulk email lists should be access-controlled to avoid spam/phishing exposure.",
    "Phone Number": "Mask phone numbers in shared/exported copies; apply consent checks before external sharing.",
    "Employee ID": "Restrict document sharing to HR/authorized personnel only.",
    "Confidential Business Information": "Apply document classification labels and DLP (data-loss-prevention) controls before external sharing.",
}


def _rule_based_summary(findings: List[Finding], risk_result: dict) -> dict:
    categories_found = sorted({f.category for f in findings})

    observations = []
    if not findings:
        observations.append("No sensitive or confidential data patterns were detected in this document.")
    else:
        for f in findings:
            observations.append(f"Detected {f.count} instance(s) of {f.category} (e.g. {f.value}).")

    security_risks = []
    for cat in categories_found:
        if cat in REMEDIATION_MAP:
            security_risks.append(f"Exposure of {cat} could enable identity theft, fraud, or unauthorized access.")

    remediation = [REMEDIATION_MAP[c] for c in categories_found if c in REMEDIATION_MAP]
    if not remediation:
        remediation.append("No specific remediation required; continue standard data-handling practices.")

    return {
        "risk_level": risk_result["level"],
        "risk_score": risk_result["score"],
        "compliance_observations": observations,
        "security_risks": security_risks or ["No significant security risks identified."],
        "remediation_steps": remediation,
    }


def _llm_enhance(rule_based: dict, raw_text_excerpt: str) -> Optional[str]:
    """Optional: use an LLM to turn the structured rule-based summary into a
    more natural narrative. Requires OPENAI_API_KEY in the environment.
    Returns None (silently) if no key is configured or the call fails, so
    the app always degrades gracefully to the rule-based summary."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        prompt = (
            "You are a data-compliance assistant. Using ONLY the structured "
            "findings below (do not invent new findings), write a concise "
            "3-paragraph compliance summary: (1) what was found, "
            "(2) associated risks, (3) recommended next steps.\n\n"
            f"Structured findings:\n{rule_based}\n\n"
            f"Document excerpt (context only, do not quote sensitive values):\n"
            f"{raw_text_excerpt[:1500]}"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception:
        return None


def generate_summary(findings: List[Finding], raw_text: str = "") -> dict:
    risk_result = classify(findings)
    rule_based = _rule_based_summary(findings, risk_result)

    llm_narrative = _llm_enhance(rule_based, raw_text)
    rule_based["llm_narrative"] = llm_narrative  # None if no API key configured
    return rule_based
