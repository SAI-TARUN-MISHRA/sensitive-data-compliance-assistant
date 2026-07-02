"""
qa_engine.py
------------
Question answering over the uploaded document.

Two layers:
1. Intent-matched answers for the common compliance questions explicitly
   listed in the assignment brief (counts, "what data exists", "summarize",
   "what are the risks"). These are answered directly from the structured
   findings/summary - fast, free, and 100% grounded (no hallucination risk).
2. Fallback: simple keyword/TF-based retrieval over the document text for
   open-ended questions, optionally handed to an LLM (if OPENAI_API_KEY is
   set) along with the retrieved snippets for a natural-language answer.
   This is a minimal RAG pattern: retrieve relevant chunks -> feed to LLM
   as grounding context -> generate answer. Without an API key it falls
   back to returning the best-matching snippet directly.
"""

from __future__ import annotations

import os
import re
from typing import List

from detector import Finding, summarize_counts
from risk import classify
from summarizer import generate_summary, _get_llm_client


def _chunk_text(text: str, chunk_size: int = 400) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)] or [""]


def _keyword_retrieve(question: str, chunks: List[str], top_k: int = 3) -> List[str]:
    q_words = set(re.findall(r"\w+", question.lower()))
    scored = []
    for chunk in chunks:
        c_words = re.findall(r"\w+", chunk.lower())
        score = sum(1 for w in c_words if w in q_words)
        scored.append((score, chunk))
    scored.sort(key=lambda x: -x[0])
    return [c for s, c in scored[:top_k] if s > 0] or chunks[:1]


def _llm_answer(question: str, context: str) -> str | None:
    """Use Groq or OpenAI to answer a question given retrieved context."""
    client, model, _ = _get_llm_client()
    if client is None:
        return None
    try:
        prompt = (
            "Answer the user's question using ONLY the context below. "
            "If the context does not contain the answer, say so plainly. "
            "Do not reproduce full sensitive values (e.g. full card/Aadhaar numbers) "
            "even if present in context — refer to them by category instead.\n\n"
            f"Context:\n{context[:3000]}\n\nQuestion: {question}"
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception:
        return None


def answer_question(question: str, text: str, findings: List[Finding]) -> str:
    q = question.lower().strip()
    counts = summarize_counts(findings)

    # --- Intent 1: "what sensitive data exists" -------------------------------
    if any(p in q for p in ["what sensitive data", "what data exists", "what pii", "list the sensitive"]):
        if not counts:
            return "No sensitive data was detected in this document."
        lines = [f"- {cat}: {n} instance(s)" for cat, n in counts.items()]
        return "The following sensitive data categories were detected:\n" + "\n".join(lines)

    # --- Intent 2: "how many X are present" ------------------------------------
    count_match = re.search(r"how many ([a-zA-Z /]+?) (?:are|is|were)", q)
    if count_match:
        target = count_match.group(1).strip()
        for cat, n in counts.items():
            if target in cat.lower() or cat.lower() in target:
                return f"There are {n} {cat} instance(s) in this document."
        return f"No matches found for '{target}' in the detected sensitive data categories."

    # --- Intent 3: summarize the document --------------------------------------
    if "summarize" in q or "summary" in q:
        summary = generate_summary(findings, text)
        if summary.get("llm_narrative"):
            return summary["llm_narrative"]
        obs = "\n".join(f"- {o}" for o in summary["compliance_observations"])
        return f"Risk Level: {summary['risk_level']}\n\nObservations:\n{obs}"

    # --- Intent 4: compliance risks ---------------------------------------------
    if "compliance risk" in q or "what risk" in q or "risks identified" in q:
        summary = generate_summary(findings, text)
        return "\n".join(f"- {r}" for r in summary["security_risks"])

    # --- Fallback: retrieval over raw text (+ optional LLM) ----------------------
    chunks = _chunk_text(text)
    top_chunks = _keyword_retrieve(question, chunks)
    context = "\n---\n".join(top_chunks)

    llm_resp = _llm_answer(question, context)
    if llm_resp:
        return llm_resp

    snippet = top_chunks[0].strip()
    if not snippet:
        return "I couldn't find relevant content in the document to answer that question."
    return f"Based on the document, the most relevant section is:\n\n\"{snippet[:500]}...\""
