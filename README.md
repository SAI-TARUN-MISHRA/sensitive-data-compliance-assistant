# Sensitive Data Detection & Compliance Assistant

An AI-powered app that ingests a document (PDF / TXT / CSV), detects sensitive
or confidential information, classifies the document's risk level, generates
a compliance/security summary, and answers natural-language questions about
what was found.

Built for the Proteccio Data AI Research Innovation Internship assignment.

---

## 1. Setup Instructions

### Prerequisites
- Python 3.10+

### Install & run

```bash
git clone <this-repo-url>
cd proteccio-sensitive-data-assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Optional: enable LLM-enhanced summaries & Q&A
By default the app runs **fully offline** using rule-based detection,
classification, summarization, and Q&A — no API key, no cost, no internet
dependency for the core evaluation criteria.

If you want the richer, natural-language layer on top:

```bash
cp .env.example .env
# edit .env and add your OpenAI API key
export OPENAI_API_KEY=sk-...      # or use `streamlit run` with the var set
```

If the key isn't set, the app silently falls back to the rule-based logic —
nothing breaks.

### Try it quickly
A sample document with fake/synthetic PII is included at
`sample_data/sample.txt` — upload it after launching the app to see the full
pipeline in action without needing a real document.

---

## 2. Architecture Overview

```
                ┌────────────────┐
   Upload  ───► │  document_parser│  PDF (pdfplumber) / TXT / CSV (pandas)
                └────────┬────────┘
                         │ raw text
                         ▼
                ┌────────────────┐
                │   detector.py   │  Regex + Luhn + context-aware rules
                │ (Aadhaar, PAN,  │  → List[Finding] (masked values only)
                │ email, phone,   │
                │ card, bank,     │
                │ API key, etc.)  │
                └────────┬────────┘
                         │ findings
                         ▼
                ┌────────────────┐
                │    risk.py      │  Weighted scoring → Low / Medium / High
                └────────┬────────┘
                         │ risk result
                         ▼
                ┌────────────────┐
                │  summarizer.py  │  Rule-based summary (always available)
                │                 │  + optional LLM narrative (if API key set)
                └────────┬────────┘
                         │
                         ▼
                ┌────────────────┐
                │   qa_engine.py  │  Intent-matched answers for the brief's
                │                 │  example questions + keyword retrieval /
                │                 │  optional LLM for open-ended questions
                └────────┬────────┘
                         │
                         ▼
                ┌────────────────┐
                │     app.py      │  Streamlit UI: Overview / Findings /
                │                 │  Compliance Summary / Ask Questions tabs
                └────────────────┘
```

**File-by-file:**

| File | Responsibility |
|---|---|
| `document_parser.py` | Extracts raw text from PDF (via `pdfplumber`, including tables), TXT (with encoding fallbacks), and CSV (via `pandas`, flattened to text). |
| `detector.py` | Regex-based detection for Aadhaar, PAN, email, phone, credit card (Luhn-validated), bank account/IFSC, API keys/secrets, passwords, employee IDs, plus keyword-based confidential-info detection. Masks all values before they ever reach the UI. |
| `risk.py` | Weighted scoring model (High/Medium/Low severity tiers, per-category caps) that maps findings → an overall Low/Medium/High risk classification with a breakdown. |
| `summarizer.py` | Builds compliance observations, security risks, and remediation steps from templates keyed to detected categories. Optionally asks an LLM to narrate the (already-verified) findings into prose. |
| `qa_engine.py` | Answers the assignment's example questions directly from structured findings (grounded, no hallucination), and falls back to keyword retrieval (+ optional LLM) for open-ended questions — a minimal RAG pattern. |
| `app.py` | Streamlit UI wiring the above into an Upload → Analyze → Overview/Findings/Summary/Q&A flow. |

---

## 3. AI/ML Approach Used

- **Detection layer — rule-based, not ML.** Aadhaar, PAN, card numbers, IFSC
  codes, and API keys all have fixed structural formats, so regex + checksum
  validation (Luhn for card numbers) is more precise and fully explainable
  than a generic NER model would be out of the box — important for a
  compliance tool where every flag needs a defensible reason. Context-window
  checks (e.g. "Account No." nearby) disambiguate overlapping formats, like a
  12-digit bank account number vs. a 12-digit Aadhaar number.
- **Risk classification — weighted scoring model.** Each category is
  assigned a severity tier (High/Medium/Low) with a weight and a per-category
  cap, so the score reflects *what kind* of data was found, not just *how
  many* fields — one Aadhaar number should outweigh fifty email addresses.
- **Summarization & Q&A — hybrid grounded-generation.** The compliance
  summary and question answering are always computed first from the
  structured, verified findings (so the "facts" are never hallucinated). An
  LLM (optional, `OPENAI_API_KEY`) is then used purely to *narrate* those
  verified facts more naturally, or to answer open-ended questions using a
  small retrieval step (chunk the document → keyword-match relevant chunks →
  pass as context) — a minimal Retrieval-Augmented Generation pattern. Without
  an API key, the retrieval step still returns the best-matching document
  snippet directly.
- **Why not a bigger model for everything?** A general-purpose NER/LLM
  approach was considered but rejected as the primary detector: it would add
  latency/cost, be less auditable ("why was this flagged?"), and regex already
  achieves near-perfect precision/recall on these highly-structured entity
  types. The LLM is used where it actually adds value — natural language
  generation and open-ended Q&A — not where deterministic rules are stronger.

---

## 4. Challenges Faced

- **Overlapping numeric formats.** A 12-digit bank account number and a
  12-digit Aadhaar number are structurally identical to a regex; disambiguation
  needed a context-window heuristic ("Account No." / "A/C" nearby) rather than
  pattern matching alone. This is still imperfect — a bank account number with
  no nearby label text could be mis-tagged as Aadhaar.
- **False positives on generic long digit strings.** Reference numbers,
  timestamps, and other long digit sequences can look like card/account
  numbers. Luhn validation for card numbers removed most false positives;
  bank account numbers currently have looser validation and are the most
  prone to over-flagging.
- **Keeping the LLM layer optional without breaking the "AI" story.** The
  brief asks for an AI-powered app, but a compliance tool that hallucinates
  findings is worse than useless. The chosen design keeps LLM usage strictly
  additive/narrative on top of verified rule-based findings, so the app is
  fully functional and demo-able with zero API keys/cost, while still having
  a real LLM integration point to show for evaluation.
- **PDF text extraction fidelity.** Tables (e.g. bank statements) sometimes
  don't extract cleanly via plain text extraction; `pdfplumber`'s table
  extraction is used as a second pass to catch tabular sensitive data.

---

## 5. Future Improvements

- Swap/augment the regex layer with a proper NER model (e.g. `spaCy` custom
  entity recognizer or a fine-tuned transformer) for unstructured PII like
  names and addresses, which regex can't reliably catch.
- Add OCR (e.g. `pytesseract`) for scanned/image-based PDFs.
- Add real data masking/redaction — export a redacted copy of the uploaded
  document, not just a findings report.
- Move from keyword retrieval to embedding-based retrieval (FAISS/ChromaDB)
  for the Q&A layer, for better recall on paraphrased questions.
- Multi-document support with a persistent findings history/dashboard.
- Audit logging (who uploaded what, when, what was flagged) for real
  compliance use.
- Dockerize and add a CI pipeline for automated regression testing of the
  detection rules against a labeled test-document set.

---

## 6. Deployment

To deploy on Streamlit Community Cloud:
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo,
   set `app.py` as the entry point.
3. (Optional) Add `OPENAI_API_KEY` under app **Secrets** if you want the
   LLM-enhanced layer live.

---

## 7. Disclaimer

`sample_data/sample.txt` contains entirely synthetic, fake data for
demonstration purposes only (no real Aadhaar/PAN/card numbers).
