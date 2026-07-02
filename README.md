# Sensitive Data Detection & Compliance Assistant

> **Live Demo:** https://sai-tarun-mishra-sensitive-data-compliance-assistant-app-iz48gg.streamlit.app
> **GitHub:** https://github.com/SAI-TARUN-MISHRA/sensitive-data-compliance-assistant

An AI-powered app that ingests a document (PDF / image / TXT / CSV), detects sensitive or confidential information using a 3-layer detection engine, classifies the document's risk level, generates a compliance/security summary, and answers natural-language questions about what was found.

Built for the **Proteccio Data AI Research Innovation Internship** assignment.

---

## 1. Setup Instructions

### Prerequisites
- Python 3.10+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed (for scanned PDF / image support)
- [Poppler](https://poppler.freedesktop.org/) installed (for pdf2image)

### Install & Run Locally

```bash
git clone https://github.com/SAI-TARUN-MISHRA/sensitive-data-compliance-assistant.git
cd sensitive-data-compliance-assistant

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Windows — Install System Dependencies

```bash
# Tesseract (OCR engine)
winget install UB-Mannheim.TesseractOCR

# Or download installer from:
# https://github.com/UB-Mannheim/tesseract/wiki
```

### Optional: Enable LLM-Enhanced Summaries & Q&A

By default the app runs **fully offline** (no API key, no cost, no internet dependency).

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-...
```

If the key isn't set, the app silently falls back to rule-based logic — nothing breaks.

### Try It Quickly

A sample document with fake/synthetic PII is at `sample_data/sample.txt` — upload it to see the full pipeline in action.

---

## 2. Architecture Overview

```
                ┌─────────────────────┐
   Upload  ───► │  document_parser.py  │  PDF (pdfplumber + OCR fallback)
   PDF/IMG/     │                     │  Image (pytesseract direct OCR)
   TXT/CSV      │                     │  TXT / CSV (pandas)
                └──────────┬──────────┘
                           │ raw text
                           ▼
                ┌─────────────────────┐
                │    detector.py      │  LAYER 1: Structural Regex
                │                     │    Aadhaar, PAN, Passport, Voter ID,
                │                     │    Driving Licence, Credit Card (Luhn),
                │                     │    Bank Account, IFSC, GST, Vehicle Reg,
                │                     │    API Key, Password, Employee ID,
                │                     │    Phone, Email, Date of Birth
                │                     │
                │                     │  LAYER 2: Labeled-Field Regex
                │                     │    "Name:", "Address:", "Father's Name:",
                │                     │    "Nationality:", "Gender:", "DOB:",
                │                     │    "Place of Birth:", "Blood Group:" ...
                │                     │
                │                     │  LAYER 3: spaCy NER (optional)
                │                     │    PERSON → Full Name
                │                     │    GPE/LOC → Place / Location
                └──────────┬──────────┘
                           │ List[Finding] (masked values only)
                           ▼
                ┌─────────────────────┐
                │      risk.py        │  Weighted scoring model
                │                     │  → Low / Medium / High
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   summarizer.py     │  Rule-based compliance summary
                │                     │  + optional LLM narrative (OpenAI)
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │    qa_engine.py     │  Intent-matched Q&A (grounded)
                │                     │  + keyword retrieval (RAG pattern)
                │                     │  + optional LLM (OpenAI)
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │      app.py         │  Streamlit UI
                │                     │  Overview / Findings / Summary / Q&A
                └─────────────────────┘
```

| File | Responsibility |
|------|---------------|
| `document_parser.py` | Text extraction from PDF (pdfplumber + OCR fallback for scanned pages), direct image OCR (JPG/PNG), TXT (encoding fallbacks), CSV (pandas). All OCR imports are lazy-loaded — app never crashes if libraries are missing. |
| `detector.py` | 4-layer PII detection: (1) structural regex for fixed-format IDs, (2) labeled-field regex for form fields like "Name:", "Address:", (3) optional spaCy NER for free-text names, and (4) **Deep LLM Extraction** (via Groq/OpenAI) for unstructured semantic PII (e.g. project names, proprietary technologies, custom health details). |
| `risk.py` | Weighted scoring: HIGH (10pts) → identity docs + credentials; MEDIUM (4pts) → contact + business info; LOW (1pts) → email, DOB, address. Per-category caps prevent one repeated low-risk field from drowning high-risk ones. |
| `summarizer.py` | Compliance observations, security risks, and remediation steps from category-keyed templates. Optional LLM (Groq/OpenAI) narrates verified findings into prose — never invents findings. |
| `qa_engine.py` | Answers assignment questions directly from structured findings (grounded, zero hallucination). Falls back to keyword retrieval + optional LLM for open-ended questions — a minimal RAG pattern. |
| `app.py` | Streamlit UI wiring Upload → Analyze → 4-tab results. Color-coded risk banners, tier labels in findings table, OCR/NER/LLM status indicators. |

---

## 3. AI/ML Approach Used

### Detection — 4-Layer Hybrid Engine

**Layer 1 — Structural Regex:** Fixed-format entities (Aadhaar, PAN, Passport, credit card numbers etc.) are detected with regex + validation (Luhn checksum for cards). Context-window heuristics disambiguate overlapping formats (e.g. 12-digit bank account vs. Aadhaar).

**Layer 2 — Labeled-Field Regex:** Documents and forms have labelled fields like `Name: John Doe`, `Address: ...`, `Father's Name: ...`. This layer anchors on the label to capture values that have no fixed structural format — solving the "name detection" problem without NER.

**Layer 3 — spaCy NER (optional):** When spaCy + `en_core_web_sm` is available, Named Entity Recognition catches `PERSON` (full names) and `GPE/LOC` (place names) in free-running text without labels. Lazy-loaded so a missing model never crashes the app.

**Layer 4 — Deep LLM Extraction:** When an API key (Groq/OpenAI) is set, a zero-shot semantic extraction pass is run. The model parses the text to identify custom confidential terms, project names (e.g. "Project Apollo"), proprietary keywords, and complex PII that rules can't detect. Findings are merged with Layers 1–3 and masked.

### Risk Classification — Weighted Scoring

Each category has a severity tier with a weight and per-category cap:

| Tier | Weight | Cap | Examples |
|------|--------|-----|---------|
| HIGH | 10 pts | 3 | Aadhaar, PAN, Passport, Credit Card, API Key, Full Name |
| MEDIUM | 4 pts | 5 | Phone, Employee ID, GST, Nationality |
| LOW | 1 pt | 5 | Email, DOB, Physical Address, Gender |

Score ≥ 20 → High Risk | 5–19 → Medium Risk | < 5 → Low Risk

### Summarization & Q&A — Grounded Generation (RAG Pattern)

Compliance summaries and Q&A answers are **always computed first from verified structured findings** (no hallucination). An LLM (if `OPENAI_API_KEY` is set) is used only to narrate those facts more naturally. For open-ended questions, relevant document chunks are retrieved by keyword overlap and passed as context — a minimal RAG (Retrieval-Augmented Generation) pattern.

### OCR — Image & Scanned PDF Support

- **Scanned PDFs** (e.g. passport photo → PDF): detected by low character count per page, then OCR'd via `pytesseract` + `pdf2image` at 300 DPI.
- **Direct image uploads** (JPG, PNG, BMP, TIFF): OCR'd directly via `pytesseract`.
- All OCR imports are lazy-loaded — if libraries are unavailable, the app warns the user but never crashes.

---

## 4. Challenges Faced

- **Overlapping numeric formats:** A 12-digit bank account number and a 12-digit Aadhaar number are structurally identical. Solved with context-window heuristics ("Account No." nearby) and label-based disambiguation.
- **False positives on long digit strings:** Reference numbers and timestamps can look like card/account numbers. Luhn validation removed most card false positives; bank accounts are filtered by digit count and context.
- **Name detection without ML:** Pure regex can't reliably detect names (they're just words). Solved with a two-pronged approach: labeled-field patterns (`Name: ...`) catch form-style documents; spaCy NER catches free-text names.
- **Scanned document leakage:** Image-based PDFs return empty text from pdfplumber — sensitive data would be completely missed. Added per-page character count check with automatic OCR fallback.
- **Crash-safety on optional libraries:** If spaCy, pytesseract, or pdf2image fail to import, the entire app must still work. Solved by lazy-loading all optional dependencies inside try/except blocks.
- **LLM hallucination in compliance context:** A compliance tool that invents findings is dangerous. LLM is used strictly as a narrator of already-verified rule-based findings.

---

## 5. Future Improvements

- **Redacted document export:** Allow downloading a version of the uploaded file with PII replaced by `[REDACTED]` markers.
- **Embedding-based RAG:** Replace keyword retrieval with FAISS/ChromaDB for better recall on paraphrased questions.
- **Multi-document support:** Upload and compare multiple documents, with a persistent findings dashboard.
- **Audit logging:** Record who uploaded what, when, and what was flagged — required for real compliance use cases.
- **Dockerization:** Package the app with all system dependencies (tesseract-ocr, poppler) for one-command deployment.
- **Fine-tuned NER:** Train a custom spaCy/transformer NER model on Indian PII for higher precision on names, addresses, and regional IDs.
- **DPDP Act / GDPR compliance rules:** Add jurisdiction-specific rule sets to flag data subject rights violations.

---

## 6. Sensitive Data Categories Detected

| Category | Risk Tier | Detection Method |
|----------|-----------|-----------------|
| Aadhaar Number | 🔴 HIGH | Structural regex + context |
| PAN Number | 🔴 HIGH | Structural regex |
| Passport Number | 🔴 HIGH | Structural regex + label |
| Voter ID (EPIC) | 🔴 HIGH | Structural regex + label |
| Driving Licence | 🔴 HIGH | Structural regex |
| Credit Card Number | 🔴 HIGH | Regex + Luhn checksum |
| Bank Account Number | 🔴 HIGH | Regex + context |
| IFSC Code | 🔴 HIGH | Structural regex |
| API Key / Secret | 🔴 HIGH | Keyword + value regex |
| Password | 🔴 HIGH | Keyword + value regex |
| Full Name | 🔴 HIGH | Labeled-field regex + NER |
| Father's / Mother's Name | 🔴 HIGH | Labeled-field regex |
| Phone Number | 🟡 MEDIUM | Structural regex |
| Employee ID | 🟡 MEDIUM | Structural regex |
| GST Number | 🟡 MEDIUM | Structural regex |
| Vehicle Registration | 🟡 MEDIUM | Structural regex |
| Place of Birth | 🟡 MEDIUM | Labeled-field regex |
| Nationality | 🟡 MEDIUM | Labeled-field regex |
| Confidential Business Info | 🟡 MEDIUM | Keyword matching |
| Email Address | 🟢 LOW | Structural regex |
| Date of Birth | 🟢 LOW | Structural regex |
| Physical Address | 🟢 LOW | Labeled-field + structural |
| Gender | 🟢 LOW | Labeled-field regex |
| Blood Group | 🟢 LOW | Labeled-field regex |
| Place / Location | 🟢 LOW | spaCy NER |

---

## 7. Bonus Features Implemented

- ✅ **OCR support** — scanned PDFs + direct image uploads (JPG/PNG)
- ✅ **Data masking/redaction** — all sensitive values masked in UI (e.g. `XXXXXXXX9012`, `jo***@email.com`)
- ✅ **RAG implementation** — keyword retrieval + optional LLM context for Q&A
- ✅ **Dashboard/UI improvements** — color-coded risk banners, tier badges, interactive legend
- ✅ **Deployment** — live on Streamlit Cloud

---

## 8. Deployment

**Live app:** https://sai-tarun-mishra-sensitive-data-compliance-assistant-app-iz48gg.streamlit.app

To redeploy:
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, set `app.py` as entry point.
3. Streamlit Cloud reads `packages.txt` automatically and installs `tesseract-ocr` and `poppler-utils`.
4. (Optional) Add `OPENAI_API_KEY` under app **Secrets** for LLM-enhanced summaries.

---

## 9. Disclaimer

`sample_data/sample.txt` contains entirely **synthetic, fake data** for demonstration purposes only. No real Aadhaar/PAN/card numbers are included.
