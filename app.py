"""
app.py
------
Streamlit UI for the Sensitive Data Detection & Compliance Assistant.

Flow: Upload -> Detect -> Classify -> Summarize -> Ask Questions
"""

import io
import streamlit as st

from document_parser import parse_document, UnsupportedFileError
from detector import detect, summarize_counts
from risk import classify
from summarizer import generate_summary
from qa_engine import answer_question

st.set_page_config(
    page_title="Sensitive Data Detection & Compliance Assistant",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in {
    "text": None,
    "findings": None,
    "risk_result": None,
    "summary": None,
    "filename": None,
    "chat_history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

RISK_COLORS  = {"Low Risk": "🟢", "Medium Risk": "🟡", "High Risk": "🔴"}
RISK_BG      = {"Low Risk": "#d4edda", "Medium Risk": "#fff3cd", "High Risk": "#f8d7da"}
RISK_FG      = {"Low Risk": "#155724", "Medium Risk": "#856404", "High Risk": "#721c24"}
TIER_BADGE   = {"high": "🔴 High",   "medium": "🟡 Medium",    "low": "🟢 Low"}

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ Compliance Assistant")
    st.caption("AI-powered sensitive data detection & compliance summary")

    uploaded_file = st.file_uploader(
        "Upload a document", type=["pdf", "txt", "csv"],
        help="Supported formats: PDF, TXT, CSV"
    )

    analyze_clicked = st.button("Analyze Document", type="primary", use_container_width=True)

    st.divider()
    st.caption("Optional: set OPENAI_API_KEY as an environment variable to "
               "enable LLM-enhanced summaries and natural-language Q&A. "
               "Without it, the app runs fully offline using rule-based logic.")
    st.divider()
    # OCR status
    try:
        import pytesseract
        from pdf2image import convert_from_bytes  # noqa: F401
        st.success("🔍 OCR enabled — scanned PDFs (e.g. passport, Aadhaar) will be analysed.")
    except ImportError:
        st.warning("⚠️ OCR not available. Scanned image PDFs won't be detected. "
                   "Install `pytesseract` and `pdf2image` to enable.")
    # NER status
    try:
        import spacy
        spacy.load("en_core_web_sm")
        st.success("🧠 NER enabled — Names, places & free-text PII will be detected.")
    except Exception:
        st.warning("⚠️ NER not available. Names/places in free text won't be detected. "
                   "Install `spacy` + `en_core_web_sm` to enable.")

    if st.session_state.filename:
        st.success(f"Loaded: {st.session_state.filename}")
        if st.button("Reset", use_container_width=True):
            for key in ["text", "findings", "risk_result", "summary", "filename", "chat_history"]:
                st.session_state[key] = None if key != "chat_history" else []
            st.rerun()

# ---------------------------------------------------------------------------
# Analyze pipeline
# ---------------------------------------------------------------------------
if analyze_clicked:
    if uploaded_file is None:
        st.sidebar.error("Please upload a file first.")
    else:
        with st.spinner("Extracting text..."):
            try:
                file_bytes = io.BytesIO(uploaded_file.getvalue())
                text = parse_document(file_bytes, uploaded_file.name)
            except UnsupportedFileError as e:
                st.sidebar.error(str(e))
                text = None
            except Exception as e:
                st.sidebar.error(f"Failed to parse document: {e}")
                text = None

        if text is not None:
            with st.spinner("Detecting sensitive data..."):
                findings = detect(text)
                risk_result = classify(findings)

            with st.spinner("Generating compliance summary..."):
                summary = generate_summary(findings, text)

            st.session_state.text = text
            st.session_state.findings = findings
            st.session_state.risk_result = risk_result
            st.session_state.summary = summary
            st.session_state.filename = uploaded_file.name
            st.session_state.chat_history = []
            # Flag if OCR was used (page marker present in extracted text)
            st.session_state.ocr_used = "[OCR]" in text
            st.rerun()

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Sensitive Data Detection & Compliance Assistant")

if st.session_state.text is None:
    st.info("👈 Upload a PDF, TXT, or CSV file in the sidebar and click **Analyze Document** to get started.")
    st.markdown("""
This tool will:
1. **Detect** sensitive data (Aadhaar, PAN, emails, phone numbers, card numbers, bank details, API keys, employee IDs, confidential markers)
2. **Classify** the document's overall risk level
3. **Generate** a compliance/security summary with remediation steps
4. **Answer questions** about what was found
    """)
else:
    findings = st.session_state.findings
    risk_result = st.session_state.risk_result
    summary = st.session_state.summary

    # OCR notice banner
    if st.session_state.get("ocr_used"):
        st.info(
            "🔍 **OCR was used on this document.** It appears to be a scanned image "
            "(e.g. a passport or ID photo converted to PDF). Text was extracted via "
            "Optical Character Recognition — detection accuracy depends on image quality."
        )

    tab_overview, tab_findings, tab_summary, tab_qa = st.tabs(
        ["📊 Overview", "🔍 Findings", "📋 Compliance Summary", "💬 Ask Questions"]
    )

    # --- Overview ------------------------------------------------------------
    with tab_overview:
        level = risk_result["level"]
        score = risk_result["score"]
        bg    = RISK_BG.get(level, "#e2e3e5")
        fg    = RISK_FG.get(level, "#383d41")
        icon  = RISK_COLORS.get(level, "")

        # Big coloured risk banner
        st.markdown(
            f"""
            <div style="background:{bg}; color:{fg}; border-radius:10px;
                        padding:20px 28px; margin-bottom:18px;">
                <h2 style="margin:0; font-size:2rem;">{icon} {level}</h2>
                <p style="margin:4px 0 0; font-size:1rem;">
                    Risk Score: <strong>{score}</strong> &nbsp;|&nbsp;
                    Thresholds: 🟢&nbsp;Low&nbsp;&lt;5 &nbsp; 🟡&nbsp;Medium&nbsp;5–19 &nbsp; 🔴&nbsp;High&nbsp;≥20
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Level", f"{icon} {level}")
        with col2:
            st.metric("Risk Score", score)
        with col3:
            st.metric("Sensitive Data Categories", len(summarize_counts(findings)))

        # Risk level legend
        with st.expander("📖 How risk levels are calculated"):
            st.markdown("""
            | Tier | Categories | Weight | Score Threshold |
            |------|-----------|--------|-----------------|
            | 🔴 **High** | Aadhaar, PAN, Passport Number, Voter ID, Driving Licence, Credit Card, Bank Account, API Key, Password, IFSC | 10 pts each | Score ≥ 20 |
            | 🟡 **Medium** | Phone Number, Employee ID, GST Number, Vehicle Registration, Confidential Business Info | 4 pts each | Score 5–19 |
            | 🟢 **Low** | Email Address, Date of Birth, Physical Address | 1 pt each | Score < 5 |
            """)

        counts = summarize_counts(findings)
        if counts:
            st.subheader("Detected Categories")
            st.bar_chart(counts)
        else:
            st.success("No sensitive data patterns detected in this document.")

    # --- Findings --------------------------------------------------------------
    with tab_findings:
        st.subheader("Detailed Findings (values masked)")
        if not findings:
            st.write("No findings to display.")
        else:
            # Build a tier lookup from the breakdown
            tier_map = {b["category"]: b["tier"] for b in risk_result["breakdown"]}
            rows = [
                {
                    "Risk Tier": TIER_BADGE.get(tier_map.get(f.category, "low"), "🟢 Low"),
                    "Category": f.category,
                    "Masked Value": f.value,
                    "Occurrences": f.count,
                }
                for f in findings
            ]
            # Sort so High → Medium → Low
            tier_order = {"🔴 High": 0, "🟡 Medium": 1, "🟢 Low": 2}
            rows.sort(key=lambda r: tier_order.get(r["Risk Tier"], 9))
            st.dataframe(rows, use_container_width=True, hide_index=True)

        with st.expander("Risk score breakdown"):
            st.json(risk_result["breakdown"])

    # --- Compliance Summary ------------------------------------------------------
    with tab_summary:
        st.subheader(f"{RISK_COLORS.get(risk_result['level'], '')} {risk_result['level']}")

        if summary.get("llm_narrative"):
            st.markdown(summary["llm_narrative"])
            st.caption("Generated with LLM assistance, grounded in structured findings below.")

        st.markdown("**Compliance Observations**")
        for obs in summary["compliance_observations"]:
            st.write(f"- {obs}")

        st.markdown("**Security Risks**")
        for r in summary["security_risks"]:
            st.write(f"- {r}")

        st.markdown("**Suggested Remediation Steps**")
        for step in summary["remediation_steps"]:
            st.write(f"- {step}")

    # --- Q&A ---------------------------------------------------------------------
    with tab_qa:
        st.subheader("Ask a question about this document")
        st.caption('Try: "What sensitive data exists in the document?", '
                    '"How many email addresses are present?", "Summarize this document."')

        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(msg)

        user_q = st.chat_input("Ask a question...")
        if user_q:
            st.session_state.chat_history.append(("user", user_q))
            with st.spinner("Thinking..."):
                answer = answer_question(user_q, st.session_state.text, findings)
            st.session_state.chat_history.append(("assistant", answer))
            st.rerun()
