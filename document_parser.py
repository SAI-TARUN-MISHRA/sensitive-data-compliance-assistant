"""
document_parser.py
-------------------
Extracts raw text content from uploaded documents (PDF, TXT, CSV).

Design notes:
- PDF parsing uses pdfplumber (pure-Python, no external binary deps, good
  text-layout preservation which helps the regex detectors below).
- CSV is parsed with pandas and flattened into a text representation so the
  same regex/NLP pipeline can run over spreadsheet data too.
- TXT is read directly with a couple of encoding fallbacks since real-world
  exports are not always clean UTF-8.
"""

from __future__ import annotations

import io
import csv
from typing import Union

import pandas as pd

try:
    import pdfplumber
except ImportError:  # pragma: no cover
    pdfplumber = None


class UnsupportedFileError(Exception):
    """Raised when a file type is not one of PDF / TXT / CSV."""


def parse_document(file: Union[str, io.BytesIO], filename: str) -> str:
    """
    Extract plain text from a document.

    Parameters
    ----------
    file : path-like str OR file-like object (e.g. Streamlit's UploadedFile)
    filename : original filename, used only to detect the extension

    Returns
    -------
    str : extracted text content
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "pdf":
        return _parse_pdf(file)
    elif ext == "txt":
        return _parse_txt(file)
    elif ext == "csv":
        return _parse_csv(file)
    else:
        raise UnsupportedFileError(
            f"Unsupported file type '.{ext}'. Please upload a PDF, TXT, or CSV file."
        )


def _parse_pdf(file) -> str:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is not installed. Run: pip install pdfplumber")

    text_chunks = []
    with pdfplumber.open(file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            text_chunks.append(f"\n--- Page {page_num} ---\n{page_text}")

            # Also pull text out of any tables (bank statements, PII tables etc.
            # often live inside PDF tables and extract_text() alone can miss them)
            for table in page.extract_tables() or []:
                for row in table:
                    row_text = " | ".join(cell or "" for cell in row)
                    text_chunks.append(row_text)

    return "\n".join(text_chunks)


def _parse_txt(file) -> str:
    if hasattr(file, "read"):
        raw = file.read()
        if isinstance(raw, bytes):
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    return raw.decode(enc)
                except UnicodeDecodeError:
                    continue
            return raw.decode("utf-8", errors="replace")
        return raw
    else:
        with open(file, "rb") as f:
            raw = f.read()
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")


def _parse_csv(file) -> str:
    try:
        df = pd.read_csv(file)
    except Exception:
        # Fall back to permissive parsing for messy delimiters/encodings
        if hasattr(file, "seek"):
            file.seek(0)
        df = pd.read_csv(file, engine="python", sep=None, on_bad_lines="skip")

    lines = [" | ".join(str(c) for c in df.columns)]
    for _, row in df.iterrows():
        lines.append(" | ".join(str(v) for v in row.values))
    return "\n".join(lines)
