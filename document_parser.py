"""
document_parser.py
-------------------
Extracts raw text content from uploaded documents (PDF, TXT, CSV).

Design notes:
- PDF parsing uses pdfplumber (pure-Python, no external binary deps, good
  text-layout preservation which helps the regex detectors below).
- SCANNED / IMAGE-BASED PDFs (e.g. a passport photo converted to PDF) have
  NO embedded text. pdfplumber returns empty strings for those pages.
  In that case we fall back to OCR via pytesseract + pdf2image so that
  sensitive data in scans is NOT missed.
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

try:
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_AVAILABLE = True
except ImportError:  # pragma: no cover
    OCR_AVAILABLE = False

# Minimum characters per page before we consider it "image-only" and need OCR
_OCR_THRESHOLD = 30


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

    # Read bytes once so we can pass to both pdfplumber and pdf2image
    if hasattr(file, "read"):
        file_bytes = file.read()
    else:
        with open(file, "rb") as f:
            file_bytes = f.read()

    text_chunks = []
    ocr_pages = []   # track page numbers that had no embedded text

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""

            # Also pull text out of any tables
            for table in page.extract_tables() or []:
                for row in table:
                    row_text = " | ".join(cell or "" for cell in row)
                    page_text += "\n" + row_text

            if len(page_text.strip()) < _OCR_THRESHOLD:
                # Page has no embedded text → mark for OCR
                ocr_pages.append(page_num)
                text_chunks.append(f"\n--- Page {page_num} (OCR pending) ---")
            else:
                text_chunks.append(f"\n--- Page {page_num} ---\n{page_text}")

    # OCR fallback for image-only pages (e.g. scanned passport, Aadhaar)
    if ocr_pages:
        if not OCR_AVAILABLE:
            # Warn but don't crash — partial text is better than an error
            text_chunks.append(
                "\n[WARNING] This PDF appears to be a scanned image. "
                "Install pytesseract and pdf2image to enable OCR and detect "
                "sensitive data inside scanned documents."
            )
        else:
            try:
                images = convert_from_bytes(file_bytes, dpi=300)
                for page_num in ocr_pages:
                    idx = page_num - 1
                    if idx < len(images):
                        ocr_text = pytesseract.image_to_string(
                            images[idx], lang="eng"
                        )
                        # Replace the placeholder we added earlier
                        placeholder = f"\n--- Page {page_num} (OCR pending) ---"
                        ocr_chunk = (
                            f"\n--- Page {page_num} [OCR] ---\n{ocr_text}"
                        )
                        text_chunks = [
                            ocr_chunk if c == placeholder else c
                            for c in text_chunks
                        ]
            except Exception as e:
                text_chunks.append(f"\n[OCR ERROR] {e}")

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
