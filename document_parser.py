"""
document_parser.py
-------------------
Extracts raw text content from uploaded documents.

Supported formats
-----------------
- PDF  : text-based (pdfplumber) with automatic OCR fallback for scanned pages
- IMAGE: JPG, JPEG, PNG, BMP, TIFF — direct OCR via pytesseract
- TXT  : plain text with encoding fallback
- CSV  : parsed with pandas, flattened to text

OCR strategy
------------
A "scanned" page is one where pdfplumber returns fewer than 30 characters.
For those pages (and for direct image uploads), we run pytesseract OCR at
300 DPI so sensitive data in photographs / scans is NEVER missed.

Crash-safety
------------
All optional imports (pytesseract, pdf2image, Pillow) are done lazily inside
functions so a missing library never prevents the module from loading.
"""

from __future__ import annotations

import io
from typing import Union

import pandas as pd

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

# Minimum characters per page before we treat it as image-only and need OCR
_OCR_THRESHOLD = 30

# Supported image extensions for direct upload
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"}


class UnsupportedFileError(Exception):
    """Raised when a file type is not supported."""


# ---------------------------------------------------------------------------
# OCR helpers — lazy-loaded, never crash at import time
# ---------------------------------------------------------------------------

def _ocr_available() -> bool:
    """Return True if both pytesseract and pdf2image are installed."""
    try:
        import pytesseract          # noqa: F401
        from pdf2image import convert_from_bytes  # noqa: F401
        return True
    except Exception:
        return False


def _ocr_image(img) -> str:
    """Run pytesseract on a PIL Image object. Returns extracted text."""
    try:
        import pytesseract
        return pytesseract.image_to_string(img, lang="eng")
    except Exception as e:
        return f"[OCR ERROR: {e}]"


def _pdf_pages_to_images(file_bytes: bytes):
    """Convert PDF bytes to list of PIL Images at 300 DPI."""
    from pdf2image import convert_from_bytes
    return convert_from_bytes(file_bytes, dpi=300)


def _open_image(file_bytes: bytes):
    """Open raw image bytes as a PIL Image."""
    from PIL import Image
    return Image.open(io.BytesIO(file_bytes))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_document(file: Union[str, io.BytesIO], filename: str) -> str:
    """
    Extract plain text from an uploaded document.

    Parameters
    ----------
    file     : file-like object (e.g. Streamlit UploadedFile) or path string
    filename : original filename — used only to detect the extension

    Returns
    -------
    str : extracted text content (may include [OCR] page markers)
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "pdf":
        return _parse_pdf(file)
    elif ext in IMAGE_EXTENSIONS:
        return _parse_image(file)
    elif ext == "txt":
        return _parse_txt(file)
    elif ext == "csv":
        return _parse_csv(file)
    else:
        raise UnsupportedFileError(
            f"Unsupported file type '.{ext}'. "
            f"Please upload a PDF, image (JPG/PNG), TXT, or CSV file."
        )


# ---------------------------------------------------------------------------
# Format-specific parsers
# ---------------------------------------------------------------------------

def _read_bytes(file) -> bytes:
    """Read bytes from a file-like object or path."""
    if hasattr(file, "read"):
        return file.read()
    with open(file, "rb") as f:
        return f.read()


def _parse_pdf(file) -> str:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is not installed.")

    file_bytes = _read_bytes(file)
    text_chunks = []
    ocr_pages   = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""

            # Also pull text from embedded tables
            for table in page.extract_tables() or []:
                for row in table:
                    row_text = " | ".join(cell or "" for cell in row)
                    page_text += "\n" + row_text

            if len(page_text.strip()) < _OCR_THRESHOLD:
                # Image-only page → queue for OCR
                ocr_pages.append(page_num)
                text_chunks.append(f"\n--- Page {page_num} (image-only, OCR pending) ---")
            else:
                text_chunks.append(f"\n--- Page {page_num} ---\n{page_text}")

    # OCR fallback for image-only pages
    if ocr_pages:
        if not _ocr_available():
            text_chunks.append(
                "\n[WARNING] This PDF contains scanned images with no embedded text. "
                "Install pytesseract and pdf2image (+ system package tesseract-ocr) "
                "to enable OCR detection of sensitive data in scanned documents."
            )
        else:
            try:
                images = _pdf_pages_to_images(file_bytes)
                for page_num in ocr_pages:
                    idx = page_num - 1
                    if idx < len(images):
                        ocr_text    = _ocr_image(images[idx])
                        placeholder = f"\n--- Page {page_num} (image-only, OCR pending) ---"
                        ocr_chunk   = f"\n--- Page {page_num} [OCR] ---\n{ocr_text}"
                        text_chunks = [
                            ocr_chunk if c == placeholder else c
                            for c in text_chunks
                        ]
            except Exception as e:
                text_chunks.append(f"\n[OCR ERROR on PDF pages: {e}]")

    return "\n".join(text_chunks)


def _parse_image(file) -> str:
    """
    Directly OCR an uploaded image file (JPG, PNG, etc.).
    This handles the case where the user uploads a photo of a passport
    or ID card directly as an image — no PDF conversion needed.
    """
    file_bytes = _read_bytes(file)

    if not _ocr_available():
        return (
            "[WARNING] Image uploaded but OCR is not available. "
            "Install pytesseract, pdf2image, and Pillow to extract text from images."
        )

    try:
        img      = _open_image(file_bytes)
        ocr_text = _ocr_image(img)
        return f"--- Image [OCR] ---\n{ocr_text}"
    except Exception as e:
        return f"[OCR ERROR on image: {e}]"


def _parse_txt(file) -> str:
    raw = _read_bytes(file)
    if isinstance(raw, bytes):
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")
    return raw


def _parse_csv(file) -> str:
    try:
        df = pd.read_csv(file)
    except Exception:
        if hasattr(file, "seek"):
            file.seek(0)
        df = pd.read_csv(file, engine="python", sep=None, on_bad_lines="skip")

    lines = [" | ".join(str(c) for c in df.columns)]
    for _, row in df.iterrows():
        lines.append(" | ".join(str(v) for v in row.values))
    return "\n".join(lines)
