"""
NexaAI Smart PDF Extractor
============================
Extraction pipeline per page:
  1. pdfplumber  — fast native text extraction (best for digital PDFs)
  2. PyMuPDF     — fallback, different rendering engine
  3. Vision OCR  — for scanned / image-only PDFs via OpenRouter

A PDF page is considered "image-only" when it yields fewer than
MIN_TEXT_PER_PAGE characters of readable text.
"""

import re
import io
import pdfplumber
import fitz           # PyMuPDF
from rag import openrouter

MIN_TEXT_PER_PAGE  = 40    # chars below this → assume image page
OCR_DPI            = 200   # resolution for rendering pages to image


def extract(filepath: str, api_key: str = "") -> tuple[str, dict]:
    """
    Extract text from a PDF file.

    Returns
    -------
    text     : str   — full extracted text (all pages combined)
    meta     : dict  — {pages, image_pages, method, warnings}
    """
    meta = {
        "pages":       0,
        "image_pages": 0,
        "method":      "pdfplumber",
        "warnings":    [],
    }

    pages_text = []

    # ── Primary: pdfplumber ─────────────────────────────────────
    try:
        with pdfplumber.open(filepath) as pdf:
            meta["pages"] = len(pdf.pages)
            plumber_pages = []
            for i, page in enumerate(pdf.pages, 1):
                raw = page.extract_text() or ""
                raw = raw.replace("\x00", "").strip()
                plumber_pages.append((i, raw))
    except Exception as e:
        meta["warnings"].append(f"pdfplumber failed: {e}")
        plumber_pages = []

    # ── Determine which pages need OCR ─────────────────────────
    needs_ocr = []
    for i, raw in plumber_pages:
        if len(raw) >= MIN_TEXT_PER_PAGE:
            pages_text.append(raw)
        else:
            needs_ocr.append(i)
            meta["image_pages"] += 1
            pages_text.append(None)   # placeholder

    # ── Secondary: PyMuPDF for pages pdfplumber couldn't read ───
    if needs_ocr:
        try:
            doc = fitz.open(filepath)
            for page_idx in [i-1 for i in needs_ocr]:
                page = doc[page_idx]
                pymupdf_text = page.get_text("text").replace("\x00", "").strip()
                if len(pymupdf_text) >= MIN_TEXT_PER_PAGE:
                    pages_text[page_idx] = pymupdf_text
                    needs_ocr = [n for n in needs_ocr if n != page_idx + 1]
                    meta["image_pages"] -= 1
            doc.close()
            if pages_text and plumber_pages:
                meta["method"] = "pdfplumber+pymupdf"
        except Exception as e:
            meta["warnings"].append(f"PyMuPDF fallback failed: {e}")

    # ── Tertiary: Vision OCR via OpenRouter ─────────────────────
    remaining_ocr = [i for i in needs_ocr if pages_text[i-1] is None]

    if remaining_ocr and openrouter.is_available(api_key):
        meta["method"] += "+vision_ocr"
        try:
            doc = fitz.open(filepath)
            for page_num in remaining_ocr:
                page = doc[page_num - 1]
                mat  = fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72)
                pix  = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")

                ocr_text = openrouter.ocr_image(img_bytes, api_key, page_num)
                if ocr_text:
                    pages_text[page_num - 1] = ocr_text
                    meta["image_pages"] -= 1
                else:
                    meta["warnings"].append(
                        f"OCR returned no text for page {page_num}."
                    )
            doc.close()
        except Exception as e:
            meta["warnings"].append(f"Vision OCR failed: {e}")

    elif remaining_ocr and not openrouter.is_available(api_key):
        skipped = len(remaining_ocr)
        meta["warnings"].append(
            f"{skipped} image-only page(s) could not be extracted "
            f"(no OpenRouter API key configured). "
            f"Add your key to .env to enable OCR."
        )

    # ── Combine all pages ────────────────────────────────────────
    full_text = "\n\n".join(p for p in pages_text if p)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text).strip()

    if not full_text:
        raise ValueError(
            "No readable text found. This PDF appears to be fully image-based. "
            + ("Add an OpenRouter API key to .env to enable Vision OCR."
               if not openrouter.is_available(api_key)
               else "Vision OCR also returned no text — the image quality may be too low.")
        )

    return full_text, meta
