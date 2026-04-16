"""
OCR fallback processor for scanned PDF pages.

Detects pages with insufficient text (image-based / scanned) and applies
Tesseract OCR via pdf2image as a best-effort fallback.

Dependencies (optional — gracefully disabled if not installed):
    pdf2image >= 1.17.0
    pytesseract >= 0.3.10
    Tesseract binary: 'tesseract-ocr' OS package
"""

import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

# Pages with fewer characters than this threshold are treated as scanned.
OCR_TEXT_THRESHOLD = 50

# DPI used when rendering PDF pages to images for OCR.
OCR_DPI = 300

# Cached availability flag so we only log the warning once.
_ocr_available: bool | None = None


def _check_ocr_available() -> bool:
    """Return True if both pdf2image and pytesseract are importable."""
    global _ocr_available
    if _ocr_available is not None:
        return _ocr_available
    try:
        import pdf2image  # noqa: F401
        import pytesseract  # noqa: F401
        _ocr_available = True
        logger.debug("OCR dependencies (pdf2image, pytesseract) are available.")
    except ImportError as exc:
        _ocr_available = False
        logger.warning(
            f"OCR dependencies not available ({exc}). "
            "Scanned pages will return empty text. "
            "Install pdf2image and pytesseract to enable OCR fallback."
        )
    return _ocr_available


def is_scanned_page(text: str) -> bool:
    """Return True if the extracted text is too short to be a real text page."""
    return len(text.strip()) < OCR_TEXT_THRESHOLD


def ocr_pdf_page(pdf_path: Path, page_number: int, dpi: int = OCR_DPI) -> str:
    """
    Render a single PDF page to an image and run Tesseract OCR on it.

    Args:
        pdf_path: Path to the PDF file.
        page_number: 1-based page number to OCR.
        dpi: Resolution for rendering (higher = better accuracy, slower).

    Returns:
        OCR text string, or empty string on failure.
    """
    if not _check_ocr_available():
        return ""

    try:
        from pdf2image import convert_from_path
        import pytesseract

        images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            first_page=page_number,
            last_page=page_number,
        )
        if not images:
            logger.warning(f"pdf2image returned no images for page {page_number}")
            return ""

        text = pytesseract.image_to_string(images[0])
        return text.strip()

    except Exception as exc:
        logger.error(f"OCR failed for page {page_number} of {pdf_path.name}: {exc}")
        return ""


def get_page_text_with_ocr_fallback(
    pdf_path: Path,
    page_number: int,
    pdfplumber_text: str,
) -> Tuple[str, bool]:
    """
    Return page text, applying OCR if pdfplumber text is below the threshold.

    Args:
        pdf_path: Path to the PDF (needed to re-render the page for OCR).
        page_number: 1-based page number.
        pdfplumber_text: Text already extracted by pdfplumber.

    Returns:
        (text, was_ocr_used) tuple.
        - text: Best available text (pdfplumber or OCR).
        - was_ocr_used: True if OCR was applied.
    """
    if not is_scanned_page(pdfplumber_text):
        return pdfplumber_text, False

    logger.info(
        f"Page {page_number}: low text yield "
        f"({len(pdfplumber_text.strip())} chars < threshold {OCR_TEXT_THRESHOLD}), "
        "attempting OCR fallback."
    )

    ocr_text = ocr_pdf_page(pdf_path, page_number)

    if ocr_text:
        logger.info(f"Page {page_number}: OCR produced {len(ocr_text)} chars.")
        return ocr_text, True

    logger.warning(
        f"Page {page_number}: OCR returned empty result, "
        "falling back to original pdfplumber text."
    )
    return pdfplumber_text, False
