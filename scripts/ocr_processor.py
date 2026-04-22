"""
OCR fallback processor for scanned PDF pages.

Detects pages with insufficient text (image-based / scanned) and applies
Tesseract OCR via pdf2image as a best-effort fallback.

Pre-processing pipeline (applied before Tesseract):
  1. Convert to grayscale
  2. Sharpen
  3. Autocontrast + global threshold → clean binary image

Dependencies (optional — gracefully disabled if not installed):
    pdf2image >= 1.17.0
    pytesseract >= 0.3.10
    Pillow >= 9.0.0  (pulled in by pdf2image)
    Tesseract binary: 'tesseract-ocr' OS package
"""

import logging
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Pages with fewer *clean* characters than this are treated as scanned.
OCR_TEXT_THRESHOLD = 50

# Minimum ratio of printable ASCII chars required to trust pdfplumber text.
_PRINTABLE_RATIO_MIN = 0.70

# DPI for initial render. Pages that still fail OCR are retried at FALLBACK_DPI.
OCR_DPI = 300
_FALLBACK_DPI = 400

# Tesseract config: LSTM engine, single uniform block layout.
_TSR_CONFIG = "--psm 6 --oem 1"
# Fallback config if the first attempt returns empty: single-column layout.
_TSR_CONFIG_FALLBACK = "--psm 4 --oem 1"

# Cached availability flag so we only log the warning once.
_ocr_available: Optional[bool] = None


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _check_ocr_available() -> bool:
    global _ocr_available
    if _ocr_available is not None:
        return _ocr_available
    try:
        import pdf2image  # noqa: F401
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
        _ocr_available = True
        logger.debug("OCR dependencies (pdf2image, pytesseract, Pillow) are available.")
    except ImportError as exc:
        _ocr_available = False
        logger.warning(
            f"OCR dependencies not available ({exc}). "
            "Scanned pages will return empty text. "
            "Install pdf2image, pytesseract, and Pillow to enable OCR fallback."
        )
    return _ocr_available


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------

def _preprocess_image(image):
    """
    Convert a PIL image to a clean binary image suitable for Tesseract.

    Steps:
      1. Grayscale — removes colour noise.
      2. Sharpen — improves edge definition on low-res scans.
      3. Autocontrast + global threshold at midpoint → clean B&W image.
    """
    from PIL import ImageFilter, ImageOps

    # 1. Grayscale
    gray = image.convert("L")

    # 2. Mild sharpening helps low-res scans before thresholding
    gray = gray.filter(ImageFilter.SHARPEN)

    # 3. Autocontrast normalises the tonal range so that the global midpoint
    #    threshold below is meaningful even on faded or low-contrast scans.
    gray = ImageOps.autocontrast(gray, cutoff=2)

    # Global threshold at midpoint: pixels above 128 → white, else → black.
    binary = gray.point(lambda px: 255 if px > 128 else 0, "1")

    # Return as RGB because Tesseract handles it slightly better than mode "1"
    return binary.convert("RGB")


# ---------------------------------------------------------------------------
# Scanned-page detection
# ---------------------------------------------------------------------------

def _printable_ratio(text: str) -> float:
    """Fraction of characters that are printable (non-control, non-PUA Unicode)."""
    if not text:
        return 0.0
    printable = sum(
        1 for ch in text
        if unicodedata.category(ch) not in ("Cc", "Co", "Cs")
    )
    return printable / len(text)


def is_scanned_page(text: str) -> bool:
    """
    Return True if the extracted text is too short or too garbled to be
    a real text page (indicating a scanned/image-only page).
    """
    stripped = text.strip()
    if len(stripped) < OCR_TEXT_THRESHOLD:
        return True
    # Enough characters, but check quality — garbage font encodings produce
    # high counts of replacement/private-use characters.
    if _printable_ratio(stripped) < _PRINTABLE_RATIO_MIN:
        return True
    return False


# ---------------------------------------------------------------------------
# Core OCR call
# ---------------------------------------------------------------------------

def _run_tesseract(image, config: str) -> str:
    """Run Tesseract on a pre-processed PIL image and return stripped text."""
    import pytesseract
    return pytesseract.image_to_string(image, config=config).strip()


def ocr_pdf_page(pdf_path: Path, page_number: int, dpi: int = OCR_DPI) -> str:
    """
    Render a single PDF page to an image, pre-process it, and run Tesseract.

    Retry strategy:
      - Attempt 1: OCR_DPI render + PSM 6 (uniform block).
      - Attempt 2 (if empty): same image, PSM 4 (single column).
      - Attempt 3 (if still empty): re-render at _FALLBACK_DPI + PSM 6.

    Args:
        pdf_path:    Path to the PDF file.
        page_number: 1-based page number to OCR.
        dpi:         Resolution for the first render attempt.

    Returns:
        OCR text string, or empty string on total failure.
    """
    if not _check_ocr_available():
        return ""

    try:
        from pdf2image import convert_from_path

        images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            first_page=page_number,
            last_page=page_number,
        )
        if not images:
            logger.warning(f"pdf2image returned no images for page {page_number}")
            return ""

        processed = _preprocess_image(images[0])

        # Attempt 1 — PSM 6
        text = _run_tesseract(processed, _TSR_CONFIG)
        if text:
            return text

        # Attempt 2 — PSM 4 (same image, different layout assumption)
        logger.debug(f"Page {page_number}: PSM 6 empty, retrying with PSM 4.")
        text = _run_tesseract(processed, _TSR_CONFIG_FALLBACK)
        if text:
            return text

        # Attempt 3 — higher DPI re-render + PSM 6
        if dpi < _FALLBACK_DPI:
            logger.debug(
                f"Page {page_number}: PSM 4 also empty, "
                f"re-rendering at {_FALLBACK_DPI} DPI."
            )
            hi_images = convert_from_path(
                str(pdf_path),
                dpi=_FALLBACK_DPI,
                first_page=page_number,
                last_page=page_number,
            )
            if hi_images:
                hi_processed = _preprocess_image(hi_images[0])
                text = _run_tesseract(hi_processed, _TSR_CONFIG)
                if text:
                    return text

        return ""

    except Exception as exc:
        logger.error(f"OCR failed for page {page_number} of {pdf_path.name}: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Batch OCR — convert all scanned pages in one pdf2image call
# ---------------------------------------------------------------------------

def ocr_pdf_pages_batch(
    pdf_path: Path,
    page_numbers: List[int],
    dpi: int = OCR_DPI,
) -> Dict[int, str]:
    """
    OCR multiple pages from the same PDF in a single pdf2image call.

    Much faster than calling ocr_pdf_page() per page because the PDF is
    only opened and decoded once.

    Args:
        pdf_path:     Path to the PDF file.
        page_numbers: Sorted list of 1-based page numbers to OCR.
        dpi:          Render resolution.

    Returns:
        Dict mapping page_number → OCR text.  Missing pages had empty results.
    """
    if not _check_ocr_available() or not page_numbers:
        return {}

    try:
        from pdf2image import convert_from_path

        # Group into contiguous ranges so we never render pages that don't
        # need OCR (e.g. scanned pages 2 and 50 → two calls, not 49 renders).
        sorted_pages = sorted(set(page_numbers))
        ranges: List[Tuple[int, int]] = []
        start = sorted_pages[0]
        prev = sorted_pages[0]
        for p in sorted_pages[1:]:
            if p == prev + 1:
                prev = p
            else:
                ranges.append((start, prev))
                start = prev = p
        ranges.append((start, prev))

        results: Dict[int, str] = {}
        for first, last in ranges:
            images = convert_from_path(
                str(pdf_path),
                dpi=dpi,
                first_page=first,
                last_page=last,
            )
            for offset, image in enumerate(images):
                page_num = first + offset
                processed = _preprocess_image(image)
                text = _run_tesseract(processed, _TSR_CONFIG)
                if not text:
                    text = _run_tesseract(processed, _TSR_CONFIG_FALLBACK)
                if text:
                    results[page_num] = text
                else:
                    logger.debug(
                        f"Page {page_num}: batch OCR returned empty, "
                        "will retry at higher DPI individually."
                    )

        # Pages that came back empty: retry individually at higher DPI
        empty_pages = [p for p in sorted_pages if p not in results]
        for page_num in empty_pages:
            text = ocr_pdf_page(pdf_path, page_num, dpi=_FALLBACK_DPI)
            if text:
                results[page_num] = text

        return results

    except Exception as exc:
        logger.error(f"Batch OCR failed for {pdf_path.name}: {exc}")
        return {}


# ---------------------------------------------------------------------------
# High-level helper (unchanged public API)
# ---------------------------------------------------------------------------

def get_page_text_with_ocr_fallback(
    pdf_path: Path,
    page_number: int,
    pdfplumber_text: str,
) -> Tuple[str, bool]:
    """
    Return page text, applying OCR if pdfplumber text is below the threshold.

    Args:
        pdf_path:         Path to the PDF (needed to re-render the page for OCR).
        page_number:      1-based page number.
        pdfplumber_text:  Text already extracted by pdfplumber.

    Returns:
        (text, was_ocr_used) tuple.
        - text:         Best available text (pdfplumber or OCR).
        - was_ocr_used: True if OCR was applied.
    """
    if not is_scanned_page(pdfplumber_text):
        return pdfplumber_text, False

    logger.info(
        f"Page {page_number}: low/garbled text yield "
        f"({len(pdfplumber_text.strip())} chars), attempting OCR fallback."
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
