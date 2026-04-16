"""
Smart PDF pre-processing for v2 extraction pipeline.

Provides four capabilities that improve on the legacy pipeline:

1. Page relevance scoring  — heuristic scoring to skip non-rate pages before
                             any LLM call is made.
2. Document-level context  — extract carrier name and effective date from the
                             cover page and inject into every chunk prompt.
3. Markdown table formatter — generic fallback that converts raw pdfplumber
                              table data into a clean Markdown table, giving
                              the LLM better-structured input for tables that
                              the existing table_processor does not recognise.
4. Continuity-aware chunker — groups consecutive scored pages into chunks
                              while respecting table continuity and character
                              limits.
"""

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

from scripts.table_processor import process_table

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration constants (v2 defaults)
# ---------------------------------------------------------------------------

# Pages with a score below this threshold are skipped (0.0 – 1.0).
SCORE_THRESHOLD = 0.20

# Maximum pages grouped into a single LLM chunk.
PAGES_PER_CHUNK_V2 = 4

# Maximum characters in a single LLM chunk before splitting.
MAX_CHARS_V2 = 12_000

# How many pages at the start of the document to scan for document context.
CONTEXT_SCAN_PAGES = 4

# ---------------------------------------------------------------------------
# Patterns used for page scoring and context extraction
# ---------------------------------------------------------------------------

_MONEY_PATTERN = re.compile(r"\$[\d,]+\.?\d*|\b\d{3,4}\.\d{2}\b")

_RATE_KEYWORDS = re.compile(
    r"\b(?:rate|premium|employee|spouse|child|family|tier|plan|coverage|"
    r"deductible|age|band|carrier|option|ppo|hmo|aca|dental|vision|"
    r"renewal|current|monthly|annual|quote|proposal)\b",
    re.IGNORECASE,
)

_CARRIER_PATTERN = re.compile(
    r"\b(?:Blue\s+Cross|Blue\s+Shield|BCBS(?:IL|MA|TX)?|Aetna|Cigna|"
    r"UnitedHealth(?:care)?|Humana|Principal|Ameritas|Nationwide|"
    r"MetLife|Guardian|Sun\s+Life|Lincoln(?:\s+Financial)?|Unum|"
    r"Anthem|Kaiser|Mutual\s+of\s+Omaha|Reliance\s+Standard|"
    r"Hartford|Trustmark|Delta\s+Dental)\b",
    re.IGNORECASE,
)

_DATE_PATTERN = re.compile(
    r"(?:effective|renewal|plan\s+year|coverage\s+period)"
    r"[\s:\-]+(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\w+\.?\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 1. Page relevance scoring
# ---------------------------------------------------------------------------

def score_page(text: str, table_count: int) -> float:
    """
    Compute a relevance score for a single page (0.0 – 1.0).

    Scoring breakdown:
      - Dollar amounts / decimal rate patterns: up to 0.40
      - Tables detected by pdfplumber:          up to 0.40
      - Rate-related keyword matches:           up to 0.20

    Args:
        text: Full text content of the page.
        table_count: Number of tables pdfplumber detected on this page.

    Returns:
        Float in [0.0, 1.0].
    """
    money_hits = len(_MONEY_PATTERN.findall(text))
    keyword_hits = len(_RATE_KEYWORDS.findall(text))

    score = (
        min(money_hits * 0.08, 0.40)
        + min(table_count * 0.20, 0.40)
        + min(keyword_hits * 0.02, 0.20)
    )
    return min(score, 1.0)


# ---------------------------------------------------------------------------
# 2. Document-level context extraction
# ---------------------------------------------------------------------------

def extract_document_context(
    pdf_path: Path,
    max_pages: int = CONTEXT_SCAN_PAGES,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Scan the first *max_pages* pages for carrier name and effective date.

    Returns:
        (carrier, effective_date) — either may be None if not found.
    """
    carriers: List[str] = []
    dates: List[str] = []

    try:
        with pdfplumber.open(pdf_path) as doc:
            for page in doc.pages[:max_pages]:
                text = page.extract_text() or ""
                carriers.extend(_CARRIER_PATTERN.findall(text))
                dates.extend(_DATE_PATTERN.findall(text))
    except Exception as exc:
        logger.warning(f"Could not extract document context from {pdf_path.name}: {exc}")
        return None, None

    carrier = Counter(carriers).most_common(1)[0][0] if carriers else None
    date = dates[0] if dates else None

    if carrier or date:
        logger.info(
            f"Document context: carrier={carrier!r}, effective_date={date!r}"
        )
    return carrier, date


# ---------------------------------------------------------------------------
# 3. Markdown table formatter (generic fallback)
# ---------------------------------------------------------------------------

def format_table_as_markdown(
    table: List[List[Any]],
    page_num: int,
    table_idx: int,
) -> Optional[str]:
    """
    Convert a raw pdfplumber table (list-of-lists) to a Markdown table string.

    This is a generic fallback used when process_table() returns None.
    It preserves all cell content without interpretation, giving the LLM a
    clean, column-aligned structure to read.

    Returns None if the table has fewer than 2 rows or 2 columns.
    """
    if not table or len(table) < 2:
        return None

    # Normalise rows: convert None cells to empty string, strip whitespace
    normalised: List[List[str]] = []
    for row in table:
        if row is None:
            continue
        cells = [
            " ".join(str(cell).split()) if cell is not None else ""
            for cell in row
        ]
        if any(cells):  # skip completely empty rows
            normalised.append(cells)

    if len(normalised) < 2:
        return None

    # Pad all rows to the same column count
    max_cols = max(len(row) for row in normalised)
    if max_cols < 2:
        return None

    padded = [row + [""] * (max_cols - len(row)) for row in normalised]

    # Build Markdown
    lines = [f"\n--- Table {table_idx} (Page {page_num}) [markdown] ---"]
    header = padded[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + " --- |" * max_cols)
    for row in padded[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: build full content string for a single page
# ---------------------------------------------------------------------------

def _build_page_content(
    page: Any,          # pdfplumber page object
    page_num: int,
    extra_text: Optional[str] = None,  # OCR text to use instead of page.extract_text()
) -> Tuple[str, int, List[List[Any]]]:
    """
    Extract and format all content from a pdfplumber page.

    Args:
        page: pdfplumber page object.
        page_num: 1-based page number (for labels in output).
        extra_text: If provided, use this text instead of page.extract_text()
                    (used when OCR was applied to a scanned page).

    Returns:
        (content_string, table_count, raw_tables)
    """
    parts: List[str] = []
    raw_tables: List[List[Any]] = page.extract_tables() or []

    for t_idx, raw_table in enumerate(raw_tables):
        # Try the existing specialised formatter first
        specialised = process_table(raw_table, page_num, t_idx + 1)
        if specialised:
            parts.append(specialised)
        else:
            # Generic Markdown fallback
            md = format_table_as_markdown(raw_table, page_num, t_idx + 1)
            if md:
                parts.append(md)

    # Plain text (or OCR substitute)
    if extra_text is not None:
        raw_text = extra_text
    else:
        raw_text = page.extract_text() or ""

    clean_text = " ".join(raw_text.split()).strip()
    if clean_text:
        parts.append(f"\n--- Page {page_num} Text ---\n{clean_text}")

    content = "\n".join(parts).strip()
    return content, len(raw_tables), raw_tables


# ---------------------------------------------------------------------------
# Helper: table continuity check
# ---------------------------------------------------------------------------

def _are_tables_continuous(
    prev_tables: List[List[Any]],
    curr_tables: List[List[Any]],
) -> bool:
    """
    Return True if the first table on *curr* page appears to continue the last
    table on *prev* page (same number of columns, no new header row detected).
    """
    if not prev_tables or not curr_tables:
        return False

    last_prev = prev_tables[-1]
    first_curr = curr_tables[0]

    if not last_prev or not first_curr:
        return False

    prev_cols = max((len(r) for r in last_prev if r), default=0)
    curr_cols = max((len(r) for r in first_curr if r), default=0)

    if prev_cols == 0 or prev_cols != curr_cols:
        return False

    # If the first row of curr looks like a header row (text only, no numbers)
    # it's a new table, not a continuation.
    first_row = first_curr[0] or []
    first_row_text = " ".join(str(c) for c in first_row if c)
    has_numbers = bool(re.search(r"\d", first_row_text))
    if not has_numbers:
        return False  # new header → new table

    return True


# ---------------------------------------------------------------------------
# 4. Continuity-aware chunker
# ---------------------------------------------------------------------------

def build_smart_chunks(
    pdf_path: Path,
    ocr_texts: Optional[Dict[int, str]] = None,
    score_threshold: float = SCORE_THRESHOLD,
    pages_per_chunk: int = PAGES_PER_CHUNK_V2,
    max_chars: int = MAX_CHARS_V2,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
) -> List[Tuple[List[int], str]]:
    """
    Build LLM-ready (page_numbers, text) chunks from a PDF using smart logic:

      - Skips pages below *score_threshold* (rate-irrelevant pages).
      - Groups consecutive scored pages into chunks up to *pages_per_chunk*.
      - Keeps tables that span page boundaries in the same chunk.
      - Falls back to processing ALL pages if no page scores above threshold
        (handles edge cases like very short or unusual documents).

    Args:
        pdf_path: Path to the PDF file.
        ocr_texts: Optional dict mapping 1-based page numbers to OCR text
                   (produced by ocr_processor for scanned pages).
        score_threshold: Minimum score for a page to be included.
        pages_per_chunk: Maximum pages per LLM chunk.
        max_chars: Maximum characters per chunk before splitting.
        start_page: First page to consider (1-based, inclusive).
        end_page: Last page to consider (1-based, inclusive).

    Returns:
        List of (page_numbers, combined_text) tuples ready for LLM processing.
    """
    if ocr_texts is None:
        ocr_texts = {}

    scored_pages: List[Tuple[int, str, float, List[List[Any]]]] = []  # (page_num, content, score, raw_tables)

    logger.info(f"Building smart chunks from {pdf_path.name}")

    with pdfplumber.open(pdf_path) as doc:
        total = len(doc.pages)
        p_start = max(1, start_page or 1)
        p_end = min(total, end_page or total)

        for idx in range(p_start, p_end + 1):
            page = doc.pages[idx - 1]
            extra_text = ocr_texts.get(idx)
            content, table_count, raw_tables = _build_page_content(
                page, idx, extra_text=extra_text
            )

            if not content:
                logger.debug(f"Page {idx}: empty content, skipping.")
                continue

            s = score_page(content, table_count)
            logger.debug(f"Page {idx}: score={s:.2f}, tables={table_count}, chars={len(content)}")
            scored_pages.append((idx, content, s, raw_tables))

    # Determine which pages pass the threshold
    passing = [(n, c, s, t) for n, c, s, t in scored_pages if s >= score_threshold]

    if not passing:
        logger.warning(
            f"No pages scored above threshold {score_threshold}. "
            "Falling back to all non-empty pages."
        )
        passing = scored_pages  # use everything rather than return empty

    logger.info(
        f"Page selection: {len(passing)}/{len(scored_pages)} pages pass "
        f"score threshold {score_threshold}."
    )

    # Group into chunks
    chunks: List[Tuple[List[int], str]] = []
    buf_pages: List[int] = []
    buf_text: List[str] = []
    buf_tables: List[List[Any]] = []

    def _flush() -> None:
        if buf_pages:
            chunks.append((list(buf_pages), "\n".join(buf_text)))
            logger.debug(
                f"Chunk: pages {buf_pages[0]}-{buf_pages[-1]} "
                f"({len(buf_pages)} pages, {sum(len(t) for t in buf_text)} chars)"
            )

    for page_num, content, _score, raw_tables in passing:
        # Check continuity with previous page in buffer
        continuous = _are_tables_continuous(buf_tables, raw_tables) if buf_tables else False

        candidate_pages = buf_pages + [page_num]
        candidate_text = "\n".join(buf_text + [content])

        over_page_limit = len(candidate_pages) > pages_per_chunk
        over_char_limit = len(candidate_text) > max_chars

        if buf_pages and (over_page_limit or over_char_limit) and not continuous:
            _flush()
            buf_pages = [page_num]
            buf_text = [content]
            buf_tables = raw_tables
        else:
            buf_pages.append(page_num)
            buf_text.append(content)
            buf_tables = raw_tables  # keep track of last page's tables

    _flush()

    logger.info(f"Built {len(chunks)} smart chunks from {pdf_path.name}.")
    return chunks
