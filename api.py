import asyncio
import logging
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

from scripts.llm_pdf_extractor import extract_pdf_with_llm
from api_utils import validate_pdf_file, read_pdf_content, open_pdf_buffer, get_pdf_info
from scripts.constants import DEFAULT_MAX_CONCURRENT, DEFAULT_MAX_TOKENS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EB3 Proposal Rate Extractor API",
    description="Extract insurance plan rates from proposal PDFs using LLM",
    version="1.0.0",
)


class ExtractResponse(BaseModel):
    file: str
    carrier: Optional[str]
    plans: list
    status: str = "success"
    timing: Optional[dict] = None


class PageCountResponse(BaseModel):
    file: str
    total_pages: int
    status: str = "success"


class PageContentResponse(BaseModel):
    file: str
    page_number: int
    total_pages: int
    content: str
    character_count: int
    tables: list = []  # List of extracted tables
    table_count: int = 0
    status: str = "success"


@app.get("/")
async def root():
    return {
        "message": "EB3 Proposal Rate Extractor API",
        "endpoints": {
            "POST /page-count": "Upload a PDF file and get total page count",
            "POST /page-content": "Upload a PDF file and get text content of a specific page",
            "POST /extract": "Extract rates (v1 pipeline)",
            "POST /extract-range": "Extract rates from a specific page range (v1 pipeline)",
            "POST /extract-v2": "Extract rates (v2 pipeline — smarter chunking, OCR fallback, caching)",
            "POST /extract-batch-v2": "Batch extract rates from multiple PDFs (v2 pipeline)",
            "GET /health": "Check API health status",
        },
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/page-count", response_model=PageCountResponse)
async def get_page_count(
    pdf: UploadFile = File(..., description="PDF proposal file"),
):
    """
    Return the total number of pages in the uploaded PDF.
    """
    try:
        total_pages, _ = await get_pdf_info(pdf)
        logger.info(f"Page count for {pdf.filename}: {total_pages}")
        return PageCountResponse(file=pdf.filename, total_pages=total_pages)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting page count for {pdf.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")


@app.post("/page-content", response_model=PageContentResponse)
async def get_page_content(
    pdf: UploadFile = File(..., description="PDF proposal file"),
    page_number: int = 1,
):
    """
    Return the text content of a specific page from the uploaded PDF.
    
    - **pdf**: The PDF file to read
    - **page_number**: Page number to extract (1-based, default: 1)
    
    Returns the raw text content extracted from the specified page.
    """
    if page_number < 1:
        raise HTTPException(status_code=400, detail="Page number must be >= 1")

    try:
        content = await read_pdf_content(pdf)
        
        with open_pdf_buffer(content) as doc:
            total_pages = len(doc.pages)
            
            if page_number > total_pages:
                raise HTTPException(
                    status_code=400,
                    detail=f"Page {page_number} does not exist. PDF has {total_pages} pages."
                )
            
            # Extract text and tables from the requested page (0-based index)
            page = doc.pages[page_number - 1]
            page_text = page.extract_text() or ""
            
            # Extract tables using pdfplumber's table detection
            tables = page.extract_tables()
            
            # Format tables as structured data
            formatted_tables = []
            for table_idx, table in enumerate(tables):
                if table and len(table) > 0:
                    formatted_table = {
                        "table_index": table_idx + 1,
                        "rows": len(table),
                        "columns": len(table[0]) if table else 0,
                        "data": table
                    }
                    formatted_tables.append(formatted_table)
            
            # Normalize whitespace for readability
            normalized_text = " ".join(page_text.split())
            char_count = len(normalized_text)

        logger.info(
            f"Page content extracted: {pdf.filename}, page {page_number}/{total_pages}, "
            f"{char_count} characters"
        )
        
        return PageContentResponse(
            file=pdf.filename,
            page_number=page_number,
            total_pages=total_pages,
            content=normalized_text,
            character_count=char_count,
            tables=formatted_tables,
            table_count=len(formatted_tables),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error getting page content for {pdf.filename}, page {page_number}: {e}",
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Error reading PDF page: {str(e)}")


async def process_pdf_upload(
    pdf: UploadFile,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    include_timing: bool = True,
) -> ExtractResponse:
    """
    Core function to process an uploaded PDF and extract rates.

    Uses default values from constants for pages_per_chunk and max_chars.

    Args:
        pdf: Uploaded PDF file
        start_page: First page to process (1-based, optional)
        end_page: Last page to process (1-based, optional)
        include_timing: If True, include per-phase timing in the response.

    Returns:
        ExtractResponse with extracted rate data
    """
    from scripts.constants import DEFAULT_PAGES_PER_CHUNK, DEFAULT_MAX_CHARS

    validate_pdf_file(pdf)

    # Create temporary file to store uploaded PDF
    temp_dir = Path(tempfile.gettempdir()) / "eb3_extractor"
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_pdf_path = None
    temp_output_path = None

    try:
        # Save uploaded file to temporary location (use UUID to avoid filename conflicts)
        file_id = str(uuid.uuid4())
        temp_pdf_path = temp_dir / f"upload_{file_id}.pdf"
        logger.info(f"Saving uploaded PDF to temporary location: {temp_pdf_path}")
        with open(temp_pdf_path, "wb") as f:
            content = await pdf.read()
            f.write(content)
        logger.info(f"PDF saved: {len(content)} bytes")

        # Always use a temporary output path (ephemeral); it will be deleted in finally.
        temp_output_path = temp_dir / f"output_{file_id}.json"
        logger.debug(f"Using temporary output path: {temp_output_path}")

        # Extract rates using LLM (optimized with parallel processing)
        logger.info("Starting LLM extraction process...")
        result = await extract_pdf_with_llm(
            pdf_path=temp_pdf_path,
            output_path=temp_output_path,
            pages_per_chunk=DEFAULT_PAGES_PER_CHUNK,
            max_chars=DEFAULT_MAX_CHARS,
            max_concurrent=DEFAULT_MAX_CONCURRENT,
            max_tokens=DEFAULT_MAX_TOKENS,
            filter_empty=False,  # Process every page; don't skip short ones
            start_page=start_page,
            end_page=end_page,
        )

        # Remove file path from result (not needed in API response)
        result.pop("file", None)

        plans_count = len(result.get("plans", []))
        logger.info(
            f"Extraction successful: {plans_count} plans found, "
            f"carrier: {result.get('carrier') or 'None'}"
        )

        return ExtractResponse(
            file=pdf.filename,
            carrier=result.get("carrier"),
            plans=result.get("plans", []),
            status="success",
            timing=result.get("timing") if include_timing else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing PDF {pdf.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

    finally:
        # Cleanup temporary files
        if temp_pdf_path and temp_pdf_path.exists():
            temp_pdf_path.unlink()
            logger.debug(f"Cleaned up temporary PDF: {temp_pdf_path}")

        if temp_output_path and temp_output_path.exists():
            temp_output_path.unlink()
            logger.debug(f"Cleaned up temporary output: {temp_output_path}")


@app.post("/extract", response_model=ExtractResponse)
async def extract_rates(
    pdf: UploadFile = File(..., description="PDF proposal file"),
    include_timing: bool = True,
):
    """
    Extract insurance plan rates from a PDF proposal.

    - **pdf**: The PDF file to process
    - **include_timing**: If true, include per-phase timing breakdown in response
    Returns extracted carrier name, plan names/IDs, rate structures, and rates.

    Uses default configuration values from constants for chunking and processing.
    """
    from scripts.constants import DEFAULT_PAGES_PER_CHUNK, DEFAULT_MAX_CHARS

    logger.info(
        f"Extract request received: file={pdf.filename}, "
        f"pages_per_chunk={DEFAULT_PAGES_PER_CHUNK}, max_chars={DEFAULT_MAX_CHARS}, "
        f"include_timing={include_timing}"
    )
    return await process_pdf_upload(pdf=pdf, include_timing=include_timing)


@app.post("/extract-range", response_model=ExtractResponse)
async def extract_rates_range(
    pdf: UploadFile = File(..., description="PDF proposal file"),
    start_page: int = 1,
    end_page: int = 1,
    include_timing: bool = True,
):
    """
    Extract rates from a specific page range of a PDF proposal.

    - **pdf**: The PDF file to process
    - **start_page**: First page to include (1-based)
    - **end_page**: Last page to include (1-based, inclusive)
    - **include_timing**: If true, include per-phase timing breakdown in response
    Uses default configuration values from constants for chunking and processing.
    """
    from scripts.constants import DEFAULT_PAGES_PER_CHUNK, DEFAULT_MAX_CHARS

    if start_page < 1 or end_page < start_page:
        raise HTTPException(
            status_code=400,
            detail="Invalid page range: ensure start_page >= 1 and end_page >= start_page",
        )

    logger.info(
        f"Extract-range request received: file={pdf.filename}, "
        f"start_page={start_page}, end_page={end_page}, "
        f"pages_per_chunk={DEFAULT_PAGES_PER_CHUNK}, max_chars={DEFAULT_MAX_CHARS}"
    )

    return await process_pdf_upload(
        pdf=pdf,
        start_page=start_page,
        end_page=end_page,
        include_timing=include_timing,
    )


# ---------------------------------------------------------------------------
# v2 endpoints — smart pipeline (same LLM, better pre-processing)
# ---------------------------------------------------------------------------

from scripts.smart_extractor import extract_pdf_smart


class ExtractBatchV2Response(BaseModel):
    results: list
    errors: list
    total: int
    successful: int
    failed: int


@app.post("/extract-v2", response_model=ExtractResponse)
async def extract_rates_v2(
    pdf: UploadFile = File(..., description="PDF proposal file"),
    include_timing: bool = True,
):
    """
    Extract insurance plan rates using the v2 smart pipeline.

    Improvements over /extract:
    - OCR fallback for scanned / image-based PDFs
    - Page relevance scoring — only rate-relevant pages sent to the LLM
    - Markdown table formatting for cleaner LLM input
    - Document-level context (carrier, date) injected into every chunk
    - Continuity-aware chunking — tables that span pages stay together
    - In-memory result cache — identical uploads return instantly

    - **include_timing**: If true, include per-phase timing breakdown in response

    Same request shape and response schema as /extract.
    """
    validate_pdf_file(pdf)

    temp_dir = Path(tempfile.gettempdir()) / "eb3_extractor_v2"
    temp_dir.mkdir(parents=True, exist_ok=True)

    file_id = str(uuid.uuid4())
    temp_pdf_path = temp_dir / f"upload_{file_id}.pdf"
    temp_output_path = temp_dir / f"output_{file_id}.json"

    try:
        pdf_bytes = await pdf.read()
        if not pdf_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        temp_pdf_path.write_bytes(pdf_bytes)
        logger.info(f"[v2] Extract request: file={pdf.filename}, size={len(pdf_bytes)} bytes")

        result = await extract_pdf_smart(
            pdf_path=temp_pdf_path,
            output_path=temp_output_path,
            pdf_bytes=pdf_bytes,
        )

        result.pop("file", None)
        plans_count = len(result.get("plans", []))
        logger.info(
            f"[v2] Extraction complete: {plans_count} plans, "
            f"carrier={result.get('carrier')!r}"
        )

        return ExtractResponse(
            file=pdf.filename,
            carrier=result.get("carrier"),
            plans=result.get("plans", []),
            status="success",
            timing=result.get("timing") if include_timing else None,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"[v2] Error processing {pdf.filename}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(exc)}")
    finally:
        if temp_pdf_path.exists():
            temp_pdf_path.unlink()
        if temp_output_path.exists():
            temp_output_path.unlink()


@app.post("/extract-batch-v2", response_model=ExtractBatchV2Response)
async def extract_batch_v2(
    pdfs: List[UploadFile] = File(..., description="PDF proposal files"),
):
    """
    Batch extraction using the v2 smart pipeline.

    Processes each uploaded PDF through the same smart pipeline as /extract-v2.
    Files are processed concurrently (up to 3 at a time).

    Returns a summary with per-file results and any per-file errors.
    """
    if not pdfs:
        raise HTTPException(status_code=400, detail="No files uploaded")

    semaphore = asyncio.Semaphore(3)

    async def _process_one(pdf: UploadFile) -> dict:
        async with semaphore:
            try:
                validate_pdf_file(pdf)

                temp_dir = Path(tempfile.gettempdir()) / "eb3_extractor_v2"
                temp_dir.mkdir(parents=True, exist_ok=True)

                file_id = str(uuid.uuid4())
                temp_pdf_path = temp_dir / f"upload_{file_id}.pdf"
                temp_output_path = temp_dir / f"output_{file_id}.json"

                try:
                    pdf_bytes = await pdf.read()
                    if not pdf_bytes:
                        return {"file": pdf.filename, "error": "Empty file"}

                    temp_pdf_path.write_bytes(pdf_bytes)

                    result = await extract_pdf_smart(
                        pdf_path=temp_pdf_path,
                        output_path=temp_output_path,
                        pdf_bytes=pdf_bytes,
                    )
                    result.pop("file", None)
                    return {
                        "file": pdf.filename,
                        "carrier": result.get("carrier"),
                        "plans": result.get("plans", []),
                        "status": "success",
                    }
                finally:
                    if temp_pdf_path.exists():
                        temp_pdf_path.unlink()
                    if temp_output_path.exists():
                        temp_output_path.unlink()

            except HTTPException as exc:
                return {"file": pdf.filename, "error": exc.detail}
            except Exception as exc:
                logger.error(f"[v2 batch] Error on {pdf.filename}: {exc}", exc_info=True)
                return {"file": pdf.filename, "error": str(exc)}

    all_results = await asyncio.gather(*[_process_one(p) for p in pdfs])

    successes = [r for r in all_results if "error" not in r]
    errors = [r for r in all_results if "error" in r]

    logger.info(
        f"[v2 batch] Complete: {len(successes)} succeeded, {len(errors)} failed "
        f"out of {len(pdfs)} files."
    )

    return ExtractBatchV2Response(
        results=successes,
        errors=errors,
        total=len(pdfs),
        successful=len(successes),
        failed=len(errors),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)

