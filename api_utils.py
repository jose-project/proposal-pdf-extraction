"""
Utility functions for API endpoints.
"""

import io
import logging
from pathlib import Path
from typing import Tuple

import pdfplumber
from fastapi import HTTPException, UploadFile

logger = logging.getLogger(__name__)


def validate_pdf_file(pdf: UploadFile) -> None:
    """
    Validate that the uploaded file is a PDF.
    
    Args:
        pdf: Uploaded file object
    
    Raises:
        HTTPException: If file is not a PDF or is empty
    """
    if not pdf.filename or not pdf.filename.lower().endswith(".pdf"):
        logger.warning(f"Invalid file type received: {pdf.filename}")
        raise HTTPException(status_code=400, detail="File must be a PDF")


async def read_pdf_content(pdf: UploadFile) -> bytes:
    """
    Read PDF content from uploaded file.
    
    Args:
        pdf: Uploaded file object
    
    Returns:
        PDF file content as bytes
    
    Raises:
        HTTPException: If file is empty
    """
    content = await pdf.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    return content


def open_pdf_buffer(content: bytes) -> pdfplumber.PDF:
    """
    Open a PDF from bytes content using pdfplumber.
    
    Args:
        content: PDF file content as bytes
    
    Returns:
        pdfplumber.PDF object
    
    Raises:
        HTTPException: If PDF cannot be read
    """
    try:
        pdf_buffer = io.BytesIO(content)
        return pdfplumber.open(pdf_buffer)
    except Exception as e:
        logger.error(f"Error reading PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")


async def get_pdf_info(pdf: UploadFile) -> Tuple[int, bytes]:
    """
    Get total page count and content from uploaded PDF.
    
    Args:
        pdf: Uploaded file object
    
    Returns:
        Tuple of (total_pages, content_bytes)
    
    Raises:
        HTTPException: If PDF is invalid or cannot be read
    """
    validate_pdf_file(pdf)
    content = await read_pdf_content(pdf)
    
    with open_pdf_buffer(content) as doc:
        total_pages = len(doc.pages)
    
    return total_pages, content

