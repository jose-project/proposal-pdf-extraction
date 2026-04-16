"""
SHA-256 keyed in-memory result cache for PDF extraction results.

Caches extraction results by the SHA-256 hash of the raw PDF bytes so that
identical files submitted multiple times return immediately without any LLM
calls.

The cache lives in process memory and is cleared on server restart.
This is intentional — PDF proposals are versioned documents; a new upload
should always produce a fresh extraction.
"""

import hashlib
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ResultCache:
    """Thread-safe (GIL-protected) in-memory cache keyed by PDF content hash."""

    def __init__(self) -> None:
        self._store: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, pdf_bytes: bytes) -> Optional[Any]:
        """
        Return cached result for *pdf_bytes*, or None if not cached.

        Args:
            pdf_bytes: Raw PDF file content.

        Returns:
            Previously cached extraction result, or None on cache miss.
        """
        key = self._hash(pdf_bytes)
        result = self._store.get(key)
        if result is not None:
            logger.info(f"Cache HIT  for PDF hash {key[:16]}…")
        else:
            logger.debug(f"Cache MISS for PDF hash {key[:16]}…")
        return result

    def set(self, pdf_bytes: bytes, result: Any) -> None:
        """
        Store *result* in the cache under the hash of *pdf_bytes*.

        Args:
            pdf_bytes: Raw PDF file content (used to compute the cache key).
            result: Extraction result dict to cache.
        """
        key = self._hash(pdf_bytes)
        self._store[key] = result
        logger.info(
            f"Cache SET  for PDF hash {key[:16]}… "
            f"(total entries: {len(self._store)})"
        )

    def clear(self) -> None:
        """Remove all cached entries."""
        count = len(self._store)
        self._store.clear()
        logger.info(f"Cache cleared ({count} entries removed).")

    @property
    def size(self) -> int:
        """Number of entries currently in the cache."""
        return len(self._store)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(pdf_bytes: bytes) -> str:
        return hashlib.sha256(pdf_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Module-level singleton — shared across all requests in the same process.
# ---------------------------------------------------------------------------

_cache = ResultCache()


def get_cache() -> ResultCache:
    """Return the global ResultCache singleton."""
    return _cache
