"""
SHA-256 keyed in-memory result cache for PDF extraction results.

Caches extraction results by the SHA-256 hash of the raw PDF bytes so that
identical files submitted multiple times return immediately without any LLM
calls.

The cache lives in process memory and is cleared on server restart.
This is intentional — PDF proposals are versioned documents; a new upload
should always produce a fresh extraction.
"""

import copy
import hashlib
import logging
from collections import OrderedDict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Maximum number of extraction results kept in memory at once.
# Each entry typically holds a few KB of JSON; 200 entries ≈ a few MB.
_MAX_CACHE_ENTRIES = 200


class ResultCache:
    """Thread-safe (GIL-protected) in-memory LRU cache keyed by PDF content hash."""

    def __init__(self, max_entries: int = _MAX_CACHE_ENTRIES) -> None:
        self._store: OrderedDict[str, Any] = OrderedDict()
        self._max_entries = max_entries

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
        if key not in self._store:
            logger.debug(f"Cache MISS for PDF hash {key[:16]}…")
            return None
        # Move to end to mark as recently used (LRU).
        self._store.move_to_end(key)
        logger.info(f"Cache HIT  for PDF hash {key[:16]}…")
        return copy.deepcopy(self._store[key])

    def set(self, pdf_bytes: bytes, result: Any) -> None:
        """
        Store *result* in the cache under the hash of *pdf_bytes*.

        Evicts the least-recently-used entry when the cache is full.

        Args:
            pdf_bytes: Raw PDF file content (used to compute the cache key).
            result: Extraction result dict to cache.
        """
        key = self._hash(pdf_bytes)
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = copy.deepcopy(result)
        if len(self._store) > self._max_entries:
            evicted_key, _ = self._store.popitem(last=False)
            logger.info(f"Cache EVICT oldest entry {evicted_key[:16]}… (limit {self._max_entries})")
        logger.info(
            f"Cache SET  for PDF hash {key[:16]}… "
            f"(total entries: {len(self._store)}/{self._max_entries})"
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
