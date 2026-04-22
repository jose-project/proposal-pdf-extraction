"""
Benchmark script: compare v1 and v2 extraction pipelines side-by-side.

Runs both pipelines on every PDF in source/proposals/ and prints a
formatted comparison table showing per-phase timings, chunk counts,
and total wall-clock time.

Runs each PDF twice through v2 to demonstrate the cache effect.

Usage
-----
    python scripts/benchmark.py
    python scripts/benchmark.py --pdf "source/proposals/Mega Polymers Inc. Proposal v2.pdf"
    python scripts/benchmark.py --dir source/proposals --detail
"""

import argparse
import asyncio
import logging
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.llm_pdf_extractor import extract_pdf_with_llm
from scripts.smart_extractor import extract_pdf_smart

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEMP_DIR = Path(tempfile.gettempdir()) / "eb3_benchmark"
_TEMP_DIR.mkdir(parents=True, exist_ok=True)


async def _run_v1(pdf_path: Path) -> Dict[str, Any]:
    """Run the v1 pipeline and return the result dict (includes 'timing')."""
    out = _TEMP_DIR / f"v1_{uuid.uuid4().hex}.json"
    try:
        result = await extract_pdf_with_llm(
            pdf_path=pdf_path,
            output_path=out,
        )
        return result
    finally:
        if out.exists():
            out.unlink()


async def _run_v2(pdf_path: Path, pdf_bytes: bytes) -> Dict[str, Any]:
    """Run the v2 pipeline and return the result dict (includes 'timing')."""
    out = _TEMP_DIR / f"v2_{uuid.uuid4().hex}.json"
    try:
        result = await extract_pdf_smart(
            pdf_path=pdf_path,
            output_path=out,
            pdf_bytes=pdf_bytes,
        )
        return result
    finally:
        if out.exists():
            out.unlink()


def _fmt(value: Optional[float], width: int = 7) -> str:
    if value is None:
        return " " * width
    return f"{value:>{width}.2f}s"


def _llm_total(timing: Dict[str, Any]) -> Optional[float]:
    llm = timing.get("llm_call_s")
    if isinstance(llm, dict):
        return llm.get("total")
    return None


def _llm_avg(timing: Dict[str, Any]) -> Optional[float]:
    llm = timing.get("llm_call_s")
    if isinstance(llm, dict):
        return llm.get("avg")
    return None


def _print_table(rows: List[Dict[str, Any]]) -> None:
    """Print a formatted comparison table to stdout."""
    col_file   = 44
    col_num    = 7
    col_time   = 8

    header = (
        f"{'PDF':<{col_file}} "
        f"{'v1 tot':>{col_time}} "
        f"{'v1 llm':>{col_time}} "
        f"{'v1 chk':>{col_num}} "
        f"{'v2 tot':>{col_time}} "
        f"{'v2 llm':>{col_time}} "
        f"{'v2 chk':>{col_num}} "
        f"{'v2 hit':>{col_time}} "
        f"{'Δ tot':>{col_time}} "
        f"{'plans':>{col_num}}"
    )
    sep = "-" * len(header)

    print()
    print(header)
    print(sep)

    v1_total_sum  = 0.0
    v2_total_sum  = 0.0
    v2_hit_sum    = 0.0

    for r in rows:
        name = r["name"]
        if len(name) > col_file:
            name = name[: col_file - 1] + "…"

        v1_t  = r.get("v1_total")
        v1_l  = r.get("v1_llm_total")
        v1_c  = r.get("v1_chunks", "-")
        v2_t  = r.get("v2_total")
        v2_l  = r.get("v2_llm_total")
        v2_c  = r.get("v2_chunks", "-")
        v2_h  = r.get("v2_cache_total")
        delta = (v2_t - v1_t) if (v1_t is not None and v2_t is not None) else None
        plans = r.get("plans", "-")

        delta_str = (
            f"{delta:>+{col_time}.2f}s"
            if delta is not None
            else " " * (col_time + 1)
        )

        print(
            f"{name:<{col_file}} "
            f"{_fmt(v1_t, col_time - 1)} "
            f"{_fmt(v1_l, col_time - 1)} "
            f"{str(v1_c):>{col_num}} "
            f"{_fmt(v2_t, col_time - 1)} "
            f"{_fmt(v2_l, col_time - 1)} "
            f"{str(v2_c):>{col_num}} "
            f"{_fmt(v2_h, col_time - 1)} "
            f"{delta_str} "
            f"{str(plans):>{col_num}}"
        )

        if v1_t is not None:
            v1_total_sum += v1_t
        if v2_t is not None:
            v2_total_sum += v2_t
        if v2_h is not None:
            v2_hit_sum += v2_h

    print(sep)

    total_delta = v2_total_sum - v1_total_sum
    print(
        f"{'TOTAL':<{col_file}} "
        f"{_fmt(v1_total_sum, col_time - 1)} "
        f"{'':>{col_time}} "
        f"{'':>{col_num}} "
        f"{_fmt(v2_total_sum, col_time - 1)} "
        f"{'':>{col_time}} "
        f"{'':>{col_num}} "
        f"{_fmt(v2_hit_sum, col_time - 1)} "
        f"{total_delta:>+{col_time}.2f}s "
        f"{'':>{col_num}}"
    )
    print()
    print("Columns: tot=total wall-clock  llm=LLM calls sum  chk=chunk count  hit=cache-hit run")
    print("Δ tot = v2_total - v1_total  (negative = v2 faster)")
    print()


def _print_detail(pdf_name: str, v1: Dict, v2: Dict) -> None:
    """Print a detailed per-phase breakdown for a single PDF."""
    print(f"\n── Detail: {pdf_name} ──")

    def _section(label: str, timing: Dict[str, Any]) -> None:
        print(f"  {label}:")
        for k, v in timing.items():
            if k in ("cache_hit", "chunks", "ocr_pages"):
                print(f"    {k:<32} {v}")
            elif isinstance(v, dict):
                print(
                    f"    {k:<32} count={v.get('count', '?')}  "
                    f"total={v.get('total', 0.0):.3f}s  "
                    f"avg={v.get('avg', 0.0):.3f}s  "
                    f"min={v.get('min', 0.0):.3f}s  "
                    f"max={v.get('max', 0.0):.3f}s"
                )
            elif isinstance(v, float):
                print(f"    {k:<32} {v:.3f}s")

    _section("v1", v1.get("timing", {}))
    _section("v2 (first run)", v2.get("timing", {}))


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

async def benchmark(
    pdf_paths: List[Path],
    detail: bool = False,
) -> None:
    rows: List[Dict[str, Any]] = []

    for pdf_path in sorted(pdf_paths):
        print(f"  Processing: {pdf_path.name} ...", end="", flush=True)
        pdf_bytes = pdf_path.read_bytes()

        # Clear v2 cache so first run is a real extraction
        from scripts.result_cache import get_cache
        get_cache().clear()

        # v1 run
        try:
            v1_result = await _run_v1(pdf_path)
            v1_timing = v1_result.get("timing", {})
        except Exception as exc:
            print(f"\n    [v1 ERROR] {exc}")
            v1_result = {}
            v1_timing = {}

        # v2 first run (cache miss)
        try:
            v2_result = await _run_v2(pdf_path, pdf_bytes)
            v2_timing = v2_result.get("timing", {})
        except Exception as exc:
            print(f"\n    [v2 ERROR] {exc}")
            v2_result = {}
            v2_timing = {}

        # v2 second run (cache hit)
        try:
            v2_cache_result = await _run_v2(pdf_path, pdf_bytes)
            v2_cache_timing = v2_cache_result.get("timing", {})
        except Exception as exc:
            print(f"\n    [v2 cache ERROR] {exc}")
            v2_cache_timing = {}

        plans = len(v2_result.get("plans") or v1_result.get("plans") or [])
        print(f" done  ({plans} plans)")

        row = {
            "name":          pdf_path.name,
            "v1_total":      v1_timing.get("total_s"),
            "v1_llm_total":  _llm_total(v1_timing),
            "v1_chunks":     v1_timing.get("chunks", "-"),
            "v2_total":      v2_timing.get("total_s"),
            "v2_llm_total":  _llm_total(v2_timing),
            "v2_chunks":     v2_timing.get("chunks", "-"),
            "v2_cache_total": v2_cache_timing.get("total_s"),
            "plans":         plans,
        }
        rows.append(row)

        if detail:
            _print_detail(pdf_path.name, v1_result, v2_result)

    print()
    _print_table(rows)


def main() -> None:
    # Silence noisy loggers so benchmark table output is readable
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("scripts").setLevel(logging.WARNING)
    logging.getLogger("pdfplumber").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(
        description="Benchmark v1 vs v2 extraction pipelines."
    )
    parser.add_argument(
        "--pdf",
        help="Path to a single PDF to benchmark (default: all PDFs in --dir).",
    )
    parser.add_argument(
        "--dir",
        default="source/proposals",
        help="Directory of PDFs to benchmark (default: source/proposals).",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Print per-phase breakdown for each PDF after the summary table.",
    )
    args = parser.parse_args()

    if args.pdf:
        pdf_paths = [Path(args.pdf).resolve()]
        if not pdf_paths[0].exists():
            print(f"Error: file not found: {pdf_paths[0]}", file=sys.stderr)
            sys.exit(1)
    else:
        proposals_dir = Path(args.dir)
        if not proposals_dir.exists():
            print(f"Error: directory not found: {proposals_dir}", file=sys.stderr)
            sys.exit(1)
        pdf_paths = sorted(proposals_dir.glob("*.pdf"))
        if not pdf_paths:
            print(f"No PDF files found in {proposals_dir}", file=sys.stderr)
            sys.exit(1)

    print(f"\nBenchmark: {len(pdf_paths)} PDF(s)\n")
    asyncio.run(benchmark(pdf_paths, detail=args.detail))


if __name__ == "__main__":
    main()
