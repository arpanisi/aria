#!/usr/bin/env python3
"""Rate-limited arXiv PDF download and bounded text extraction."""

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pdfplumber
import requests


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


_LAST_PDF_FETCH_TS = 0.0


def fetch_arxiv_pdf(
    paper_id: str,
    *,
    cache_dir: Path,
    min_interval_seconds: float = 3.0,
) -> dict[str, Any]:
    """Fetch an arXiv PDF only after abstract-level evidence passes the gate."""
    global _LAST_PDF_FETCH_TS
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", paper_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = cache_dir / f"{safe_id}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return {
            "status": "cached",
            "paper_id": paper_id,
            "pdf_path": str(pdf_path),
            "bytes": int(pdf_path.stat().st_size),
        }

    wait = min_interval_seconds - (time.monotonic() - _LAST_PDF_FETCH_TS)
    if wait > 0:
        print(f"{_ts()} [pdf] rate-limiting {wait:.1f}s before fetching {paper_id}", flush=True)
        time.sleep(wait)

    pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    print(f"{_ts()} [pdf] downloading {pdf_url}", flush=True)
    try:
        response = requests.get(pdf_url, timeout=30)
        _LAST_PDF_FETCH_TS = time.monotonic()
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type.lower() and not response.content.startswith(b"%PDF"):
            return {
                "status": "not_pdf",
                "paper_id": paper_id,
                "pdf_url": pdf_url,
                "content_type": content_type,
            }
        pdf_path.write_bytes(response.content)
        return {
            "status": "downloaded",
            "paper_id": paper_id,
            "pdf_url": pdf_url,
            "pdf_path": str(pdf_path),
            "bytes": len(response.content),
        }
    except Exception as exc:  # noqa: BLE001
        _LAST_PDF_FETCH_TS = time.monotonic()
        return {
            "status": "failed",
            "paper_id": paper_id,
            "pdf_url": pdf_url,
            "error": str(exc),
        }


def extract_pdf_text(pdf_path: Path, *, max_pages: int = 24, max_chars: int = 60000) -> dict[str, Any]:
    """Extract bounded text from a selected arXiv PDF for method-spec reading."""
    try:
        pages: list[str] = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_index, page in enumerate(pdf.pages[:max_pages]):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"[page {page_index + 1}]\n{text}")
                if sum(len(item) for item in pages) >= max_chars:
                    break
        text = "\n\n".join(pages)[:max_chars]
        return {
            "status": "ok" if text.strip() else "empty",
            "pdf_path": str(pdf_path),
            "pages_read": min(max_pages, len(pages)),
            "chars": len(text),
            "text": text,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "pdf_path": str(pdf_path),
            "error": str(exc),
            "text": "",
        }
