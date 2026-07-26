"""Download and extract documentation bundles from devdocs.io."""

from __future__ import annotations

import gzip
import io
import json
import logging
import tarfile
from pathlib import Path
from typing import Any

import httpx

from .http_utils import http_download_with_retry

logger = logging.getLogger(__name__)

DOWNLOAD_BASE = "https://downloads.devdocs.io"


def download_doc(slug: str, docs_dir: Path, timeout: float = 120.0) -> dict[str, Any]:
    """Download a doc bundle (tar.gz) and extract it to docs_dir/{slug}/.

    Returns the extracted metadata (index.json + meta.json).
    
    Args:
        slug: Documentation slug
        docs_dir: Directory to store downloads
        timeout: Download timeout in seconds
        
    Returns:
        Metadata dictionary with 'index' and 'meta' keys
        
    Raises:
        httpx.HTTPError: If download fails
        tarfile.TarError: If extraction fails
    """
    url = f"{DOWNLOAD_BASE}/{slug}.tar.gz"
    target_dir = docs_dir / slug

    if target_dir.exists() and (target_dir / "db.json").exists():
        logger.info("Doc %s already downloaded, skipping", slug)
        return _read_local_metadata(target_dir)

    logger.info("Downloading %s → %s", slug, target_dir)
    content = http_download_with_retry(url, timeout=timeout)

    # Extract tar.gz in-memory
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tf:
        tf.extractall(target_dir)

    logger.info("Extracted %s (%d files)", slug, _count_files(target_dir))
    return _read_local_metadata(target_dir)


def download_docs_batch(
    slugs: list[str],
    docs_dir: Path,
    max_workers: int = 4
) -> dict[str, str]:
    """Download multiple doc bundles sequentially with progress tracking.

    Args:
        slugs: List of documentation slugs to download
        docs_dir: Directory to store downloads
        max_workers: Unused (kept for API compatibility)
        
    Returns:
        Dictionary mapping slug to status ('ok' or 'error: ...')
    """
    results: dict[str, str] = {}
    total = len(slugs)

    for i, slug in enumerate(slugs):
        try:
            download_doc(slug, docs_dir)
            results[slug] = "ok"
        except Exception as exc:
            logger.error("Failed to download %s: %s", slug, exc)
            results[slug] = f"error: {exc}"
        finally:
            logger.info("[%d/%d] %s → %s", i + 1, total, slug, results[slug])

    return results


def _count_files(dir_path: Path) -> int:
    """Count files in directory tree."""
    return sum(1 for f in dir_path.rglob("*") if f.is_file())


def _read_local_metadata(target_dir: Path) -> dict[str, Any]:
    """Read metadata from a locally extracted doc bundle."""
    index_json = target_dir / "index.json"
    meta_json = target_dir / "meta.json"

    data: dict[str, Any] = {}
    if index_json.exists():
        with open(index_json) as f:
            data["index"] = json.load(f)
    if meta_json.exists():
        with open(meta_json) as f:
            data["meta"] = json.load(f)

    return data


def get_doc_pages(slug: str, docs_dir: Path) -> dict[str, Any]:
    """Read the full page content (db.json) for a downloaded doc.

    Returns {path: html_content} mapping.
    """
    db_path = docs_dir / slug / "db.json"
    if not db_path.exists():
        return {}

    with open(db_path) as f:
        return json.load(f)


def get_doc_index(slug: str, docs_dir: Path) -> dict[str, Any] | None:
    """Read the index.json for a downloaded doc."""
    idx_path = docs_dir / slug / "index.json"
    if not idx_path.exists():
        return None

    with open(idx_path) as f:
        return json.load(f)


def list_downloaded_docs(docs_dir: Path) -> dict[str, bool]:
    """List all downloaded doc slugs. Returns {slug: True}."""
    docs = {}
    if not docs_dir.exists():
        return docs

    for entry in docs_dir.iterdir():
        if entry.is_dir() and (entry / "db.json").exists():
            docs[entry.name] = True

    return docs


def remove_doc(slug: str, docs_dir: Path) -> bool:
    """Remove a downloaded doc bundle."""
    target_dir = docs_dir / slug
    if not target_dir.exists():
        return False

    import shutil
    shutil.rmtree(target_dir)
    logger.info("Removed doc %s", slug)
    return True
