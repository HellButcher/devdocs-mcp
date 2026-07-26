"""Catalog management — fetching and merging doc metadata."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from .config import Config, DevdocsSource, LocalSource, WebSource
from .http_utils import http_get_with_retry

logger = logging.getLogger(__name__)

DOCS_JSON_URL = "https://devdocs.io/docs.json"
CATALOG_CACHE_TTL = timedelta(hours=24)  # Cache catalog for 24 hours


@dataclass
class DocEntry:
    """A single documentation entry in the catalog."""
    slug: str
    name: str
    version: str
    type: str  # scraper type (simple, bash, angular, etc.)
    db_size: int  # compressed size in bytes
    release: str
    mtime: int | None = None
    alias: str | None = None
    home_url: str | None = None
    code_url: str | None = None
    source_type: str = "devdocs"  # devdocs | local

    @property
    def size_mb(self) -> float:
        return self.db_size / (1024 * 1024)

    @property
    def is_large(self) -> bool:
        """Docs larger than 50 MB are considered large."""
        return self.db_size > 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Fetching from devdocs.io
# ---------------------------------------------------------------------------

def fetch_devdocs_catalog(timeout: float = 30.0, cache_dir: Path | None = None) -> list[DocEntry]:
    """Fetch the official devdocs.io catalog and return parsed DocEntries.
    
    Uses local cache if available and fresh (< 24 hours old).
    
    Args:
        timeout: Request timeout in seconds
        cache_dir: Directory for caching catalog (uses default if None)
        
    Returns:
        List of documentation entries
        
    Raises:
        httpx.HTTPError: If catalog fetch fails after retries
    """
    # Check cache first
    if cache_dir:
        cache_file = cache_dir / "catalog_cache.json"
        if cache_file.exists():
            try:
                cache_data = json.loads(cache_file.read_text())
                cached_time = datetime.fromisoformat(cache_data.get("timestamp", ""))
                
                if datetime.now() - cached_time < CATALOG_CACHE_TTL:
                    logger.info("Using cached catalog (age: %s)", datetime.now() - cached_time)
                    return _parse_catalog_entries(cache_data["catalog"])
                else:
                    logger.info("Catalog cache expired (age: %s)", datetime.now() - cached_time)
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning("Failed to load catalog cache: %s", e)
    
    logger.info("Fetching devdocs catalog from %s", DOCS_JSON_URL)
    resp = http_get_with_retry(DOCS_JSON_URL, timeout=timeout)
    raw = resp.json()
    
    # Cache the response
    if cache_dir:
        cache_file = cache_dir / "catalog_cache.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "catalog": raw,
        }
        cache_file.write_text(json.dumps(cache_data, indent=2))
        logger.info("Cached catalog to %s", cache_file)
    
    return _parse_catalog_entries(raw)


def _parse_catalog_entries(raw: list[dict[str, Any]]) -> list[DocEntry]:
    """Parse raw catalog JSON into DocEntry objects."""

def _parse_catalog_entries(raw: list[dict[str, Any]]) -> list[DocEntry]:
    """Parse raw catalog JSON into DocEntry objects."""
    entries: list[DocEntry] = []
    for item in raw:
        slug = item["slug"]
        # Handle nested version (e.g. "angular~21" → name="Angular", version="21")
        parts = slug.split("~")
        base_slug = parts[0]

        entries.append(DocEntry(
            slug=slug,
            name=item.get("name", base_slug),
            version=item.get("version", ""),
            type=item.get("type", "unknown"),
            db_size=item.get("db_size", 0),
            release=item.get("release", ""),
            mtime=item.get("mtime"),
            alias=item.get("alias"),
            home_url=item.get("links", {}).get("home"),
            code_url=item.get("links", {}).get("code"),
            source_type="devdocs",
        ))

    logger.info("Parsed %d entries from devdocs catalog", len(entries))
    return entries


# ---------------------------------------------------------------------------
# Local directory indexing
# ---------------------------------------------------------------------------

def index_local_directory(
    path: str,
    slug_prefix: str = "",
) -> list[DocEntry]:
    """Scan a local directory for static HTML documentation and create DocEntries.

    Expected structure:
        <path>/
            page1.html
            page2.html
            ...

    Each .html file becomes one entry with slug = "<prefix><filename_without_ext>".
    
    Args:
        path: Directory path to scan
        slug_prefix: Optional prefix for generated slugs
        
    Returns:
        List of DocEntry objects for HTML files found
    """
    p = Path(path)
    if not p.is_dir():
        logger.warning("Local source path %s is not a directory", path)
        return []

    entries: list[DocEntry] = []
    html_files = sorted(p.glob("*.html")) + sorted(p.glob("*.htm"))

    for f in html_files:
        slug = slug_prefix + f.stem if slug_prefix else f.stem
        # Avoid collisions
        base_slug = slug
        counter = 1
        while any(e.slug == slug for e in entries):
            slug = f"{base_slug}_{counter}"
            counter += 1

        entries.append(DocEntry(
            slug=slug,
            name=f.stem.replace("_", " ").title(),
            version="",
            type="local_html",
            db_size=f.stat().st_size,
            release="",
            source_type="local",
            home_url=None,
        ))

    logger.info("Indexed %d HTML files from %s → %d entries", len(html_files), path, len(entries))
    return entries


# ---------------------------------------------------------------------------
# Merged catalog
# ---------------------------------------------------------------------------

def get_merged_catalog(config: Config) -> list[DocEntry]:
    """Build the full catalog by merging devdocs.io + custom sources.
    
    Uses cached entries for local sources to avoid rescanning the filesystem.
    """
    all_entries: list[DocEntry] = []

    for src in config.sources:
        if isinstance(src, DevdocsSource) and src.enabled:
            try:
                entries = fetch_devdocs_catalog(cache_dir=config.cache_dir)
                all_entries.extend(entries)
            except Exception as exc:
                logger.warning("Failed to fetch devdocs catalog: %s", exc)

        elif isinstance(src, LocalSource):
            # Each LocalSource is a single catalog entry
            all_entries.append(DocEntry(
                slug=src.slug,
                name=src.name,
                version="",
                type="local",
                db_size=0,  # Calculated on-demand
                release="",
                mtime=0,
                alias="",
                home_url=None,
                code_url=None,
                source_type="local",
            ))
        
        elif isinstance(src, WebSource):
            # Each WebSource is a single catalog entry
            all_entries.append(DocEntry(
                slug=src.slug,
                name=src.name,
                version="",
                type="web",
                db_size=0,  # Calculated on-demand
                release="",
                mtime=0,
                alias="",
                home_url=src.url,
                code_url=None,
                source_type="web",
            ))

    # Deduplicate by slug (keep first occurrence)
    seen: set[str] = set()
    unique: list[DocEntry] = []
    for e in all_entries:
        if e.slug not in seen:
            seen.add(e.slug)
            unique.append(e)

    return unique


def find_doc_by_slug(catalog: list[DocEntry], slug: str) -> DocEntry | None:
    """Find a doc entry by exact slug match."""
    for e in catalog:
        if e.slug == slug:
            return e
    return None
