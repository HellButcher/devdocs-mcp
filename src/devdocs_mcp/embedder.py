"""Embedding and indexing — create semantic search from downloaded docs."""

from __future__ import annotations

import html as html_module
import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from .chunking import chunk_document

logger = logging.getLogger(__name__)


@dataclass
class SearchDocument:
    """A single searchable document chunk."""
    id: str  # unique id: {slug}#{entry_path} or {slug}/{file_stem}
    slug: str  # doc source (e.g. "async", "javascript")
    title: str  # page/entry title
    content: str  # cleaned text content
    type: str = ""  # entry type from index.json (e.g. "Control Flow")
    path: str = ""  # original path in doc (e.g. "index#apply")
    source_type: str = "devdocs"  # devdocs | local


def extract_text_from_db(
    db_json: dict[str, Any], 
    slug: str, 
    max_chunk_tokens: int = 512,
    index_json: dict[str, Any] | None = None
) -> list[SearchDocument]:
    """Extract searchable documents from a db.json content map.

    If index_json is provided with entries, extracts individual entries.
    Otherwise falls back to page-based extraction.
    
    Args:
        db_json: Page content map (path → HTML)
        slug: Documentation slug
        max_chunk_tokens: Maximum tokens per document chunk
        index_json: Optional index.json with entry metadata
    """
    # Use entry-based extraction if index available
    if index_json and "entries" in index_json:
        return _extract_entries_from_index(db_json, index_json, slug, max_chunk_tokens)
    
    # Fallback to page-based extraction
    return _extract_pages_from_db(db_json, slug, max_chunk_tokens)


def _extract_pages_from_db(db_json: dict[str, Any], slug: str, max_chunk_tokens: int = 512) -> list[SearchDocument]:
    """Extract searchable documents from pages (original behavior).

    The db.json maps page paths to HTML content strings.
    We clean the HTML and create one SearchDocument per page.
    Large documents are chunked into smaller pieces.
    
    Args:
        db_json: Page content map (path → HTML)
        slug: Documentation slug
        max_chunk_tokens: Maximum tokens per document chunk
    """
    docs: list[SearchDocument] = []

    for path, raw_html in db_json.items():
        text = _clean_html(raw_html)
        if not text.strip():
            continue

        # Extract title from HTML
        title = _extract_title(raw_html, slug)

        # Chunk large documents
        chunks = chunk_document(text, max_tokens=max_chunk_tokens)
        
        for i, chunk_text in enumerate(chunks):
            doc_id_suffix = f"#{i}" if len(chunks) > 1 else ""
            doc_id = f"{slug}#{path}{doc_id_suffix}" if "#" in path else f"{slug}/{path.replace('/', '#')}{doc_id_suffix}"

            docs.append(SearchDocument(
                id=doc_id,
                slug=slug,
                title=f"{title} (part {i+1}/{len(chunks)})" if len(chunks) > 1 else title,
                content=chunk_text.strip(),
                source_type="devdocs",
            ))

    logger.info("Extracted %d documents from %s (page-based, with chunking)", len(docs), slug)
    return docs


def _extract_entries_from_index(
    db_json: dict[str, Any],
    index_json: dict[str, Any],
    slug: str,
    max_chunk_tokens: int = 512
) -> list[SearchDocument]:
    """Extract individual entries using index.json metadata.
    
    Uses anchor information to extract specific sections from pages.
    Falls back to full page if anchor extraction fails.
    
    Args:
        db_json: Page content map (path → HTML)
        index_json: Index with entries array
        slug: Documentation slug
        max_chunk_tokens: Maximum tokens per document chunk
    """
    docs: list[SearchDocument] = []
    entries = index_json.get("entries", [])
    
    # Pre-parse all pages to avoid re-parsing for each entry (MAJOR OPTIMIZATION)
    parsed_pages = {}
    for page_path, page_html in db_json.items():
        parsed_pages[page_path] = BeautifulSoup(page_html, 'html.parser')
    
    logger.info("Processing %d entries from %s...", len(entries), slug)
    
    # Process entries in batches to show progress
    batch_size = 50
    for batch_idx in range(0, len(entries), batch_size):
        batch = entries[batch_idx:batch_idx + batch_size]
        
        for entry in batch:
            name = entry.get("name", "")
            path = entry.get("path", "")
            entry_type = entry.get("type", "")
            
            if not name or not path:
                continue
            
            # Split path into page and anchor (e.g., "index#_vkcreateshadermodule_3")
            page_path = path.split("#")[0] if "#" in path else path
            anchor = path.split("#")[1] if "#" in path else None
            
            # Get pre-parsed page
            soup = parsed_pages.get(page_path)
            if not soup:
                logger.debug("Page not found for entry %s: %s", name, page_path)
                continue
            
            # Extract section for this entry
            if anchor:
                entry_html = _extract_section_by_anchor_from_soup(soup, anchor)
            else:
                entry_html = str(soup)
            
            # Clean and chunk
            text = _clean_html(entry_html)
            if not text.strip():
                continue
            
            chunks = chunk_document(text, max_tokens=max_chunk_tokens)
            
            for i, chunk_text in enumerate(chunks):
                doc_id_suffix = f"#{i}" if len(chunks) > 1 else ""
                doc_id = f"{slug}#{path}{doc_id_suffix}"
                
                docs.append(SearchDocument(
                    id=doc_id,
                    slug=slug,
                    title=f"{name} (part {i+1}/{len(chunks)})" if len(chunks) > 1 else name,
                    content=chunk_text.strip(),
                    type=entry_type,
                    path=path,
                    source_type="devdocs",
                ))
        
        # Log progress every batch
        if batch_idx + batch_size < len(entries):
            logger.info("Processed %d/%d entries...", batch_idx + len(batch), len(entries))
    
    logger.info("Extracted %d documents from %s (%d entries in index.json)", len(docs), slug, len(entries))
    return docs


def _extract_section_by_anchor_from_soup(soup: Any, anchor: str) -> str:
    """Extract HTML section for a specific anchor ID from pre-parsed soup.
    
    Finds the element with the matching ID and collects content
    until the next heading of the same or higher level.
    
    Args:
        soup: Pre-parsed BeautifulSoup object
        anchor: Anchor ID to search for
        
    Returns:
        HTML section for the anchor, or empty string if anchor not found
    """
    # Try finding with and without leading underscore
    element = soup.find(id=anchor)
    if not element and anchor.startswith('_'):
        element = soup.find(id=anchor[1:])
    elif not element and not anchor.startswith('_'):
        element = soup.find(id=f'_{anchor}')
    
    if not element:
        return ""
    
    # Collect content from this element until the next same-level heading
    section_parts = []
    current = element
    
    # Determine the heading level to stop at
    heading_levels = {'h1': 1, 'h2': 2, 'h3': 3, 'h4': 4, 'h5': 5, 'h6': 6}
    stop_level = None
    
    # If element is a heading, stop at same level
    if element.name in heading_levels:
        stop_level = heading_levels[element.name]
    # Otherwise stop at h2 or higher
    else:
        stop_level = 2
    
    while current:
        section_parts.append(str(current))
        current = current.find_next_sibling()
        
        # Stop at next heading of same or higher level
        if current and current.name in heading_levels:
            if heading_levels[current.name] <= stop_level:
                break
    
    return ''.join(section_parts)


def _extract_title(html: str, default: str) -> str:
    """Extract title from HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Try h1 first
    h1 = soup.find('h1')
    if h1:
        return h1.get_text(strip=True)
    
    # Try title tag
    title_tag = soup.find('title')
    if title_tag:
        return title_tag.get_text(strip=True)
    
    # Fallback to default
    return default.replace("_", " ").title()


def extract_text_from_local(docs_dir: Path, slug: str) -> list[SearchDocument]:
    """Extract documents from a local HTML directory."""
    local_dir = docs_dir / "local" / slug
    if not local_dir.exists():
        return []

    docs: list[SearchDocument] = []
    for f in sorted(local_dir.glob("*.html")):
        text = _clean_html(f.read_text())
        if not text.strip():
            continue

        title = f.stem.replace("_", " ").title()
        docs.append(SearchDocument(
            id=f"local/{slug}/{f.stem}",
            slug=slug,
            title=title,
            content=text.strip(),
            source_type="local",
        ))

    return docs


def _filter_nested_elements(elements: list) -> list:
    """Filter out elements that are nested inside other elements in the list.
    
    Args:
        elements: List of BeautifulSoup elements
        
    Returns:
        List of only top-level (non-nested) elements
        
    Example:
        If element A contains element B, only A is returned.
    """
    if not elements:
        return []
    
    top_level = []
    for element in elements:
        # Check if this element is nested inside any other element in the list
        is_nested = any(
            element in other.descendants 
            for other in elements 
            if other != element
        )
        if not is_nested:
            top_level.append(element)
    
    return top_level


def _clean_html(raw: str) -> str:
    """Strip HTML tags and extract plain text from documentation pages.
    
    Uses BeautifulSoup for robust HTML parsing. Prioritizes <main> content
    if available to exclude navigation, headers, footers, and sidebars.
    Handles multiple main containers without nesting duplicates.
    """
    soup = BeautifulSoup(raw, 'html.parser')
    
    # Prioritize <main> tag content if present (semantic HTML for main content)
    main_tags = soup.find_all('main')
    
    if main_tags:
        top_level_mains = _filter_nested_elements(main_tags)
        
        if top_level_mains:
            logger.debug(
                "Found %d top-level <main> tag(s), extracting main content only", 
                len(top_level_mains)
            )
            # Create a new soup with only the main content containers
            combined_soup = BeautifulSoup('<div></div>', 'html.parser')
            container = combined_soup.div
            for main in top_level_mains:
                container.append(main)
            soup = container
    else:
        # If no <main>, look for common content containers
        # Try to find ALL matching containers of the same type
        for selector_func in [
            lambda s: s.find_all('article'),
            lambda s: s.find_all(class_='content'),
            lambda s: s.find_all(id='content'),
            lambda s: s.find_all(class_='main-content'),
            lambda s: s.find_all(class_='documentation'),
            lambda s: s.find_all(role='main'),
        ]:
            containers = selector_func(soup)
            if containers:
                top_level = _filter_nested_elements(containers)
                
                if top_level:
                    logger.debug(
                        "Found %d content container(s): %s", 
                        len(top_level), 
                        top_level[0].name if top_level else "none"
                    )
                    # Create combined soup
                    combined_soup = BeautifulSoup('<div></div>', 'html.parser')
                    wrapper = combined_soup.div
                    for cont in top_level:
                        wrapper.append(cont)
                    soup = wrapper
                    break
    
    # Remove script, style, and navigation elements
    for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        element.decompose()
    
    # Remove common noise classes
    noise_classes = [
        'sidebar', 'navigation', 'nav', 'breadcrumb', 'breadcrumbs',
        'toc', 'table-of-contents', 'menu', 'ad', 'advertisement',
        'social', 'share', 'comments', 'related', 'suggested'
    ]
    for noise_class in noise_classes:
        for element in soup.find_all(class_=lambda c: c and noise_class in c.lower()):
            element.decompose()
    
    # Get text with some structure preservation
    text = soup.get_text(separator='\n', strip=True)
    
    # Normalize whitespace
    lines = [line.strip() for line in text.splitlines()]
    text = '\n'.join(line for line in lines if line)
    
    return text


def _strip_tags(html_str: str) -> str:
    """Remove all HTML tags from a string."""
    soup = BeautifulSoup(html_str, 'html.parser')
    return soup.get_text(strip=True)


# ---------------------------------------------------------------------------
# Metadata database (SQLite)
# ---------------------------------------------------------------------------

def init_metadata_db(db_path: Path) -> sqlite3.Connection:
    """Initialize the SQLite metadata database.
    
    Uses explicit INTEGER PRIMARY KEY as FAISS ID directly.
    This is an alias for ROWID but makes the schema clearer.
    """
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(db_path))
    db.execute("PRAGMA journal_mode=WAL")
    
    # Documents table with id as FAISS ID
    db.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT UNIQUE NOT NULL,
            slug TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            type TEXT DEFAULT '',
            path TEXT DEFAULT '',
            source_type TEXT DEFAULT 'devdocs',
            chunk_index INTEGER DEFAULT 0,
            total_chunks INTEGER DEFAULT 1
        )
    """)
    
    # Indexes for fast lookups
    db.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_docs_doc_id ON documents(doc_id)
    """)
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_docs_slug ON documents(slug)
    """)
    db.execute("""
        CREATE INDEX IF NOT EXISTS idx_docs_source ON documents(source_type)
    """)
    
    db.commit()
    return db


def upsert_documents(db: sqlite3.Connection, docs: list[SearchDocument]) -> int:
    """Insert or update documents in the metadata database.
    
    Returns both the number of documents inserted and their FAISS IDs (ROWIDs).
    
    Args:
        db: Database connection
        docs: List of SearchDocument objects
        
    Returns:
        Number of documents inserted/updated
    """
    db.executemany("""
        INSERT INTO documents (doc_id, slug, title, content, type, path, source_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
            slug=excluded.slug,
            title=excluded.title,
            content=excluded.content,
            type=excluded.type,
            path=excluded.path,
            source_type=excluded.source_type
    """, [
        (d.id, d.slug, d.title, d.content, d.type, d.path, d.source_type)
        for d in docs
    ])
    db.commit()
    return len(docs)


def get_faiss_ids_for_doc_ids(db: sqlite3.Connection, doc_ids: list[str]) -> dict[str, int]:
    """Get FAISS IDs (integer PKs) for document IDs.
    
    Args:
        db: Database connection
        doc_ids: List of document IDs
        
    Returns:
        Dictionary mapping doc_id to FAISS ID (id column)
    """
    if not doc_ids:
        return {}
    
    placeholders = ','.join('?' * len(doc_ids))
    cursor = db.execute(
        f"SELECT doc_id, id FROM documents WHERE doc_id IN ({placeholders})",
        doc_ids
    )
    return {row[0]: row[1] for row in cursor.fetchall()}


def get_all_doc_ids(db: sqlite3.Connection) -> list[str]:
    """Get all document IDs from the database."""
    cursor = db.execute("SELECT doc_id FROM documents ORDER BY id")
    return [row[0] for row in cursor.fetchall()]


def get_document_by_id(db: sqlite3.Connection, doc_id: str) -> dict[str, Any] | None:
    """Get a single document by its document ID (not integer PK).
    
    Args:
        db: Database connection
        doc_id: Document ID (e.g., "javascript#promises")
        
    Returns:
        Document dictionary or None
    """
    cursor = db.execute(
        "SELECT id, doc_id, slug, title, content, type, path, source_type FROM documents WHERE doc_id = ?",
        (doc_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {
        "faiss_id": row[0],  # id is the FAISS ID
        "id": row[1],        # doc_id
        "slug": row[2],
        "title": row[3],
        "content": row[4],
        "type": row[5],
        "path": row[6],
        "source_type": row[7],
    }


def get_document_by_faiss_id(db: sqlite3.Connection, faiss_id: int) -> dict[str, Any] | None:
    """Get a single document by its FAISS ID (integer PK).
    
    Args:
        db: Database connection
        faiss_id: FAISS ID (id column)
        
    Returns:
        Document dictionary or None
    """
    cursor = db.execute(
        "SELECT id, doc_id, slug, title, content, type, path, source_type FROM documents WHERE id = ?",
        (faiss_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {
        "faiss_id": row[0],
        "id": row[1],
        "slug": row[2],
        "title": row[3],
        "content": row[4],
        "type": row[5],
        "path": row[6],
        "source_type": row[7],
    }


def get_documents_by_slug(db: sqlite3.Connection, slug: str) -> list[dict[str, Any]]:
    """Get all documents for a given slug."""
    cursor = db.execute(
        "SELECT id, doc_id, slug, title, content, type, path, source_type FROM documents WHERE slug = ?",
        (slug,),
    )
    return [
        {
            "faiss_id": r[0],
            "id": r[1],
            "slug": r[2],
            "title": r[3],
            "content": r[4],
            "type": r[5],
            "path": r[6],
            "source_type": r[7],
        }
        for r in cursor.fetchall()
    ]


def count_documents(db: sqlite3.Connection) -> int:
    """Count total documents."""
    row = db.execute("SELECT COUNT(*) FROM documents").fetchone()
    return row[0] if row else 0


def delete_documents_by_ids(db: sqlite3.Connection, doc_ids: list[str]) -> int:
    """Delete documents by their document IDs.
    
    Args:
        db: Database connection
        doc_ids: List of document IDs to delete
        
    Returns:
        Number of documents deleted
    """
    if not doc_ids:
        return 0
    
    placeholders = ','.join('?' * len(doc_ids))
    cursor = db.execute(
        f"DELETE FROM documents WHERE doc_id IN ({placeholders})",
        doc_ids
    )
    db.commit()
    return cursor.rowcount
