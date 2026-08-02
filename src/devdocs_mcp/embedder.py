"""Embedding and indexing — create semantic search from downloaded docs."""

from __future__ import annotations

import html as html_module
import json
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from lxml import html as lxml_html
from lxml import etree

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
    index_json: dict[str, Any] | None = None,
    progress_callback: Optional[Callable] = None
) -> list[SearchDocument]:
    """Extract searchable documents from a db.json content map.

    If index_json is provided with entries, extracts individual entries.
    Otherwise falls back to page-based extraction.
    
    Args:
        db_json: Page content map (path → HTML)
        slug: Documentation slug
        max_chunk_tokens: Maximum tokens per document chunk
        index_json: Optional index.json with entry metadata
        progress_callback: Optional callback(current, total, slug) for progress reporting
    """
    # Use entry-based extraction if index available
    if index_json and "entries" in index_json:
        return _extract_entries_from_index(db_json, index_json, slug, max_chunk_tokens, progress_callback)
    
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
        # Parse once with lxml.html (much faster than BeautifulSoup)
        tree = lxml_html.fromstring(raw_html)
        
        # Extract text and title from the same tree
        text = _clean_html(tree)
        if not text.strip():
            continue

        title = _extract_title(tree, slug)

        # Chunk large documents
        chunks = chunk_document(text, max_tokens=max_chunk_tokens)
        
        for i, chunk in enumerate(chunks):
            # Generate section identifier from title hint if available
            if len(chunks) > 1:
                if chunk.title_hint:
                    section = _slugify_section(chunk.title_hint)
                    doc_id_suffix = f"#{section}"
                else:
                    doc_id_suffix = f"#{i}"
            else:
                doc_id_suffix = ""
            
            doc_id = f"{slug}#{path}{doc_id_suffix}" if "#" in path else f"{slug}/{path.replace('/', '#')}{doc_id_suffix}"

            # Generate better chunk title
            if len(chunks) > 1:
                if chunk.title_hint:
                    chunk_title = f"{title}: {chunk.title_hint}"
                else:
                    chunk_title = f"{title} (part {i+1}/{len(chunks)})"
            else:
                chunk_title = title

            docs.append(SearchDocument(
                id=doc_id,
                slug=slug,
                title=chunk_title,
                content=chunk.text.strip(),
                source_type="devdocs",
            ))

    logger.info("Extracted %d documents from %s (page-based, with chunking)", len(docs), slug)
    return docs


def _extract_entries_from_index(
    db_json: dict[str, Any],
    index_json: dict[str, Any],
    slug: str,
    max_chunk_tokens: int = 512,
    progress_callback: Optional[Callable] = None
) -> list[SearchDocument]:
    """Extract individual entries using index.json metadata.
    
    Uses anchor information to extract specific sections from pages.
    Falls back to full page if anchor extraction fails.
    
    Args:
        db_json: Page content map (path → HTML)
        index_json: Index with entries array
        slug: Documentation slug
        max_chunk_tokens: Maximum tokens per document chunk
        progress_callback: Optional callback(current, total, slug) for progress reporting
    """
    docs: list[SearchDocument] = []
    entries = index_json.get("entries", [])
    
    # Pre-parse all pages to avoid re-parsing for each entry (MAJOR OPTIMIZATION)
    parsed_pages = {}
    for page_path, page_html in db_json.items():
        parsed_pages[page_path] = lxml_html.fromstring(page_html)
    
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
            page_tree = parsed_pages.get(page_path)
            if page_tree is None:
                logger.debug("Page not found for entry %s: %s", name, page_path)
                continue
            
            # Extract section for this entry (returns lxml element)
            if anchor:
                entry_tree = _extract_section_by_anchor_from_tree(page_tree, anchor)
                if entry_tree is None:
                    logger.debug("Anchor not found for entry %s: %s", name, anchor)
                    continue
            else:
                entry_tree = page_tree
            
            # Clean and chunk (no re-parsing needed)
            text = _clean_html(entry_tree)
            if not text.strip():
                continue
            
            chunks = chunk_document(text, max_tokens=max_chunk_tokens)
            
            for i, chunk in enumerate(chunks):
                # Generate section identifier from title hint if available
                if len(chunks) > 1:
                    if chunk.title_hint:
                        section = _slugify_section(chunk.title_hint)
                        doc_id_suffix = f"#{section}"
                    else:
                        doc_id_suffix = f"#{i}"
                else:
                    doc_id_suffix = ""
                
                doc_id = f"{slug}#{path}{doc_id_suffix}"
                
                # Generate better chunk title
                if len(chunks) > 1:
                    if chunk.title_hint:
                        chunk_title = f"{name}: {chunk.title_hint}"
                    else:
                        chunk_title = f"{name} (part {i+1}/{len(chunks)})"
                else:
                    chunk_title = name
                
                docs.append(SearchDocument(
                    id=doc_id,
                    slug=slug,
                    title=chunk_title,
                    content=chunk.text.strip(),
                    type=entry_type,
                    path=path,
                    source_type="devdocs",
                ))
        
        # Log progress every batch and call progress callback
        processed = batch_idx + len(batch)
        if progress_callback:
            progress_callback(processed, len(entries), slug)
        
        if batch_idx + batch_size < len(entries):
            logger.info("Processed %d/%d entries...", processed, len(entries))
    
    logger.info("Extracted %d documents from %s (%d entries in index.json)", len(docs), slug, len(entries))
    return docs


def _extract_section_by_anchor_from_tree(tree: Any, anchor: str) -> Any:
    """Extract lxml element section for a specific anchor ID from pre-parsed tree.
    
    Finds the element with the matching ID and collects content
    until the next heading of the same or higher level.
    
    Args:
        tree: Pre-parsed lxml element tree
        anchor: Anchor ID to search for
        
    Returns:
        lxml element containing the section, or None if anchor not found
    """
    # Try finding with and without leading underscore using XPath
    element = tree.get_element_by_id(anchor, None)
    if element is None and anchor.startswith('_'):
        element = tree.get_element_by_id(anchor[1:], None)
    elif element is None and not anchor.startswith('_'):
        element = tree.get_element_by_id(f'_{anchor}', None)
    
    if element is None:
        return None
    
    # For simplicity, just return the element itself
    # lxml will extract text from it and all descendants
    return element

def _extract_title(tree: Any, default: str) -> str:
    """Extract title from HTML.
    
    Args:
        tree: lxml element tree
        default: Default title if extraction fails
    """
    # Try h1 first
    h1_elements = tree.xpath('.//h1')
    if h1_elements:
        return h1_elements[0].text_content().strip()
    
    # Try title tag
    title_elements = tree.xpath('.//title')
    if title_elements:
        return title_elements[0].text_content().strip()
    
    # Fallback to default
    return default.replace("_", " ").title()


def _slugify_section(text: str, max_length: int = 50) -> str:
    """Convert text to a URL-friendly section identifier.
    
    Args:
        text: Text to slugify
        max_length: Maximum length of the slug
        
    Returns:
        Slugified text suitable for use in doc IDs
    """
    import re
    
    # Convert to lowercase
    slug = text.lower()
    
    # Replace spaces and special chars with hyphens
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[-\s]+', '-', slug)
    
    # Remove leading/trailing hyphens
    slug = slug.strip('-')
    
    # Truncate to max length at word boundary
    if len(slug) > max_length:
        slug = slug[:max_length]
        # Try to break at last hyphen
        last_hyphen = slug.rfind('-')
        if last_hyphen > max_length // 2:  # Only if not too short
            slug = slug[:last_hyphen]
    
    return slug or "section"  # Fallback if empty


def scan_local_directory(base_path: Path) -> list[Path]:
    """Scan a local directory for HTM/HTML files to index."""
    html_files: list[Path] = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.htm', '.html')):
                html_files.append(Path(root) / file)
        for dir in dirs:
            if dir.startswith('.') or "node_modules" in dir or "__pycache__" in dir:
                dirs.remove(dir)  # Skip hidden directories
    return html_files

def extract_text_from_local(
    f: Path, 
    slug: str, 
    source_type: str,
    max_chunk_tokens: int = 512
) -> list[SearchDocument]:
    """Extract documents from a local HTML file with chunking.
    
    Args:
        f: Path to HTML file
        slug: Documentation slug
        source_type: Source type identifier (e.g., "local", "web")
        max_chunk_tokens: Maximum tokens per chunk (default: 512)
        
    Returns:
        List of SearchDocument chunks with extracted titles
    """
    import sys
    docs: list[SearchDocument] = []
    
    # Log file size for large files
    file_size_mb = f.stat().st_size / (1024 * 1024)
    if file_size_mb > 5:
        logger.info("  Reading %s (%.1f MB)...", f.name, file_size_mb)
        sys.stderr.flush()
    
    # Read HTML content
    html_content = f.read_text(encoding='utf-8')
    
    if file_size_mb > 5:
        logger.info("  Parsing HTML with lxml (fast)...")
        sys.stderr.flush()
    
    # Parse HTML once with lxml (much faster than BeautifulSoup)
    tree = lxml_html.fromstring(html_content)
    
    # Extract title from parsed tree
    title = _extract_title(tree, default=f.stem)
    
    # Clean HTML to get text (reuses the same tree object)
    text = _clean_html(tree)
    
    if not text.strip():
        return []
    
    if file_size_mb > 5:
        logger.info("  Chunking document...")
        sys.stderr.flush()
    
    # Chunk large documents
    chunks = chunk_document(text, max_tokens=max_chunk_tokens)
    
    if file_size_mb > 5:
        logger.info("  Created %d chunks", len(chunks))
        sys.stderr.flush()
    
    for i, chunk in enumerate(chunks):
        # Generate unique ID for each chunk
        # Use consistent format across all source types: {slug}#{path}#{section}
        
        # For web sources, use the filename as the path (without .html extension)
        # For local sources, also use the filename as the path
        # This matches the devdocs format: slug#path#section
        path = f.stem  # Remove .html/.htm extension
        
        # Generate section identifier from title hint if available
        if len(chunks) > 1:
            if chunk.title_hint:
                # Convert title hint to a URL-friendly section name
                section = _slugify_section(chunk.title_hint)
                doc_id = f"{slug}#{path}#{section}"
            else:
                # Fallback to numeric index if no title hint
                doc_id = f"{slug}#{path}#{i}"
        else:
            doc_id = f"{slug}#{path}"
        
        # Generate better chunk title
        if len(chunks) > 1:
            if chunk.title_hint:
                chunk_title = f"{title}: {chunk.title_hint}"
            else:
                chunk_title = f"{title} (part {i+1}/{len(chunks)})"
        else:
            chunk_title = title
        
        docs.append(SearchDocument(
            id=doc_id,
            slug=slug,
            title=chunk_title,
            content=chunk.text.strip(),
            source_type=source_type,
            path=path,  # Add path field for consistency
        ))
    
    return docs


def _clean_html(tree: Any) -> str:
    """Strip HTML tags and extract plain text from documentation pages.
    
    Uses lxml for fast HTML parsing. Prioritizes <main> content
    if available to exclude navigation, headers, footers, and sidebars.
    
    Args:
        tree: lxml element tree
    """
    # First, find the content container (before removing anything)
    content_element = None
    
    # Prioritize <main> tag content if present
    main_elements = tree.xpath('.//main')
    if main_elements:
        content_element = main_elements[0]
    else:
        # Try common content containers
        for xpath in [
            './/article',
            './/*[@id="content"]',
            './/*[contains(@class, "content")]',
            './/*[contains(@class, "main-content")]',
            './/*[contains(@class, "documentation")]',
            './/*[@role="main"]',
        ]:
            containers = tree.xpath(xpath)
            if containers:
                content_element = containers[0]
                break
    
    # If no content container found, use the whole tree
    if content_element is None:
        content_element = tree
    
    # Now remove noise elements from the content element
    # Remove script, style, and navigation elements
    for tag in ['script', 'style', 'nav', 'header', 'footer', 'aside']:
        for element in content_element.xpath(f'.//{tag}'):
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)
    
    # Remove common noise classes using XPath
    noise_classes = [
        'sidebar', 'navigation', 'nav', 'breadcrumb', 'breadcrumbs',
        'toc', 'table-of-contents', 'menu', 'ad', 'advertisement',
        'social', 'share', 'comments', 'related', 'suggested'
    ]
    for noise_class in noise_classes:
        xpath = f'.//*[contains(concat(" ", normalize-space(@class), " "), " {noise_class} ")]'
        for element in content_element.xpath(xpath):
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)
    
    # Extract text
    text = content_element.text_content()
    
    # Normalize whitespace
    lines = [line.strip() for line in text.splitlines()]
    text = '\n'.join(line for line in lines if line)
    
    return text


def _strip_tags(html_str: str) -> str:
    """Remove all HTML tags from a string."""
    tree = lxml_html.fromstring(html_str)
    return tree.text_content()


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
    
    # SQLite has a limit on the number of host parameters per statement
    # (SQLITE_MAX_VARIABLE_NUMBER, historically 999, up to ~32766 on newer
    # builds). Chunk large lists to stay well within that limit.
    batch_size = 900
    result: dict[str, int] = {}
    for i in range(0, len(doc_ids), batch_size):
        batch = doc_ids[i:i + batch_size]
        placeholders = ','.join('?' * len(batch))
        cursor = db.execute(
            f"SELECT doc_id, id FROM documents WHERE doc_id IN ({placeholders})",
            batch
        )
        result.update({row[0]: row[1] for row in cursor.fetchall()})
    return result


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
    
    batch_size = 900
    deleted = 0
    for i in range(0, len(doc_ids), batch_size):
        batch = doc_ids[i:i + batch_size]
        placeholders = ','.join('?' * len(batch))
        cursor = db.execute(
            f"DELETE FROM documents WHERE doc_id IN ({placeholders})",
            batch
        )
        deleted += cursor.rowcount
    db.commit()
    return deleted
