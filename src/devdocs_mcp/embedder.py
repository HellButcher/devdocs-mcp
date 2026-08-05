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

import html2text

from .chunking import chunk_document

logger = logging.getLogger(__name__)


def _make_html2text_converter() -> html2text.HTML2Text:
    """Build an HTML2Text converter configured for documentation content.

    Disables line-wrapping (body_width=0) so Markdown output isn't
    hard-wrapped mid-sentence -- chunking/embedding work on logical lines,
    not a fixed terminal width. unicode_snob keeps literal Unicode
    characters (e.g. arrows, smart quotes) instead of html2text's default
    ASCII-escaping, since we want faithful text for embedding/search
    rather than a terminal-safe rendering.
    """
    converter = html2text.HTML2Text()
    converter.body_width = 0
    converter.unicode_snob = True
    return converter


# _clean_html() is called once per page/entry (potentially thousands of
# times for large docsets), so the HTML2Text converter is built once and
# reused rather than re-instantiated on every call. HTML2Text instances
# are safely reusable across handle() calls -- internal state is reset
# each time via HTMLParser.reset()/HTML2Text.__init__ conventions.
_HTML2TEXT_CONVERTER = _make_html2text_converter()


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
    """Strip HTML noise and convert documentation pages to Markdown text.
    
    Uses lxml for fast HTML parsing to locate and prune the relevant
    content (prioritizing <main> content if available, and removing
    navigation, headers, footers, and sidebars), then hands the
    remaining markup to html2text so headings, lists, links, emphasis,
    and code blocks survive as Markdown instead of being flattened into
    unstructured plain text.
    
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
    
    # Convert the cleaned markup to Markdown, preserving headings, lists,
    # links, emphasis, and code blocks instead of flattening to plain text.
    # with_tail=False: content_element may be an anchor-extracted fragment
    # (e.g. via get_element_by_id), not just a whole-page container -- its
    # .tail is unrelated sibling text that follows it in the DOM and must
    # not leak into this element's own extracted content.
    inner_html = etree.tostring(
        content_element, encoding="unicode", method="html", with_tail=False
    )
    text = _HTML2TEXT_CONVERTER.handle(inner_html)
    
    # Normalize whitespace: collapse blank-line runs left behind by
    # html2text's block-element spacing, and strip leading/trailing
    # whitespace from each line.
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned_lines: list[str] = []
    blank_run = False
    for line in lines:
        if line.strip() == "":
            if not blank_run:
                cleaned_lines.append("")
            blank_run = True
        else:
            cleaned_lines.append(line)
            blank_run = False
    text = "\n".join(cleaned_lines).strip()
    
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

    _init_fts(db)

    db.commit()
    return db


def _init_fts(db: sqlite3.Connection) -> None:
    """Create the FTS5 full-text index and keep-in-sync triggers.

    Uses an "external content" FTS5 table backed by the `documents` table
    (content_rowid='id') so `documents_fts` never duplicates storage and
    stays automatically in sync via triggers on INSERT/UPDATE/DELETE of
    `documents` — no changes needed to upsert_documents/delete_documents_by_ids.

    This provides a lexical/keyword search path (BM25) to complement the
    dense vector search, which struggles with short/bare keyword queries
    (e.g. a single word like "Unstable") that don't carry enough semantic
    context for the embedding model to match well.
    """
    db.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            title, content, content='documents', content_rowid='id'
        )
    """)

    db.execute("""
        CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
            INSERT INTO documents_fts(rowid, title, content)
            VALUES (new.id, new.title, new.content);
        END
    """)
    db.execute("""
        CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, title, content)
            VALUES ('delete', old.id, old.title, old.content);
        END
    """)
    db.execute("""
        CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
            INSERT INTO documents_fts(documents_fts, rowid, title, content)
            VALUES ('delete', old.id, old.title, old.content);
            INSERT INTO documents_fts(rowid, title, content)
            VALUES (new.id, new.title, new.content);
        END
    """)

    # Backfill for pre-existing databases that had documents inserted
    # before the FTS table/triggers existed.
    #
    # NOTE: `documents_fts` is an "external content" FTS5 table, so a plain
    # (non-MATCH) `SELECT COUNT(*) FROM documents_fts` is answered by proxy
    # from the `documents` content table itself — it does NOT reflect
    # whether the FTS5 index (segments) actually contain any data. The
    # `documents_fts_docsize` shadow table is the correct way to check
    # whether the index has been populated.
    doc_count = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    indexed_count = db.execute("SELECT COUNT(*) FROM documents_fts_docsize").fetchone()[0]
    if doc_count > 0 and indexed_count == 0:
        logger.info("Building full-text index for %d existing documents...", doc_count)
        # The 'rebuild' command is FTS5's documented way to (re)populate an
        # external-content table's index from its content table.
        db.execute("INSERT INTO documents_fts(documents_fts) VALUES('rebuild')")


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


def get_faiss_ids_by_filter(
    db: sqlite3.Connection,
    slugs: list[str] | None = None,
    source_type: str | None = None,
) -> list[int] | None:
    """Get FAISS IDs (documents.id) matching slug/source_type filters.

    Used to scope vector search to a subset of the index (via a FAISS
    IDSelector) *before* ranking, instead of taking the globally-ranked
    top-k and filtering afterwards. Filtering afterwards can silently
    drop every relevant hit when the requested slug is a small fraction
    of a much larger combined index (e.g. searching a single doc's slug
    within an index containing hundreds of thousands of chunks from many
    docs) — relevant chunks simply never make it into the initial
    unfiltered top-k candidate pool.

    Args:
        db: Database connection
        slugs: Optional list of slugs to restrict to
        source_type: Optional source_type to restrict to

    Returns:
        List of FAISS ids to restrict search to, or None if no filter
        was requested (meaning: search the whole index).
    """
    if not slugs and not source_type:
        return None

    query = "SELECT id FROM documents WHERE 1=1"
    params: list[Any] = []

    if slugs:
        placeholders = ",".join("?" * len(slugs))
        query += f" AND slug IN ({placeholders})"
        params.extend(slugs)

    if source_type:
        query += " AND source_type = ?"
        params.append(source_type)

    cursor = db.execute(query, params)
    return [row[0] for row in cursor.fetchall()]


def keyword_search(
    db: sqlite3.Connection,
    query: str,
    slugs: list[str] | None = None,
    source_type: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Lexical/keyword search over title+content using SQLite FTS5 (BM25).

    Complements the dense vector search: bare keyword or exact-term queries
    (e.g. a single word like "Unstable") often don't carry enough semantic
    context for an embedding model to score well, even when the term
    appears verbatim in a highly-relevant document. FTS5's BM25 ranking
    reliably surfaces exact/near-exact term matches that semantic search
    can miss.

    Args:
        db: Database connection
        query: Search query (raw user text; sanitized into an FTS5 query)
        slugs: Optional list of slugs to restrict to
        source_type: Optional source_type to restrict to
        limit: Maximum number of results

    Returns:
        List of dicts with doc_id, faiss_id, and the raw BM25 score,
        ordered best-match-first (BM25 is more negative for better
        matches). Callers combine this ranked list with the semantic
        search's ranked list via Reciprocal Rank Fusion (see
        operations.search_docs_impl) rather than trying to compare BM25
        and cosine-similarity magnitudes directly — the two scores live on
        unrelated scales and BM25 magnitude isn't reliably comparable
        across corpora/queries, so rank position is the more robust signal
        to fuse on.
    """
    fts_query = _build_fts_query(query)
    if not fts_query:
        return []

    sql = """
        SELECT d.id, d.doc_id, bm25(documents_fts, 10.0, 1.0) AS rank
        FROM documents_fts
        JOIN documents d ON d.id = documents_fts.rowid
        WHERE documents_fts MATCH ?
    """
    # Column weights: title=10.0, content=1.0. A term matching the *title*
    # (e.g. "Unstable" in "The Rust Unstable Book") is a far stronger
    # relevance signal than the same term merely appearing once inside a
    # large content chunk (e.g. a code sample using the word "unstable"),
    # so it's weighted much higher to rank the on-topic doc first.
    params: list[Any] = [fts_query]

    if slugs:
        placeholders = ",".join("?" * len(slugs))
        sql += f" AND d.slug IN ({placeholders})"
        params.extend(slugs)
    if source_type:
        sql += " AND d.source_type = ?"
        params.append(source_type)

    # bm25() returns negative values; lower (more negative) is a better match
    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    try:
        cursor = db.execute(sql, params)
        rows = cursor.fetchall()
    except sqlite3.OperationalError as e:
        logger.debug("FTS5 keyword search failed for query %r: %s", query, e)
        return []

    results = []
    for faiss_id, doc_id, rank in rows:
        results.append({"faiss_id": faiss_id, "doc_id": doc_id, "bm25": rank})

    return results


# Common English stopwords excluded from FTS queries. Bare/short-word
# queries like "Unstable" are exactly the case keyword_search exists to
# help with, but including high-frequency function words (e.g. "how",
# "to", "a") in an OR-joined FTS query would make it "match" a huge
# fraction of any corpus and pollute lexical ranking for ordinary
# natural-language queries (e.g. "how to make HTTP requests").
_FTS_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "of", "in", "on", "at",
    "to", "for", "with", "without", "is", "are", "was", "were", "be",
    "been", "being", "do", "does", "did", "how", "what", "when", "where",
    "why", "which", "who", "whom", "this", "that", "these", "those",
    "it", "its", "as", "by", "from", "into", "about", "can", "could",
    "should", "would", "will", "shall", "may", "might", "must", "not",
    "you", "your", "i", "me", "my", "we", "our", "they", "their",
})


def _build_fts_query(query: str) -> str:
    """Sanitize free-form user text into a safe FTS5 MATCH query.

    Wraps each significant (non-stopword) token as a quoted phrase and
    joins with OR so that any matching token contributes to relevance
    ranking, while avoiding FTS5 query syntax errors from special
    characters in the raw user input. Stopwords are dropped so that
    ordinary natural-language queries aren't diluted into effectively
    matching most of the corpus.

    Returns "" if there are no significant tokens (e.g. the query is only
    stopwords), signaling callers to skip lexical search for this query.
    """
    tokens = re.findall(r"[\w][\w'-]*", query)
    significant = [t for t in tokens if t.lower() not in _FTS_STOPWORDS]
    if not significant:
        return ""
    # Escape embedded double quotes, then quote each token as an FTS5 phrase
    quoted = ['"{}"'.format(t.replace('"', '""')) for t in significant]
    return " OR ".join(quoted)


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
