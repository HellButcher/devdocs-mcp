"""Core operations shared between CLI and MCP implementations.

This module contains the business logic for all operations, separated from
the presentation layer. Both CLI and MCP call these functions but format
the output differently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .config import Config
from .sources import SourceOperationResult, DocumentInfo, SourceType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes for operation results
# ---------------------------------------------------------------------------

@dataclass
class ReindexResult:
    """Result of a reindex operation."""
    success: bool
    total_docs: int
    total_embeddings: int
    total_available: int
    devdocs_count: int
    local_count: int
    new_docs_added: int
    slugs_processed: list[str]
    error: str | None = None
    mode: str = "incremental"  # "incremental", "clean", or "specific"


@dataclass
class SearchResult:
    """Result of a search operation."""
    success: bool
    query: str
    results: list[dict[str, Any]]
    total_found: int
    filtered_by: dict[str, Any]
    error: str | None = None


@dataclass
class ListDocsResult:
    """Result of a list_docs operation."""
    success: bool
    entries: list[tuple[Any, bool]]  # (DocEntry, is_downloaded)
    indexed_slugs: set[str]
    index_exists: bool
    total_count: int
    error: str | None = None


# Alias for backwards compatibility (DownloadResult is now SourceOperationResult)
DownloadResult = SourceOperationResult
AddLocalSourceResult = SourceOperationResult


@dataclass
class DocInfoResult:
    """Result of a doc_info operation."""
    success: bool
    slug: str
    name: str
    version: str
    type: str
    size_mb: float
    release: str
    downloaded: bool
    indexed: bool
    home_url: str | None
    code_url: str | None
    page_count: int | None
    content_size_kb: float | None
    error: str | None = None


@dataclass
class GetDocumentResult:
    """Result of a get_document operation."""
    success: bool
    doc_id: str | None = None
    slug: str | None = None
    title: str | None = None
    content: str | None = None
    type: str | None = None
    path: str | None = None
    source_type: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Reindex operation
# ---------------------------------------------------------------------------

def rebuild_index_impl(
    config: Config,
    clean: bool = False,
    slugs: list[str] | None = None,
    progress_callback: Optional[Callable] = None,
) -> ReindexResult:
    """Rebuild the embedding index from available documentation.
    
    Uses source abstraction layer - delegates extraction to source handlers.
    
    Args:
        config: Configuration object
        clean: If True, rebuild entire index from scratch
        slugs: Optional list of specific slugs to re-index
        progress_callback: Optional callback(current, total, slug) for progress reporting
        
    Returns:
        ReindexResult with operation details
    """
    from .catalog import get_merged_catalog
    from .embedder import get_documents_by_slug, init_metadata_db, upsert_documents
    from .faiss_index import EmbeddingIndex
    from .sources import SourceType, detect_source_type, get_source_handler
    
    # Clean mode: drop and recreate database
    if clean:
        logger.info("Clean mode: rebuilding entire index...")
        if config.faiss_index_path.exists():
            config.faiss_index_path.unlink()
            logger.info("Dropped old FAISS index")
        if config.metadata_db_path.exists():
            # Ensure file is closed before unlinking
            import gc
            gc.collect()  # Force garbage collection to close any open handles
            
            # Remove SQLite database and its WAL/SHM files
            config.metadata_db_path.unlink()
            # Remove WAL (Write-Ahead Log) file if exists
            wal_path = config.metadata_db_path.with_suffix('.db-wal')
            if wal_path.exists():
                wal_path.unlink()
            # Remove SHM (Shared Memory) file if exists
            shm_path = config.metadata_db_path.with_suffix('.db-shm')
            if shm_path.exists():
                shm_path.unlink()
            logger.info("Dropped old database")

    # Initialize metadata DB if needed
    db = init_metadata_db(config.metadata_db_path)

    # Get embedding index
    idx = EmbeddingIndex(config.embeddings_dir, config.metadata_db_path)
    idx.load_or_create_index()

    # Get all available sources (devdocs + local + web)
    catalog = get_merged_catalog(config)
    
    # Build availability map using source handlers
    all_available_slugs = set()
    devdocs_count = 0
    local_count = 0
    web_count = 0
    
    for entry in catalog:
        # Use entry.source_type from catalog to avoid repeated detect_source_type() calls
        source_type = SourceType(entry.source_type)
        handler = get_source_handler(source_type)
        if handler.is_available(config, entry.slug):
            all_available_slugs.add(entry.slug)
            if source_type == SourceType.DEVDOCS:
                devdocs_count += 1
            elif source_type == SourceType.LOCAL:
                local_count += 1
            elif source_type == SourceType.WEB:
                web_count += 1
    
    total_docs = 0
    total_embeddings = 0

    # If specific slugs requested, filter to only those
    if slugs:
        # Validate that requested slugs are available
        not_available = [s for s in slugs if s not in all_available_slugs]
        if not_available:
            return ReindexResult(
                success=False,
                total_docs=0,
                total_embeddings=0,
                total_available=len(all_available_slugs),
                devdocs_count=devdocs_count,
                local_count=local_count,
                new_docs_added=0,
                slugs_processed=[],
                error=f"The following docs are not available: {', '.join(not_available)}",
                mode="specific",
            )
        
        slugs_to_index = slugs
        logger.info("Re-indexing specific docs: %s", ', '.join(slugs))
        
        # Delete existing entries for these slugs
        for slug in slugs:
            existing_docs = get_documents_by_slug(db, slug)
            if existing_docs:
                doc_ids_to_delete = [doc['id'] for doc in existing_docs]
                logger.info("Removing %d existing documents for %s", len(doc_ids_to_delete), slug)
                
                # Remove from FAISS index and database
                idx.remove_documents(doc_ids_to_delete)
        
        mode = "specific"
    else:
        # Get already-indexed slugs (unless clean mode)
        indexed_slugs = set()
        if not clean:
            for slug in all_available_slugs:
                existing_docs = get_documents_by_slug(db, slug)
                if existing_docs:
                    indexed_slugs.add(slug)

        # Filter to only missing slugs (unless clean)
        slugs_to_index = [s for s in all_available_slugs if s not in indexed_slugs]

        if not slugs_to_index and not clean:
            return ReindexResult(
                success=True,
                total_docs=0,
                total_embeddings=0,
                total_available=len(all_available_slugs),
                devdocs_count=devdocs_count,
                local_count=local_count,
                new_docs_added=0,
                slugs_processed=[],
                error="All available documentation is already indexed.",
                mode="incremental",
            )

        if indexed_slugs and not clean:
            logger.info("Skipping %d already-indexed docs", len(indexed_slugs))
        
        mode = "clean" if clean else "incremental"

    logger.info("Indexing %d documentation bundles...", len(slugs_to_index))
    
    # Process each slug using its source handler
    for i, slug in enumerate(slugs_to_index):
        logger.info("Processing %d/%d: %s", i + 1, len(slugs_to_index), slug)
        
        # Detect source type and get handler
        source_type = detect_source_type(config, slug)
        if not source_type:
            logger.warning("Could not detect source type for slug %s", slug)
            continue
        
        try:
            handler = get_source_handler(source_type)
            
            # Use handler to extract documents (with progress callback)
            docs = handler.extract_documents(config, slug, progress_callback=progress_callback)
            if not docs:
                logger.warning("No documents extracted for %s", slug)
                continue
            
            # Insert into metadata DB
            count = upsert_documents(db, docs)
            total_docs += count
            
            # Prepare texts for embedding
            doc_ids = [d.id for d in docs]
            texts = [d.content for d in docs]
            
            # Add to FAISS index
            emb_count = idx.add_documents(texts, doc_ids)
            total_embeddings += emb_count
            
        except Exception as e:
            logger.error("Failed to index %s: %s", slug, e)
            continue

    # Save the index
    idx.save_index()

    return ReindexResult(
        success=True,
        total_docs=total_docs,
        total_embeddings=total_embeddings,
        total_available=len(all_available_slugs),
        devdocs_count=devdocs_count,
        local_count=local_count,
        new_docs_added=len(slugs_to_index),
        slugs_processed=slugs_to_index,
        mode=mode,
    )



# ---------------------------------------------------------------------------
# Search operation
# ---------------------------------------------------------------------------

def search_docs_impl(
    config: Config,
    query: str,
    top_k: int = 10,
    min_score: float = 0.3,
    slugs: list[str] | None = None,
    source_type: str | None = None,
) -> SearchResult:
    """Search documentation using semantic similarity.
    
    This is the core implementation used by both CLI and MCP.
    
    Args:
        config: Configuration object
        query: Natural language search query
        top_k: Number of results to return
        min_score: Minimum relevance score 0.0-1.0
        slugs: Optional list of doc slugs to filter by
        source_type: Optional source type filter ('devdocs' or 'local')
        
    Returns:
        SearchResult with search results
    """
    from .embedder import get_document_by_id
    from .faiss_index import EmbeddingIndex
    import sqlite3
    
    # Check if index exists
    if not (config.cache_dir / "embeddings" / "index.faiss").exists():
        return SearchResult(
            success=False,
            query=query,
            results=[],
            total_found=0,
            filtered_by={},
            error="No embeddings found. Please download and index documentation first.",
        )
    
    # Load index
    idx = EmbeddingIndex(config.embeddings_dir, config.metadata_db_path)
    try:
        idx.load_or_create_index()
    except Exception as e:
        return SearchResult(
            success=False,
            query=query,
            results=[],
            total_found=0,
            filtered_by={},
            error=f"Failed to load embedding index: {e}",
        )
    
    # Search
    results = idx.search(query, top_k=top_k * 3, min_score=min_score)
    
    if not results:
        return SearchResult(
            success=True,
            query=query,
            results=[],
            total_found=0,
            filtered_by={"slugs": slugs, "source_type": source_type},
        )
    
    # Fetch full document content and apply filters
    db_results = []
    try:
        with sqlite3.connect(str(config.metadata_db_path)) as conn:
            for r in results:
                doc_id = r["doc_id"]
                doc = get_document_by_id(conn, doc_id)
                if doc:
                    # Apply filters
                    if slugs and doc.get("slug") not in slugs:
                        continue
                    if source_type and doc.get("source_type") != source_type:
                        continue
                    
                    db_results.append({**r, **doc})
                    
                    # Stop once we have enough filtered results
                    if len(db_results) >= top_k:
                        break
    except Exception as e:
        logger.warning("Failed to fetch document metadata: %s", e)
        # Fallback to search results without full content
        db_results = [{"doc_id": r["doc_id"], "score": r["score"]} for r in results[:top_k]]
    
    return SearchResult(
        success=True,
        query=query,
        results=db_results,
        total_found=len(results),
        filtered_by={"slugs": slugs, "source_type": source_type},
    )


# ---------------------------------------------------------------------------
# Get document operation
# ---------------------------------------------------------------------------

def get_document_impl(
    config: Config,
    doc_id: str,
) -> GetDocumentResult:
    """Get full document content by document ID.
    
    Args:
        config: Configuration object
        doc_id: Document ID from search results
        
    Returns:
        GetDocumentResult with document content
    """
    from .embedder import get_document_by_id
    import sqlite3
    
    # Check if database exists
    if not config.metadata_db_path.exists():
        return GetDocumentResult(
            success=False,
            error="No metadata database found. Please index documentation first.",
        )
    
    try:
        with sqlite3.connect(str(config.metadata_db_path)) as conn:
            doc = get_document_by_id(conn, doc_id)
            
            if not doc:
                return GetDocumentResult(
                    success=False,
                    error=f"Document '{doc_id}' not found in index.",
                )
            
            return GetDocumentResult(
                success=True,
                doc_id=doc.get("id"),
                slug=doc.get("slug"),
                title=doc.get("title"),
                content=doc.get("content"),
                type=doc.get("type"),
                path=doc.get("path"),
                source_type=doc.get("source_type"),
            )
    except Exception as e:
        logger.error("Failed to get document: %s", e)
        return GetDocumentResult(
            success=False,
            error=f"Failed to get document: {e}",
        )


# ---------------------------------------------------------------------------
# List docs operation
# ---------------------------------------------------------------------------

def list_docs_impl(
    config: Config,
    source_type: str | None = None,
    include_large: bool = False,
    downloaded_only: bool = False,
    query: str | None = None,
) -> ListDocsResult:
    """List available documentation sources with optional filtering.
    
    Uses source abstraction - availability checks delegated to handlers.
    
    Args:
        config: Configuration object
        source_type: Filter by 'devdocs', 'local', or None for all
        include_large: Include docs larger than 50 MB
        downloaded_only: Only include downloaded/available docs
        query: Optional fuzzy text filter on doc metadata
        
    Returns:
        ListDocsResult with documentation entries
    """
    from .catalog import get_merged_catalog
    from .embedder import get_documents_by_slug
    from .sources import SourceType, get_source_handler
    import sqlite3
    
    catalog = get_merged_catalog(config)
    
    # Check index status
    index_exists = (config.cache_dir / "embeddings" / "index.faiss").exists()
    indexed_slugs = set()
    
    if index_exists:
        try:
            with sqlite3.connect(str(config.metadata_db_path)) as db:
                for entry in catalog:
                    # Use source handler to check availability
                    try:
                        source_type_enum = SourceType(entry.source_type)
                        handler = get_source_handler(source_type_enum)
                        if handler.is_available(config, entry.slug):
                            docs = get_documents_by_slug(db, entry.slug)
                            if docs:
                                indexed_slugs.add(entry.slug)
                    except ValueError:
                        logger.error(f"Unknown source type '{entry.source_type}' for slug '{entry.slug}'")
                        raise
        except Exception as e:
            logger.warning("Failed to check index status: %s", e)
    
    # Apply filters
    filtered_entries = []
    for entry in catalog:
        if source_type and entry.source_type != source_type:
            continue
        if not include_large and entry.is_large:
            continue
        
        # Use source handler to check availability (generic approach)
        try:
            source_type_enum = SourceType(entry.source_type)
            handler = get_source_handler(source_type_enum)
            downloaded = handler.is_available(config, entry.slug)
        except ValueError:
            logger.error(f"Unknown source type '{entry.source_type}' for slug '{entry.slug}'")
            raise
        
        if downloaded_only and not downloaded:
            continue
        
        # Apply fuzzy text filter on metadata only (not content)
        if query:
            query_lower = query.lower()
            searchable_fields = [
                entry.slug or "",
                entry.name or "",
                entry.type or "",
                entry.release or "",
                entry.alias or "",
            ]
            if not any(query_lower in field.lower() for field in searchable_fields if field):
                continue
        
        filtered_entries.append((entry, downloaded))
    
    return ListDocsResult(
        success=True,
        entries=filtered_entries,
        indexed_slugs=indexed_slugs,
        index_exists=index_exists,
        total_count=len(filtered_entries),
    )


# ---------------------------------------------------------------------------
# Download operation
# ---------------------------------------------------------------------------

def download_docs_impl(
    config: Config,
    slugs: list[str],
    version: str | None = None,
) -> SourceOperationResult:
    """Download one or more documentation bundles from devdocs.io.
    
    Uses source abstraction - delegates to DevDocsSourceHandler.
    
    Args:
        config: Configuration object
        slugs: List of documentation slugs to download
        version: Optional version specifier (only for single slug)
        
    Returns:
        SourceOperationResult with download status
    """
    from .sources import SourceType, get_source_handler
    
    handler = get_source_handler(SourceType.DEVDOCS)
    return handler.add_source(config, slugs=slugs, version=version)


# ---------------------------------------------------------------------------
# Add local source operation
# ---------------------------------------------------------------------------

def add_local_source_impl(
    config: Config,
    path: str,
    slug_prefix: str = "",
) -> SourceOperationResult:
    """Add a local directory as a documentation source.
    
    Uses source abstraction - delegates to LocalSourceHandler.
    
    Args:
        config: Configuration object
        path: Absolute filesystem path to the docs directory
        slug_prefix: Optional prefix for generated slugs
        
    Returns:
        SourceOperationResult with operation status
    """
    from .sources import SourceType, get_source_handler
    
    handler = get_source_handler(SourceType.LOCAL)
    return handler.add_source(config, path=path, slug_prefix=slug_prefix)


# ---------------------------------------------------------------------------
# Add web source operation
# ---------------------------------------------------------------------------

def add_web_source_impl(
    config: Config,
    url: str | None,
    slug: str,
    name: str | None = None,
    max_depth: int = 2,
    pattern: str = r".*\.html?$",
    url_prefix: str | None = None,
) -> SourceOperationResult:
    r"""Fetch documentation from a web URL.
    
    If url is None, re-downloads an existing web source by slug.
    
    Uses source abstraction - delegates to WebSourceHandler.
    
    Args:
        config: Configuration object
        url: Base URL to fetch from (optional for re-download)
        slug: Unique identifier for this web source
        name: Display name (defaults to slug)
        max_depth: Recursion depth for crawling (default: 2)
        pattern: Regex pattern for URLs to fetch (default: r".*\.html?$")
        url_prefix: Optional URL prefix to restrict crawling (default: directory of initial URL)
        
    Returns:
        SourceOperationResult with operation status
    """
    from .sources import SourceType, get_source_handler
    
    handler = get_source_handler(SourceType.WEB)
    return handler.add_source(
        config,
        url=url,
        slug=slug,
        name=name,
        max_depth=max_depth,
        pattern=pattern,
        url_prefix=url_prefix,
    )


# ---------------------------------------------------------------------------
# Doc info operation
# ---------------------------------------------------------------------------

def doc_info_impl(
    config: Config,
    slug: str,
) -> DocInfoResult:
    """Get detailed information about a specific documentation.
    
    Uses source abstraction - delegates to source handlers.
    
    Args:
        config: Configuration object
        slug: Documentation slug (e.g. 'javascript', 'async')
        
    Returns:
        DocInfoResult with documentation details
    """
    from .sources import detect_source_type, get_source_handler
    
    # Detect which source this slug belongs to
    source_type = detect_source_type(config, slug)
    
    if not source_type:
        return DocInfoResult(
            success=False,
            slug=slug,
            name="",
            version="",
            type="",
            size_mb=0.0,
            release="",
            downloaded=False,
            indexed=False,
            home_url=None,
            code_url=None,
            page_count=None,
            content_size_kb=None,
            error=f"Documentation '{slug}' not found in any catalog.",
        )
    
    # Get source handler and retrieve info
    try:
        handler = get_source_handler(source_type)
        doc_info = handler.get_info(config, slug)
        
        if not doc_info:
            return DocInfoResult(
                success=False,
                slug=slug,
                name="",
                version="",
                type="",
                size_mb=0.0,
                release="",
                downloaded=False,
                indexed=False,
                home_url=None,
                code_url=None,
                page_count=None,
                content_size_kb=None,
                error=f"Documentation '{slug}' not found.",
            )
        
        # Convert DocumentInfo to DocInfoResult
        return DocInfoResult(
            success=True,
            slug=doc_info.slug,
            name=doc_info.name,
            version=doc_info.version,
            type=doc_info.type,
            size_mb=doc_info.size_mb,
            release=doc_info.release,
            downloaded=doc_info.downloaded,
            indexed=doc_info.indexed,
            home_url=doc_info.home_url,
            code_url=doc_info.code_url,
            page_count=doc_info.page_count,
            content_size_kb=doc_info.content_size_kb,
        )
    except Exception as e:
        logger.error("Failed to get info for %s: %s", slug, e)
        return DocInfoResult(
            success=False,
            slug=slug,
            name="",
            version="",
            type="",
            size_mb=0.0,
            release="",
            downloaded=False,
            indexed=False,
            home_url=None,
            code_url=None,
            page_count=None,
            content_size_kb=None,
            error=str(e),
        )
