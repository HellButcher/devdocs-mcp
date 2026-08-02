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

# Reciprocal Rank Fusion constant. 60 is the standard value used by Azure AI
# Search's hybrid search scoring and other hybrid FTS5+embedding
# implementations; it flattens the difference between adjacent top ranks
# while still favoring higher ranks. See _reciprocal_rank_fusion below.
_RRF_K = 60


def _reciprocal_rank_fusion(
    ranked_doc_id_lists: list[list[str]], k: int = _RRF_K
) -> dict[str, float]:
    """Fuse multiple ranked doc_id lists into one score per doc via RRF.

    Reciprocal Rank Fusion: RRF(d) = sum, over each ranked list L that
    contains d, of 1/(k + rank_L(d)) (rank is 1-indexed; best match first).

    This is the standard technique for combining rankings from
    heterogeneous retrieval methods (e.g. BM25 keyword search + dense
    vector/embedding search) whose raw scores live on unrelated,
    corpus-dependent scales and can't be reliably compared or calibrated
    against each other directly. Used by e.g. Azure AI Search's hybrid
    search scoring and other hybrid FTS5+embedding implementations:
    - https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking
    - https://dev.to/tofutim/how-we-built-a-hybrid-fts5-embedding-search-for-code-and-why-you-need-both-4ec2

    k=60 is the constant these implementations converge on; it has the
    effect of flattening the difference between e.g. rank 1 and rank 2
    (1/61 vs 1/62) so that top-of-list matches from *any* contributing
    signal have similar overall influence, while still favoring the
    highest ranks.

    Args:
        ranked_doc_id_lists: One or more ranked lists of doc_ids
            (best match first). Empty lists are ignored.
        k: RRF constant (default 60, the commonly used value)

    Returns:
        Dict of doc_id -> fused RRF score (higher is better; not bounded
        to [0, 1] — see search_docs_impl for how this is normalized into
        an approximate confidence score for min_score thresholding).
    """
    scores: dict[str, float] = {}
    for doc_ids in ranked_doc_id_lists:
        for rank, doc_id in enumerate(doc_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def search_docs_impl(
    config: Config,
    query: str,
    top_k: int = 10,
    min_score: float = 0.3,
    slugs: list[str] | None = None,
    source_type: str | None = None,
) -> SearchResult:
    """Search documentation using hybrid semantic + keyword search.

    This is the core implementation used by both CLI and MCP.

    Combines dense vector (semantic) search with an FTS5/BM25 keyword
    search over titles+content, fused via Reciprocal Rank Fusion (RRF —
    see `_reciprocal_rank_fusion`). This matters for bare keyword queries
    (e.g. a single word like "Unstable") which often don't carry enough
    semantic context for the embedding model to score well, even when the
    term appears verbatim in a highly relevant document/title — the
    keyword search's BM25 ranking reliably surfaces those, and RRF lets
    them contribute to the final ranking without having to calibrate BM25
    scores against cosine similarities (which live on unrelated scales).

    When `slugs`/`source_type` filters are given, the vector search is
    scoped to just the matching documents *before* ranking (via a FAISS
    ID selector), rather than taking the globally-ranked top-k and
    filtering afterwards — the latter can silently drop every relevant
    result when the filtered subset is a small fraction of a much larger
    combined index.

    Args:
        config: Configuration object
        query: Natural language search query
        top_k: Number of results to return
        min_score: Minimum vector-similarity (cosine) score, 0.0-1.0,
            applied to the semantic/vector search leg only — so weak
            dense-vector matches never enter the candidate pool at all.
            It does not gate the final fused ranking: results found via
            keyword/BM25 search but with low (or no) semantic similarity
            can still surface, since a bare keyword like "Unstable" often
            has weak embedding similarity even to a highly relevant doc.
            Final result selection is governed by Reciprocal Rank Fusion
            order (see `_reciprocal_rank_fusion`) plus `top_k`.
        slugs: Optional list of doc slugs to filter by
        source_type: Optional source type filter ('devdocs' or 'local')
        
    Returns:
        SearchResult with search results
    """
    from .embedder import get_document_by_id, get_faiss_ids_by_filter, keyword_search
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

    # Schema (including the FTS5 keyword-search index) is initialized once at
    # process startup via init_metadata_db — use a plain connection here to
    # avoid repeating that setup/backfill work on every search query.
    with sqlite3.connect(str(config.metadata_db_path)) as conn:
        # Scope the vector search to the filtered subset (if any) so
        # relevant docs in a small slug can't be crowded out by the rest
        # of a much larger combined index.
        allowed_faiss_ids = get_faiss_ids_by_filter(conn, slugs=slugs, source_type=source_type)

        # Semantic (dense vector) search, scoped by filter. min_score is
        # applied here as a vector-similarity threshold (comparable to
        # Azure AI Search's per-query vector minimum-similarity thresholds)
        # so weak dense-vector noise never enters the candidate pool.
        # idx.search() returns results already sorted best-first.
        semantic_results = idx.search(
            query, top_k=top_k * 3, min_score=min_score, allowed_faiss_ids=allowed_faiss_ids
        )

        # Lexical/keyword search (BM25 via FTS5), same scope. Also already
        # ordered best-first (lowest/most-negative BM25 first).
        lexical_results = keyword_search(
            conn, query, slugs=slugs, source_type=source_type, limit=top_k * 3
        )

    semantic_doc_ids = [r["doc_id"] for r in semantic_results]
    lexical_doc_ids = [r["doc_id"] for r in lexical_results]
    ranked_lists = [lst for lst in (semantic_doc_ids, lexical_doc_ids) if lst]

    if not ranked_lists:
        return SearchResult(
            success=True,
            query=query,
            results=[],
            total_found=0,
            filtered_by={"slugs": slugs, "source_type": source_type},
        )

    fused = _reciprocal_rank_fusion(ranked_lists, k=_RRF_K)

    # Normalize into an approximate [0, 1] confidence score purely for
    # display purposes: a doc ranked #1 by every contributing signal
    # scores ~1.0. NOTE: min_score is intentionally *not* reapplied here.
    # min_score was already used above as a real relevance threshold (a
    # vector-similarity floor) on the semantic leg. Reapplying it here
    # against the normalized RRF score would make it behave as an
    # implicit rank-position cutoff whose meaning shifts with top_k (since
    # each leg's candidate pool size is top_k*3) rather than a stable
    # quality bar — e.g. the same min_score would cut off a very
    # different effective rank depending on top_k. Final result selection
    # is governed by RRF rank order + the top_k truncation below instead.
    max_possible = len(ranked_lists) * (1.0 / (_RRF_K + 1))
    merged = {
        doc_id: min(1.0, score / max_possible) for doc_id, score in fused.items()
    }

    # Sort by fused (RRF) score, best first
    ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)

    if not ranked:
        return SearchResult(
            success=True,
            query=query,
            results=[],
            total_found=0,
            filtered_by={"slugs": slugs, "source_type": source_type},
        )

    # Fetch full document content
    db_results = []
    try:
        with sqlite3.connect(str(config.metadata_db_path)) as conn:
            for doc_id, score in ranked:
                doc = get_document_by_id(conn, doc_id)
                if doc:
                    db_results.append({"doc_id": doc_id, "score": score, **doc})
                    if len(db_results) >= top_k:
                        break
    except Exception as e:
        logger.warning("Failed to fetch document metadata: %s", e)
        db_results = [{"doc_id": doc_id, "score": score} for doc_id, score in ranked[:top_k]]

    return SearchResult(
        success=True,
        query=query,
        results=db_results,
        total_found=len(ranked),
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
    slug: str,
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
    return handler.add_source(config, path=path, slug=slug)


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
# Remove doc operation
# ---------------------------------------------------------------------------

def remove_doc_impl(
    config: Config,
    slug: str,
) -> SourceOperationResult:
    """Remove documentation identified by slug, regardless of source type.
    
    Detects which source (devdocs, local, web) the slug belongs to and
    delegates to that source's handler. Also removes any indexed
    embeddings/metadata for the slug from the search index.
    
    Args:
        config: Configuration object
        slug: Documentation slug to remove
        
    Returns:
        SourceOperationResult with removal status
    """
    from .sources import SourceType, detect_source_type, get_source_handler

    source_type = detect_source_type(config, slug)
    if not source_type:
        return SourceOperationResult(
            success=False,
            source_type=SourceType.DEVDOCS,
            slugs=[slug],
            errors={slug: f"Documentation '{slug}' not found."},
            metadata={},
        )

    handler = get_source_handler(source_type)
    result = handler.remove_source(config, slug=slug)

    if result.success:
        result.metadata["removed_from_index"] = _remove_slug_from_index(config, slug)

    return result


def _remove_slug_from_index(config: Config, slug: str) -> int:
    """Remove all indexed documents/embeddings for a slug.
    
    Safe to call even if no index exists yet or ML dependencies aren't
    installed - in that case it's a no-op.
    
    Args:
        config: Configuration object
        slug: Documentation slug whose index entries should be removed
        
    Returns:
        Number of documents removed from the index
    """
    if not config.metadata_db_path.exists():
        return 0

    from .faiss_index import _ML_DEPS_OK
    if not _ML_DEPS_OK:
        logger.warning("ML dependencies not installed; skipping index cleanup for '%s'", slug)
        return 0

    import sqlite3
    from .embedder import get_documents_by_slug
    from .faiss_index import EmbeddingIndex

    with sqlite3.connect(str(config.metadata_db_path)) as db:
        existing_docs = get_documents_by_slug(db, slug)

    if not existing_docs:
        return 0

    doc_ids_to_delete = [doc["id"] for doc in existing_docs]

    idx = EmbeddingIndex(config.embeddings_dir, config.metadata_db_path)
    idx.load_or_create_index()
    removed = idx.remove_documents(doc_ids_to_delete)
    idx.save_index()

    logger.info("Removed %d indexed document(s) for '%s'", removed, slug)
    return removed


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
