"""MCP server for devdocs.io semantic search."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

# Check ML dependencies availability (don't auto-install)
def _check_ml_deps():
    """Check if ML dependencies are available."""
    try:
        import faiss  # noqa: F401
        from sentence_transformers import SentenceTransformer  # noqa: F401
        return True
    except ImportError:
        return False

ML_AVAILABLE = _check_ml_deps()

from mcp.server.fastmcp import FastMCP  # noqa: E402
import httpx  # noqa: E402

from .config import (  # noqa: E402
    APP_NAME,
    CACHE_DIR,
    CONFIG_DIR,
    POPULAR_DOCS,
    Config,
    DevdocsSource,
    LocalSource,
    get_config,
)
from .catalog import (  # noqa: E402
    DocEntry,
    fetch_devdocs_catalog,
    find_doc_by_slug,
    get_merged_catalog,
    index_local_directory,
)
from .download import (  # noqa: E402
    download_doc as _download_doc_impl,
    download_docs_batch,
    get_doc_index,
    get_doc_pages,
    list_downloaded_docs,
    remove_doc as _remove_doc_impl,
)
from .embedder import (  # noqa: E402
    SearchDocument,
    count_documents,
    extract_text_from_db,
    extract_text_from_local,
    get_all_doc_ids,
    get_document_by_id,
    init_metadata_db,
    upsert_documents,
)

logger = logging.getLogger(__name__)

# Create MCP server
mcp = FastMCP(
    APP_NAME,
    instructions="Semantic search over devdocs.io documentation and custom static docs. Download docs from devdocs.io catalog, index them for semantic search, add local HTML doc directories as sources.",
)


# ---------------------------------------------------------------------------
# Helper: Initialize index on first run
# ---------------------------------------------------------------------------

def _get_index():
    """Lazy-load the embedding index."""
    if not ML_AVAILABLE:
        raise RuntimeError(
            "ML dependencies not available. Install with:\nuv sync --extra ml"
        )
    from .faiss_index import EmbeddingIndex  # noqa: E402
    config = get_config()
    return EmbeddingIndex(config.embeddings_dir, config.metadata_db_path)


def _ensure_initialized():
    """Ensure the embedding index is loaded and metadata DB exists."""
    from .faiss_index import EmbeddingIndex  # noqa: E402
    config = get_config()

    if not (config.cache_dir / "embeddings" / "index.faiss").exists():
        logger.info("Embedding index not found, creating...")
        return None  # Will be created on demand

    idx = EmbeddingIndex(config.embeddings_dir, config.metadata_db_path)
    idx.load_or_create_index()
    return idx


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_docs(
    source_type: str | None = None,
    include_large: bool = False,
    downloaded_only: bool = True,
    query: str | None = None,
) -> str:
    """List all available documentation sources with optional filtering.

    Args:
        source_type: Filter by 'devdocs', 'local', or None for all
        include_large: Include docs larger than 50 MB (default: False = exclude large)
        downloaded_only: only include downloaded docs (default: True = only show downloaded docs)
        query: Optional fuzzy text filter on doc metadata (slug, name, type, release, alias)
               Case-insensitive substring match. Does NOT search document content.
    """
    config = get_config()
    
    # Use shared implementation
    from .operations import list_docs_impl
    result = list_docs_impl(
        config,
        source_type=source_type,
        include_large=include_large,
        downloaded_only=downloaded_only,
        query=query,
    )
    
    if not result.entries:
        filter_info = []
        if query:
            filter_info.append(f"query='{query}'")
        if source_type:
            filter_info.append(f"source={source_type}")
        if downloaded_only:
            filter_info.append("downloaded_only=True")
        
        filter_str = " with filters: " + ", ".join(filter_info) if filter_info else ""
        return f"No documentation sources found{filter_str}."
    
    # Build output
    results = []
    header_parts = [f"Documentation sources ({len(result.entries)})"]
    
    # Add index status to header
    if not result.index_exists:
        header_parts.append("[INDEX NOT BUILT]")
    elif result.indexed_slugs:
        total_downloaded = len([e for e, d in result.entries if d])
        indexed_count = len([e for e, _ in result.entries if e.slug in result.indexed_slugs])
        if indexed_count < total_downloaded:
            header_parts.append(f"[{indexed_count}/{total_downloaded} indexed]")
    
    if query:
        header_parts.append(f"(matching '{query}')")
    
    results.append(" ".join(header_parts) + ":\n")
    
    for entry, downloaded in result.entries:
        # Status markers
        markers = []
        if not downloaded_only and downloaded:
            markers.append("downloaded")
        if downloaded and entry.slug in result.indexed_slugs:
            markers.append("indexed")
        elif downloaded and result.index_exists:
            markers.append("NOT indexed")
        
        marker_str = f" [{', '.join(markers)}]" if markers else ""
        
        results.append(
            f"{entry.slug}{marker_str} - {entry.name}"
            + (f" v{entry.version}" if entry.version else "")
            + f" ({entry.size_mb:.1f} MB)"
        )
    
    # Add helpful tip if index not built
    footer = []
    if not result.index_exists and any(d for _, d in result.entries):
        footer.append("\nRun 'rebuild_index' to create the search index for downloaded docs.")
    elif result.indexed_slugs and any(d and e.slug not in result.indexed_slugs for e, d in result.entries):
        unindexed = [e.slug for e, d in result.entries if d and e.slug not in result.indexed_slugs]
        footer.append(f"\nUnindexed docs: {', '.join(unindexed)}")
        footer.append("Run 'rebuild_index' to index them.")
    
    return "\n".join(results + footer)


@mcp.tool()
def search_docs(
    query: str,
    top_k: int = 10,
    min_score: float = 0.3,
    slugs: list[str] | None = None,
    source_type: str | None = None,
) -> str:
    """Search documentation using semantic similarity.

    Args:
        query: Natural language search query (e.g. "how to make HTTP requests")
        top_k: Number of results to return (default: 10)
        min_score: Minimum relevance score 0.0-1.0 (default: 0.3)
        slugs: Optional list of doc slugs to filter by (e.g. ['javascript', 'python'])
        source_type: Optional source type filter ('devdocs' or 'local')
    """
    if not ML_AVAILABLE:
        return (
            "Semantic search requires ML dependencies.\n\n"
            "Install with:\nuv sync --extra ml\n\n"
            "Alternatively, use list_docs and doc_info for basic browsing."
        )

    config = get_config()
    
    # Use shared implementation
    from .operations import search_docs_impl
    result = search_docs_impl(
        config,
        query=query,
        top_k=top_k,
        min_score=min_score,
        slugs=slugs,
        source_type=source_type,
    )
    
    # Format for MCP (string output)
    if not result.success:
        if "No embeddings found" in (result.error or ""):
            return (
                f"{result.error}\n\n"
                "Run 'download_doc' to download a doc, then 'rebuild_index' to create embeddings."
            )
        return result.error or "Search failed."
    
    if not result.results:
        filter_msg = ""
        if result.filtered_by.get("slugs"):
            filter_msg += f" (filtered by slugs: {', '.join(result.filtered_by['slugs'])})"
        if result.filtered_by.get("source_type"):
            filter_msg += f" (filtered by source_type: {result.filtered_by['source_type']})"
        return f"No matching documents found{filter_msg}. Try adjusting your filters or query."
    
    # Format results
    filter_info = []
    if result.filtered_by.get("slugs"):
        filter_info.append(f"slugs: {', '.join(result.filtered_by['slugs'])}")
    if result.filtered_by.get("source_type"):
        filter_info.append(f"source: {result.filtered_by['source_type']}")
    
    header = f"Found {len(result.results)} matching documents"
    if filter_info:
        header += f" ({'; '.join(filter_info)})"
    header += ":\n\n"
    
    output_parts = [header]
    for i, r in enumerate(result.results):
        title = r.get("title", "Unknown")
        slug = r.get("slug", "?")
        score = r["score"]
        doc_type = r.get("type", "")
        path = r.get("path", "")
        doc_source = r.get("source_type", "")

        # Truncate content for display
        content_preview = r.get("content", "")[:300]

        output_parts.append(
            f"### {i+1}. [{title}]({slug}) [score: {score:.3f}]"
            + (f" ({doc_type})" if doc_type else "")
            + (f" [{doc_source}]" if doc_source else "")
            + (f"\nPath: {path}" if path else "")
            + f"\n\n{content_preview}\n"
        )

    return "\n".join(output_parts)


@mcp.tool()
def download_doc(slugs: list[str] | str, version: str | None = None) -> str:
    """Download one or more documentation bundles from devdocs.io.

    Args:
        slugs: Single slug string or list of documentation slugs 
               (e.g. 'javascript' or ['javascript', 'python', 'react'])
        version: Optional version specifier for single slug (ignored for multiple slugs, uses latest)
    """
    config = get_config()
    
    # Normalize to list
    if isinstance(slugs, str):
        slug_list = [slugs]
        single_mode = True
    else:
        slug_list = slugs
        single_mode = False
        version = None  # Ignore version for batch mode

    if not slug_list:
        return "No slugs provided. Specify at least one documentation slug."
    
    # Use shared implementation
    from .operations import download_docs_impl
    result = download_docs_impl(config, slug_list, version=version)
    
    # Format for MCP (string output)
    if single_mode:
        slug = slug_list[0]
        
        if slug in result.successful_slugs:
            from .catalog import fetch_devdocs_catalog, find_doc_by_slug
            catalog = fetch_devdocs_catalog(cache_dir=config.cache_dir)
            doc_entry = find_doc_by_slug(catalog, slug)
            total_size_mb = result.metadata.get("total_size_mb", 0.0)
            size_mb = doc_entry.size_mb if doc_entry else total_size_mb
            return (
                f"Successfully downloaded '{slug}' ({size_mb:.1f} MB)\n\n"
                "Run 'rebuild_index' to create embeddings for semantic search."
            )
        elif slug in result.failed_slugs:
            error_msg = result.errors.get(slug, "Unknown error")
            if "not found" in error_msg:
                return f"Documentation '{slug}' not found. Run 'list_docs' to see available docs."
            return f"Failed to download '{slug}': {error_msg}"
        else:
            return f"Documentation '{slug}' is already downloaded."
    else:
        # Batch mode response
        success_count = len(result.successful_slugs)
        failed_count = len(result.failed_slugs)
        total = len(slug_list)
        
        msg_parts = []
        if success_count > 0:
            msg_parts.append(f"Downloaded {success_count}/{total} docs successfully")
        
        msg = ", ".join(msg_parts) + ".\n\n" if msg_parts else ""
        
        if failed_count > 0:
            failed_list = [f"{s}: {result.errors.get(s, 'error')}" for s in result.failed_slugs]
            msg += "Failed:\n" + "\n".join(f"  - {f}" for f in failed_list) + "\n\n"
        
        if success_count > 0:
            msg += "Run 'rebuild_index' to create embeddings for semantic search."
        
        return msg if msg_parts else f"All downloads failed."


@mcp.tool()
def rebuild_index(clean: bool = False, slugs: list[str] | None = None) -> str:
    """Rebuild the embedding index from downloaded documentation.

    By default, only indexes newly downloaded documentation that hasn't been
    indexed yet. Use clean=True to rebuild everything from scratch.

    Args:
        clean: If True, rebuild entire index from scratch. If False (default),
               only index missing documents.
        slugs: Optional list of specific doc slugs to re-index (e.g., ['vulkan', 'python']).
               If provided, only these docs will be processed. Ignores already-indexed
               status and re-indexes them.
    """
    if not ML_AVAILABLE:
        return (
            "Embedding requires ML dependencies.\n\n"
            "Install with:\nuv sync --extra ml\n\n"
            "Docs are still downloaded and stored, just without semantic search."
        )

    config = get_config()
    
    # Use shared implementation
    from .operations import rebuild_index_impl
    result = rebuild_index_impl(config, clean=clean, slugs=slugs)
    
    # Format result for MCP (string output)
    if not result.success:
        if result.error:
            return (
                f"{result.error}\n\n"
                f"Download them first with: download_doc(slug) or add as local source\n\n"
                f"Total available docs: {result.total_available}"
            )
        else:
            return result.error or "Index operation failed."
    
    # Build appropriate message based on mode
    if result.mode == "specific":
        slug_list = ', '.join(result.slugs_processed)
        return (
            f"Successfully re-indexed {len(result.slugs_processed)} specific doc(s): {slug_list}\n"
            f"- {result.total_docs} documents indexed\n"
            f"- {result.total_embeddings} embeddings created\n\n"
            "You can now use 'search_docs' to find information."
        )
    else:
        mode_msg = "rebuilt from scratch" if result.mode == "clean" else "updated with new documents"
        return (
            f"Index {mode_msg} successfully!\n"
            f"- {result.total_docs} documents indexed\n"
            f"- {result.total_embeddings} new embeddings created\n"
            f"- Total available docs: {result.total_available} ({result.devdocs_count} devdocs, {result.local_count} local)\n"
            f"- New docs added: {result.new_docs_added}\n\n"
            "You can now use 'search_docs' to find information."
        )


@mcp.tool()
def remove_doc(slug: str) -> str:
    """Remove a downloaded documentation bundle.

    Args:
        slug: Documentation slug to remove (e.g. 'async')
    """
    config = get_config()

    if slug not in list_downloaded_docs(config.docs_dir):
        return f"Documentation '{slug}' is not downloaded."

    # Remove from downloads
    success = _remove_doc_impl(slug, config.docs_dir)

    # Update config
    config.downloaded_slugs.discard(slug)
    config.save()

    # Note: Rebuild index to clean up embeddings
    return (
        f"Removed '{slug}'.\n\n"
        "Run 'rebuild_index' to update the embedding index."
    )


@mcp.tool()
def add_local_source(path: str, slug_prefix: str = "") -> str:
    """Add a local directory as a documentation source.

    The directory should contain HTML files that will be indexed for search.

    Args:
        path: Absolute filesystem path to the docs directory
        slug_prefix: Optional prefix for generated slugs (e.g. 'mylib/')
    """
    config = get_config()
    
    # Use shared implementation
    from .operations import add_local_source_impl
    result = add_local_source_impl(config, path, slug_prefix)
    
    # Format for MCP (string output)
    if not result.success:
        error_msg = list(result.errors.values())[0] if result.errors else "Failed to add local source."
        return error_msg
    
    path_str = result.metadata.get("path", path)
    num_files = result.metadata.get("num_files", len(result.successful_slugs))
    return (
        f"Added local source '{path_str}'\n"
        f"- {num_files} HTML files found\n"
        f"Run 'rebuild_index' to create embeddings for these docs."
    )


@mcp.tool()
def add_web_source(
    slug: str,
    url: str | None = None,
    name: str | None = None,
    max_depth: int = 2,
    pattern: str = r".*\.html?$",
    url_prefix: str | None = None,
) -> str:
    r"""Fetch documentation from a web URL (or re-download existing source).
    
    If url is not provided, re-downloads an existing web source by slug.
    
    By default, only fetches URLs in the same directory or below the initial URL.
    For example, fetching https://example.com/docs/api/index.html will only fetch
    files under https://example.com/docs/api/ and not https://example.com/docs/other/.
    
    Only HTML files (.html, .htm, or no extension) are fetched and followed.
    Non-HTML files (CSS, JS, images, fonts, etc.) are automatically skipped.

    Args:
        slug: Unique identifier for this web source
        url: Base URL to fetch from (optional for re-download)
        name: Display name (defaults to slug)
        max_depth: Recursion depth for crawling (default: 2)
        pattern: Regex pattern for URLs to fetch (default: .*\.html?$)
        url_prefix: Optional URL prefix to restrict crawling (default: directory of initial URL)
    """
    config = get_config()
    
    # Use shared implementation
    from .operations import add_web_source_impl
    result = add_web_source_impl(config, url, slug, name, max_depth, pattern, url_prefix)
    
    # Format for MCP (string output)
    if not result.success:
        error_msg = list(result.errors.values())[0] if result.errors else "Failed to fetch web source."
        return error_msg
    
    url_str = result.metadata.get("url", url or "existing source")
    num_files = result.metadata.get("num_files", len(result.successful_slugs))
    
    action = "Re-downloaded" if not url else "Fetched"
    return (
        f"{action} web source from '{url_str}'\n"
        f"- {num_files} HTML files downloaded\n"
        f"- Slug: {slug}\n"
        f"Run 'rebuild_index' to create embeddings for these docs."
    )


@mcp.tool()
def remove_local_source(path: str) -> str:
    """Remove a local documentation source.

    Args:
        path: The absolute filesystem path of the source to remove
    """
    config = get_config()

    original_count = len(config.sources)
    config.sources = [
        s for s in config.sources
        if not (isinstance(s, LocalSource) and s.path == path)
    ]

    if len(config.sources) >= original_count:
        return f"No local source found at '{path}'."

    config.save()
    return f"Removed local source '{path}'. Run 'rebuild_index' to update embeddings."


@mcp.tool()
def doc_info(slug: str) -> str:
    """Get detailed information about a specific documentation.

    Args:
        slug: Documentation slug (e.g. 'javascript', 'async')
    """
    config = get_config()
    
    # Use shared implementation
    from .operations import doc_info_impl
    result = doc_info_impl(config, slug)
    
    # Format for MCP (string output)
    if not result.success:
        return result.error or f"Documentation '{slug}' not found."
    
    info_lines = [
        f"Name: {result.name}",
        f"Slug: {slug}" + (f"~{result.version}" if result.version else ""),
        f"Type: {result.type}",
        f"Size: {result.size_mb:.1f} MB",
        f"Release: {result.release}",
        f"Downloaded: {'Yes' if result.downloaded else 'No'}",
    ]
    
    if result.indexed:
        info_lines.append(f"Indexed: Yes")
    elif result.downloaded:
        info_lines.append(f"Indexed: No (run 'rebuild_index' to index)")

    if result.home_url:
        info_lines.append(f"Homepage: {result.home_url}")
    if result.code_url:
        info_lines.append(f"Repository: {result.code_url}")

    if result.page_count is not None:
        info_lines.append(f"\nPages: {result.page_count}")
    if result.content_size_kb is not None:
        info_lines.append(f"Total content size: {result.content_size_kb:.1f} KB")

    return "\n".join(info_lines)

