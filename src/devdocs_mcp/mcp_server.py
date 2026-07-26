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
    catalog = get_merged_catalog(config)
    
    # Check index status
    index_exists = (config.cache_dir / "embeddings" / "index.faiss").exists()
    indexed_slugs = set()
    
    if index_exists:
        try:
            import sqlite3
            with sqlite3.connect(str(config.metadata_db_path)) as db:
                from .embedder import get_documents_by_slug
                for entry in catalog:
                    if entry.slug in config.downloaded_slugs:
                        docs = get_documents_by_slug(db, entry.slug)
                        if docs:
                            indexed_slugs.add(entry.slug)
        except Exception as e:
            logger.warning("Failed to check index status: %s", e)
    
    # Apply filters
    filtered_entries = []
    for entry in catalog:
        if source_type and entry.source_type != source_type:
            continue
        if not include_large and entry.is_large:
            continue
        downloaded = entry.slug in config.downloaded_slugs
        if downloaded_only and not downloaded:
            continue
        
        # Apply fuzzy text filter on metadata only (not content)
        if query:
            query_lower = query.lower()
            # Search in: slug, name, type, release, alias
            searchable_fields = [
                entry.slug,
                entry.name,
                entry.type,
                entry.release or "",
                entry.alias or "",
            ]
            if not any(query_lower in field.lower() for field in searchable_fields):
                continue
        
        filtered_entries.append((entry, downloaded))
    
    if not filtered_entries:
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
    header_parts = [f"Documentation sources ({len(filtered_entries)})"]
    
    # Add index status to header
    if not index_exists:
        header_parts.append("[INDEX NOT BUILT]")
    elif indexed_slugs:
        total_downloaded = len([e for e, d in filtered_entries if d])
        indexed_count = len([e for e, _ in filtered_entries if e.slug in indexed_slugs])
        if indexed_count < total_downloaded:
            header_parts.append(f"[{indexed_count}/{total_downloaded} indexed]")
    
    if query:
        header_parts.append(f"(matching '{query}')")
    
    results.append(" ".join(header_parts) + ":\n")
    
    for entry, downloaded in filtered_entries:
        # Status markers
        markers = []
        if not downloaded_only and downloaded:
            markers.append("downloaded")
        if downloaded and entry.slug in indexed_slugs:
            markers.append("indexed")
        elif downloaded and index_exists:
            markers.append("NOT indexed")
        
        marker_str = f" [{', '.join(markers)}]" if markers else ""
        
        results.append(
            f"{entry.slug}{marker_str} - {entry.name}"
            + (f" v{entry.version}" if entry.version else "")
            + f" ({entry.size_mb:.1f} MB)"
        )
    
    # Add helpful tip if index not built
    footer = []
    if not index_exists and any(d for _, d in filtered_entries):
        footer.append("\nRun 'rebuild_index' to create the search index for downloaded docs.")
    elif indexed_slugs and any(d and e.slug not in indexed_slugs for e, d in filtered_entries):
        unindexed = [e.slug for e, d in filtered_entries if d and e.slug not in indexed_slugs]
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

    # Check if we have any indexed documents
    if not (config.cache_dir / "embeddings" / "index.faiss").exists():
        return (
            "No embeddings found. Please download and index documentation first.\n\n"
            "Run 'download_doc' to download a doc, then 'rebuild_index' to create embeddings."
        )

    idx = _get_index()
    if not idx._loaded:
        try:
            idx.load_or_create_index()
        except Exception as e:
            return f"Failed to load embedding index: {e}"

    results = idx.search(query, top_k=top_k * 3, min_score=min_score)  # Get more for filtering

    if not results:
        return "No matching documents found. Try a different query or download more documentation."

    # Fetch full document content from metadata DB
    import sqlite3 as _sqlite3
    db_results = []
    try:
        with _sqlite3.connect(str(config.metadata_db_path)) as conn:
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

    if not db_results:
        filter_msg = ""
        if slugs:
            filter_msg += f" (filtered by slugs: {', '.join(slugs)})"
        if source_type:
            filter_msg += f" (filtered by source_type: {source_type})"
        return f"No matching documents found{filter_msg}. Try adjusting your filters or query."

    # Format results
    filter_info = []
    if slugs:
        filter_info.append(f"slugs: {', '.join(slugs)}")
    if source_type:
        filter_info.append(f"source: {source_type}")
    
    header = f"Found {len(db_results)} matching documents"
    if filter_info:
        header += f" ({'; '.join(filter_info)})"
    header += ":\n\n"
    
    output_parts = [header]
    for i, r in enumerate(db_results):
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
    catalog = fetch_devdocs_catalog(cache_dir=config.cache_dir)
    
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

    # Initialize DB for document extraction
    db = init_metadata_db(config.metadata_db_path)
    
    results = {}
    total = len(slug_list)
    extracted_docs_count = 0
    
    for i, slug in enumerate(slug_list):
        if total > 1:
            logger.info("Processing %d/%d: %s", i + 1, total, slug)
        
        # Validate slug exists in catalog
        doc_entry = find_doc_by_slug(catalog, slug)
        if not doc_entry:
            results[slug] = "not found"
            continue

        # Apply version if specified (single mode only)
        if single_mode and version and version != "":
            full_slug = f"{slug}~{version}"
        else:
            full_slug = slug

        # Check if already downloaded
        if full_slug in config.downloaded_slugs:
            results[slug] = "already downloaded"
            continue

        try:
            # Download the documentation
            metadata = _download_doc_impl(full_slug, config.docs_dir)
            config.downloaded_slugs.add(full_slug)
            
            # Extract text for indexing
            db_pages = get_doc_pages(full_slug, config.docs_dir)
            if not db_pages:
                results[slug] = "downloaded but empty"
                continue

            # Get index.json for entry-level extraction
            index_data = get_doc_index(full_slug, config.docs_dir)
            documents = extract_text_from_db(db_pages, full_slug, index_json=index_data)
            
            # Count entries vs pages for better reporting
            entry_count = len(index_data.get("entries", [])) if index_data else 0
            page_count = len(db_pages)
            
            logger.info("Extracted %d documents from '%s' (%d entries, %d pages)", 
                       len(documents), slug, entry_count, page_count)

            # Store documents in metadata DB
            upsert_documents(db, documents)
            extracted_docs_count += len(documents)
            
            results[slug] = "ok"
        except Exception as e:
            results[slug] = f"error: {e}"

    # Save config if any downloads succeeded
    if any(r == "ok" for r in results.values()):
        config.save()

    # Build response message
    if single_mode:
        slug = slug_list[0]
        status = results[slug]
        
        if status == "ok":
            doc_entry = find_doc_by_slug(catalog, slug)
            return (
                f"Successfully downloaded '{slug}' ({doc_entry.size_mb:.1f} MB)\n"
                f"Extracted {extracted_docs_count} documents\n\n"
                "Run 'rebuild_index' to create embeddings for semantic search."
            )
        elif status == "already downloaded":
            return f"Documentation '{slug}' is already downloaded."
        elif status == "not found":
            return f"Documentation '{slug}' not found. Run 'list_docs' to see available docs."
        elif status == "downloaded but empty":
            return (
                f"Downloaded '{slug}' but no page content found.\n"
                "The doc may be empty or in an unexpected format."
            )
        else:
            return f"Failed to download '{slug}': {status}"
    else:
        # Batch mode response
        success_count = sum(1 for v in results.values() if v == "ok")
        already_count = sum(1 for v in results.values() if v == "already downloaded")
        failed = [f"{s}: {r}" for s, r in results.items() if r not in ["ok", "already downloaded"]]
        
        msg_parts = []
        if success_count > 0:
            msg_parts.append(f"Downloaded {success_count}/{total} docs successfully")
        if already_count > 0:
            msg_parts.append(f"{already_count} already downloaded")
        
        msg = ", ".join(msg_parts) + ".\n\n"
        
        if failed:
            msg += "Failed:\n" + "\n".join(f"  - {f}" for f in failed) + "\n\n"
        
        if success_count > 0:
            msg += "Run 'rebuild_index' to create embeddings for semantic search."
        
        return msg if msg_parts else f"All downloads failed: {', '.join(failed)}"


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

    # Clean mode: drop and recreate database
    if clean:
        logger.info("Clean mode: rebuilding entire index...")
        if config.metadata_db_path.exists():
            config.metadata_db_path.unlink()
            logger.info("Dropped old database")
        if config.faiss_index_path.exists():
            config.faiss_index_path.unlink()
            logger.info("Dropped old FAISS index")

    # Initialize metadata DB if needed
    db = init_metadata_db(config.metadata_db_path)

    # Get embedding index (will auto-install dependencies if needed)
    idx = _get_index()
    idx.load_or_create_index()

    # Process all downloaded docs from devdocs.io
    total_docs = 0
    total_embeddings = 0
    downloaded = list_downloaded_docs(config.docs_dir)
    total_slugs = len(downloaded)

    # If specific slugs requested, filter to only those
    if slugs:
        # Validate that requested slugs are downloaded
        not_downloaded = [s for s in slugs if s not in downloaded]
        if not_downloaded:
            return (
                f"The following docs are not downloaded: {', '.join(not_downloaded)}\n\n"
                f"Download them first with: download_doc(slug)\n\n"
                f"Available docs: {', '.join(sorted(downloaded))}"
            )
        slugs_to_index = slugs
        logger.info("Re-indexing specific docs: %s", ', '.join(slugs))
        
        # Delete existing entries for these slugs
        from .embedder import get_documents_by_slug
        for slug in slugs:
            existing_docs = get_documents_by_slug(db, slug)
            if existing_docs:
                doc_ids_to_delete = [doc['id'] for doc in existing_docs]
                logger.info("Removing %d existing documents for %s", len(doc_ids_to_delete), slug)
                
                # Remove from FAISS index and database
                idx.remove_documents(doc_ids_to_delete)
    else:
        # Get already-indexed slugs (unless clean mode)
        from .embedder import get_documents_by_slug
        indexed_slugs = set()
        if not clean:
            for slug in downloaded:
                existing_docs = get_documents_by_slug(db, slug)
                if existing_docs:
                    indexed_slugs.add(slug)

        # Filter to only missing slugs (unless clean)
        slugs_to_index = [s for s in downloaded if s not in indexed_slugs]

        if not slugs_to_index and not clean:
            return (
                "All downloaded documentation is already indexed.\n\n"
                "Use clean=True to rebuild the entire index from scratch,\n"
                "or specify slugs=['doc1', 'doc2'] to re-index specific docs."
            )

        if indexed_slugs and not clean:
            logger.info("Skipping %d already-indexed docs", len(indexed_slugs))

    logger.info("Indexing %d documentation bundles...", len(slugs_to_index))
    
    for i, slug in enumerate(slugs_to_index):
        logger.info("Processing %d/%d: %s", i + 1, len(slugs_to_index), slug)
        
        db_pages = get_doc_pages(slug, config.docs_dir)
        if not db_pages:
            continue

        # Get index.json for entry-level extraction
        index_data = get_doc_index(slug, config.docs_dir)

        # Extract documents from devdocs bundle
        docs = extract_text_from_db(db_pages, slug, index_json=index_data)
        if not docs:
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

    # Also process local sources
    for src in config.sources:
        if isinstance(src, LocalSource):
            local_docs_dir = Path(src.path) / "docs"
            if not local_docs_dir.exists():
                continue
            for slug_entry in local_docs_dir.iterdir():
                if not slug_entry.is_dir():
                    continue
                docs = extract_text_from_local(config.cache_dir, slug_entry.name)
                if not docs:
                    continue

                count = upsert_documents(db, docs)
                total_docs += count

                doc_ids = [d.id for d in docs]
                texts = [d.content for d in docs]
                emb_count = idx.add_documents(texts, doc_ids)
                total_embeddings += emb_count

    # Save the index
    idx.save_index()

    # Build appropriate message based on mode
    if slugs:
        mode_msg = f"re-indexed {len(slugs)} specific doc(s)"
        slug_list = ', '.join(slugs)
        return (
            f"Successfully {mode_msg}: {slug_list}\n"
            f"- {total_docs} documents indexed\n"
            f"- {total_embeddings} embeddings created\n\n"
            "You can now use 'search_docs' to find information."
        )
    else:
        mode_msg = "rebuilt from scratch" if clean else "updated with new documents"
        return (
            f"Index {mode_msg} successfully!\n"
            f"- {total_docs} documents indexed\n"
            f"- {total_embeddings} new embeddings created\n"
            f"- Total docs in index: {total_slugs}\n"
            f"- New docs added: {len(slugs_to_index)}\n\n"
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

    # Validate path exists and is a directory
    path_obj = Path(path).expanduser().resolve()
    if not path_obj.exists():
        return f"Path '{path}' does not exist."
    
    if not path_obj.is_dir():
        return f"Path '{path}' is not a directory."
    
    # Check if directory contains HTML files
    html_files = list(path_obj.glob("*.html")) + list(path_obj.glob("*.htm"))
    if not html_files:
        return f"No HTML files found in '{path}'. Add at least one .html or .htm file."

    # Add to sources
    source = LocalSource(path=str(path_obj), slug_prefix=slug_prefix)
    config.sources.append(source)
    config.save()

    # Index the local docs
    entries = index_local_directory(str(path_obj), slug_prefix)
    return (
        f"Added local source '{path_obj}'\n"
        f"- {len(entries)} HTML files indexed\n"
        "Run 'rebuild_index' to create embeddings for these docs."
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
    catalog = fetch_devdocs_catalog(cache_dir=config.cache_dir)
    doc_entry = find_doc_by_slug(catalog, slug)

    if not doc_entry:
        return f"Documentation '{slug}' not found in the devdocs.io catalog."

    downloaded = slug in config.downloaded_slugs
    db_pages = get_doc_pages(slug, config.docs_dir) if downloaded else None

    info_lines = [
        f"Name: {doc_entry.name}",
        f"Slug: {slug}" + (f"~{doc_entry.version}" if doc_entry.version else ""),
        f"Type: {doc_entry.type}",
        f"Size: {doc_entry.size_mb:.1f} MB",
        f"Release: {doc_entry.release or 'N/A'}",
        f"Downloaded: {'Yes' if downloaded else 'No'}",
    ]

    if doc_entry.home_url:
        info_lines.append(f"Homepage: {doc_entry.home_url}")
    if doc_entry.code_url:
        info_lines.append(f"Repository: {doc_entry.code_url}")

    if db_pages:
        page_count = len(db_pages)
        total_size = sum(len(str(v)) for v in db_pages.values())
        info_lines.append(f"\nPages: {page_count}")
        info_lines.append(f"Total content size: {total_size / 1024:.1f} KB")

    return "\n".join(info_lines)

