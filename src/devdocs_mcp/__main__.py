"""Main entry point for devdocs-mcp with CLI subcommands."""

import argparse
import logging
import sys

from .config import CACHE_DIR, CONFIG_DIR


def main():
    """Run the devdocs-mcp CLI."""
    parser = argparse.ArgumentParser(
        prog="devdocs-mcp",
        description="MCP server for semantic search over devdocs.io documentation",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # MCP server command
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Run as MCP server (default)",
    )
    mcp_parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    mcp_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP port (default: 8000, only for http transport)",
    )
    
    # Query command
    query_parser = subparsers.add_parser(
        "query",
        help="Search documentation from command line",
    )
    query_parser.add_argument(
        "search_query",
        help="Search query",
    )
    query_parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=5,
        help="Number of results (default: 5)",
    )
    query_parser.add_argument(
        "-s", "--min-score",
        type=float,
        default=0.3,
        help="Minimum similarity score (default: 0.3)",
    )
    query_parser.add_argument(
        "--slugs",
        nargs="+",
        help="Filter by doc slugs (e.g., javascript python)",
    )
    query_parser.add_argument(
        "--source",
        choices=["devdocs", "local"],
        help="Filter by source type",
    )
    
    # Add command
    add_parser = subparsers.add_parser(
        "add",
        help="Download documentation or add local sources",
    )
    add_subparsers = add_parser.add_subparsers(dest="add_type", help="What to add")
    
    # Add -> download
    download_parser = add_subparsers.add_parser(
        "download",
        help="Download documentation from devdocs.io",
    )
    download_parser.add_argument(
        "slugs",
        nargs="+",
        help="Documentation slugs to download (e.g., javascript python)",
    )
    
    # Add -> local
    local_parser = add_subparsers.add_parser(
        "local",
        help="Add local HTML directory as documentation source",
    )
    local_parser.add_argument(
        "path",
        help="Path to local HTML directory",
    )
    local_parser.add_argument(
        "--prefix",
        default="",
        help="Slug prefix for generated entries",
    )
    
    # Reindex command
    reindex_parser = subparsers.add_parser(
        "reindex",
        help="Rebuild the search index from downloaded docs",
    )
    reindex_parser.add_argument(
        "--clean",
        action="store_true",
        help="Drop existing database and recreate from scratch",
    )
    reindex_parser.add_argument(
        "--slugs",
        nargs="+",
        help="Re-index specific doc slugs only (e.g., vulkan python)",
    )
    
    # List command
    list_parser = subparsers.add_parser(
        "list",
        help="List available or downloaded documentation",
    )
    list_parser.add_argument(
        "--downloaded",
        action="store_true",
        help="Show only downloaded docs",
    )
    list_parser.add_argument(
        "--large",
        action="store_true",
        help="Include large docs (>50MB)",
    )
    list_parser.add_argument(
        "--query",
        help="Fuzzy filter on doc metadata (slug, name, type, etc.)",
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger = logging.getLogger(__name__)
    
    # Default to mcp command if none specified
    if not args.command:
        args.command = "mcp"
    
    try:
        if args.command == "mcp":
            run_mcp_server(args)
        elif args.command == "query":
            run_query(args)
        elif args.command == "add":
            run_add(args)
        elif args.command == "reindex":
            run_reindex(args)
        elif args.command == "list":
            run_list(args)
        else:
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception("Fatal error: %s", e)
        sys.exit(1)


def run_mcp_server(args):
    """Run the MCP server with proper signal handling."""
    import asyncio
    import signal
    import sys
    from .mcp_server import mcp
    
    logger = logging.getLogger(__name__)
    logger.info("devdocs-mcp starting...")
    logger.info("Config dir: %s", CONFIG_DIR)
    logger.info("Cache dir:  %s", CACHE_DIR)
    logger.info("Reading from stdin (press CTRL+D to close or CTRL+C to interrupt)")
    
    # Wrap the FastMCP run to ensure proper signal handling
    async def run_with_signal_handling():
        """Run server with proper SIGINT/SIGTERM handling."""
        # Get the event loop
        loop = asyncio.get_running_loop()
        
        # Create a future to signal shutdown
        shutdown_future = loop.create_future()
        
        def handle_signal():
            """Handle shutdown signals."""
            if not shutdown_future.done():
                logger.info("Shutdown signal received, stopping server...")
                shutdown_future.set_result(None)
        
        # Register signal handlers for clean shutdown
        loop.add_signal_handler(signal.SIGINT, handle_signal)
        loop.add_signal_handler(signal.SIGTERM, handle_signal)
        
        # Create task for the MCP server
        server_task = asyncio.create_task(mcp.run_stdio_async())
        
        try:
            # Wait for either the server to finish or shutdown signal
            done, pending = await asyncio.wait(
                [server_task, shutdown_future],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # If shutdown was signaled, cancel the server task
            if shutdown_future in done:
                server_task.cancel()
                try:
                    await server_task
                except asyncio.CancelledError:
                    pass
            
        finally:
            # Remove signal handlers
            loop.remove_signal_handler(signal.SIGINT)
            loop.remove_signal_handler(signal.SIGTERM)
    
    try:
        # Use asyncio.run which properly handles cancellation in Python 3.11+
        asyncio.run(run_with_signal_handling())
        logger.info("Server stopped")
    except KeyboardInterrupt:
        # Fallback for when KeyboardInterrupt is raised
        logger.info("Server stopped")
    except Exception as e:
        logger.exception("Server error: %s", e)
        sys.exit(1)


def run_query(args):
    """Run a search query from the command line."""
    from .config import get_config
    from .faiss_index import EmbeddingIndex, _ML_DEPS_OK
    from .embedder import get_document_by_id
    import sqlite3
    
    logger = logging.getLogger(__name__)
    
    if not _ML_DEPS_OK:
        logger.error("ML dependencies not installed.")
        logger.error("Install with: uv sync --extra ml")
        sys.exit(1)
    
    config = get_config()
    
    # Check if index exists
    if not (config.cache_dir / "embeddings" / "index.faiss").exists():
        logger.error("No search index found.")
        logger.error("Run 'devdocs-mcp reindex' to create the index first.")
        sys.exit(1)
    
    # Load index
    idx = EmbeddingIndex(config.embeddings_dir, config.metadata_db_path)
    idx.load_or_create_index()
    
    # Search
    results = idx.search(args.search_query, top_k=args.top_k * 3, min_score=args.min_score)
    
    logger = logging.getLogger(__name__)
    
    if not results:
        logger.info("No results found for: %s", args.search_query)
        sys.exit(0)
    
    # Filter results if needed
    filtered_results = []
    with sqlite3.connect(str(config.metadata_db_path)) as db:
        for r in results:
            doc = get_document_by_id(db, r["doc_id"])
            if not doc:
                continue
            
            # Apply filters
            if args.slugs and doc.get("slug") not in args.slugs:
                continue
            if args.source and doc.get("source_type") != args.source:
                continue
            
            filtered_results.append({**r, **doc})
            
            if len(filtered_results) >= args.top_k:
                break
    
    # Display results
    logger = logging.getLogger(__name__)
    logger.info("Found %d results for: %s\n", len(filtered_results), args.search_query)
    
    for i, r in enumerate(filtered_results, 1):
        logger.info("%d. [%s] (%s) - Score: %.3f", i, r['title'], r['slug'], r['score'])
        if r.get('path'):
            logger.info("   Path: %s", r['path'])
        preview = r.get('content', '')[:200].replace('\n', ' ')
        logger.info("   %s...\n", preview)


def run_add(args):
    """Add documentation (download or local)."""
    from .config import get_config
    from .catalog import fetch_devdocs_catalog, find_doc_by_slug, index_local_directory
    from .download import download_doc as download_doc_impl
    from .mcp_server import add_local_source
    from .config import LocalSource
    
    logger = logging.getLogger(__name__)
    config = get_config()
    
    if not args.add_type:
        logger.error("Specify what to add: 'download' or 'local'")
        sys.exit(1)
    
    if args.add_type == "download":
        catalog = fetch_devdocs_catalog(cache_dir=config.cache_dir)
        
        for slug in args.slugs:
            doc_entry = find_doc_by_slug(catalog, slug)
            if not doc_entry:
                logger.error("Documentation '%s' not found in catalog", slug)
                continue
            
            try:
                logger.info("Downloading %s...", slug)
                download_doc_impl(slug, config.docs_dir)
                config.downloaded_slugs.add(slug)
                logger.info("✓ Downloaded %s (%.1f MB)", slug, doc_entry.size_mb)
            except Exception as e:
                logger.error("✗ Failed to download %s: %s", slug, e)
        
        config.save()
        logger.info("Downloaded %d docs.", len(args.slugs))
        logger.info("Run 'devdocs-mcp reindex' to make them searchable.")
        
    elif args.add_type == "local":
        from pathlib import Path
        
        path = Path(args.path).expanduser().resolve()
        
        if not path.exists() or not path.is_dir():
            logger.error("Path '%s' does not exist or is not a directory", path)
            sys.exit(1)
        
        html_files = list(path.glob("*.html")) + list(path.glob("*.htm"))
        if not html_files:
            logger.error("No HTML files found in '%s'", path)
            sys.exit(1)
        
        source = LocalSource(path=str(path), slug_prefix=args.prefix)
        config.sources.append(source)
        config.save()
        
        logger.info("✓ Added local source: %s", path)
        logger.info("  Found %d HTML files", len(html_files))
        logger.info("Run 'devdocs-mcp reindex' to index them.")


def run_reindex(args):
    """Rebuild the search index."""
    from .config import get_config
    from .embedder import init_metadata_db, upsert_documents, extract_text_from_db
    from .download import get_doc_pages, get_doc_index, list_downloaded_docs
    from .faiss_index import EmbeddingIndex, _ML_DEPS_OK
    from .config import LocalSource
    from pathlib import Path
    
    logger = logging.getLogger(__name__)
    
    if not _ML_DEPS_OK:
        logger.error("ML dependencies not installed.")
        logger.error("Install with: uv sync --extra ml")
        sys.exit(1)
    
    config = get_config()
    
    # Clean mode: drop and recreate database
    if args.clean:
        logger.info("Cleaning existing database...")
        if config.metadata_db_path.exists():
            config.metadata_db_path.unlink()
            logger.info("✓ Dropped old database")
        if config.faiss_index_path.exists():
            config.faiss_index_path.unlink()
            logger.info("✓ Dropped old FAISS index")
        logger.info("")
    
    # Initialize database
    logger.info("Initializing metadata database...")
    db = init_metadata_db(config.metadata_db_path)
    logger.info("✓ Database ready")
    logger.info("")
    
    # Initialize FAISS index
    logger.info("Loading embedding model...")
    idx = EmbeddingIndex(config.embeddings_dir, config.metadata_db_path)
    idx.load_or_create_index()
    logger.info("✓ Index ready")
    logger.info("")
    
    # Process downloaded docs
    total_docs = 0
    total_embeddings = 0
    downloaded = list_downloaded_docs(config.docs_dir)
    
    # If specific slugs requested, filter to only those
    if args.slugs:
        # Validate that requested slugs are downloaded
        not_downloaded = [s for s in args.slugs if s not in downloaded]
        if not_downloaded:
            logger.error("The following docs are not downloaded: %s", ', '.join(not_downloaded))
            logger.error("Download them first with: devdocs-mcp add download <slug>")
            logger.error("")
            logger.info("Available docs: %s", ', '.join(sorted(downloaded)))
            sys.exit(1)
        
        slugs_to_index = args.slugs
        logger.info("Re-indexing specific docs: %s", ', '.join(slugs_to_index))
        logger.info("")
        
        # Delete existing entries for these slugs
        from .embedder import get_documents_by_slug
        for slug in slugs_to_index:
            existing_docs = get_documents_by_slug(db, slug)
            if existing_docs:
                doc_ids_to_delete = [doc['id'] for doc in existing_docs]
                logger.info("Removing %d existing documents for %s", len(doc_ids_to_delete), slug)
                
                # Remove from FAISS index and database
                idx.remove_documents(doc_ids_to_delete)
        
        if any(get_documents_by_slug(db, s) for s in slugs_to_index):
            logger.info("")
    else:
        # Get already-indexed slugs (unless --clean was used)
        from .embedder import get_documents_by_slug
        indexed_slugs = set()
        if not args.clean:
            for slug in downloaded:
                existing_docs = get_documents_by_slug(db, slug)
                if existing_docs:
                    indexed_slugs.add(slug)
        
        # Filter to only missing slugs (unless --clean)
        slugs_to_index = [s for s in downloaded if s not in indexed_slugs]
        
        if not slugs_to_index:
            logger.info("All downloaded documentation is already indexed.")
            logger.info("Use --clean to rebuild from scratch.")
            logger.info("Use --slugs to re-index specific docs.")
            logger.info("")
            return
        
        if indexed_slugs:
            logger.info("Skipping %d already-indexed docs (use --clean to rebuild all)", len(indexed_slugs))
    
    logger.info("Processing %d documentation bundle(s)...", len(slugs_to_index))
    logger.info("")
    
    for i, slug in enumerate(slugs_to_index, 1):
        logger.info("[%d/%d] %s...", i, len(slugs_to_index), slug)
        
        db_pages = get_doc_pages(slug, config.docs_dir)
        if not db_pages:
            logger.info("(no pages)")
            continue
        
        # Get index.json for entry-level extraction
        index_data = get_doc_index(slug, config.docs_dir)
        
        docs = extract_text_from_db(db_pages, slug, index_json=index_data)
        if not docs:
            logger.info("(no content)")
            continue
        
        # Insert into database first (to get auto-generated IDs)
        count = upsert_documents(db, docs)
        total_docs += count
        
        # Re-read documents to get the auto-generated IDs
        stored_docs = get_documents_by_slug(db, slug)
        
        # Add to FAISS index using stored IDs
        if stored_docs:
            faiss_ids = [d["faiss_id"] for d in stored_docs]  # Use faiss_id column (INTEGER id)
            texts = [d["content"] for d in stored_docs]
            
            # Directly encode and add to FAISS
            import numpy as np
            if texts:
                embeddings = idx.model.encode(texts, normalize_embeddings=True)
                idx.faiss_index.add_with_ids(
                    embeddings.astype("float32"),
                    np.array(faiss_ids, dtype=np.int64)
                )
                total_embeddings += len(texts)
                emb_count = len(texts)
            else:
                emb_count = 0
        else:
            emb_count = 0
        
        # Show entry count if available
        if index_data and 'entries' in index_data:
            entry_count = len(index_data['entries'])
            logger.info("(%d docs from %d entries, %d embeddings)", count, entry_count, emb_count)
        else:
            logger.info("(%d docs, %d embeddings)", count, emb_count)
    
    # Save index
    logger.info("")
    logger.info("Saving FAISS index...")
    idx.save_index()
    logger.info("✓ Index saved")
    logger.info("")
    
    logger.info("=" * 60)
    if args.slugs:
        logger.info("Re-index Complete for: %s", ', '.join(args.slugs))
    else:
        logger.info("Reindex Complete!")
    logger.info("=" * 60)
    logger.info("Documents indexed: %d", total_docs)
    logger.info("Embeddings created: %d", total_embeddings)
    if args.slugs:
        logger.info("Docs re-indexed: %d", len(slugs_to_index))
    else:
        logger.info("Total docs in index: %d", len(downloaded))
        logger.info("New docs added: %d", len(slugs_to_index))
    logger.info("")


def run_list(args):
    """List available or downloaded documentation."""
    from .config import get_config
    from .catalog import get_merged_catalog
    from .download import list_downloaded_docs
    from .embedder import get_documents_by_slug
    import sqlite3
    
    logger = logging.getLogger(__name__)
    config = get_config()
    
    # Check index status
    index_exists = (config.cache_dir / "embeddings" / "index.faiss").exists()
    indexed_slugs = set()
    
    if index_exists:
        try:
            with sqlite3.connect(str(config.metadata_db_path)) as db:
                catalog = get_merged_catalog(config)
                for entry in catalog:
                    if entry.slug in config.downloaded_slugs:
                        docs = get_documents_by_slug(db, entry.slug)
                        if docs:
                            indexed_slugs.add(entry.slug)
        except Exception as e:
            logger.warning("Failed to check index status: %s", e)
    
    if args.downloaded:
        # List only downloaded docs
        downloaded = list_downloaded_docs(config.docs_dir)
        catalog = get_merged_catalog(config)
        
        # Apply query filter if provided
        if args.query:
            query_lower = args.query.lower()
            filtered = []
            for slug in sorted(downloaded.keys()):
                entry = next((e for e in catalog if e.slug == slug), None)
                if entry:
                    searchable = [
                        entry.slug,
                        entry.name,
                        entry.type,
                        entry.release or "",
                        entry.alias or "",
                    ]
                    if any(query_lower in field.lower() for field in searchable):
                        filtered.append((slug, entry))
                else:
                    # No catalog entry, just check slug
                    if query_lower in slug.lower():
                        filtered.append((slug, None))
        else:
            filtered = [(slug, next((e for e in catalog if e.slug == slug), None)) 
                       for slug in sorted(downloaded.keys())]
        
        # Build header
        header_parts = [f"Downloaded documentation ({len(filtered)})"]
        if not index_exists:
            header_parts.append("[INDEX NOT BUILT]")
        elif indexed_slugs:
            indexed_count = len([s for s, _ in filtered if s in indexed_slugs])
            if indexed_count < len(filtered):
                header_parts.append(f"[{indexed_count}/{len(filtered)} indexed]")
        if args.query:
            header_parts.append(f"(matching '{args.query}')")
        
        logger.info(" ".join(header_parts) + ":\n")
        
        for slug, entry in filtered:
            size_str = f" ({entry.size_mb:.1f} MB)" if entry else ""
            indexed_marker = " [indexed]" if slug in indexed_slugs else " [NOT indexed]" if index_exists else ""
            logger.info("  %s%s%s", slug, indexed_marker, size_str)
        
        # Show helpful tips
        if not index_exists and filtered:
            logger.info("\nRun 'devdocs-mcp reindex' to create the search index.")
        elif indexed_slugs and any(s not in indexed_slugs for s, _ in filtered):
            unindexed = [s for s, _ in filtered if s not in indexed_slugs]
            logger.info("\nUnindexed docs: %s", ', '.join(unindexed))
            logger.info("Run 'devdocs-mcp reindex' to index them.")
    else:
        # List all available docs
        catalog = get_merged_catalog(config)
        
        if not args.large:
            catalog = [e for e in catalog if not e.is_large]
        
        # Apply query filter
        if args.query:
            query_lower = args.query.lower()
            catalog = [
                e for e in catalog
                if any(query_lower in field.lower() for field in [
                    e.slug, e.name, e.type, e.release or "", e.alias or ""
                ])
            ]
        
        # Build header
        header_parts = [f"Available documentation ({len(catalog)})"]
        if not index_exists:
            header_parts.append("[INDEX NOT BUILT]")
        elif indexed_slugs:
            downloaded_count = len([e for e in catalog if e.slug in config.downloaded_slugs])
            indexed_count = len([e for e in catalog if e.slug in indexed_slugs])
            if downloaded_count > 0 and indexed_count < downloaded_count:
                header_parts.append(f"[{indexed_count}/{downloaded_count} indexed]")
        if args.query:
            header_parts.append(f"(matching '{args.query}')")
        
        logger.info(" ".join(header_parts) + ":\n")
        
        for entry in catalog:
            downloaded = "✓" if entry.slug in config.downloaded_slugs else " "
            indexed_marker = ""
            if entry.slug in config.downloaded_slugs:
                if entry.slug in indexed_slugs:
                    indexed_marker = " [indexed]"
                elif index_exists:
                    indexed_marker = " [NOT indexed]"
            
            logger.info("  [%s] %s%s - %s (%.1f MB)", 
                       downloaded, entry.slug, indexed_marker, entry.name, entry.size_mb)
        
        # Show helpful tips
        downloaded_entries = [e for e in catalog if e.slug in config.downloaded_slugs]
        if not index_exists and downloaded_entries:
            logger.info("\nRun 'devdocs-mcp reindex' to create the search index.")
        elif indexed_slugs and any(e.slug not in indexed_slugs for e in downloaded_entries):
            unindexed = [e.slug for e in downloaded_entries if e.slug not in indexed_slugs]
            logger.info("\nUnindexed docs: %s", ', '.join(unindexed))
            logger.info("Run 'devdocs-mcp reindex' to index them.")


if __name__ == "__main__":
    main()
