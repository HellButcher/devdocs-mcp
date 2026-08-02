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
    
    # Add -> web
    web_parser = add_subparsers.add_parser(
        "web",
        help="Fetch documentation from a web URL (or re-download by slug)",
    )
    web_parser.add_argument(
        "slug",
        help="Unique identifier for this web source",
    )
    web_parser.add_argument(
        "url",
        nargs="?",
        help="Base URL to fetch from (optional for re-download)",
    )
    web_parser.add_argument(
        "--name",
        help="Display name (defaults to slug)",
    )
    web_parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Recursion depth for crawling (default: 2)",
    )
    web_parser.add_argument(
        "--pattern",
        default=r".*\.html?$",
        help=r"Regex pattern for URLs to fetch (default: .*\.html?$)",
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
    
    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show detailed information about a specific documentation",
    )
    info_parser.add_argument(
        "slug",
        help="Documentation slug (e.g., javascript, python)",
    )
    
    # Get command
    get_parser = subparsers.add_parser(
        "get",
        help="Get full document content by document ID",
    )
    get_parser.add_argument(
        "doc_id",
        help="Document ID from search results",
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
        elif args.command == "info":
            run_info(args)
        elif args.command == "get":
            run_get(args)
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
    
    # Wrap the MCPServer run to ensure proper signal handling
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
    from .faiss_index import _ML_DEPS_OK
    from .operations import search_docs_impl
    
    logger = logging.getLogger(__name__)
    
    if not _ML_DEPS_OK:
        logger.error("ML dependencies not installed.")
        logger.error("Install with: uv sync --extra ml")
        sys.exit(1)
    
    config = get_config()
    
    # Use shared implementation
    result = search_docs_impl(
        config,
        query=args.search_query,
        top_k=args.top_k,
        min_score=args.min_score,
        slugs=args.slugs if args.slugs else None,
        source_type=args.source if hasattr(args, 'source') else None,
    )
    
    # Format for CLI (nice user-friendly output)
    if not result.success:
        if "No embeddings found" in (result.error or ""):
            logger.error("No search index found.")
            logger.error("Run 'devdocs-mcp reindex' to create the index first.")
        else:
            logger.error(result.error or "Search failed.")
        sys.exit(1)
    
    if not result.results:
        logger.info("No results found for: %s", args.search_query)
        sys.exit(0)
    
    # Display results
    logger.info("Found %d results for: %s\n", len(result.results), args.search_query)
    
    for i, r in enumerate(result.results, 1):
        logger.info("%d. [%s] (%s) - Score: %.3f", i, r['title'], r['slug'], r['score'])
        if r.get('id'):
            logger.info("   Doc ID: %s", r['id'])
        if r.get('path'):
            logger.info("   Path: %s", r['path'])
        preview = r.get('content', '')[:200].replace('\n', ' ')
        logger.info("   %s...", preview)
        logger.info("   → Use: devdocs-mcp get \"%s\"\n", r.get('id', ''))


def run_add(args):
    """Add documentation (download, local, or web)."""
    from .config import get_config
    from .operations import download_docs_impl, add_local_source_impl, add_web_source_impl
    
    logger = logging.getLogger(__name__)
    config = get_config()
    
    if not args.add_type:
        logger.error("Specify what to add: 'download', 'local', or 'web'")
        sys.exit(1)
    
    if args.add_type == "download":
        # Use shared implementation
        result = download_docs_impl(config, args.slugs)
        
        # Format for CLI
        for slug in result.successful_slugs:
            logger.info("✓ Downloaded %s", slug)
        
        for slug in result.failed_slugs:
            logger.error("✗ Failed to download %s: %s", slug, result.errors.get(slug, "Unknown error"))
        
        if result.successful_slugs:
            logger.info("")
            total_size_mb = result.metadata.get("total_size_mb", 0.0)
            logger.info("Downloaded %d doc(s) (%.1f MB total)", len(result.successful_slugs), total_size_mb)
            logger.info("Run 'devdocs-mcp reindex' to make them searchable.")
        
        if result.failed_slugs and not result.successful_slugs:
            sys.exit(1)
        
    elif args.add_type == "local":
        # Use shared implementation
        result = add_local_source_impl(config, args.path, args.prefix or "")
        
        # Format for CLI
        if not result.success:
            error_msg = list(result.errors.values())[0] if result.errors else "Failed to add local source"
            logger.error(error_msg)
            sys.exit(1)
        
        path = result.metadata.get("path", args.path)
        num_files = result.metadata.get("num_files", len(result.successful_slugs))
        logger.info("✓ Added local source: %s", path)
        logger.info("  Found %d HTML files", num_files)
        logger.info("Run 'devdocs-mcp reindex' to index them.")
    
    elif args.add_type == "web":
        # Use shared implementation
        result = add_web_source_impl(
            config,
            args.url,  # May be None for re-download
            args.slug,
            name=args.name,
            max_depth=args.max_depth,
            pattern=args.pattern,
        )
        
        # Format for CLI
        if not result.success:
            error_msg = list(result.errors.values())[0] if result.errors else "Failed to fetch web source"
            logger.error(error_msg)
            sys.exit(1)
        
        url = result.metadata.get("url", args.url or "existing source")
        num_files = result.metadata.get("num_files", len(result.successful_slugs))
        
        if args.url:
            logger.info("✓ Fetched web source: %s", url)
        else:
            logger.info("✓ Re-downloaded web source: %s", url)
        
        logger.info("  Downloaded %d HTML files", num_files)
        logger.info("Run 'devdocs-mcp reindex' to index them.")


def run_reindex(args):
    """Rebuild the search index."""
    from .config import get_config
    from .faiss_index import _ML_DEPS_OK
    from .operations import rebuild_index_impl
    import sys
    
    logger = logging.getLogger(__name__)
    
    if not _ML_DEPS_OK:
        logger.error("ML dependencies not installed.")
        logger.error("Install with: uv sync --extra ml")
        sys.exit(1)
    
    config = get_config()
    
    # Progress tracking state
    progress_state = {
        "current_slug": None,
        "current": 0,
        "total": 0,
        "last_percent": -1,
    }
    
    def progress_callback(current: int, total: int, slug: str):
        """Display progress bar for document extraction."""
        # Update state
        if progress_state["current_slug"] != slug:
            progress_state["current_slug"] = slug
            progress_state["current"] = 0
            progress_state["total"] = total
            progress_state["last_percent"] = -1
        
        progress_state["current"] = current
        progress_state["total"] = total
        
        # Calculate percentage
        percent = int((current / total) * 100) if total > 0 else 0
        
        # Only update if percentage changed (avoid spamming)
        if percent != progress_state["last_percent"]:
            progress_state["last_percent"] = percent
            
            # Draw progress bar
            bar_width = 40
            filled = int(bar_width * current / total) if total > 0 else 0
            bar = "=" * filled + "-" * (bar_width - filled)
            
            # Print with carriage return to overwrite
            sys.stderr.write(f"\r  Extracting {slug}: [{bar}] {current}/{total} ({percent}%)")
            sys.stderr.flush()
            
            # Newline when complete
            if current >= total:
                sys.stderr.write("\n")
                sys.stderr.flush()
    
    # Use shared implementation with progress callback
    result = rebuild_index_impl(
        config,
        clean=args.clean,
        slugs=args.slugs if args.slugs else None,
        progress_callback=progress_callback,
    )
    
    # Format result for CLI (nice user-friendly output)
    if not result.success:
        if result.error and "already indexed" in result.error:
            logger.info(result.error)
            logger.info("Use --clean to rebuild from scratch.")
            logger.info("Use --slugs to re-index specific docs.")
            logger.info("")
            return
        else:
            logger.error(result.error or "Index operation failed.")
            if "not available" in (result.error or ""):
                logger.error("Download them first with: devdocs-mcp add download <slug>")
                logger.error("")
                logger.info("Total available docs: %d", result.total_available)
            sys.exit(1)
    
    # Show success output
    logger.info("✓ Index saved")
    logger.info("")
    
    logger.info("=" * 60)
    if result.mode == "specific":
        logger.info("Re-index Complete for: %s", ', '.join(result.slugs_processed))
    else:
        logger.info("Reindex Complete!")
    logger.info("=" * 60)
    logger.info("Documents indexed: %d", result.total_docs)
    logger.info("Embeddings created: %d", result.total_embeddings)
    if result.mode == "specific":
        logger.info("Docs re-indexed: %d", len(result.slugs_processed))
    else:
        logger.info("Total available docs: %d (%d devdocs, %d local)", 
                   result.total_available, result.devdocs_count, result.local_count)
        logger.info("New docs added: %d", result.new_docs_added)
    logger.info("")


def run_list(args):
    """List available or downloaded documentation."""
    from .config import get_config
    from .operations import list_docs_impl
    
    logger = logging.getLogger(__name__)
    config = get_config()
    
    if args.downloaded:
        # List only downloaded/available docs (CLI default behavior different from MCP)
        result = list_docs_impl(
            config,
            source_type=None,
            include_large=args.large,
            downloaded_only=True,
            query=args.query if args.query else None,
        )
        
        # Build CLI-specific output
        header_parts = [f"Downloaded documentation ({result.total_count})"]
        if not result.index_exists:
            header_parts.append("[INDEX NOT BUILT]")
        elif result.indexed_slugs:
            indexed_count = len([e for e, _ in result.entries if e.slug in result.indexed_slugs])
            if indexed_count < result.total_count:
                header_parts.append(f"[{indexed_count}/{result.total_count} indexed]")
        if args.query:
            header_parts.append(f"(matching '{args.query}')")
        
        logger.info(" ".join(header_parts) + ":\n")
        
        for entry, _ in result.entries:
            indexed_marker = " [indexed]" if entry.slug in result.indexed_slugs else " [NOT indexed]" if result.index_exists else ""
            logger.info("  %s%s (%.1f MB)", entry.slug, indexed_marker, entry.size_mb)
        
        # Show helpful tips
        if not result.index_exists and result.entries:
            logger.info("\nRun 'devdocs-mcp reindex' to create the search index.")
        elif result.indexed_slugs and any(e.slug not in result.indexed_slugs for e, _ in result.entries):
            unindexed = [e.slug for e, _ in result.entries if e.slug not in result.indexed_slugs]
            logger.info("\nUnindexed docs: %s", ', '.join(unindexed))
            logger.info("Run 'devdocs-mcp reindex' to index them.")
    else:
        # List all available docs (CLI shows all by default, MCP shows only downloaded by default)
        result = list_docs_impl(
            config,
            source_type=None,
            include_large=args.large,
            downloaded_only=False,
            query=args.query if args.query else None,
        )
        
        # Build CLI-specific output
        header_parts = [f"Available documentation ({result.total_count})"]
        if not result.index_exists:
            header_parts.append("[INDEX NOT BUILT]")
        elif result.indexed_slugs:
            available_count = len([e for e, d in result.entries if d])
            indexed_count = len([e for e, _ in result.entries if e.slug in result.indexed_slugs])
            if available_count > 0 and indexed_count < available_count:
                header_parts.append(f"[{indexed_count}/{available_count} indexed]")
        if args.query:
            header_parts.append(f"(matching '{args.query}')")
        
        logger.info(" ".join(header_parts) + ":\n")
        
        for entry, is_available in result.entries:
            downloaded = "✓" if is_available else " "
            indexed_marker = ""
            if is_available:
                if entry.slug in result.indexed_slugs:
                    indexed_marker = " [indexed]"
                elif result.index_exists:
                    indexed_marker = " [NOT indexed]"
            
            logger.info("  [%s] %s%s - %s (%.1f MB)", 
                       downloaded, entry.slug, indexed_marker, entry.name, entry.size_mb)
        
        # Show helpful tips
        available_entries = [e for e, d in result.entries if d]
        if not result.index_exists and available_entries:
            logger.info("\nRun 'devdocs-mcp reindex' to create the search index.")
        elif result.indexed_slugs and any(e.slug not in result.indexed_slugs for e in available_entries):
            unindexed = [e.slug for e in available_entries if e.slug not in result.indexed_slugs]
            logger.info("\nUnindexed docs: %s", ', '.join(unindexed))
            logger.info("Run 'devdocs-mcp reindex' to index them.")


def run_info(args):
    """Show detailed information about a specific documentation."""
    from .config import get_config
    from .operations import doc_info_impl
    
    logger = logging.getLogger(__name__)
    config = get_config()
    
    # Use shared implementation
    result = doc_info_impl(config, args.slug)
    
    # Format for CLI (nice user-friendly output)
    if not result.success:
        logger.error(result.error or f"Documentation '{args.slug}' not found.")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("Documentation Info: %s", result.name)
    logger.info("=" * 60)
    logger.info("Slug:       %s%s", args.slug, f"~{result.version}" if result.version else "")
    logger.info("Type:       %s", result.type)
    logger.info("Size:       %.1f MB", result.size_mb)
    logger.info("Release:    %s", result.release)
    logger.info("Downloaded: %s", "Yes ✓" if result.downloaded else "No")
    
    if result.indexed:
        logger.info("Indexed:    Yes ✓")
    elif result.downloaded:
        logger.info("Indexed:    No (run 'devdocs-mcp reindex' to index)")
    
    if result.home_url:
        logger.info("Homepage:   %s", result.home_url)
    if result.code_url:
        logger.info("Repository: %s", result.code_url)
    
    if result.page_count is not None:
        logger.info("")
        logger.info("Pages:      %d", result.page_count)
    if result.content_size_kb is not None:
        logger.info("Content:    %.1f KB", result.content_size_kb)
    
    logger.info("")


def run_get(args):
    """Get full document content by document ID."""
    from .config import get_config
    from .operations import get_document_impl
    
    logger = logging.getLogger(__name__)
    config = get_config()
    
    # Use shared implementation
    result = get_document_impl(config, args.doc_id)
    
    # Format for CLI
    if not result.success:
        logger.error(result.error or f"Failed to get document '{args.doc_id}'.")
        sys.exit(1)
    
    # Print document
    print("=" * 80)
    if result.title:
        print(f"# {result.title}")
        print()
    
    # Metadata
    if result.slug:
        print(f"**Slug:** {result.slug}")
    if result.type:
        print(f"**Type:** {result.type}")
    if result.source_type:
        print(f"**Source:** {result.source_type}")
    if result.path:
        print(f"**Path:** {result.path}")
    if result.doc_id:
        print(f"**Doc ID:** {result.doc_id}")
    
    print()
    print("-" * 80)
    print()
    
    # Content
    if result.content:
        print(result.content)
    else:
        print("(No content available)")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
