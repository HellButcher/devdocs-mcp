"""Source abstraction layer for different documentation sources.

This module provides a unified interface for handling different documentation
sources (devdocs.io, local directories, git repos, web-fetched docs, etc.).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import logging

from devdocs_mcp.config import Config, LocalSource, WebSource
from devdocs_mcp.embedder import extract_text_from_local, scan_local_directory

logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Types of documentation sources."""
    DEVDOCS = "devdocs"
    LOCAL = "local"
    WEB = "web"


@dataclass
class SourceOperationResult:
    """Unified result for source operations (add/download)."""
    success: bool
    source_type: SourceType
    slugs: list[str]  # Slugs that were added/downloaded
    errors: dict[str, str]  # slug -> error message
    metadata: dict[str, Any]  # source-specific metadata
    
    @property
    def failed_slugs(self) -> list[str]:
        """List of slugs that failed."""
        return list(self.errors.keys())
    
    @property
    def successful_slugs(self) -> list[str]:
        """List of slugs that succeeded."""
        return [s for s in self.slugs if s not in self.errors]


@dataclass
class DocumentInfo:
    """Information about a documentation source."""
    slug: str
    name: str
    version: str
    type: str
    source_type: SourceType
    size_mb: float
    release: str
    downloaded: bool
    indexed: bool
    home_url: str | None = None
    code_url: str | None = None
    page_count: int | None = None
    content_size_kb: float | None = None


class SourceHandler(ABC):
    """Abstract base class for documentation source handlers."""
    
    @abstractmethod
    def get_source_type(self) -> SourceType:
        """Return the source type this handler manages."""
        pass
    
    @abstractmethod
    def add_source(self, config: Any, **kwargs) -> SourceOperationResult:
        """Add/download documentation from this source.
        
        Args:
            config: Configuration object
            **kwargs: Source-specific parameters
            
        Returns:
            SourceOperationResult with operation details
        """
        pass
    
    @abstractmethod
    def extract_documents(self, config: Any, slug: str, progress_callback: Optional[Callable] = None) -> list[Any]:
        """Extract SearchDocument objects from this source.
        
        Args:
            config: Configuration object
            slug: Documentation slug
            progress_callback: Optional callback(current, total, slug) for progress reporting
            
        Returns:
            List of SearchDocument objects ready for indexing
        """
        pass
    
    @abstractmethod
    def get_info(self, config: Any, slug: str) -> DocumentInfo | None:
        """Get detailed information about a documentation.
        
        Args:
            config: Configuration object
            slug: Documentation slug
            
        Returns:
            DocumentInfo or None if not found
        """
        pass
    
    @abstractmethod
    def is_available(self, config: Any, slug: str) -> bool:
        """Check if a documentation is available/downloaded.
        
        Args:
            config: Configuration object
            slug: Documentation slug
            
        Returns:
            True if available, False otherwise
        """
        pass


class DevDocsSourceHandler(SourceHandler):
    """Handler for devdocs.io documentation sources."""
    
    def get_source_type(self) -> SourceType:
        return SourceType.DEVDOCS
    
    def add_source(self, config: Any, **kwargs) -> SourceOperationResult:
        """Download documentation from devdocs.io.
        
        Expected kwargs:
            slugs: list[str] - slugs to download
        """
        from .download import download_doc, get_doc_pages
        
        slugs = kwargs.get("slugs", [])
        if not slugs:
            return SourceOperationResult(
                success=False,
                source_type=SourceType.DEVDOCS,
                slugs=[],
                errors={"": "No slugs provided"},
                metadata={},
            )
        
        downloaded = []
        errors = {}
        total_size_mb = 0.0
        
        for slug in slugs:
            try:
                size_mb = download_doc(slug, config.docs_dir)
                downloaded.append(slug)
                total_size_mb += size_mb
                logger.info(f"Downloaded {slug} ({size_mb:.1f} MB)")
            except Exception as e:
                errors[slug] = str(e)
                logger.error(f"Failed to download {slug}: {e}")
        
        # Update config
        if downloaded:
            config.downloaded_slugs.update(downloaded)
            config.save()
        
        return SourceOperationResult(
            success=len(downloaded) > 0,
            source_type=SourceType.DEVDOCS,
            slugs=downloaded,
            errors=errors,
            metadata={"total_size_mb": total_size_mb},
        )
    
    def extract_documents(self, config: Any, slug: str, progress_callback: Optional[Callable] = None) -> list[Any]:
        """Extract documents from devdocs.io source."""
        from .download import get_doc_index, get_doc_pages
        from .embedder import extract_text_from_db
        
        doc_index = get_doc_index(slug, config.docs_dir)
        db_pages = get_doc_pages(slug, config.docs_dir)
        
        if not doc_index or not db_pages:
            return []
        
        # Use extract_text_from_db which handles chunking and title extraction
        documents = extract_text_from_db(db_pages, slug, index_json=doc_index, progress_callback=progress_callback)
        
        return documents
    
    def get_info(self, config: Any, slug: str) -> DocumentInfo | None:
        """Get info about a devdocs.io documentation."""
        from .catalog import fetch_devdocs_catalog, find_doc_by_slug
        from .download import get_doc_pages
        from .embedder import get_documents_by_slug
        
        catalog = fetch_devdocs_catalog(cache_dir=config.cache_dir)
        doc_entry = find_doc_by_slug(catalog, slug)
        
        if not doc_entry:
            return None
        
        downloaded = slug in config.downloaded_slugs
        indexed = False
        page_count = None
        content_size_kb = None
        
        if downloaded:
            # Check if indexed
            try:
                import sqlite3
                with sqlite3.connect(str(config.metadata_db_path)) as db:
                    indexed_docs = get_documents_by_slug(db, slug)
                    indexed = len(indexed_docs) > 0
            except:
                indexed = False
            
            # Get page count and size
            db_pages = get_doc_pages(slug, config.docs_dir)
            if db_pages:
                page_count = len(db_pages)
                total_size = sum(len(str(v)) for v in db_pages.values())
                content_size_kb = total_size / 1024
        
        return DocumentInfo(
            slug=slug,
            name=doc_entry.name,
            version=doc_entry.version or "",
            type=doc_entry.type,
            source_type=SourceType.DEVDOCS,
            size_mb=doc_entry.size_mb,
            release=doc_entry.release or "N/A",
            downloaded=downloaded,
            indexed=indexed,
            home_url=doc_entry.home_url,
            code_url=doc_entry.code_url,
            page_count=page_count,
            content_size_kb=content_size_kb,
        )
    
    def is_available(self, config: Any, slug: str) -> bool:
        """Check if a devdocs.io doc is downloaded."""
        return slug in config.downloaded_slugs


class LocalSourceHandler(SourceHandler):
    """Handler for local HTML directory sources."""
    
    def get_source_type(self) -> SourceType:
        return SourceType.LOCAL
    
    def add_source(self, config: Any, **kwargs) -> SourceOperationResult:
        """Add a local HTML directory as a source.
        
        Expected kwargs:
            path: str - path to local HTML directory
            slug: str - unique identifier for this local source
            name: str - display name (optional, defaults to slug)
        """
        from .config import LocalSource
        
        path = kwargs.get("path")
        slug = kwargs.get("slug")
        
        if not path:
            return SourceOperationResult(
                success=False,
                source_type=SourceType.LOCAL,
                slugs=[],
                errors={"": "No path provided"},
                metadata={},
            )
        abs_path = Path(path).resolve()
        if not abs_path.exists():
            return SourceOperationResult(
                success=False,
                source_type=SourceType.LOCAL,
                slugs=[],
                errors={str(abs_path): "Path does not exist"},
                metadata={},
            )

        if not slug:
            # fallback to basename (without file extension)
            slug = abs_path.stem if abs_path.is_file() else abs_path.name
            if not slug:
                return SourceOperationResult(
                    success=False,
                    source_type=SourceType.LOCAL,
                    slugs=[],
                    errors={str(abs_path): "Could not determine slug from path"},
                    metadata={},
                )

        name = kwargs.get("name", slug)
        
        # Support both directories and single files
        if abs_path.is_file():
            if abs_path.suffix.lower() not in ['.html', '.htm']:
                return SourceOperationResult(
                    success=False,
                    source_type=SourceType.LOCAL,
                    slugs=[],
                    errors={str(abs_path): "File must have .html or .htm extension"},
                    metadata={},
                )
        elif not abs_path.is_dir():
            return SourceOperationResult(
                success=False,
                source_type=SourceType.LOCAL,
                slugs=[],
                errors={str(abs_path): "Path is not a directory"},
                metadata={},
            )

        entries = scan_local_directory(abs_path)
        if not entries:
            return SourceOperationResult(
                success=False,
                source_type=SourceType.LOCAL,
                slugs=[],
                errors={str(abs_path): "No HTML files found"},
                metadata={},
            )
        
        # Remove existing local source with same path (deduplication)
        config.sources = [
            src for src in config.sources
            if not (isinstance(src, LocalSource) and Path(src.path).resolve() == abs_path)
        ]
        
        # Create LocalSource
        local_src = LocalSource(
            path=str(abs_path),
            slug=slug,
            name=name,
        )
        
        # Add to config
        config.sources.append(local_src)
        config.save()
        
        return SourceOperationResult(
            success=True,
            source_type=SourceType.LOCAL,
            slugs=[slug],
            errors={},
            metadata={
                "path": str(abs_path),
                "num_files": len(entries),
            },
        )
    
    def get_source_by_slug(self, config: Config, slug: str) -> LocalSource | None:
        """Get a LocalSource by slug."""
        for local_src in config.local_sources:
            if local_src.slug == slug:
                return local_src
        return None

    def extract_documents(self, config: Config, slug: str, progress_callback: Optional[Callable] = None) -> list[Any]:
        """Extract documents from local HTML source."""
        # Find the local source with this slug
        source = self.get_source_by_slug(config, slug)
        if not source:
            logger.warning(f"Could not find local source for slug {slug}")
            return []
        
        # Read the HTML file
        file_path = Path(source.path)
        if not file_path.exists():
            logger.warning(f"Local file not found: {file_path}")
            return []

        entries = scan_local_directory(file_path)
        documents = []
        total = len(entries)
        
        for i, entry in enumerate(entries):
            documents.extend(extract_text_from_local(entry, slug, "local"))
            if progress_callback and i % 10 == 0:  # Report every 10 files
                progress_callback(i + 1, total, slug)
        
        if progress_callback:
            progress_callback(total, total, slug)
        
        return documents
    
    def get_info(self, config: Any, slug: str) -> DocumentInfo | None:
        """Get info about a local documentation."""
        from .embedder import get_documents_by_slug
        
        # Find in local sources directly
        source = self.get_source_by_slug(config, slug)
        if not source:
            return None
        
        # Check if indexed
        indexed = False
        try:
            import sqlite3
            with sqlite3.connect(str(config.metadata_db_path)) as db:
                indexed_docs = get_documents_by_slug(db, slug)
                indexed = len(indexed_docs) > 0
        except:
            indexed = False
        
        # Get file path and size
        file_path = Path(source.path)

        total_size_bytes = 0
        if file_path.exists():
            entries = scan_local_directory(file_path)
            for entry in entries:
                total_size_bytes += entry.stat().st_size
        else:
            entries = []
        
        return DocumentInfo(
            slug=slug,
            name=source.name,
            version="",
            type="local",
            source_type=SourceType.LOCAL,
            size_mb=total_size_bytes / (1024 * 1024),
            release="N/A",
            downloaded=True,  # Local sources are always "available"
            indexed=indexed,
            home_url=None,
            code_url=None,
            page_count=len(entries),
            content_size_kb=total_size_bytes / 1024,
        )
    
    def is_available(self, config: Any, slug: str) -> bool:
        """Local sources are always available (no download needed)."""
        source = self.get_source_by_slug(config, slug)
        return source is not None and Path(source.path).exists()


class WebSourceHandler(SourceHandler):
    """Handler for web-fetched documentation sources."""
    
    def get_source_type(self) -> SourceType:
        return SourceType.WEB
    
    def add_source(self, config: Any, **kwargs) -> SourceOperationResult:
        r"""Fetch documentation from a URL using httpx.
        
        If url is not provided but slug is, re-downloads an existing web source.
        
        Expected kwargs:
            url: str - base URL to fetch (optional if re-downloading)
            slug: str - unique identifier for this web source
            name: str - display name (optional, defaults to slug)
            max_depth: int - recursion depth (default: 2)
            pattern: str - regex pattern to match HTML files (default: r".*\.html?$")
            url_prefix: str - optional URL prefix to restrict crawling (default: directory of initial URL)
        """
        import re
        from urllib.parse import urljoin, urlparse
        from .config import WebSource
        
        try:
            import httpx
        except ImportError:
            return SourceOperationResult(
                success=False,
                source_type=SourceType.WEB,
                slugs=[],
                errors={"": "httpx not installed. Run: pip install httpx"},
                metadata={},
            )
        
        url = kwargs.get("url")
        slug = kwargs.get("slug")
        name = kwargs.get("name")
        max_depth = kwargs.get("max_depth", 2)
        pattern = kwargs.get("pattern", r".*\.html?$")
        url_prefix = kwargs.get("url_prefix")  # Optional URL prefix restriction
        
        # If no URL provided, try to find existing web source and re-download
        if not url:
            if not slug:
                return SourceOperationResult(
                    success=False,
                    source_type=SourceType.WEB,
                    slugs=[],
                    errors={"": "Either 'url' or 'slug' (for re-download) is required"},
                    metadata={},
                )
            
            # Find existing web source
            existing_source = None
            for web_src in config.web_sources:
                if web_src.slug == slug:
                    existing_source = web_src
                    break
            
            if not existing_source:
                return SourceOperationResult(
                    success=False,
                    source_type=SourceType.WEB,
                    slugs=[],
                    errors={slug: f"Web source '{slug}' not found. Provide a URL to add a new source."},
                    metadata={},
                )
            
            # Use existing source's URL and settings
            url = existing_source.url
            name = name or existing_source.name
            logger.info(f"Re-downloading existing web source '{slug}' from {url}")
        
        # Validate required params
        if not slug:
            return SourceOperationResult(
                success=False,
                source_type=SourceType.WEB,
                slugs=[],
                errors={"": "'slug' is required"},
                metadata={},
            )
        
        # Default name to slug if not provided
        if not name:
            name = slug
        
        # Create cache directory for this web source
        cache_path = config.web_docs_dir / slug
        
        # Clear existing cache if re-downloading
        if cache_path.exists():
            import shutil
            logger.info(f"Clearing existing cache at {cache_path}")
            shutil.rmtree(cache_path)
        
        cache_path.mkdir(parents=True, exist_ok=True)
        
        # Determine URL prefix for restricting crawling
        # If not specified, default to the directory of the initial URL
        if not url_prefix:
            parsed_url = urlparse(url)
            # Get directory path (everything before the last /)
            path_parts = parsed_url.path.rsplit('/', 1)
            if len(path_parts) > 1:
                # URL points to a file, use its directory
                url_prefix = f"{parsed_url.scheme}://{parsed_url.netloc}{path_parts[0]}/"
            else:
                # URL points to a directory or root
                url_prefix = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
                if not url_prefix.endswith('/'):
                    url_prefix += '/'
        
        logger.info(f"URL prefix restriction: {url_prefix}")
        
        # Fetch pages recursively
        visited = set()
        to_visit = [(url, 0)]  # (url, depth)
        downloaded_files = []
        pattern_re = re.compile(pattern)
        base_domain = urlparse(url).netloc
        
        logger.info(f"Fetching web docs from {url} (max_depth={max_depth})")
        
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            while to_visit:
                current_url, depth = to_visit.pop(0)
                
                if current_url in visited or depth > max_depth:
                    continue
                
                visited.add(current_url)
                
                # Check if URL looks like an HTML file
                parsed_current = urlparse(current_url)
                path_lower = parsed_current.path.lower()
                is_html = (
                    path_lower.endswith(('.html', '.htm')) or
                    not any(path_lower.endswith(ext) for ext in [
                        '.css', '.js', '.json', '.xml', '.txt', '.pdf', 
                        '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico',
                        '.woff', '.woff2', '.ttf', '.eot', '.otf'
                    ])
                )
                
                # Skip non-HTML files entirely
                if not is_html:
                    continue
                
                # Only fetch if matches pattern (for saving) or we need to crawl for links
                should_fetch = pattern_re.search(current_url) or depth < max_depth
                if not should_fetch:
                    continue
                
                try:
                    logger.info(f"Fetching {current_url} (depth={depth})")
                    response = client.get(current_url)
                    response.raise_for_status()
                    
                    # Check Content-Type to ensure it's HTML
                    content_type = response.headers.get('content-type', '').lower()
                    if 'text/html' not in content_type:
                        logger.info(f"Skipping {current_url} - not HTML (content-type: {content_type})")
                        continue
                    
                    # Only save HTML files that match the pattern
                    if pattern_re.search(current_url):
                        # Generate filename from URL
                        parsed = urlparse(current_url)
                        filename = parsed.path.strip("/").replace("/", "_") or "index"
                        if not filename.endswith((".html", ".htm")):
                            filename += ".html"
                        
                        file_path = cache_path / filename
                        file_path.write_text(response.text, encoding="utf-8")
                        downloaded_files.append({
                            "slug": f"{slug}/{filename}",
                            "name": filename.replace("_", " ").replace(".html", "").title(),
                            "path": str(file_path),
                            "url": current_url,
                        })
                    
                    # Extract links for next level (only from HTML content)
                    if depth < max_depth:
                        content = response.text
                        # Simple link extraction
                        import re
                        links = re.findall(r'href=["\'](.*?)["\']', content, re.IGNORECASE)
                        
                        for link in links:
                            absolute_url = urljoin(current_url, link)
                            parsed_link = urlparse(absolute_url)
                            
                            # Only follow links on same domain AND under the URL prefix
                            if parsed_link.netloc == base_domain:
                                # Remove fragments and query parameters
                                clean_url = f"{parsed_link.scheme}://{parsed_link.netloc}{parsed_link.path}"
                                
                                # Check if URL looks like HTML
                                link_path_lower = parsed_link.path.lower()
                                link_is_html = (
                                    link_path_lower.endswith(('.html', '.htm')) or
                                    not any(link_path_lower.endswith(ext) for ext in [
                                        '.css', '.js', '.json', '.xml', '.txt', '.pdf',
                                        '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico',
                                        '.woff', '.woff2', '.ttf', '.eot', '.otf'
                                    ])
                                )
                                
                                # Check if URL is under the allowed prefix and looks like HTML
                                if (link_is_html and 
                                    clean_url.startswith(url_prefix) and 
                                    clean_url not in visited):
                                    to_visit.append((clean_url, depth + 1))
                
                except Exception as e:
                    logger.warning(f"Failed to fetch {current_url}: {e}")
                    continue
        
        if not downloaded_files:
            return SourceOperationResult(
                success=False,
                source_type=SourceType.WEB,
                slugs=[],
                errors={url: "No HTML files found at URL"},
                metadata={},
            )
        
        # Remove existing web source with same slug (deduplication)
        config.sources = [
            src for src in config.sources
            if not (isinstance(src, WebSource) and src.slug == slug)
        ]
        
        # Create WebSource with cached entries
        web_src = WebSource(
            url=url,
            slug=slug,
            name=name,
            cache_path=str(cache_path),
        )
        
        config.sources.append(web_src)
        config.save()
        
        logger.info(f"Downloaded {len(downloaded_files)} files from {url}")
        
        return SourceOperationResult(
            success=True,
            source_type=SourceType.WEB,
            slugs=[slug],
            errors={},
            metadata={
                "url": url,
                "num_files": len(downloaded_files),
                "cache_path": str(cache_path),
            },
        )

    def get_source_by_slug(self, config: Config, slug: str) -> WebSource | None:
        """Get a WebSource by slug."""
        for web_src in config.web_sources:
            if web_src.slug == slug:
                return web_src
        return None
    
    def extract_documents(self, config: Any, slug: str, progress_callback: Optional[Callable] = None) -> list[Any]:
        """Extract documents from web-cached HTML files."""
        # Find the web source
        source = self.get_source_by_slug(config, slug)
        if not source:
            logger.warning(f"Could not find web source for slug {slug}")
            return []
        
        cache_path = Path(source.cache_path)
        if not cache_path.exists():
            logger.warning(f"Cached file not found: {file_path}")
            return []
        entries = scan_local_directory(cache_path)
        documents = []
        total = len(entries)
        
        for i, entry in enumerate(entries):
            documents.extend(extract_text_from_local(entry, slug, "web"))
            if progress_callback and i % 10 == 0:  # Report every 10 files
                progress_callback(i + 1, total, slug)
        
        if progress_callback:
            progress_callback(total, total, slug)
        
        return documents
    
    def get_info(self, config: Any, slug: str) -> DocumentInfo | None:
        """Get info about a web documentation source."""
        from .embedder import get_documents_by_slug
        
        # Find in web sources
        source = self.get_source_by_slug(config, slug)
        if not source:
            return None
        
        # Check if indexed
        indexed = False
        try:
            import sqlite3
            with sqlite3.connect(str(config.metadata_db_path)) as db:
                indexed_docs = get_documents_by_slug(db, slug)
                indexed = len(indexed_docs) > 0
        except:
            indexed = False
        
        # Calculate total size
        cache_path = Path(source.cache_path)
        total_size_bytes = 0
        if cache_path.exists():
            entries = scan_local_directory(cache_path)
            for entry in entries:
                total_size_bytes += entry.stat().st_size
        else:
            entries = []
        
        return DocumentInfo(
            slug=slug,
            name=source.name,
            version="",
            type="web",
            source_type=SourceType.WEB,
            size_mb=total_size_bytes / (1024 * 1024),
            release="N/A",
            downloaded=True,  # Web sources are always "available" once cached
            indexed=indexed,
            home_url=source.url,
            code_url=None,
            page_count=len(entries),
            content_size_kb=total_size_bytes / 1024,
        )
    
    def is_available(self, config: Any, slug: str) -> bool:
        """Check if web docs are cached."""
        source = self.get_source_by_slug(config, slug)
        return source is not None and Path(source.cache_path).exists()


# Registry of source handlers
_HANDLERS: dict[SourceType, SourceHandler] = {
    SourceType.DEVDOCS: DevDocsSourceHandler(),
    SourceType.LOCAL: LocalSourceHandler(),
    SourceType.WEB: WebSourceHandler(),
}


def get_source_handler(source_type: SourceType | str) -> SourceHandler:
    """Get the appropriate source handler for a source type.
    
    Args:
        source_type: SourceType enum or string ("devdocs", "local")
        
    Returns:
        SourceHandler instance
        
    Raises:
        ValueError: If source type is not supported
    """
    if isinstance(source_type, str):
        try:
            source_type = SourceType(source_type)
        except ValueError:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    handler = _HANDLERS.get(source_type)
    if not handler:
        raise ValueError(f"No handler registered for source type: {source_type}")
    
    return handler


def detect_source_type(config: Any, slug: str) -> SourceType | None:
    """Detect which source type a slug belongs to.
    
    Checks in order: web, local, downloaded devdocs, devdocs catalog.
    
    Args:
        config: Configuration object
        slug: Documentation slug
        
    Returns:
        SourceType or None if not found
    """
    # Check web sources first (fast, config-based)
    for web_src in config.web_sources:
        if web_src.slug == slug:
            return SourceType.WEB
    
    # Check local sources (fast, no catalog fetch needed)
    for local_src in config.local_sources:
        if local_src.slug == slug:
            return SourceType.LOCAL
    
    # Check if it's a downloaded devdocs doc
    if slug in config.downloaded_slugs:
        return SourceType.DEVDOCS
    
    # Finally check devdocs catalog (may fetch from network)
    from .catalog import fetch_devdocs_catalog, find_doc_by_slug
    try:
        catalog = fetch_devdocs_catalog(cache_dir=config.cache_dir)
        if find_doc_by_slug(catalog, slug):
            return SourceType.DEVDOCS
    except Exception:
        pass
    
    return None
