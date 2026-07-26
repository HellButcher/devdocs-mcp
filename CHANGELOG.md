# Changelog

## [Unreleased] - 2026-07-25

### Added

#### CLI with Subcommands
- **New CLI structure** with 5 subcommands:
  - `mcp` - Run as MCP server (default)
  - `query` - Search documentation from command line
  - `add` - Download docs from devdocs.io or add local HTML directories
  - `reindex` - Rebuild search index (with `--clean` flag for fresh rebuild)
  - `list` - List available/downloaded documentation

- **Query command features:**
  - `-k/--top-k` - Limit number of results
  - `--min-score` - Set similarity threshold
  - `--slugs` - Filter by documentation slugs
  - `--source` - Filter by source type (devdocs/local)

- **Add command features:**
  - `add download SLUG [SLUG ...]` - Download from devdocs.io
  - `add local PATH [--prefix PREFIX]` - Add local HTML directory

- **List command features:**
  - `--downloaded` - Show only downloaded docs
  - `--large` - Include large documentation (>50MB)

- **Reindex command features:**
  - Incremental indexing by default (only new docs)
  - `--clean` flag for dropping database and rebuilding from scratch

#### Database Improvements
- **Explicit `id` column** in documents table (INTEGER PRIMARY KEY AUTOINCREMENT)
  - Replaces implicit rowid usage for clarity
  - Serves as FAISS ID directly (no mapping table needed)
- **Efficient incremental reindexing** - only processes new documents
- **Better progress reporting** during reindex

### Changed

#### Architecture
- **Improved HTML cleaning** with BeautifulSoup4:
  - Prioritizes `<main>` tag content
  - Filters nested elements to avoid redundancy
  - Removes navigation, ads, and boilerplate
- **Document chunking** with configurable strategies:
  - SentenceChunker (sentence-based splitting)
  - ParagraphChunker (paragraph-based splitting)
  - 512 tokens per chunk with 50-token overlap
- **HTTP retry logic** extracted to `http_utils.py`:
  - Exponential backoff (1s, 2s, 4s)
  - Shared across catalog fetching and downloads
  - Configurable retry count and timeout

#### MCP Server
- **Fixed function name collisions** using import aliases
  - `download_doc as _download_doc_impl`
  - `remove_doc as _remove_doc_impl`
- **Search filters** implemented:
  - `slugs: list[str]` parameter for filtering by documentation
  - `source_type: str` parameter for filtering by source (devdocs/local)

### Fixed
- **Config path construction** now handles missing directories properly
- **Catalog caching** with 24-hour TTL (was fetching on every startup)
- **Path validation** in `add_local_source()` - better error messages
- **FAISS IndexIDMap** for efficient document deletion without rebuild
- **Auto-download removed** from config.py (was causing unwanted behavior)

### Technical Details

#### Database Schema
```sql
CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,  -- FAISS ID
    doc_id TEXT UNIQUE NOT NULL,           -- logical document ID
    slug TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    type TEXT DEFAULT '',
    path TEXT DEFAULT '',
    source_type TEXT DEFAULT 'devdocs',
    chunk_index INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 1
);
```

#### File Structure Changes
```
After:
  src/devdocs_mcp/
    __main__.py                      # New CLI entry point
    chunking.py                      # Document chunking strategies
    http_utils.py                    # Shared HTTP retry logic
```

#### Dependencies
- Added: `beautifulsoup4` for HTML parsing
- Existing: `mcp`, `httpx`, `platformdirs`
- Optional (ml): `faiss-cpu`, `sentence-transformers`, `numpy`

### Documentation
- **CLI_GUIDE.md** - Comprehensive CLI usage guide
- **SEARCH_FILTERS.md** - Search filter implementation details

### Performance
- **First query**: ~2-3 seconds (model loading)
- **Subsequent queries**: <200ms for most queries
- **Indexing speed**: ~500-1000 docs/second
- **Memory usage**: ~100-200MB for typical setup (5-10 documentation bundles)

### Example Usage

```bash
# Download and index documentation
devdocs-mcp add download javascript python~3.13
devdocs-mcp reindex

# Search from command line
devdocs-mcp query "async await" -k 5

# Run as MCP server
devdocs-mcp mcp

# List downloaded docs
devdocs-mcp list --downloaded
```

## Previous Versions

No previous tagged releases.
