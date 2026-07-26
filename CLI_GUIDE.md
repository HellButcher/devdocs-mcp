# devdocs-mcp CLI Guide

## Overview

The `devdocs-mcp` CLI provides a comprehensive command-line interface for managing and searching developer documentation. It supports running as an MCP server, downloading documentation, indexing content, and performing semantic searches.

## Installation

```bash
# Core dependencies only (MCP server without ML features)
uv sync

# With ML dependencies (semantic search enabled)
uv sync --extra ml
```

## Commands

### 1. MCP Server Mode (Default)

Run devdocs-mcp as an MCP server:

```bash
# Default stdio transport
devdocs-mcp mcp

# Or explicitly specify transport
devdocs-mcp mcp --transport stdio

# HTTP transport (experimental)
devdocs-mcp mcp --transport http --port 8000
```

If no command is specified, `mcp` is used by default:

```bash
devdocs-mcp  # Same as: devdocs-mcp mcp
```

### 2. Query Documentation

Search indexed documentation from the command line:

```bash
# Basic search
devdocs-mcp query "async HTTP request"

# Limit results
devdocs-mcp query "python list comprehension" -k 3

# Set minimum similarity score
devdocs-mcp query "javascript promises" --min-score 0.5

# Filter by documentation slug(s)
devdocs-mcp query "list comprehension" --slugs python~3.13

# Filter by source type
devdocs-mcp query "async await" --source devdocs
```

**Options:**
- `-k, --top-k N`: Number of results to return (default: 5)
- `-s, --min-score SCORE`: Minimum similarity score (default: 0.3)
- `--slugs SLUG [SLUG ...]`: Filter by documentation slugs
- `--source {devdocs,local}`: Filter by source type

### 3. Add Documentation

#### Download from devdocs.io

```bash
# Download single documentation
devdocs-mcp add download javascript

# Download multiple documentations
devdocs-mcp add download python~3.13 javascript react

# Check available docs first
devdocs-mcp list
```

#### Add Local HTML Directory

```bash
# Add local HTML documentation
devdocs-mcp add local /path/to/docs

# With slug prefix
devdocs-mcp add local /path/to/docs --prefix myproject_
```

### 4. Reindex

Rebuild the search index after downloading or adding documentation:

```bash
# Incremental reindex (add new docs to existing index)
devdocs-mcp reindex

# Clean rebuild (drop database and recreate from scratch)
devdocs-mcp reindex --clean
```

**Clean mode** is useful when:
- Changing the database schema
- Fixing corrupted index
- Starting fresh after removing docs

### 5. List Documentation

View available and downloaded documentation:

```bash
# List all available docs (excluding large >50MB)
devdocs-mcp list

# Include large documentation packages
devdocs-mcp list --large

# Show only downloaded docs
devdocs-mcp list --downloaded
```

Output format:
```
Available documentation (819):

  [ ] angular~17 - Angular 17 (6.8 MB)
  [✓] async - Async (0.2 MB)
  [✓] axios - Axios (0.1 MB)
  ...
```

Checkmark `✓` indicates downloaded documentation.

## Typical Workflow

### Initial Setup

```bash
# 1. Install with ML dependencies
uv sync --extra ml

# 2. Browse available documentation
devdocs-mcp list

# 3. Download desired docs
devdocs-mcp add download javascript python~3.13 react

# 4. Build search index
devdocs-mcp reindex
```

### Daily Usage

```bash
# Search documentation
devdocs-mcp query "how to use async await in javascript" -k 5

# Add more documentation
devdocs-mcp add download vue~3

# Update index with new docs
devdocs-mcp reindex
```

### MCP Integration

```bash
# Run as MCP server (for Claude Desktop, IDEs, etc.)
devdocs-mcp mcp
```

## Configuration

Configuration is stored in XDG-compliant directories:

- **Config:** `~/.config/devdocs-mcp/config.json`
- **Cache:** `~/.cache/devdocs-mcp/`
  - `docs/` - Downloaded documentation bundles
  - `embeddings/` - FAISS vector index
  - `metadata.db` - SQLite metadata database
  - `catalog_cache.json` - Cached catalog (24h TTL)

## Database Architecture

The system uses:

1. **SQLite** for metadata storage
   - `documents` table with `id INTEGER PRIMARY KEY AUTOINCREMENT`
   - The `id` column serves as FAISS ID directly
   - No separate mapping table needed

2. **FAISS** for vector similarity search
   - `IndexIDMap` wrapper for efficient document removal
   - Uses Inner Product (cosine similarity with normalized vectors)
   - Stored in `~/.cache/devdocs-mcp/embeddings/index.faiss`

3. **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
   - Can be changed via `DEVD_EMBEDDING_MODEL` env variable

## Examples

### Search Specific Documentation

```bash
# Only search Python docs
devdocs-mcp query "list comprehension" --slugs python~3.13

# Search multiple docs
devdocs-mcp query "HTTP request" --slugs javascript axios
```

### Filter by Quality

```bash
# Only high-confidence results
devdocs-mcp query "neural networks" --min-score 0.7 -k 3
```

### Add Custom Documentation

```bash
# 1. Prepare HTML directory structure:
#    /path/to/myproject/
#      page1.html
#      page2.html
#      ...

# 2. Add to config
devdocs-mcp add local /path/to/myproject --prefix myproject_

# 3. Index
devdocs-mcp reindex

# 4. Search
devdocs-mcp query "myproject feature" --source local
```

## Troubleshooting

### No Results Found

1. Check if docs are indexed:
   ```bash
   devdocs-mcp list --downloaded
   ```

2. If downloaded but not indexed:
   ```bash
   devdocs-mcp reindex
   ```

3. Try lower similarity threshold:
   ```bash
   devdocs-mcp query "your query" --min-score 0.2
   ```

### ML Dependencies Error

```
Error: ML dependencies not installed.
```

**Solution:**
```bash
uv sync --extra ml
```

### Corrupted Index

```bash
# Clean rebuild
devdocs-mcp reindex --clean
```

### Download Failures

Check internet connection and retry:
```bash
devdocs-mcp add download python~3.13
```

If a specific doc consistently fails, it may not exist in the catalog. Check available slugs:
```bash
devdocs-mcp list | grep python
```

## Performance Notes

- **First query is slow:** Embedding model loads into memory (~90MB)
- **Subsequent queries are fast:** Model stays loaded
- **Large docs:** Python, JavaScript are 10-20MB compressed
- **Index build time:** ~1 second per MB of documentation
- **Memory usage:** Proportional to index size (~100MB for 5-10 docs)

## Advanced Usage

### Batch Download and Index

```bash
# Download multiple docs
devdocs-mcp add download \
  javascript \
  python~3.13 \
  react \
  vue~3 \
  node~22_lts

# Build index
devdocs-mcp reindex
```

### Custom Embedding Model

```bash
export DEVD_EMBEDDING_MODEL="sentence-transformers/paraphrase-MiniLM-L6-v2"
devdocs-mcp reindex --clean
```

### Programmatic Access

All CLI commands use the same underlying Python API:

```python
from devdocs_mcp.config import get_config
from devdocs_mcp.faiss_index import EmbeddingIndex

config = get_config()
idx = EmbeddingIndex(config.embeddings_dir, config.metadata_db_path)
idx.load_or_create_index()

results = idx.search("async await", top_k=5)
for r in results:
    print(f"{r['title']}: {r['score']:.3f}")
```

## Next Steps

- See `REFACTORING.md` for technical architecture details
- See `ROWID_ARCHITECTURE.md` for database schema information
- Check `pyproject.toml` for available configuration options
