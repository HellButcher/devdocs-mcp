# devdocs-mcp Development Guidelines

## Project Structure
```
src/devdocs_mcp/
├── __init__.py      # Entry point
├── __main__.py      # CLI entry point with error handling
├── config.py        # XDG paths (platformdirs), settings, sources
├── catalog.py       # Fetches 819 docs from devdocs.io API
├── download.py      # Downloads/extracts .tar.gz bundles
├── embedder.py      # HTML→text extraction + SQLite metadata storage
├── chunking.py      # Document chunking strategies (sentence/paragraph)
├── http_utils.py    # Shared HTTP retry logic with exponential backoff
├── faiss_index.py   # FAISS vector index with IndexIDMap
└── mcp_server.py    # 10 MCP tools via FastMCP
```

## Setup & Dependencies
- Python >=3.13
- `uv sync` for core deps (mcp, httpx, platformdirs)
- `uv sync --extra ml` for ML deps (faiss-cpu, sentence-transformers, numpy)
- ML deps are optional — server runs without them but semantic search requires them

## Building New Tools
1. Add tool function with `@mcp.tool()` decorator in `mcp_server.py`
2. Import dependencies at top of file
3. Keep tool descriptions clear and parameter types explicit
4. Return strings for simple responses; use structured formatting for multi-field output

## Code Style
- Type hints on all public functions
- Docstrings with Args/Returns sections
- No print() in library code — use logging.getLogger(__name__)
- Import order: stdlib → third-party → local (with noqa: E402 if needed)

## Testing
```bash
uv run python -c "from devdocs_mcp.mcp_server import ML_AVAILABLE, list_docs; ..."
```

## Adding New Doc Scrapers
devdocs.io bundles are served from `https://downloads.devdocs.io/{slug}.tar.gz` containing:
- `index.json` — entry index with path/type info
- `db.json` — page content map (path → HTML)
- `meta.json` — metadata

New slugs can be added by downloading the tarball and extracting. The embedder handles all formats automatically.

## Configuration
Config stored at XDG paths:
- Config: `~/.config/devdocs-mcp/config.json`
- Cache: `~/.cache/devdocs-mcp/` (docs + embeddings)

Environment variable override: `DEVD_EMBEDDING_MODEL` to change the embedding model.

## Common Tasks
- Download docs: `download_doc('python')` for single doc, `download_doc(['python', 'javascript'])` for multiple
- Rebuild index after downloading docs: `rebuild_index()` tool
- List available docs: `list_docs(include_large=True)` for full catalog
- List with filter: `list_docs(query='python')` to fuzzy-match on slug/name/type
- Check index status: `list_docs()` shows if index is built and which docs are indexed
- Add local HTML dir: `add_local_source("/path/to/docs")`
