# devdocs-mcp
KnowledgeBase MCP server for Developer Documentation

## Installation

### Quick Start with uvx (Recommended)

Run directly from GitHub with ML dependencies (required for semantic search):

```bash
uvx --from "devdocs-mcp[ml] @ git+https://github.com/HellButcher/devdocs-mcp.git" devdocs-mcp
```

### MCP Server Configuration

Add to your MCP client configuration (e.g., Claude Desktop, Cline):

```json
{
  "mcpServers": {
    "devdocs": {
      "command": "uvx",
      "args": [
        "--from",
        "devdocs-mcp[ml] @ git+https://github.com/HellButcher/devdocs-mcp.git",
        "devdocs-mcp"
      ]
    }
  }
}
```

### Development Setup

Clone and install with ML dependencies:

```bash
git clone https://github.com/HellButcher/devdocs-mcp.git
cd devdocs-mcp
uv sync --extra ml
uv run devdocs-mcp
```

### Alternative Installation Methods

Install with uv:
```bash
uv pip install "devdocs-mcp[ml] @ git+https://github.com/HellButcher/devdocs-mcp.git"
devdocs-mcp
```

Install without ML dependencies (semantic search disabled):
```bash
uvx --from "git+https://github.com/HellButcher/devdocs-mcp.git" devdocs-mcp
```

## Features

- Access to 819+ documentation sources from devdocs.io
- Semantic search with vector embeddings (requires ML dependencies)
- Custom local documentation sources
- Efficient caching and incremental updates
- 10 MCP tools for documentation queries

## Configuration

Configuration is stored at XDG paths:
- Config: `~/.config/devdocs-mcp/config.json`
- Cache: `~/.cache/devdocs-mcp/` (docs + embeddings)

Environment variables:
- `DEVD_EMBEDDING_MODEL` - Override the default embedding model
