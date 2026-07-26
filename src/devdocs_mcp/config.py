"""Configuration for devdocs-mcp."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import platformdirs


# ---------------------------------------------------------------------------
# Popular docs — small set that covers most developers' needs (~5-10 MB total)
# ---------------------------------------------------------------------------
POPULAR_DOCS = [
    "javascript",
    "node",
    "react",
    "python",
    "django",
    "flask",
    "rust",
    "go",
    "typescript",
    "bash",
]

# Docs we consider "large" and skip from the default popular set even if listed.
SKIP_LARGE = frozenset({
    "ansible", "scala~2.12_library", "scala~2.13_library", "qt~6.9",
    "openjdk~25", "qt~6.8", "scikit_learn", "man", "pandas",
})


# ---------------------------------------------------------------------------
# Platform-specific directories (XDG / Apple / Windows)
# ---------------------------------------------------------------------------

APP_NAME = "devdocs-mcp"

CACHE_DIR: Path = Path(platformdirs.user_cache_dir(APP_NAME))
CONFIG_DIR: Path = Path(platformdirs.user_config_dir(APP_NAME))
EMBEDDING_MODEL = os.environ.get(
    "DEVD_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


# ---------------------------------------------------------------------------
# Config file (JSON stored at config_dir/config.json)
# ---------------------------------------------------------------------------

@dataclass
class LocalSource:
    """A local directory containing static HTML documentation."""
    path: str  # absolute filesystem path
    slug: str  # unique slug for this local source
    name: str  # display name


@dataclass
class WebSource:
    """A web-fetched documentation source."""
    url: str  # base URL that was fetched
    slug: str  # unique slug for this web source
    name: str  # display name
    cache_path: str  # where the downloaded files are stored


@dataclass
class DevdocsSource:
    """devdocs.io catalog source."""
    enabled: bool = True
    exclude_patterns: list[str] = field(default_factory=list)  # Patterns to exclude from catalog (e.g., ["angular*", "react*"])


@dataclass
class Config:
    cache_dir: Path = field(default_factory=lambda: CACHE_DIR)
    config_dir: Path = field(default_factory=lambda: CONFIG_DIR)
    sources: list[LocalSource | WebSource | DevdocsSource] = field(default_factory=list)

    # Track which devdocs slugs are downloaded
    downloaded_slugs: set[str] = field(default_factory=set)

    def __post_init__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    @property
    def config_path(self) -> Path:
        return self.config_dir / "config.json"

    @property
    def docs_dir(self) -> Path:
        return self.cache_dir / "docs"
    
    @property
    def web_docs_dir(self) -> Path:
        """Directory for web-fetched documentation."""
        return self.cache_dir / "web"

    @property
    def embeddings_dir(self) -> Path:
        return self.cache_dir / "embeddings"

    @property
    def faiss_index_path(self) -> Path:
        return self.embeddings_dir / "index.faiss"

    @property
    def metadata_db_path(self) -> Path:
        return self.embeddings_dir / "metadata.db"
    
    @property
    def local_sources(self) -> list[LocalSource]:
        """Return only LocalSource instances."""
        return [s for s in self.sources if isinstance(s, LocalSource)]
    
    @property
    def web_sources(self) -> list[WebSource]:
        """Return only WebSource instances."""
        return [s for s in self.sources if isinstance(s, WebSource)]

    @classmethod
    def load(cls, config_dir: Path | None = None) -> Config:
        """Load config from disk or create default config."""
        p = (config_dir or CONFIG_DIR) / "config.json"
        if not p.exists():
            cfg = cls()
            cfg.sources = [DevdocsSource(enabled=True)]
            cfg.save(p)
            return cfg

        data = json.loads(p.read_text())
        sources: list[LocalSource | WebSource | DevdocsSource] = []
        for s in data.get("sources", []):
            if s.get("type") == "local":
                sources.append(LocalSource(
                    path=s["path"],
                    slug=s["slug"],
                    name=s["name"],
                ))
            elif s.get("type") == "web":
                sources.append(WebSource(
                    url=s["url"],
                    slug=s["slug"],
                    name=s["name"],
                    cache_path=s["cache_path"],
                ))
            else:
                sources.append(DevdocsSource(
                    enabled=s.get("enabled", True),
                    exclude_patterns=s.get("exclude_patterns", [])
                ))

        return cls(
            cache_dir=CACHE_DIR,
            config_dir=config_dir or CONFIG_DIR,
            sources=sources,
            downloaded_slugs=set(data.get("downloaded_slugs", [])),
        )

    def save(self, path: Path | None = None) -> None:
        p = path or self.config_path
        # Serialize sources
        src_list = []
        for s in self.sources:
            if isinstance(s, LocalSource):
                src_list.append({
                    "type": "local",
                    "path": s.path,
                    "slug": s.slug,
                    "name": s.name,
                })
            elif isinstance(s, WebSource):
                src_list.append({
                    "type": "web",
                    "url": s.url,
                    "slug": s.slug,
                    "name": s.name,
                    "cache_path": s.cache_path,
                })
            else:
                src_list.append({
                    "type": "devdocs",
                    "enabled": s.enabled,
                    "exclude_patterns": s.exclude_patterns,
                })

        json.dump({
            "sources": src_list,
            "downloaded_slugs": sorted(self.downloaded_slugs),
        }, p.open("w"), indent=2)


def get_config() -> Config:
    """Load config from the default data directory."""
    return Config.load()
