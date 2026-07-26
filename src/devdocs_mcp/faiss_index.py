"""FAISS-based semantic search index for devdocs."""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Configure transformers/sentence-transformers caching
# Cache models for 24h to reduce HEAD requests to HuggingFace
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")  # Allow initial download
os.environ.setdefault("HF_HUB_OFFLINE", "0")  # Allow initial download
# Reduce verbosity of HuggingFace HTTP logs
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

try:
    import faiss  # noqa: F401
    from sentence_transformers import SentenceTransformer  # noqa: F401
    import numpy as np  # noqa: F401

    _ML_DEPS_OK = True
except ImportError as e:
    _ML_DEPS_OK = False
    _IMPORT_ERROR = str(e)


def _check_ml_deps():
    """Check if ML dependencies are available."""
    if not _ML_DEPS_OK:
        raise RuntimeError(
            f"ML dependencies not installed.\n\n"
            f"Install with:\nuv sync --extra ml\n\n"
            f"(Original error: {_IMPORT_ERROR})"
        )


class EmbeddingIndex:
    """FAISS-based embedding index with SQLite metadata.
    
    Uses IndexIDMap wrapper for efficient document removal.
    Document table's INTEGER PRIMARY KEY (id) serves as FAISS ID directly.
    """

    def __init__(self, embeddings_dir: Path, metadata_db_path: Path):
        """Initialize the embedding index.
        
        Args:
            embeddings_dir: Directory to store FAISS index files
            metadata_db_path: Path to SQLite metadata database
        """
        _check_ml_deps()

        import faiss
        from sentence_transformers import SentenceTransformer

        self.embeddings_dir = embeddings_dir
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_db_path = metadata_db_path

        # Load or create embedding model
        from .config import EMBEDDING_MODEL
        
        # Configure model caching to avoid frequent HEAD requests
        # Use local_files_only=True if model is cached, otherwise download once
        model_cache_dir = embeddings_dir / "model_cache"
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Try loading from cache first
            self.model = SentenceTransformer(
                EMBEDDING_MODEL,
                cache_folder=str(model_cache_dir),
                local_files_only=True
            )
            logger.debug("Loaded model from cache")
        except (OSError, ValueError):
            # Download if not in cache
            logger.info("Downloading model (first time only)...")
            self.model = SentenceTransformer(
                EMBEDDING_MODEL,
                cache_folder=str(model_cache_dir),
                local_files_only=False
            )
            logger.info("Model downloaded and cached")
        
        try:
            self.dim = self.model.get_embedding_dimension()
        except AttributeError:
            self.dim = self.model.get_sentence_embedding_dimension()

        # FAISS index with ID mapping for efficient removal
        # Inner index uses Inner Product (cosine similarity with normalized vectors)
        inner_index = faiss.IndexFlatIP(self.dim)
        self.faiss_index = faiss.IndexIDMap(inner_index)

        self._loaded = False

    def load_or_create_index(self):
        """Load existing index from disk or create new one.
        
        ID mappings are in SQLite (documents.id column), not a separate file.
        """
        import faiss

        idx_path = self.embeddings_dir / "index.faiss"

        if idx_path.exists():
            # Load FAISS index
            self.faiss_index = faiss.read_index(str(idx_path))
            self._loaded = True

            # IDs are in SQLite documents table
            with sqlite3.connect(str(self.metadata_db_path)) as db:
                count = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            
            logger.info(
                "Loaded index with %d embeddings from %s",
                count, idx_path
            )
        else:
            # Create new empty index
            self._loaded = False

    def save_index(self):
        """Save current index to disk.
        
        ID mappings are in SQLite (documents.id), not a separate file.
        Only the FAISS index file is written here.
        """
        import faiss

        idx_path = self.embeddings_dir / "index.faiss"
        faiss.write_index(self.faiss_index, str(idx_path))
        
        logger.info("Saved FAISS index to %s", idx_path)

    def add_documents(self, texts: list[str], doc_ids: list[str]) -> int:
        """Add new documents to the embedding index.
        
        Documents are inserted into SQLite first to get auto-generated IDs,
        which are then used as FAISS IDs directly.

        Args:
            texts: Plain text content for each document
            doc_ids: Unique IDs matching each text

        Returns: number of documents added
        """
        if not texts:
            return 0

        import numpy as np
        from .embedder import get_faiss_ids_for_doc_ids

        # Check which docs are already indexed
        new_texts = []
        new_ids = []
        
        with sqlite3.connect(str(self.metadata_db_path)) as db:
            existing = get_faiss_ids_for_doc_ids(db, doc_ids)
            
            for text, doc_id in zip(texts, doc_ids):
                if doc_id not in existing:
                    new_texts.append(text)
                    new_ids.append(doc_id)

        if not new_texts:
            logger.info("No new documents to embed")
            return 0

        # Generate embeddings
        embeddings = self.model.encode(new_texts, normalize_embeddings=True)

        # Get FAISS IDs (id column) for the new documents
        with sqlite3.connect(str(self.metadata_db_path)) as db:
            faiss_ids_map = get_faiss_ids_for_doc_ids(db, new_ids)
        
        faiss_ids = [faiss_ids_map[doc_id] for doc_id in new_ids]

        # Add to FAISS index with document IDs
        self.faiss_index.add_with_ids(
            embeddings.astype("float32"),
            np.array(faiss_ids, dtype=np.int64)
        )

        logger.info(
            "Added %d documents to index (total: %d)",
            len(new_ids), self.get_doc_count()
        )
        return len(new_ids)

    def remove_documents(self, doc_ids: list[str]) -> int:
        """Remove documents from the embedding index.

        With IndexIDMap, removal is efficient (no rebuild needed).
        Uses document.id as FAISS ID directly.
        
        Args:
            doc_ids: Document IDs to remove
            
        Returns:
            Number of documents removed
        """
        if not doc_ids:
            return 0

        import numpy as np
        from .embedder import get_faiss_ids_for_doc_ids, delete_documents_by_ids

        # Get FAISS IDs (id column) for documents to remove
        with sqlite3.connect(str(self.metadata_db_path)) as db:
            faiss_ids_map = get_faiss_ids_for_doc_ids(db, doc_ids)
            
            if not faiss_ids_map:
                logger.info("No documents to remove")
                return 0
            
            # Delete from database (this removes the ID)
            deleted_count = delete_documents_by_ids(db, list(faiss_ids_map.keys()))

        # Remove from FAISS index
        faiss_ids_to_remove = list(faiss_ids_map.values())
        self.faiss_index.remove_ids(np.array(faiss_ids_to_remove, dtype=np.int64))

        logger.info("Removed %d documents from index (total: %d)", deleted_count, self.get_doc_count())
        return deleted_count

    def search(
        self, query: str, top_k: int = 10, min_score: float = 0.0
    ) -> list[dict[str, Any]]:
        """Search for documents similar to the query.
        
        Uses FAISS IDs (documents.id) to directly lookup documents.

        Args:
            query: Search query string
            top_k: Maximum number of results
            min_score: Minimum similarity score threshold

        Returns: List of {doc_id, score}
        """
        import numpy as np
        from .embedder import get_document_by_faiss_id, count_documents

        doc_count = 0
        with sqlite3.connect(str(self.metadata_db_path)) as db:
            doc_count = count_documents(db)
        
        if doc_count == 0:
            return []

        # Encode query
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        query_vec = np.array([query_embedding], dtype="float32")

        # Search FAISS index
        scores, faiss_ids = self.faiss_index.search(
            query_vec, min(top_k, doc_count)
        )

        results = []
        with sqlite3.connect(str(self.metadata_db_path)) as db:
            for score, faiss_id in zip(scores[0], faiss_ids[0]):
                if faiss_id < 0:
                    continue  # Invalid index (padding)

                # FAISS ID is documents.id - direct lookup!
                doc = get_document_by_faiss_id(db, int(faiss_id))
                if doc is None:
                    continue  # Document not found
                
                if float(score) < min_score:
                    continue

                results.append({
                    "doc_id": doc["id"],  # Return the doc_id, not integer id
                    "score": float(score),
                })

        return results

    def get_doc_count(self) -> int:
        """Get total number of documents in the index."""
        from .embedder import count_documents
        
        with sqlite3.connect(str(self.metadata_db_path)) as db:
            return count_documents(db)

    @property
    def is_empty(self) -> bool:
        """Check if the index is empty."""
        return self.get_doc_count() == 0
