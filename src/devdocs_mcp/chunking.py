"""Document chunking strategies for large content."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A chunk of document with metadata."""
    text: str
    title_hint: str | None = None  # Suggested title from first heading or sentence


class ChunkingStrategy(Protocol):
    """Protocol for document chunking strategies."""

    def chunk(self, text: str, max_tokens: int = 512) -> list[str]:
        """Split text into chunks of max_tokens size."""
        ...


class SentenceChunker:
    """Chunk documents by sentences, respecting token limits."""

    def __init__(self, overlap: int = 50):
        """Initialize chunker with overlap between chunks.
        
        Args:
            overlap: Number of tokens to overlap between chunks
        """
        self.overlap = overlap

    def chunk(self, text: str, max_tokens: int = 512) -> list[str]:
        """Split text into chunks by sentences.
        
        Args:
            text: Input text to chunk
            max_tokens: Maximum tokens per chunk (approximate, using word count)
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []

        # Split into sentences
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = self._estimate_tokens(sentence)
            
            # If single sentence exceeds max, split it by words
            if sentence_length > max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long sentence into word-based chunks
                word_chunks = self._chunk_by_words(sentence, max_tokens)
                chunks.extend(word_chunks)
                continue
            
            # Add sentence to current chunk if it fits
            if current_length + sentence_length <= max_tokens:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # Start new chunk
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                
                # Add overlap from previous chunk
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk, self.overlap
                )
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(
                    self._estimate_tokens(s) for s in current_chunk
                )

        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        logger.debug(
            "Chunked text of %d chars into %d chunks",
            len(text), len(chunks)
        )
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using basic heuristics."""
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: words * 1.3)."""
        words = len(text.split())
        return int(words * 1.3)

    def _chunk_by_words(self, text: str, max_tokens: int) -> list[str]:
        """Split long text by words when sentences are too long."""
        words = text.split()
        chunks = []
        current = []
        current_length = 0

        for word in words:
            word_tokens = self._estimate_tokens(word)
            if current_length + word_tokens > max_tokens and current:
                chunks.append(" ".join(current))
                # Add overlap
                overlap_words = current[-min(len(current), self.overlap // 2):]
                current = overlap_words + [word]
                current_length = sum(self._estimate_tokens(w) for w in current)
            else:
                current.append(word)
                current_length += word_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _get_overlap_sentences(
        self, sentences: list[str], target_tokens: int
    ) -> list[str]:
        """Get last N sentences that fit within target_tokens."""
        overlap = []
        total_tokens = 0

        for sentence in reversed(sentences):
            tokens = self._estimate_tokens(sentence)
            if total_tokens + tokens > target_tokens:
                break
            overlap.insert(0, sentence)
            total_tokens += tokens

        return overlap


class ParagraphChunker:
    """Chunk documents by paragraphs, respecting token limits."""

    def __init__(self, overlap: int = 50):
        self.overlap = overlap

    def chunk(self, text: str, max_tokens: int = 512) -> list[str]:
        """Split text into chunks by paragraphs.
        
        Args:
            text: Input text to chunk
            max_tokens: Maximum tokens per chunk
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []

        # Split by paragraphs (double newline)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Fall back to sentence chunking if needed
        sentence_chunker = SentenceChunker(overlap=self.overlap)
        
        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)
            
            # If paragraph is too long, split it
            if para_tokens > max_tokens:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Use sentence chunker for long paragraphs
                para_chunks = sentence_chunker.chunk(para, max_tokens)
                chunks.extend(para_chunks)
                continue
            
            # Add paragraph to current chunk if it fits
            if current_length + para_tokens <= max_tokens:
                current_chunk.append(para)
                current_length += para_tokens
            else:
                # Start new chunk
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_length = para_tokens

        # Add remaining chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        words = len(text.split())
        return int(words * 1.3)


def chunk_document(
    text: str,
    max_tokens: int = 512,
    strategy: str = "sentence",
    overlap: int = 50,
) -> list[DocumentChunk]:
    """Chunk a document using the specified strategy.
    
    Args:
        text: Input text to chunk
        max_tokens: Maximum tokens per chunk
        strategy: Chunking strategy ('sentence' or 'paragraph')
        overlap: Token overlap between chunks
        
    Returns:
        List of DocumentChunk objects with text and title hints
    """
    if strategy == "paragraph":
        chunker = ParagraphChunker(overlap=overlap)
    else:
        chunker = SentenceChunker(overlap=overlap)
    
    text_chunks = chunker.chunk(text, max_tokens=max_tokens)
    
    # Convert to DocumentChunk with title hints
    chunks_with_metadata = []
    for chunk_text in text_chunks:
        title_hint = _extract_title_hint(chunk_text)
        chunks_with_metadata.append(DocumentChunk(
            text=chunk_text,
            title_hint=title_hint
        ))
    
    return chunks_with_metadata


def _extract_title_hint(text: str, max_length: int = 80) -> str | None:
    """Extract a title hint from chunk text.
    
    Tries to find:
    1. First heading-like line (all caps, or ends with colon)
    2. First sentence
    3. First line
    
    Args:
        text: Chunk text
        max_length: Maximum length for title hint
        
    Returns:
        Title hint string or None
    """
    if not text.strip():
        return None
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return None
    
    # Look for heading-like patterns in first few lines
    for line in lines[:3]:
        # Skip very short lines
        if len(line) < 3:
            continue
            
        # Check if it looks like a heading
        # - All caps (with some tolerance for punctuation)
        # - Ends with colon
        # - Short line followed by longer content
        words = line.split()
        if len(words) > 0 and len(words) <= 10:
            # All caps check (at least 60% uppercase)
            alpha_chars = [c for c in line if c.isalpha()]
            if alpha_chars:
                upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
                if upper_ratio > 0.6:
                    return line[:max_length]
            
            # Ends with colon
            if line.endswith(':'):
                return line[:-1][:max_length]  # Remove colon
    
    # Fallback: use first sentence or first line
    first_line = lines[0]
    
    # Try to extract first sentence
    sentences = re.split(r'[.!?]\s+', first_line)
    if sentences:
        first_sentence = sentences[0].strip()
        if first_sentence:
            # Truncate if too long
            if len(first_sentence) > max_length:
                return first_sentence[:max_length-3] + "..."
            return first_sentence
    
    # Fallback to first line
    if len(first_line) > max_length:
        return first_line[:max_length-3] + "..."
    return first_line

