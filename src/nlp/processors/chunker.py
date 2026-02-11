"""
Text Chunker for handling long documents.
Splits text into overlapping chunks for processing.
"""

import re
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from ..config import get_nlp_settings

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a chunk of text."""
    text: str
    start_idx: int
    end_idx: int
    chunk_idx: int
    total_chunks: int
    metadata: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.text)


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    chunk_size: int = 450  # tokens/words
    chunk_overlap: int = 50
    min_chunk_size: int = 50
    split_on_sentences: bool = True
    preserve_paragraphs: bool = True


class TextChunker:
    """
    Text chunker for handling long documents.

    Splits text into overlapping chunks while respecting
    sentence and paragraph boundaries when possible.
    """

    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+')
    PARAGRAPH_BREAK = re.compile(r'\n\s*\n')

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """
        Initialize chunker.

        Args:
            config: Chunking configuration
        """
        if config is None:
            settings = get_nlp_settings()
            config = ChunkingConfig(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )
        self.config = config

    def chunk(self, text: str) -> List[TextChunk]:
        """
        Split text into chunks.

        Args:
            text: Input text

        Returns:
            List of TextChunk objects
        """
        if not text:
            return []

        # If text is short enough, return as single chunk
        if len(text.split()) <= self.config.chunk_size:
            return [TextChunk(
                text=text,
                start_idx=0,
                end_idx=len(text),
                chunk_idx=0,
                total_chunks=1
            )]

        # Split into sentences or paragraphs
        if self.config.preserve_paragraphs:
            segments = self._split_paragraphs(text)
        elif self.config.split_on_sentences:
            segments = self._split_sentences(text)
        else:
            segments = text.split()

        # Group segments into chunks
        chunks = self._create_chunks(segments, text)

        return chunks

    def chunk_batch(self, texts: List[str]) -> List[List[TextChunk]]:
        """
        Chunk multiple texts.

        Args:
            texts: List of texts

        Returns:
            List of chunk lists
        """
        return [self.chunk(text) for text in texts]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = self.SENTENCE_ENDINGS.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = self.PARAGRAPH_BREAK.split(text)
        result = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If paragraph is too long, split into sentences
            if len(para.split()) > self.config.chunk_size:
                sentences = self._split_sentences(para)
                result.extend(sentences)
            else:
                result.append(para)

        return result

    def _create_chunks(
        self,
        segments: List[str],
        original_text: str
    ) -> List[TextChunk]:
        """
        Create chunks from segments with overlap.

        Args:
            segments: List of text segments
            original_text: Original full text

        Returns:
            List of TextChunk objects
        """
        chunks = []
        current_chunk = []
        current_size = 0

        for segment in segments:
            segment_words = len(segment.split())

            # Check if adding segment exceeds chunk size
            if current_size + segment_words > self.config.chunk_size:
                if current_chunk:
                    # Create chunk from current segments
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(chunk_text)

                    # Keep overlap segments
                    overlap_segments = []
                    overlap_size = 0

                    for seg in reversed(current_chunk):
                        seg_words = len(seg.split())
                        if overlap_size + seg_words <= self.config.chunk_overlap:
                            overlap_segments.insert(0, seg)
                            overlap_size += seg_words
                        else:
                            break

                    current_chunk = overlap_segments
                    current_size = overlap_size

            current_chunk.append(segment)
            current_size += segment_words

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.split()) >= self.config.min_chunk_size:
                chunks.append(chunk_text)

        # Convert to TextChunk objects with position info
        result = []
        current_pos = 0

        for i, chunk_text in enumerate(chunks):
            # Find position in original text
            start_idx = original_text.find(chunk_text[:50], current_pos)
            if start_idx == -1:
                start_idx = current_pos

            end_idx = start_idx + len(chunk_text)

            result.append(TextChunk(
                text=chunk_text,
                start_idx=start_idx,
                end_idx=end_idx,
                chunk_idx=i,
                total_chunks=len(chunks)
            ))

            current_pos = start_idx + 1

        return result

    def rechunk_with_context(
        self,
        chunks: List[TextChunk],
        context_before: int = 1,
        context_after: int = 1
    ) -> List[TextChunk]:
        """
        Add context from neighboring chunks.

        Args:
            chunks: List of chunks
            context_before: Number of chunks to include before
            context_after: Number of chunks to include after

        Returns:
            Chunks with added context
        """
        result = []

        for i, chunk in enumerate(chunks):
            context_parts = []

            # Add context from previous chunks
            for j in range(max(0, i - context_before), i):
                context_parts.append(f"[PREV] {chunks[j].text}")

            # Add current chunk
            context_parts.append(chunk.text)

            # Add context from following chunks
            for j in range(i + 1, min(len(chunks), i + context_after + 1)):
                context_parts.append(f"[NEXT] {chunks[j].text}")

            result.append(TextChunk(
                text='\n'.join(context_parts),
                start_idx=chunk.start_idx,
                end_idx=chunk.end_idx,
                chunk_idx=chunk.chunk_idx,
                total_chunks=chunk.total_chunks,
                metadata={'has_context': True}
            ))

        return result


def chunk_text(
    text: str,
    chunk_size: int = 450,
    overlap: int = 50
) -> List[str]:
    """
    Convenience function for text chunking.

    Args:
        text: Input text
        chunk_size: Maximum words per chunk
        overlap: Overlap between chunks

    Returns:
        List of chunk texts
    """
    config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    chunker = TextChunker(config)
    chunks = chunker.chunk(text)
    return [c.text for c in chunks]


class TranscriptChunker(TextChunker):
    """Specialized chunker for earnings call transcripts."""

    SPEAKER_PATTERN = re.compile(r'\[([^\]]+)\]:')

    def __init__(self):
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=50,
            preserve_paragraphs=True
        )
        super().__init__(config)

    def chunk_by_speaker(self, text: str) -> List[TextChunk]:
        """
        Chunk transcript by speaker turns.

        Args:
            text: Transcript text

        Returns:
            List of chunks, one per speaker turn
        """
        # Split by speaker labels
        parts = self.SPEAKER_PATTERN.split(text)

        chunks = []
        current_speaker = None

        for i, part in enumerate(parts):
            if i % 2 == 1:  # Speaker name
                current_speaker = part
            elif part.strip():  # Content
                chunk_text = part.strip()

                # If content is too long, sub-chunk it
                if len(chunk_text.split()) > self.config.chunk_size:
                    sub_chunks = self.chunk(chunk_text)
                    for sub in sub_chunks:
                        sub.metadata['speaker'] = current_speaker
                        chunks.append(sub)
                else:
                    chunks.append(TextChunk(
                        text=chunk_text,
                        start_idx=0,
                        end_idx=len(chunk_text),
                        chunk_idx=len(chunks),
                        total_chunks=0,  # Will update
                        metadata={'speaker': current_speaker}
                    ))

        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks
