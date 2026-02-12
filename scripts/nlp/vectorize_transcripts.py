#!/usr/bin/env python3
"""
Vectorize Earnings Transcripts.
Creates embeddings for semantic search over 42K+ earnings call transcripts.

Usage:
    py -3 scripts/nlp/vectorize_transcripts.py --limit 1000  # Test with 1000
    py -3 scripts/nlp/vectorize_transcripts.py               # Process all
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import psycopg2
from psycopg2.extras import execute_batch
from tqdm import tqdm

from src.nlp.config import get_nlp_settings
from src.nlp.services.embedding_service import get_embedding_service
from src.nlp.processors.chunker import TextChunker, ChunkingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TranscriptVectorizer:
    """Vectorize earnings transcripts for semantic search."""

    def __init__(self, db_url: str = None):
        settings = get_nlp_settings()
        self.db_url = db_url or settings.database_url
        self.embedding_service = get_embedding_service()

        config = ChunkingConfig(
            chunk_size=450,
            chunk_overlap=50,
            min_chunk_size=100
        )
        self.chunker = TextChunker(config)
        self.batch_size = 32

    def get_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.db_url)

    def get_unprocessed_transcripts(self, limit: int = None) -> list:
        """Get transcripts that haven't been vectorized yet."""
        conn = self.get_connection()
        try:
            cur = conn.cursor()

            # Find transcripts not yet in nlp_embeddings
            query = """
                SELECT t.id, t.symbol, t.year, t.quarter, t.date, t.content
                FROM fmp_earnings_transcripts t
                WHERE t.content IS NOT NULL
                    AND LENGTH(t.content) > 100
                    AND NOT EXISTS (
                        SELECT 1 FROM nlp_embeddings e
                        WHERE e.source_type = 'transcript'
                        AND e.source_id = t.id
                    )
                ORDER BY t.date DESC
            """

            if limit:
                query += f" LIMIT {limit}"

            cur.execute(query)
            columns = ['id', 'symbol', 'year', 'quarter', 'date', 'content']
            return [dict(zip(columns, row)) for row in cur.fetchall()]

        finally:
            conn.close()

    def chunk_transcript(self, content: str, symbol: str, year: int, quarter: str) -> list:
        """Split transcript into chunks with metadata."""
        # Identify sections in the transcript
        sections = self._identify_sections(content)

        chunks = []
        for section_name, section_text in sections:
            if len(section_text.strip()) < 50:
                continue

            section_chunks = self.chunker.chunk(section_text)

            for i, chunk_obj in enumerate(section_chunks):
                # chunk_obj is a TextChunk object, get the text
                chunk_text = chunk_obj.text if hasattr(chunk_obj, 'text') else str(chunk_obj)
                chunks.append({
                    'text': chunk_text,
                    'section': section_name,
                    'chunk_index': i,
                    'prefix': f"[{symbol} Q{quarter} {year}] "
                })

        return chunks

    def _identify_sections(self, content: str) -> list:
        """Identify sections in transcript (prepared remarks, Q&A, etc.)."""
        sections = []
        content_lower = content.lower()

        # Common section markers
        qa_markers = ['question-and-answer', 'q&a session', 'questions and answers',
                      'operator: thank you', 'we will now take questions']
        guidance_markers = ['guidance', 'outlook', 'looking ahead', 'forward-looking']

        # Find Q&A section start
        qa_start = -1
        for marker in qa_markers:
            pos = content_lower.find(marker)
            if pos > 0 and (qa_start == -1 or pos < qa_start):
                qa_start = pos

        # Split into sections
        if qa_start > 0:
            prepared = content[:qa_start].strip()
            qa = content[qa_start:].strip()

            if prepared:
                sections.append(('prepared_remarks', prepared))
            if qa:
                sections.append(('qa', qa))
        else:
            # No clear Q&A section, treat as single section
            sections.append(('full_transcript', content))

        return sections

    def save_embeddings(self, records: list) -> int:
        """Save embeddings to database."""
        if not records:
            return 0

        conn = self.get_connection()
        try:
            cur = conn.cursor()

            query = """
                INSERT INTO nlp_embeddings
                (source_type, source_id, symbol, text_content, chunk_index,
                 embedding, year, quarter, section, model_name, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            data = []
            for r in records:
                embedding_list = r['embedding'].tolist() if hasattr(r['embedding'], 'tolist') else r['embedding']
                data.append((
                    'transcript',
                    r['source_id'],
                    r['symbol'],
                    r['text_content'],
                    r['chunk_index'],
                    embedding_list,
                    r['year'],
                    r['quarter'],
                    r['section'],
                    r['model_name'],
                    '{}'
                ))

            execute_batch(cur, query, data, page_size=100)
            conn.commit()
            return len(data)

        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()

    def process_batch(self, transcripts: list) -> int:
        """Process a batch of transcripts."""
        all_chunks = []

        for t in transcripts:
            chunks = self.chunk_transcript(
                t['content'],
                t['symbol'],
                t['year'],
                t['quarter']
            )

            for chunk in chunks:
                all_chunks.append({
                    'source_id': t['id'],
                    'symbol': t['symbol'],
                    'year': t['year'],
                    'quarter': t['quarter'],
                    'section': chunk['section'],
                    'chunk_index': chunk['chunk_index'],
                    'text': chunk['prefix'] + chunk['text']
                })

        if not all_chunks:
            return 0

        # Generate embeddings in batch
        texts = [c['text'] for c in all_chunks]
        embeddings = self.embedding_service.embed_batch(texts, batch_size=self.batch_size)

        if embeddings is None:
            logger.error("Failed to generate embeddings")
            return 0

        # Prepare records for saving
        records = []
        model_name = get_nlp_settings().embedding_model
        for i, chunk in enumerate(all_chunks):
            records.append({
                'source_id': chunk['source_id'],
                'symbol': chunk['symbol'],
                'year': chunk['year'],
                'quarter': chunk['quarter'],
                'section': chunk['section'],
                'chunk_index': chunk['chunk_index'],
                'text_content': chunk['text'],
                'embedding': embeddings[i],
                'model_name': model_name
            })

        return self.save_embeddings(records)

    def run(self, limit: int = None, batch_size: int = 50) -> dict:
        """Run vectorization process."""
        logger.info("Starting transcript vectorization...")

        # Check embedding service
        if not self.embedding_service.is_available():
            logger.error("Embedding service not available")
            return {'error': 'Embedding service not available'}

        # Get unprocessed transcripts
        transcripts = self.get_unprocessed_transcripts(limit)
        total = len(transcripts)
        logger.info(f"Found {total} unprocessed transcripts")

        if total == 0:
            return {'transcripts': 0, 'embeddings': 0}

        # Process in batches
        total_embeddings = 0
        start_time = time.time()

        with tqdm(total=total, desc="Vectorizing transcripts") as pbar:
            for i in range(0, total, batch_size):
                batch = transcripts[i:i + batch_size]
                embeddings_saved = self.process_batch(batch)
                total_embeddings += embeddings_saved
                pbar.update(len(batch))
                pbar.set_postfix({'embeddings': total_embeddings})

        elapsed = time.time() - start_time
        rate = total_embeddings / elapsed if elapsed > 0 else 0

        result = {
            'transcripts': total,
            'embeddings': total_embeddings,
            'elapsed_seconds': round(elapsed, 1),
            'rate_per_second': round(rate, 1)
        }

        logger.info(f"Vectorization complete: {result}")
        return result


def main():
    parser = argparse.ArgumentParser(description='Vectorize earnings transcripts')
    parser.add_argument('--limit', type=int, help='Limit number of transcripts to process')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')
    args = parser.parse_args()

    vectorizer = TranscriptVectorizer()
    result = vectorizer.run(limit=args.limit, batch_size=args.batch_size)

    print("\n=== RESULTS ===")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
