#!/usr/bin/env python3
"""
Vectorize Company Profiles.
Creates embeddings for semantic search over 92K+ company profiles.

Usage:
    py -3 scripts/nlp/vectorize_profiles.py --limit 1000  # Test with 1000
    py -3 scripts/nlp/vectorize_profiles.py               # Process all
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import psycopg2
from psycopg2.extras import execute_batch
from tqdm import tqdm

from src.nlp.config import get_nlp_settings
from src.nlp.services.embedding_service import get_embedding_service

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProfileVectorizer:
    """Vectorize company profiles for semantic search."""

    def __init__(self, db_url: str = None):
        settings = get_nlp_settings()
        self.db_url = db_url or settings.database_url
        self.embedding_service = get_embedding_service()
        self.batch_size = 64  # Higher batch size for shorter texts

    def get_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.db_url)

    def get_unprocessed_profiles(self, limit: int = None) -> list:
        """Get profiles that haven't been vectorized yet."""
        conn = self.get_connection()
        try:
            cur = conn.cursor()

            # Find profiles not yet in nlp_embeddings
            query = """
                SELECT p.symbol, p.company_name, p.description, p.sector,
                       p.industry, p.country, p.ceo, p.employees, p.mkt_cap
                FROM fmp_profiles p
                WHERE p.description IS NOT NULL
                    AND LENGTH(p.description) > 50
                    AND NOT EXISTS (
                        SELECT 1 FROM nlp_embeddings e
                        WHERE e.source_type = 'profile'
                        AND e.symbol = p.symbol
                    )
                ORDER BY p.mkt_cap DESC NULLS LAST
            """

            if limit:
                query += f" LIMIT {limit}"

            cur.execute(query)
            columns = ['symbol', 'company_name', 'description', 'sector',
                       'industry', 'country', 'ceo', 'employees', 'mkt_cap']
            return [dict(zip(columns, row)) for row in cur.fetchall()]

        finally:
            conn.close()

    def create_profile_text(self, profile: dict) -> str:
        """Create searchable text from profile."""
        parts = []

        # Company name and symbol
        if profile.get('company_name'):
            parts.append(f"{profile['company_name']} ({profile['symbol']})")

        # Sector and industry
        if profile.get('sector'):
            parts.append(f"Sector: {profile['sector']}")
        if profile.get('industry'):
            parts.append(f"Industry: {profile['industry']}")

        # Country
        if profile.get('country'):
            parts.append(f"Country: {profile['country']}")

        # CEO and employees
        if profile.get('ceo'):
            parts.append(f"CEO: {profile['ceo']}")
        if profile.get('employees'):
            parts.append(f"Employees: {profile['employees']:,}")

        # Market cap
        if profile.get('mkt_cap'):
            mkt_cap_b = profile['mkt_cap'] / 1e9
            if mkt_cap_b >= 1:
                parts.append(f"Market Cap: ${mkt_cap_b:.1f}B")
            else:
                parts.append(f"Market Cap: ${profile['mkt_cap'] / 1e6:.0f}M")

        # Description (main content)
        if profile.get('description'):
            parts.append(f"\n{profile['description']}")

        return " | ".join(parts[:7]) + (parts[7] if len(parts) > 7 else "")

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
                    'profile',
                    None,  # source_id not applicable for profiles
                    r['symbol'],
                    r['text_content'],
                    0,  # chunk_index = 0 (single chunk per profile)
                    embedding_list,
                    None,  # year
                    None,  # quarter
                    'description',  # section
                    r['model_name'],
                    json.dumps(r.get('metadata', {}))
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

    def process_batch(self, profiles: list) -> int:
        """Process a batch of profiles."""
        if not profiles:
            return 0

        # Create texts for embedding
        texts = []
        profile_data = []

        for p in profiles:
            text = self.create_profile_text(p)
            texts.append(text)
            profile_data.append({
                'symbol': p['symbol'],
                'text_content': text,
                'metadata': {
                    'sector': p.get('sector'),
                    'industry': p.get('industry'),
                    'country': p.get('country')
                }
            })

        # Generate embeddings in batch
        embeddings = self.embedding_service.embed_batch(texts, batch_size=self.batch_size)

        if embeddings is None:
            logger.error("Failed to generate embeddings")
            return 0

        # Prepare records for saving
        model_name = get_nlp_settings().embedding_model
        records = []
        for i, data in enumerate(profile_data):
            records.append({
                'symbol': data['symbol'],
                'text_content': data['text_content'],
                'embedding': embeddings[i],
                'model_name': model_name,
                'metadata': data['metadata']
            })

        return self.save_embeddings(records)

    def run(self, limit: int = None, batch_size: int = 200) -> dict:
        """Run vectorization process."""
        logger.info("Starting profile vectorization...")

        # Check embedding service
        if not self.embedding_service.is_available():
            logger.error("Embedding service not available")
            return {'error': 'Embedding service not available'}

        # Get unprocessed profiles
        profiles = self.get_unprocessed_profiles(limit)
        total = len(profiles)
        logger.info(f"Found {total} unprocessed profiles")

        if total == 0:
            return {'profiles': 0, 'embeddings': 0}

        # Process in batches
        total_embeddings = 0
        start_time = time.time()

        with tqdm(total=total, desc="Vectorizing profiles") as pbar:
            for i in range(0, total, batch_size):
                batch = profiles[i:i + batch_size]
                embeddings_saved = self.process_batch(batch)
                total_embeddings += embeddings_saved
                pbar.update(len(batch))
                pbar.set_postfix({'embeddings': total_embeddings})

        elapsed = time.time() - start_time
        rate = total_embeddings / elapsed if elapsed > 0 else 0

        result = {
            'profiles': total,
            'embeddings': total_embeddings,
            'elapsed_seconds': round(elapsed, 1),
            'rate_per_second': round(rate, 1)
        }

        logger.info(f"Vectorization complete: {result}")
        return result


def main():
    parser = argparse.ArgumentParser(description='Vectorize company profiles')
    parser.add_argument('--limit', type=int, help='Limit number of profiles to process')
    parser.add_argument('--batch-size', type=int, default=200, help='Batch size for processing')
    args = parser.parse_args()

    vectorizer = ProfileVectorizer()
    result = vectorizer.run(limit=args.limit, batch_size=args.batch_size)

    print("\n=== RESULTS ===")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
