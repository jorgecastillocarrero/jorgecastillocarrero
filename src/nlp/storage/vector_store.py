"""
Vector Store Abstraction.
Supports ChromaDB, pgvector, and future vector databases.
"""

import logging
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import numpy as np

from ..config import get_nlp_settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store abstraction for semantic search.

    Supports:
    - ChromaDB (local, default)
    - pgvector (PostgreSQL)
    - Pinecone/Weaviate (future)
    """

    def __init__(
        self,
        store_type: Optional[str] = None,
        collection_name: str = "nlp_embeddings"
    ):
        """
        Initialize vector store.

        Args:
            store_type: Type of store ('chromadb', 'pgvector')
            collection_name: Name of the collection
        """
        settings = get_nlp_settings()
        self.store_type = store_type or settings.vector_store_type
        self.collection_name = collection_name

        self._client = None
        self._collection = None
        self._is_initialized = False

    def _ensure_initialized(self) -> bool:
        """Ensure store is initialized."""
        if self._is_initialized:
            return True

        if self.store_type == 'chromadb':
            return self._init_chromadb()
        elif self.store_type == 'pgvector':
            return self._init_pgvector()
        else:
            logger.error(f"Unknown vector store type: {self.store_type}")
            return False

    def _init_chromadb(self) -> bool:
        """Initialize ChromaDB."""
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            settings = get_nlp_settings()
            persist_dir = Path(settings.chromadb_path)
            persist_dir.mkdir(parents=True, exist_ok=True)

            self._client = chromadb.Client(ChromaSettings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(persist_dir),
                anonymized_telemetry=False
            ))

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            self._is_initialized = True
            logger.info(f"ChromaDB initialized at {persist_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            return False

    def _init_pgvector(self) -> bool:
        """Initialize pgvector (uses PostgresStorage)."""
        try:
            from .postgres import PostgresStorage

            self._client = PostgresStorage()

            if not self._client.has_pgvector():
                logger.warning("pgvector extension not available")
                return False

            self._is_initialized = True
            logger.info("pgvector initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize pgvector: {e}")
            return False

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict]] = None
    ) -> bool:
        """
        Add embeddings to the store.

        Args:
            ids: List of unique identifiers
            embeddings: List of embedding vectors
            documents: Optional list of source documents
            metadatas: Optional list of metadata dicts

        Returns:
            True if successful
        """
        if not self._ensure_initialized():
            return False

        try:
            if self.store_type == 'chromadb':
                self._collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
            elif self.store_type == 'pgvector':
                records = []
                for i, (id_, emb) in enumerate(zip(ids, embeddings)):
                    meta = metadatas[i] if metadatas else {}
                    records.append({
                        'source_type': meta.get('source_type', ''),
                        'source_id': meta.get('source_id'),
                        'symbol': meta.get('symbol', ''),
                        'embedding': emb,
                        'model_name': meta.get('model_name', '')
                    })
                self._client.save_embeddings(records)

            return True

        except Exception as e:
            logger.error(f"Error adding to vector store: {e}")
            return False

    def query(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results
            where: Optional filter conditions

        Returns:
            Dict with ids, distances, documents, metadatas
        """
        if not self._ensure_initialized():
            return {'ids': [], 'distances': [], 'documents': [], 'metadatas': []}

        try:
            if self.store_type == 'chromadb':
                results = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    where=where
                )
                return {
                    'ids': results['ids'][0] if results['ids'] else [],
                    'distances': results['distances'][0] if results.get('distances') else [],
                    'documents': results['documents'][0] if results.get('documents') else [],
                    'metadatas': results['metadatas'][0] if results.get('metadatas') else []
                }

            elif self.store_type == 'pgvector':
                source_type = where.get('source_type') if where else None
                symbol = where.get('symbol') if where else None

                results = self._client.search_similar(
                    query_embedding,
                    source_type=source_type,
                    symbol=symbol,
                    k=n_results
                )

                return {
                    'ids': [str(r['id']) for r in results],
                    'distances': [1 - r['similarity'] for r in results],
                    'documents': [],
                    'metadatas': results
                }

        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            return {'ids': [], 'distances': [], 'documents': [], 'metadatas': []}

    def search_text(
        self,
        text: str,
        n_results: int = 10,
        where: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Search by text (requires embedding service).

        Args:
            text: Query text
            n_results: Number of results
            where: Optional filter conditions

        Returns:
            Dict with search results
        """
        from ..services.embedding_service import get_embedding_service

        service = get_embedding_service()
        embedding = service.embed(text)

        if embedding is None:
            return {'ids': [], 'distances': [], 'documents': [], 'metadatas': []}

        return self.query(
            embedding.tolist(),
            n_results=n_results,
            where=where
        )

    def search_hybrid(
        self,
        text: str,
        n_results: int = 10,
        source_type: Optional[str] = None,
        symbol: Optional[str] = None,
        year: Optional[int] = None,
        quarter: Optional[str] = None,
        section: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Hybrid search combining semantic search with SQL filters.

        Args:
            text: Query text
            n_results: Number of results
            source_type: Filter by source ('transcript', 'profile')
            symbol: Filter by stock symbol
            year: Filter by year
            quarter: Filter by quarter
            section: Filter by section

        Returns:
            Dict with results including text_content and metadata
        """
        if self.store_type != 'pgvector':
            # Fall back to standard search for non-pgvector stores
            where = {}
            if source_type:
                where['source_type'] = source_type
            if symbol:
                where['symbol'] = symbol
            return self.search_text(text, n_results, where)

        from ..services.embedding_service import get_embedding_service

        service = get_embedding_service()
        embedding = service.embed(text)

        if embedding is None:
            return {'results': [], 'total': 0}

        if not self._ensure_initialized():
            return {'results': [], 'total': 0}

        try:
            conn = self._client.get_connection()
            cur = conn.cursor()

            # Build hybrid query
            sql = """
                SELECT
                    e.id,
                    e.source_type,
                    e.symbol,
                    e.text_content,
                    e.year,
                    e.quarter,
                    e.section,
                    e.metadata,
                    1 - (e.embedding <=> %s::vector) as similarity
                FROM nlp_embeddings e
                WHERE 1=1
            """
            params = [embedding.tolist()]

            if source_type:
                sql += " AND e.source_type = %s"
                params.append(source_type)

            if symbol:
                sql += " AND e.symbol = %s"
                params.append(symbol.upper())

            if year:
                sql += " AND e.year = %s"
                params.append(year)

            if quarter:
                sql += " AND e.quarter = %s"
                params.append(quarter)

            if section:
                sql += " AND e.section = %s"
                params.append(section)

            sql += " ORDER BY e.embedding <=> %s::vector LIMIT %s"
            params.extend([embedding.tolist(), n_results])

            cur.execute(sql, params)

            results = []
            for row in cur.fetchall():
                results.append({
                    'id': row[0],
                    'source_type': row[1],
                    'symbol': row[2],
                    'text_content': row[3],
                    'year': row[4],
                    'quarter': row[5],
                    'section': row[6],
                    'metadata': row[7] if row[7] else {},
                    'similarity': float(row[8]) if row[8] else 0
                })

            conn.close()

            return {
                'results': results,
                'total': len(results)
            }

        except Exception as e:
            logger.error(f"Hybrid search error: {e}")
            return {'results': [], 'total': 0}

    def count(self) -> int:
        """Get number of embeddings in store."""
        if not self._ensure_initialized():
            return 0

        try:
            if self.store_type == 'chromadb':
                return self._collection.count()
            elif self.store_type == 'pgvector':
                stats = self._client.get_processing_stats()
                return stats.get('embeddings_stored', 0)
        except Exception:
            return 0

    def delete(self, ids: List[str]) -> bool:
        """
        Delete embeddings by ID.

        Args:
            ids: List of IDs to delete

        Returns:
            True if successful
        """
        if not self._ensure_initialized():
            return False

        try:
            if self.store_type == 'chromadb':
                self._collection.delete(ids=ids)
                return True
            elif self.store_type == 'pgvector':
                # Would need to implement deletion in PostgresStorage
                logger.warning("pgvector deletion not implemented")
                return False

        except Exception as e:
            logger.error(f"Error deleting from vector store: {e}")
            return False

    def persist(self) -> bool:
        """Persist store to disk (for ChromaDB)."""
        if not self._is_initialized:
            return False

        try:
            if self.store_type == 'chromadb' and hasattr(self._client, 'persist'):
                self._client.persist()
            return True
        except Exception as e:
            logger.error(f"Error persisting vector store: {e}")
            return False


def get_vector_store(
    store_type: Optional[str] = None,
    collection_name: str = "nlp_embeddings"
) -> VectorStore:
    """
    Get a vector store instance.

    Args:
        store_type: Type of store
        collection_name: Collection name

    Returns:
        VectorStore instance
    """
    return VectorStore(store_type, collection_name)
