"""
Portfolio RAG (Retrieval Augmented Generation) Module - LangChain Version.
Advanced semantic search over portfolio data using LangChain + ChromaDB.

Features:
- LangChain integration for advanced retrieval
- Multiple embedding options (HuggingFace free, OpenAI paid)
- Conversation memory for context
- Hybrid search (semantic + filters)
- SQL agent for complex queries
"""

import os
import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Any
import hashlib

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


class PortfolioRAG:
    """
    Advanced RAG system using LangChain + ChromaDB.
    Supports semantic search, conversation memory, and SQL queries.
    """

    COLLECTION_NAME = "portfolio_data"

    def __init__(self, persist_directory: str = "data/chroma_db"):
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.embeddings = None
        self.retriever = None
        self.llm = None
        self.qa_chain = None
        self._initialized = False

        self._init_components()

    def _init_components(self):
        """Initialize LangChain components."""
        try:
            # Create persist directory
            os.makedirs(self.persist_directory, exist_ok=True)

            # Initialize embeddings (free option with HuggingFace)
            self._init_embeddings()

            # Initialize vector store
            self._init_vectorstore()

            # Initialize LLM
            self._init_llm()

            # Create retrieval chain
            self._init_qa_chain()

            self._initialized = True
            logger.info("Portfolio RAG initialized with LangChain")

        except Exception as e:
            logger.error(f"Error initializing RAG: {e}")
            self._initialized = False

    def _init_embeddings(self):
        """Initialize embedding model (free HuggingFace or paid OpenAI)."""
        # Try HuggingFace embeddings first (free, local)
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Using HuggingFace embeddings (free, multilingual)")
            return
        except ImportError:
            pass

        # Fallback to OpenAI if available
        if os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import OpenAIEmbeddings
                self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                logger.info("Using OpenAI embeddings")
                return
            except ImportError:
                pass

        # Last resort: simple ChromaDB default
        logger.warning("Using default ChromaDB embeddings")
        self.embeddings = None

    def _init_vectorstore(self):
        """Initialize ChromaDB vector store."""
        from langchain_chroma import Chroma

        self.vectorstore = Chroma(
            collection_name=self.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )

    def _init_llm(self):
        """Initialize LLM for RAG responses."""
        # Try Claude first
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                from langchain_anthropic import ChatAnthropic
                self.llm = ChatAnthropic(
                    model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
                    temperature=0
                )
                logger.info("RAG using Claude LLM")
                return
            except ImportError:
                pass

        # Try Gemini
        if os.getenv("GOOGLE_API_KEY"):
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm = ChatGoogleGenerativeAI(
                    model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                    temperature=0
                )
                logger.info("RAG using Gemini LLM")
                return
            except ImportError:
                pass

        # Try Groq
        if os.getenv("GROQ_API_KEY"):
            try:
                from langchain_groq import ChatGroq
                self.llm = ChatGroq(
                    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                    temperature=0
                )
                logger.info("RAG using Groq LLM")
                return
            except ImportError:
                pass

        logger.warning("No LLM available for RAG chain")

    def _init_qa_chain(self):
        """Initialize the QA chain for retrieval-augmented answers."""
        if not self.llm or not self.retriever:
            return

        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            from langchain_core.output_parsers import StrOutputParser

            # RAG prompt template
            template = """Eres un asistente financiero experto analizando un portfolio de inversiones.

Usa el siguiente contexto para responder la pregunta. Si no encuentras la información en el contexto,
di que no tienes esa información disponible.

CONTEXTO DEL PORTFOLIO:
{context}

PREGUNTA: {question}

RESPUESTA (en español, clara y concisa):"""

            prompt = ChatPromptTemplate.from_template(template)

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            # Build RAG chain using LCEL (LangChain Expression Language)
            self.qa_chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
            )
            logger.info("QA chain initialized with LCEL")

        except Exception as e:
            logger.error(f"Error creating QA chain: {e}")

    def is_available(self) -> bool:
        """Check if RAG system is available."""
        return self._initialized and self.vectorstore is not None

    def _generate_id(self, text: str) -> str:
        """Generate unique ID for a document."""
        return hashlib.md5(text.encode()).hexdigest()[:16]

    # =========================================================================
    # INDEXING METHODS
    # =========================================================================

    def index_holdings(self, fecha: date = None) -> int:
        """Index current holdings from database."""
        if not self.is_available():
            return 0

        from src.database import get_db_manager
        from sqlalchemy import text
        from langchain_core.documents import Document

        fecha = fecha or date.today()
        db = get_db_manager()

        documents = []

        with db.get_session() as session:
            result = session.execute(text("""
                SELECT account_code, symbol, shares, currency, asset_type
                FROM holding_diario
                WHERE fecha = :fecha
            """), {'fecha': fecha})

            for row in result.fetchall():
                account, symbol, shares, currency, asset_type = row

                position_type = "posición larga" if shares > 0 else "posición corta"
                content = (
                    f"Cuenta {account}: {abs(shares)} acciones de {symbol} "
                    f"({position_type}) en {currency}. "
                    f"Tipo de activo: {asset_type or 'Otros'}."
                )

                doc = Document(
                    page_content=content,
                    metadata={
                        "type": "holding",
                        "account": account,
                        "symbol": symbol,
                        "shares": int(shares),
                        "currency": currency,
                        "asset_type": asset_type or "Otros",
                        "fecha": str(fecha),
                        "position_type": "long" if shares > 0 else "short"
                    }
                )
                documents.append(doc)

        if documents:
            # Clear existing holdings and add new
            self._delete_by_metadata("type", "holding")
            self.vectorstore.add_documents(documents)
            logger.info(f"Indexed {len(documents)} holdings for {fecha}")

        return len(documents)

    def index_trades(self, days: int = 90) -> int:
        """Index recent trades from database."""
        if not self.is_available():
            return 0

        from src.database import get_db_manager
        from sqlalchemy import text
        from langchain_core.documents import Document

        db = get_db_manager()
        start_date = date.today() - timedelta(days=days)

        documents = []

        with db.get_session() as session:
            result = session.execute(text("""
                SELECT t.trade_date, t.account_code, t.symbol, t.trade_type,
                       t.shares, t.price, t.currency, t.commission
                FROM stock_trades t
                WHERE t.trade_date >= :start_date
                ORDER BY t.trade_date DESC
            """), {'start_date': start_date})

            for row in result.fetchall():
                trade_date, account, symbol, trade_type, shares, price, currency, commission = row

                action = "compró" if trade_type == 'BUY' else "vendió"
                content = (
                    f"El {trade_date}, en cuenta {account}, se {action} "
                    f"{shares} acciones de {symbol} a {price} {currency}. "
                    f"Comisión: {commission or 0} {currency}."
                )

                doc = Document(
                    page_content=content,
                    metadata={
                        "type": "trade",
                        "account": account,
                        "symbol": symbol,
                        "trade_type": trade_type,
                        "shares": int(shares) if shares else 0,
                        "price": float(price) if price else 0,
                        "currency": currency or "USD",
                        "fecha": str(trade_date)
                    }
                )
                documents.append(doc)

        if documents:
            self._delete_by_metadata("type", "trade")
            self.vectorstore.add_documents(documents)
            logger.info(f"Indexed {len(documents)} trades")

        return len(documents)

    def index_cash(self, fecha: date = None) -> int:
        """Index cash positions from database."""
        if not self.is_available():
            return 0

        from src.database import get_db_manager
        from sqlalchemy import text
        from langchain_core.documents import Document

        fecha = fecha or date.today()
        db = get_db_manager()

        documents = []

        with db.get_session() as session:
            result = session.execute(text("""
                SELECT account_code, currency, saldo
                FROM cash_diario
                WHERE fecha = :fecha
            """), {'fecha': fecha})

            for row in result.fetchall():
                account, currency, saldo = row

                content = f"Cuenta {account}: {saldo:,.2f} {currency} en efectivo."

                doc = Document(
                    page_content=content,
                    metadata={
                        "type": "cash",
                        "account": account,
                        "currency": currency or "EUR",
                        "balance": float(saldo) if saldo else 0,
                        "fecha": str(fecha)
                    }
                )
                documents.append(doc)

        if documents:
            self._delete_by_metadata("type", "cash")
            self.vectorstore.add_documents(documents)
            logger.info(f"Indexed {len(documents)} cash positions for {fecha}")

        return len(documents)

    def index_portfolio_summary(self, fecha: date = None) -> int:
        """Index portfolio summary by account."""
        if not self.is_available():
            return 0

        from src.database import get_db_manager
        from sqlalchemy import text
        from langchain_core.documents import Document

        fecha = fecha or date.today()
        db = get_db_manager()

        documents = []

        with db.get_session() as session:
            result = session.execute(text("""
                SELECT account_code,
                       COUNT(DISTINCT symbol) as num_positions,
                       SUM(CASE WHEN shares > 0 THEN 1 ELSE 0 END) as long_positions,
                       SUM(CASE WHEN shares < 0 THEN 1 ELSE 0 END) as short_positions
                FROM holding_diario
                WHERE fecha = :fecha
                GROUP BY account_code
            """), {'fecha': fecha})

            for row in result.fetchall():
                account, num_positions, long_pos, short_pos = row

                content = (
                    f"Resumen cuenta {account} ({fecha}): "
                    f"{num_positions} posiciones totales, "
                    f"{long_pos} posiciones largas, "
                    f"{short_pos} posiciones cortas."
                )

                doc = Document(
                    page_content=content,
                    metadata={
                        "type": "summary",
                        "account": account,
                        "num_positions": int(num_positions),
                        "long_positions": int(long_pos),
                        "short_positions": int(short_pos),
                        "fecha": str(fecha)
                    }
                )
                documents.append(doc)

        if documents:
            self._delete_by_metadata("type", "summary")
            self.vectorstore.add_documents(documents)
            logger.info(f"Indexed {len(documents)} portfolio summaries")

        return len(documents)

    def _get_latest_date(self) -> Optional[date]:
        """Get the latest date with data in the database."""
        try:
            from src.database import get_db_manager
            from sqlalchemy import text

            db = get_db_manager()
            with db.get_session() as session:
                result = session.execute(text(
                    "SELECT MAX(fecha) FROM holding_diario"
                ))
                row = result.fetchone()
                if row and row[0]:
                    # Parse date string to date object
                    if isinstance(row[0], str):
                        return datetime.strptime(row[0], "%Y-%m-%d").date()
                    return row[0]
        except Exception as e:
            logger.warning(f"Could not get latest date: {e}")
        return None

    def index_all(self, fecha: date = None, trade_days: int = 90) -> Dict[str, int]:
        """Index all portfolio data."""
        # Use provided date, or latest available, or today
        if fecha is None:
            fecha = self._get_latest_date() or date.today()
            logger.info(f"Using date: {fecha}")

        return {
            "holdings": self.index_holdings(fecha),
            "trades": self.index_trades(trade_days),
            "cash": self.index_cash(fecha),
            "summaries": self.index_portfolio_summary(fecha)
        }

    # =========================================================================
    # SEARCH METHODS
    # =========================================================================

    def search(self, query: str, n_results: int = 10,
               filter_type: str = None, filter_account: str = None) -> List[Dict]:
        """
        Search portfolio data using semantic search.

        Args:
            query: Natural language query
            n_results: Maximum number of results
            filter_type: Filter by type (holding, trade, cash, summary)
            filter_account: Filter by account code

        Returns:
            List of matching documents with metadata
        """
        if not self.is_available():
            return []

        try:
            # Build filter
            search_filter = {}
            if filter_type:
                search_filter["type"] = filter_type
            if filter_account:
                search_filter["account"] = filter_account

            if search_filter:
                docs = self.vectorstore.similarity_search(
                    query, k=n_results, filter=search_filter
                )
            else:
                docs = self.vectorstore.similarity_search(query, k=n_results)

            return [
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]

        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question using the RAG chain (retrieval + LLM answer).

        Args:
            question: Natural language question

        Returns:
            Dictionary with 'answer' and 'sources'
        """
        # Get source documents first
        sources = self.search(question, n_results=5)

        if not self.qa_chain:
            # Fallback to simple search (no LLM)
            context = "\n".join([r['text'] for r in sources])
            return {
                "answer": f"Contexto encontrado:\n{context}",
                "sources": sources
            }

        try:
            # Use the LCEL chain
            answer = self.qa_chain.invoke(question)
            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            logger.error(f"QA chain error: {e}")
            return {"answer": f"Error: {str(e)}", "sources": sources}

    def get_context_for_query(self, query: str, max_tokens: int = 2000) -> str:
        """
        Get relevant context for a query, formatted for LLM consumption.

        Args:
            query: User's question
            max_tokens: Approximate maximum length

        Returns:
            Formatted context string
        """
        if not self.is_available():
            return ""

        results = self.search(query, n_results=15)

        if not results:
            return ""

        context_parts = ["=== CONTEXTO DEL PORTFOLIO (RAG) ===\n"]

        # Group by type
        by_type = {}
        for r in results:
            doc_type = r['metadata'].get('type', 'other')
            if doc_type not in by_type:
                by_type[doc_type] = []
            by_type[doc_type].append(r)

        type_labels = {
            'holding': 'POSICIONES',
            'trade': 'OPERACIONES',
            'cash': 'EFECTIVO',
            'summary': 'RESUMEN'
        }

        for doc_type, docs in by_type.items():
            label = type_labels.get(doc_type, doc_type.upper())
            context_parts.append(f"\n[{label}]")
            for d in docs[:5]:
                context_parts.append(f"- {d['text']}")

        context = "\n".join(context_parts)

        max_chars = max_tokens * 4
        if len(context) > max_chars:
            context = context[:max_chars] + "\n... (truncado)"

        return context

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _delete_by_metadata(self, key: str, value: str):
        """Delete documents by metadata filter."""
        if not self.is_available():
            return

        try:
            # Get IDs to delete
            results = self.vectorstore._collection.get(
                where={key: value}
            )
            if results and results['ids']:
                self.vectorstore._collection.delete(ids=results['ids'])
        except Exception as e:
            logger.warning(f"Could not delete documents: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed data."""
        if not self.is_available():
            return {"available": False}

        try:
            count = self.vectorstore._collection.count()
            return {
                "available": True,
                "total_documents": count,
                "persist_directory": self.persist_directory,
                "has_qa_chain": self.qa_chain is not None,
                "embedding_type": type(self.embeddings).__name__ if self.embeddings else "default"
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    def clear_all(self):
        """Clear all indexed data."""
        if not self.is_available():
            return

        try:
            # Delete and recreate collection
            from langchain_chroma import Chroma

            self.vectorstore._client.delete_collection(self.COLLECTION_NAME)
            self.vectorstore = Chroma(
                collection_name=self.COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
            logger.info("Cleared all indexed data")
        except Exception as e:
            logger.error(f"Error clearing data: {e}")


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_rag_instance: Optional[PortfolioRAG] = None


def get_portfolio_rag() -> PortfolioRAG:
    """Get singleton instance of PortfolioRAG."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = PortfolioRAG()
    return _rag_instance


# =============================================================================
# CLI FOR TESTING
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=== Portfolio RAG Module (LangChain) ===\n")

    rag = get_portfolio_rag()

    if not rag.is_available():
        print("ERROR: RAG not available")
        sys.exit(1)

    print(f"Stats: {rag.get_stats()}")

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "index":
            print("\nIndexing all portfolio data...")
            results = rag.index_all()
            print(f"Indexed: {results}")

        elif cmd == "search":
            query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "posiciones AAPL"
            print(f"\nSearching: {query}")
            results = rag.search(query)
            for r in results:
                print(f"  - {r['text'][:100]}...")

        elif cmd == "ask":
            question = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "cuanto efectivo tengo"
            print(f"\nQuestion: {question}")
            result = rag.ask(question)
            print(f"\nAnswer: {result['answer']}")
            print(f"\nSources: {len(result['sources'])} documents")

        elif cmd == "clear":
            print("Clearing all data...")
            rag.clear_all()
            print("Done")
