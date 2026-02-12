"""
Financial RAG (Retrieval Augmented Generation) Module.
Unified RAG system combining:
- Earnings transcripts (pgvector)
- Company profiles (pgvector)
- Portfolio data (ChromaDB)
- SQL data (prices, fundamentals)
"""

import os
import logging
from datetime import date, datetime
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


class FinancialRAG:
    """
    Unified RAG system for financial data.

    Combines semantic search over:
    - 42K+ earnings call transcripts
    - 92K+ company profiles
    - Portfolio positions and trades
    - Market data and fundamentals
    """

    def __init__(self):
        """Initialize Financial RAG."""
        self.search_service = None
        self.portfolio_rag = None
        self.db = None
        self.llm = None
        self._initialized = False

        self._init_components()

    def _init_components(self):
        """Initialize all RAG components."""
        # Initialize pgvector search service
        try:
            from src.nlp.services.financial_search_service import get_financial_search_service
            self.search_service = get_financial_search_service()
            if self.search_service.is_available():
                logger.info("Financial search service initialized (pgvector)")
            else:
                logger.warning("Financial search service not available")
        except Exception as e:
            logger.warning(f"Could not initialize search service: {e}")

        # Initialize portfolio RAG (ChromaDB)
        try:
            from src.portfolio_rag import get_portfolio_rag
            self.portfolio_rag = get_portfolio_rag()
            if self.portfolio_rag.is_available():
                logger.info("Portfolio RAG initialized (ChromaDB)")
        except Exception as e:
            logger.warning(f"Could not initialize portfolio RAG: {e}")

        # Initialize database analyzer
        try:
            from src.db_analysis_tools import DatabaseAnalyzer
            self.db = DatabaseAnalyzer()
            logger.info("Database analyzer initialized")
        except Exception as e:
            logger.warning(f"Could not initialize database analyzer: {e}")

        # Initialize LLM
        self._init_llm()

        self._initialized = (
            (self.search_service and self.search_service.is_available()) or
            (self.portfolio_rag and self.portfolio_rag.is_available())
        )

    def _init_llm(self):
        """Initialize LLM for RAG responses."""
        # Try Claude first
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                import anthropic
                self.llm = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                self.llm_type = "claude"
                logger.info("Using Claude for RAG")
                return
            except ImportError:
                pass

        # Try Gemini
        if os.getenv("GOOGLE_API_KEY"):
            try:
                from google import genai
                self.llm = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
                self.llm_type = "gemini"
                logger.info("Using Gemini for RAG")
                return
            except ImportError:
                pass

        logger.warning("No LLM available for RAG responses")
        self.llm = None
        self.llm_type = None

    def is_available(self) -> bool:
        """Check if RAG system is available."""
        return self._initialized

    # =========================================================================
    # SEARCH METHODS
    # =========================================================================

    def search_transcripts(
        self,
        query: str,
        symbol: str = None,
        year: int = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Search earnings call transcripts.

        Args:
            query: Search query
            symbol: Filter by stock symbol
            year: Filter by year
            limit: Maximum results

        Returns:
            List of search results
        """
        if not self.search_service or not self.search_service.is_available():
            return []

        response = self.search_service.search_transcripts(
            query=query,
            symbol=symbol,
            year=year,
            limit=limit
        )

        return [
            {
                'text': r.text,
                'symbol': r.symbol,
                'score': r.score,
                'year': r.metadata.get('year'),
                'quarter': r.metadata.get('quarter'),
                'section': r.metadata.get('section'),
                'source': 'transcript'
            }
            for r in response.results
        ]

    def search_companies(
        self,
        query: str,
        sector: str = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Search company profiles.

        Args:
            query: Search query
            sector: Filter by sector
            limit: Maximum results

        Returns:
            List of search results
        """
        if not self.search_service or not self.search_service.is_available():
            return []

        response = self.search_service.search_companies(
            query=query,
            sector=sector,
            limit=limit
        )

        return [
            {
                'text': r.text,
                'symbol': r.symbol,
                'score': r.score,
                'sector': r.metadata.get('sector'),
                'industry': r.metadata.get('industry'),
                'source': 'profile'
            }
            for r in response.results
        ]

    def search_portfolio(
        self,
        query: str,
        limit: int = 5
    ) -> List[Dict]:
        """
        Search portfolio data.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of search results
        """
        if not self.portfolio_rag or not self.portfolio_rag.is_available():
            return []

        results = self.portfolio_rag.search(query, n_results=limit)

        return [
            {
                'text': r['text'],
                'symbol': r['metadata'].get('symbol'),
                'score': 1.0,  # ChromaDB doesn't return scores in same way
                'type': r['metadata'].get('type'),
                'account': r['metadata'].get('account'),
                'source': 'portfolio'
            }
            for r in results
        ]

    def search_all(
        self,
        query: str,
        symbol: str = None,
        limit: int = 10
    ) -> List[Dict]:
        """
        Search across all sources.

        Args:
            query: Search query
            symbol: Filter by stock symbol
            limit: Maximum results per source

        Returns:
            Combined list of search results
        """
        results = []

        # Search transcripts
        transcript_results = self.search_transcripts(query, symbol=symbol, limit=limit)
        results.extend(transcript_results)

        # Search companies
        company_results = self.search_companies(query, limit=limit)
        results.extend(company_results)

        # Search portfolio
        portfolio_results = self.search_portfolio(query, limit=limit)
        results.extend(portfolio_results)

        # Sort by score
        results.sort(key=lambda x: x.get('score', 0), reverse=True)

        return results[:limit * 2]  # Return top results

    # =========================================================================
    # CONTEXT BUILDING
    # =========================================================================

    def get_context_for_query(
        self,
        query: str,
        include_transcripts: bool = True,
        include_companies: bool = True,
        include_portfolio: bool = True,
        include_fundamentals: bool = True,
        max_results: int = 5
    ) -> str:
        """
        Build context for RAG from all sources.

        Args:
            query: User's question
            include_transcripts: Include earnings call context
            include_companies: Include company profile context
            include_portfolio: Include portfolio context
            include_fundamentals: Include fundamental data
            max_results: Maximum results per source

        Returns:
            Formatted context string
        """
        context_parts = []

        # Extract symbol from query if present
        import re
        symbol_match = re.search(r'\b([A-Z]{1,5})\b', query)
        symbol = symbol_match.group(1) if symbol_match else None
        if symbol in ['Q1', 'Q2', 'Q3', 'Q4', 'EUR', 'USD', 'ETF', 'CEO']:
            symbol = None

        # Transcript context
        if include_transcripts:
            transcripts = self.search_transcripts(query, symbol=symbol, limit=max_results)
            if transcripts:
                context_parts.append("\n=== EARNINGS CALL TRANSCRIPTS ===")
                for t in transcripts[:3]:
                    period = f"Q{t.get('quarter', '?')} {t.get('year', '?')}"
                    section = t.get('section', '')
                    context_parts.append(
                        f"\n[{t['symbol']} - {period}] ({section}, relevance: {t['score']:.2f})"
                    )
                    context_parts.append(t['text'][:800])

        # Company context
        if include_companies:
            companies = self.search_companies(query, limit=max_results)
            if companies:
                context_parts.append("\n=== COMPANY PROFILES ===")
                for c in companies[:3]:
                    context_parts.append(
                        f"\n[{c['symbol']}] (relevance: {c['score']:.2f})"
                    )
                    context_parts.append(c['text'][:600])

        # Portfolio context
        if include_portfolio and self.portfolio_rag:
            portfolio_context = self.portfolio_rag.get_context_for_query(query, max_tokens=1000)
            if portfolio_context:
                context_parts.append("\n" + portfolio_context)

        # Fundamental data context
        if include_fundamentals and self.db and symbol:
            try:
                info = self.db.get_symbol_info(symbol)
                if info:
                    context_parts.append(f"\n=== FUNDAMENTALS {symbol} ===")
                    context_parts.append(f"Company: {info.get('name')}")
                    context_parts.append(f"Sector: {info.get('sector')}")
                    context_parts.append(f"Market Cap: ${info.get('market_cap_b', 0):.1f}B")
                    context_parts.append(f"P/E Ratio: {info.get('pe_ratio', 'N/A')}")

                tech = self.db.get_technical_indicators(symbol)
                if tech and 'error' not in tech:
                    context_parts.append(f"\n=== TECHNICAL {symbol} ===")
                    for k, v in list(tech.items())[:10]:
                        if v is not None:
                            context_parts.append(f"  {k}: {v}")
            except Exception as e:
                logger.debug(f"Could not get fundamentals: {e}")

        return "\n".join(context_parts)

    # =========================================================================
    # RAG ANSWER
    # =========================================================================

    def ask(
        self,
        question: str,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Ask a question using RAG (retrieval + LLM answer).

        Args:
            question: Natural language question
            include_sources: Include source documents in response

        Returns:
            Dictionary with 'answer' and optionally 'sources'
        """
        # Build context
        context = self.get_context_for_query(question)

        if not context:
            return {
                'answer': "No encontre informacion relevante para tu pregunta.",
                'sources': []
            }

        # Get sources for reference
        sources = self.search_all(question, limit=5) if include_sources else []

        # If no LLM, return context directly
        if not self.llm:
            return {
                'answer': f"Contexto encontrado:\n{context[:2000]}",
                'sources': sources
            }

        # Generate answer with LLM
        try:
            system_prompt = """Eres un asistente financiero experto con acceso a:
- Transcripciones de earnings calls de empresas
- Perfiles de empresas
- Datos de portfolio personal
- Metricas fundamentales y tecnicas

Responde en espanol de forma clara y concisa.
Basa tus respuestas UNICAMENTE en el contexto proporcionado.
Si no encuentras la informacion, dilo claramente."""

            user_prompt = f"""CONTEXTO:
{context}

PREGUNTA: {question}

RESPUESTA:"""

            if self.llm_type == "claude":
                response = self.llm.messages.create(
                    model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
                    max_tokens=2048,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                answer = response.content[0].text

            elif self.llm_type == "gemini":
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = self.llm.models.generate_content(
                    model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                    contents=full_prompt
                )
                answer = response.text

            else:
                answer = f"Contexto encontrado:\n{context[:2000]}"

            return {
                'answer': answer,
                'sources': sources
            }

        except Exception as e:
            logger.error(f"RAG answer error: {e}")
            return {
                'answer': f"Error generando respuesta: {str(e)}",
                'sources': sources
            }

    # =========================================================================
    # STATS AND UTILITIES
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed data."""
        stats = {
            'available': self._initialized,
            'components': {}
        }

        # Search service stats
        if self.search_service:
            try:
                search_stats = self.search_service.get_stats()
                stats['components']['pgvector'] = search_stats
            except Exception as e:
                stats['components']['pgvector'] = {'error': str(e)}

        # Portfolio RAG stats
        if self.portfolio_rag:
            try:
                portfolio_stats = self.portfolio_rag.get_stats()
                stats['components']['chromadb'] = portfolio_stats
            except Exception as e:
                stats['components']['chromadb'] = {'error': str(e)}

        # Database stats
        if self.db:
            try:
                market_stats = self.db.get_market_stats()
                stats['components']['database'] = market_stats
            except Exception as e:
                stats['components']['database'] = {'error': str(e)}

        # LLM info
        stats['llm'] = {
            'type': self.llm_type,
            'available': self.llm is not None
        }

        return stats

    def refresh_portfolio_index(self) -> Dict[str, int]:
        """Refresh portfolio RAG index."""
        if not self.portfolio_rag or not self.portfolio_rag.is_available():
            return {'error': 'Portfolio RAG not available'}

        return self.portfolio_rag.index_all()


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_rag_instance: Optional[FinancialRAG] = None


def get_financial_rag() -> FinancialRAG:
    """Get singleton instance of FinancialRAG."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = FinancialRAG()
    return _rag_instance


# =============================================================================
# CLI FOR TESTING
# =============================================================================

if __name__ == "__main__":
    import sys

    print("=== Financial RAG Module ===\n")

    rag = get_financial_rag()

    print(f"Available: {rag.is_available()}")
    print(f"Stats: {rag.get_stats()}")

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "search":
            query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Apple iPhone revenue"
            print(f"\nSearching: {query}")
            results = rag.search_all(query)
            for r in results[:5]:
                print(f"  [{r['source']}] {r['symbol']}: {r['text'][:100]}...")

        elif cmd == "ask":
            question = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Que dijo Apple sobre iPhone?"
            print(f"\nQuestion: {question}")
            result = rag.ask(question)
            print(f"\nAnswer: {result['answer']}")
            print(f"\nSources: {len(result['sources'])} documents")

        elif cmd == "transcripts":
            query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "AI artificial intelligence growth"
            print(f"\nSearching transcripts: {query}")
            results = rag.search_transcripts(query, limit=5)
            for r in results:
                print(f"  [{r['symbol']} Q{r['quarter']} {r['year']}] {r['text'][:150]}...")

        elif cmd == "companies":
            query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "semiconductor chips"
            print(f"\nSearching companies: {query}")
            results = rag.search_companies(query, limit=5)
            for r in results:
                print(f"  [{r['symbol']}] {r['text'][:150]}...")
    else:
        print("\nUsage:")
        print("  py -3 src/financial_rag.py search <query>")
        print("  py -3 src/financial_rag.py ask <question>")
        print("  py -3 src/financial_rag.py transcripts <query>")
        print("  py -3 src/financial_rag.py companies <query>")
