"""
Cache Manager.
Provides caching for sentiment and embedding results.
"""

import logging
import json
import hashlib
from typing import Optional, Any, Dict
from datetime import datetime, timedelta

from ..config import get_nlp_settings

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Cache manager for NLP results.

    Supports:
    - In-memory cache (default)
    - Redis (for distributed caching)
    """

    def __init__(self, prefix: str = "nlp"):
        """
        Initialize cache manager.

        Args:
            prefix: Key prefix for cache entries
        """
        self.settings = get_nlp_settings()
        self.prefix = prefix
        self._redis_client = None
        self._memory_cache: Dict[str, Dict] = {}
        self._is_redis_available = None

    def _get_redis(self):
        """Get Redis client if available."""
        if not self.settings.enable_cache:
            return None

        if self._is_redis_available is False:
            return None

        if self._redis_client is not None:
            return self._redis_client

        if not self.settings.redis_url:
            self._is_redis_available = False
            return None

        try:
            import redis
            self._redis_client = redis.from_url(self.settings.redis_url)
            self._redis_client.ping()
            self._is_redis_available = True
            logger.info("Redis cache connected")
            return self._redis_client
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self._is_redis_available = False
            return None

    def _make_key(self, key: str) -> str:
        """Create full cache key."""
        return f"{self.prefix}:{key}"

    def _hash_text(self, text: str) -> str:
        """Create hash of text for cache key."""
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        full_key = self._make_key(key)

        # Try Redis first
        redis_client = self._get_redis()
        if redis_client:
            try:
                value = redis_client.get(full_key)
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.warning(f"Redis get error: {e}")

        # Fall back to memory cache
        if full_key in self._memory_cache:
            entry = self._memory_cache[full_key]
            if entry['expires_at'] > datetime.now():
                return entry['value']
            else:
                del self._memory_cache[full_key]

        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds

        Returns:
            True if successful
        """
        full_key = self._make_key(key)
        ttl = ttl_seconds or self.settings.cache_ttl_seconds

        # Try Redis first
        redis_client = self._get_redis()
        if redis_client:
            try:
                redis_client.setex(
                    full_key,
                    ttl,
                    json.dumps(value)
                )
                return True
            except Exception as e:
                logger.warning(f"Redis set error: {e}")

        # Fall back to memory cache
        self._memory_cache[full_key] = {
            'value': value,
            'expires_at': datetime.now() + timedelta(seconds=ttl)
        }

        # Cleanup old entries periodically
        self._cleanup_memory_cache()

        return True

    def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        full_key = self._make_key(key)

        redis_client = self._get_redis()
        if redis_client:
            try:
                redis_client.delete(full_key)
            except Exception:
                pass

        if full_key in self._memory_cache:
            del self._memory_cache[full_key]

        return True

    def get_sentiment(self, text: str) -> Optional[Dict]:
        """
        Get cached sentiment result for text.

        Args:
            text: Input text

        Returns:
            Cached sentiment dict or None
        """
        key = f"sentiment:{self._hash_text(text)}"
        return self.get(key)

    def set_sentiment(self, text: str, result: Dict) -> bool:
        """
        Cache sentiment result for text.

        Args:
            text: Input text
            result: Sentiment result dict

        Returns:
            True if cached
        """
        key = f"sentiment:{self._hash_text(text)}"
        return self.set(key, result)

    def get_embedding(self, text: str) -> Optional[list]:
        """
        Get cached embedding for text.

        Args:
            text: Input text

        Returns:
            Cached embedding list or None
        """
        key = f"embedding:{self._hash_text(text)}"
        return self.get(key)

    def set_embedding(self, text: str, embedding: list) -> bool:
        """
        Cache embedding for text.

        Args:
            text: Input text
            embedding: Embedding list

        Returns:
            True if cached
        """
        key = f"embedding:{self._hash_text(text)}"
        # Longer TTL for embeddings (less likely to change)
        return self.set(key, embedding, ttl_seconds=86400)

    def _cleanup_memory_cache(self) -> None:
        """Clean up expired memory cache entries."""
        if len(self._memory_cache) < 1000:
            return

        now = datetime.now()
        expired_keys = [
            k for k, v in self._memory_cache.items()
            if v['expires_at'] <= now
        ]

        for key in expired_keys:
            del self._memory_cache[key]

        # If still too many, remove oldest
        if len(self._memory_cache) > 5000:
            sorted_keys = sorted(
                self._memory_cache.keys(),
                key=lambda k: self._memory_cache[k]['expires_at']
            )
            for key in sorted_keys[:1000]:
                del self._memory_cache[key]

    def clear(self) -> bool:
        """
        Clear all cache entries.

        Returns:
            True if cleared
        """
        redis_client = self._get_redis()
        if redis_client:
            try:
                # Delete all keys with prefix
                pattern = f"{self.prefix}:*"
                cursor = 0
                while True:
                    cursor, keys = redis_client.scan(cursor, pattern, 100)
                    if keys:
                        redis_client.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.warning(f"Redis clear error: {e}")

        self._memory_cache.clear()
        return True

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats
        """
        stats = {
            'memory_cache_size': len(self._memory_cache),
            'redis_available': self._is_redis_available or False,
            'cache_enabled': self.settings.enable_cache,
        }

        redis_client = self._get_redis()
        if redis_client:
            try:
                info = redis_client.info('memory')
                stats['redis_memory_used'] = info.get('used_memory_human', 'unknown')
            except Exception:
                pass

        return stats


# Singleton instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get singleton cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
