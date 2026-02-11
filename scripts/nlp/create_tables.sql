-- =============================================================================
-- NLP Sentiment Analysis Tables
-- Database: FMP PostgreSQL (localhost:5433/fmp_data)
-- =============================================================================

-- Enable pgvector extension (for embeddings - run as superuser if needed)
-- CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- 1. NEWS SENTIMENT TABLE (Partitioned by year)
-- =============================================================================

-- Main table
CREATE TABLE IF NOT EXISTS nlp_sentiment_news (
    id BIGSERIAL,
    news_id BIGINT,
    symbol VARCHAR(20),
    published_date DATE,

    -- Scores by model
    finbert_score FLOAT,
    roberta_score FLOAT,
    ensemble_score FLOAT,
    ensemble_label VARCHAR(20),
    confidence FLOAT,

    -- Metadata
    processed_at TIMESTAMP DEFAULT NOW(),
    model_version VARCHAR(50),

    PRIMARY KEY (id, published_date),
    CONSTRAINT unique_news_id UNIQUE (news_id)
) PARTITION BY RANGE (published_date);

-- Create partitions by year
CREATE TABLE IF NOT EXISTS nlp_sentiment_news_2023 PARTITION OF nlp_sentiment_news
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

CREATE TABLE IF NOT EXISTS nlp_sentiment_news_2024 PARTITION OF nlp_sentiment_news
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE IF NOT EXISTS nlp_sentiment_news_2025 PARTITION OF nlp_sentiment_news
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

CREATE TABLE IF NOT EXISTS nlp_sentiment_news_2026 PARTITION OF nlp_sentiment_news
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

-- Default partition for future data
CREATE TABLE IF NOT EXISTS nlp_sentiment_news_default PARTITION OF nlp_sentiment_news
    DEFAULT;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_sentiment_news_symbol ON nlp_sentiment_news(symbol);
CREATE INDEX IF NOT EXISTS idx_sentiment_news_date ON nlp_sentiment_news(published_date);
CREATE INDEX IF NOT EXISTS idx_sentiment_news_symbol_date ON nlp_sentiment_news(symbol, published_date);
CREATE INDEX IF NOT EXISTS idx_sentiment_news_processed ON nlp_sentiment_news(processed_at);


-- =============================================================================
-- 2. TRANSCRIPT SENTIMENT TABLE
-- =============================================================================

CREATE TABLE IF NOT EXISTS nlp_sentiment_transcript (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    year INTEGER NOT NULL,
    quarter VARCHAR(10) NOT NULL,
    earnings_date DATE,

    -- Sentiment by section
    overall_score FLOAT,
    prepared_remarks_score FLOAT,
    qa_section_score FLOAT,
    guidance_score FLOAT,

    -- Q&A vs prepared delta (reveals tension)
    qa_prepared_delta FLOAT,

    -- Topics detected
    topics JSONB,

    -- Metadata
    num_segments INTEGER,
    processed_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT unique_transcript UNIQUE (symbol, year, quarter)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_sentiment_transcript_symbol ON nlp_sentiment_transcript(symbol);
CREATE INDEX IF NOT EXISTS idx_sentiment_transcript_date ON nlp_sentiment_transcript(earnings_date);
CREATE INDEX IF NOT EXISTS idx_sentiment_transcript_year_qtr ON nlp_sentiment_transcript(year, quarter);


-- =============================================================================
-- 3. DAILY SENTIMENT FEATURES TABLE (Partitioned by year)
-- =============================================================================

CREATE TABLE IF NOT EXISTS features_sentiment_daily (
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,

    -- News sentiment
    news_sentiment FLOAT,
    news_count INTEGER,
    news_sentiment_ma7 FLOAT,
    news_sentiment_momentum FLOAT,

    -- Transcript sentiment (forward-filled)
    transcript_sentiment FLOAT,
    days_since_earnings INTEGER,

    -- Macro (from existing features)
    fear_greed FLOAT,
    aaii_spread FLOAT,

    -- Combined
    combined_sentiment FLOAT,
    sentiment_zone VARCHAR(20),

    PRIMARY KEY (symbol, date)
) PARTITION BY RANGE (date);

-- Create partitions
CREATE TABLE IF NOT EXISTS features_sentiment_daily_2023 PARTITION OF features_sentiment_daily
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

CREATE TABLE IF NOT EXISTS features_sentiment_daily_2024 PARTITION OF features_sentiment_daily
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE IF NOT EXISTS features_sentiment_daily_2025 PARTITION OF features_sentiment_daily
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

CREATE TABLE IF NOT EXISTS features_sentiment_daily_2026 PARTITION OF features_sentiment_daily
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

CREATE TABLE IF NOT EXISTS features_sentiment_daily_default PARTITION OF features_sentiment_daily
    DEFAULT;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_features_sentiment_date ON features_sentiment_daily(date);
CREATE INDEX IF NOT EXISTS idx_features_sentiment_symbol ON features_sentiment_daily(symbol);
CREATE INDEX IF NOT EXISTS idx_features_sentiment_zone ON features_sentiment_daily(sentiment_zone);


-- =============================================================================
-- 4. EMBEDDINGS TABLE (Requires pgvector extension)
-- =============================================================================

-- Only create if pgvector is available
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        CREATE TABLE IF NOT EXISTS nlp_embeddings (
            id BIGSERIAL PRIMARY KEY,
            source_type VARCHAR(20),  -- 'news', 'transcript', 'filing'
            source_id BIGINT,
            symbol VARCHAR(20),

            -- Vector embedding (768 dimensions)
            embedding VECTOR(768),

            -- Metadata
            model_name VARCHAR(100),
            created_at TIMESTAMP DEFAULT NOW()
        );

        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_embeddings_source ON nlp_embeddings(source_type, source_id);
        CREATE INDEX IF NOT EXISTS idx_embeddings_symbol ON nlp_embeddings(symbol);

        -- IVFFlat index for fast similarity search
        -- Note: Requires some data before creating
        -- CREATE INDEX idx_embeddings_vector ON nlp_embeddings
        --     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

        RAISE NOTICE 'Created nlp_embeddings table with pgvector support';
    ELSE
        RAISE NOTICE 'pgvector extension not available, skipping nlp_embeddings table';
    END IF;
END $$;


-- =============================================================================
-- 5. PROCESSING LOG TABLE (for tracking processing status)
-- =============================================================================

CREATE TABLE IF NOT EXISTS nlp_processing_log (
    id BIGSERIAL PRIMARY KEY,
    process_type VARCHAR(50),  -- 'news', 'transcript', 'aggregate'
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    records_processed INTEGER,
    records_failed INTEGER,
    status VARCHAR(20),  -- 'running', 'completed', 'failed'
    error_message TEXT,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_processing_log_type ON nlp_processing_log(process_type);
CREATE INDEX IF NOT EXISTS idx_processing_log_status ON nlp_processing_log(status);
CREATE INDEX IF NOT EXISTS idx_processing_log_started ON nlp_processing_log(started_at);


-- =============================================================================
-- 6. HELPER FUNCTIONS
-- =============================================================================

-- Function to get latest sentiment for a symbol
CREATE OR REPLACE FUNCTION get_latest_sentiment(p_symbol VARCHAR)
RETURNS TABLE (
    news_sentiment FLOAT,
    news_count INTEGER,
    transcript_sentiment FLOAT,
    combined_sentiment FLOAT,
    sentiment_zone VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        f.news_sentiment,
        f.news_count,
        f.transcript_sentiment,
        f.combined_sentiment,
        f.sentiment_zone
    FROM features_sentiment_daily f
    WHERE f.symbol = p_symbol
    ORDER BY f.date DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Function to get sentiment history
CREATE OR REPLACE FUNCTION get_sentiment_history(
    p_symbol VARCHAR,
    p_days INTEGER DEFAULT 30
)
RETURNS TABLE (
    date DATE,
    news_sentiment FLOAT,
    transcript_sentiment FLOAT,
    combined_sentiment FLOAT,
    sentiment_zone VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        f.date,
        f.news_sentiment,
        f.transcript_sentiment,
        f.combined_sentiment,
        f.sentiment_zone
    FROM features_sentiment_daily f
    WHERE f.symbol = p_symbol
        AND f.date >= CURRENT_DATE - p_days
    ORDER BY f.date;
END;
$$ LANGUAGE plpgsql;


-- =============================================================================
-- GRANTS (adjust as needed)
-- =============================================================================

-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO fmp;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO fmp;


-- =============================================================================
-- VERIFICATION QUERIES
-- =============================================================================

-- Check table creation
-- SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename LIKE 'nlp_%';

-- Check partitions
-- SELECT inhrelid::regclass AS partition FROM pg_inherits WHERE inhparent = 'nlp_sentiment_news'::regclass;

-- Check indexes
-- SELECT indexname FROM pg_indexes WHERE tablename LIKE 'nlp_%' OR tablename LIKE 'features_sentiment%';
