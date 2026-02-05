# Financial Data Project - Production Dockerfile
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker cache optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir psycopg2-binary gunicorn

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /app/data

# Create non-root user for security
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Expose Streamlit port
EXPOSE 8501

# SERVICE_MODE: "web" (default) or "scheduler"
ENV SERVICE_MODE=web

# Start: web (Streamlit) or scheduler depending on SERVICE_MODE
CMD ["sh", "-c", "if [ \"$SERVICE_MODE\" = \"scheduler\" ]; then python -m src.scheduler --daemon; else streamlit run web/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true; fi"]
