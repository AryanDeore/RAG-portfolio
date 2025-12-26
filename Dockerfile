FROM python:3.13-slim

# Create and change to the app directory
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files and source code 
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Install project dependencies
RUN uv sync --frozen --no-dev

# Copy rest of local code to the container image
COPY . .

# Expose port (Railway sets PORT env var automatically)
EXPOSE 8000

# Run the web service on container startup
CMD ["sh", "-c", "uv run uvicorn server.main:app --host 0.0.0.0 --port ${PORT:-8000}"]