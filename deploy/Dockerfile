

FROM python:3.11-slim

# System dependencies for Azure SDKs, FastAPI, and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libffi-dev \
        libssl-dev \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# Copy requirements
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Expose FastAPI port
EXPOSE 8000

# Default environment variables (can be overridden in deployment)
ENV PORT=8000

# Required Azure and CRM environment variables (documented for deployment)
# ENV CRM_API_ENDPOINT=
# ENV API_AUTH_TOKEN=
# ENV AZURE_SEARCH_ENDPOINT=
# ENV AZURE_SEARCH_API_KEY=
# ENV AZURE_SEARCH_INDEX_NAME=
# ENV AZURE_OPENAI_ENDPOINT=
# ENV AZURE_OPENAI_API_KEY=
# ENV AZURE_OPENAI_EMBEDDING_DEPLOYMENT=

# Entrypoint
CMD ["uvicorn", "agent:app", "--host", "0.0.0.0", "--port", "8000"]

