FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install minimal OS dependencies if needed for PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security best practices
RUN useradd --create-home --shell /bin/bash myuser

WORKDIR /home/myuser/app

# Copy dependency definition and install
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY --chown=myuser:myuser . ./

USER myuser

# Run actor
CMD ["python3", "-m", "src.main"]
