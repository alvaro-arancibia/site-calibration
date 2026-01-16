FROM python:3.12-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy local package
COPY . .

# Install the package and its dependencies
RUN pip install --no-cache-dir .

# Expose port 8080 for Google Cloud Run
EXPOSE 8080

# Command to run the API
CMD ["uvicorn", "sitecal.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
