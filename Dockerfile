# --- Dockerfile for Streamlit app ---
# Use a lightweight Python base image
FROM python:3.11-slim

# Prevents Python from writing .pyc files and enables unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# System deps (optional; kept minimal since scikit-learn has wheels)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better caching
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the project files
COPY . /app

# Streamlit config: run headless and listen on all interfaces
ENV STREAMLIT_SERVER_HEADLESS=true
ENV PORT=8501

EXPOSE 8501

# Default command
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
