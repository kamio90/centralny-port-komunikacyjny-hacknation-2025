# CPK - Klasyfikator Chmur Punktow v2.0
# Dockerfile dla HackNation 2025
# Multi-platform: AMD64 + ARM64 (Mac M1/M2/M3/M4)
#
# Funkcjonalnosci:
# - 36 klas infrastruktury (ASPRS, Railway, Road, BIM, CPK)
# - Modul ML (Random Forest, PointNet, Ensemble, Active Learning)
# - Modul Railway (Catenary, Track, Pole, Signal detection)
# - Modul BIM (Building extraction, Clash detection, IFC export, LOD)
# - Web UI (Streamlit) + CLI

FROM python:3.11-slim

# Metadane
LABEL maintainer="HackNation 2025 - CPK Team"
LABEL description="CPK Point Cloud Classifier - Automatyczna klasyfikacja infrastruktury z modulami ML, Railway i BIM"
LABEL version="2.0"
LABEL features="ML,Railway,BIM,36-classes,IFC-export"

# Zmienne srodowiskowe
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501

# Katalog roboczy
WORKDIR /app

# Zainstaluj zaleznosci systemowe (curl dla healthcheck + build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Skopiuj requirements i zainstaluj zaleznosci Pythona
COPY requirements-docker.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Skopiuj kod zrodlowy
COPY app.py .
COPY cli.py .
COPY src/ ./src/

# Utworz katalogi dla danych
RUN mkdir -p /app/data /app/output /app/assets

# Skopiuj assets (logo)
COPY assets/ ./assets/

# Expose port Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Domyslnie uruchom Streamlit UI
# Mozna nadpisac przez: docker run ... python cli.py input.las output.las
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]

# ============================================================================
# UZYCIE:
#
# === BUILD ===
# docker build -t cpk-classifier .
#
# === URUCHOM WEB UI ===
# docker run -d -p 8501:8501 \
#   -v $(pwd)/data:/app/data \
#   -v $(pwd)/output:/app/output \
#   --name cpk \
#   cpk-classifier
#
# Aplikacja: http://localhost:8501
#
# === URUCHOM CLI ===
# docker run --rm \
#   -v $(pwd)/data:/app/data \
#   -v $(pwd)/output:/app/output \
#   cpk-classifier \
#   python cli.py /app/data/input.las /app/output/output.las --report /app/output/raport.json
#
# === LOGI ===
# docker logs -f cpk
#
# === ZATRZYMAJ ===
# docker stop cpk && docker rm cpk
#
# ============================================================================
