# CPK - Klasyfikator Chmur Punktów v2.0
# Dockerfile dla HackNation 2025

# Bazowy obraz Python 3.9 (slim dla mniejszego rozmiaru)
FROM python:3.9-slim

# Metadane
LABEL maintainer="HackNation 2025 - CPK Team"
LABEL description="CPK Point Cloud Classifier - Automatyczna klasyfikacja 45 klas infrastruktury"
LABEL version="2.0"

# Ustaw zmienne środowiskowe
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Utwórz katalog roboczy
WORKDIR /app

# Zainstaluj zależności systemowe (potrzebne dla laspy i numpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Skopiuj requirements.txt i zainstaluj zależności Pythona
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Skopiuj kod źródłowy aplikacji
COPY app.py .
COPY src/ ./src/

# Utwórz katalogi dla danych wejściowych i wyjściowych
RUN mkdir -p /app/data /app/output

# Expose port dla Streamlit (domyślnie 8501)
EXPOSE 8501

# Healthcheck - sprawdź czy Streamlit działa
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Zainstaluj curl dla healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Uruchom aplikację Streamlit
# --server.address=0.0.0.0 - słuchaj na wszystkich interfejsach (dla Docker)
# --server.port=8501 - standardowy port Streamlit
# --server.headless=true - tryb headless (bez przeglądarki)
CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]

# ============================================================================
# UŻYCIE:
#
# 1. Build image:
#    docker build -t cpk-classifier:v2.0 .
#
# 2. Uruchom kontener (z montowaniem katalogów):
#    docker run -d \
#      -p 8501:8501 \
#      -v $(pwd)/data:/app/data \
#      -v $(pwd)/output:/app/output \
#      --name cpk-classifier \
#      cpk-classifier:v2.0
#
# 3. Sprawdź logi:
#    docker logs -f cpk-classifier
#
# 4. Zatrzymaj kontener:
#    docker stop cpk-classifier
#
# 5. Usuń kontener:
#    docker rm cpk-classifier
#
# Aplikacja będzie dostępna pod: http://localhost:8501
# ============================================================================
