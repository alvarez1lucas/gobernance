# ==============================================================================
# Dockerfile — AI Governance Framework (Lightweight & Auditable)
# ==============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalamos curl para el Healthcheck (no requerimos git ni build-essential aquí)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar EXCLUSIVAMENTE las dependencias de Gobernanza
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copiar la estructura del framework (incluyendo los submódulos clonados en el CI)
COPY . .

# Asegurar la estructura de directorios para que el data_loader no falle por I/O
RUN mkdir -p reports \
             submodules/credit-risk/reports \
             submodules/market-risk/reports \
             submodules/market-risk/data/raw/sentiment \
             demo/credit_risk \
             demo/market_risk

# Ejecutar el motor de políticas en el build para auditoría
RUN python run_governance.py

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=4s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
