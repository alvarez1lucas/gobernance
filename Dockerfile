# ==============================================================================
# Dockerfile — AI Governance Framework
# ==============================================================================
FROM python:3.11-slim

# Evita que Python escriba archivos .pyc y fuerza el buffer de salida para logs en vivo
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias del sistema esenciales para compilación de ciertas librerías numéricas
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar primero los requerimientos para aprovechar el cache de capas de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copiar la estructura del proyecto
COPY . .

# Crear directorios obligatorios para outputs si no existen por persistencia o volumenes
RUN mkdir -p reports demo/credit_risk demo/market_risk

# Validar el registro de modelos y correr el pipeline principal de gobernanza durante el build
# Esto asegura que si hay un error de Rego o parsing de YAML, el build falle de inmediato.
RUN python run_governance.py

# Exponer el puerto por defecto de Streamlit
EXPOSE 8501

# Configuración de salud del contenedor (Healthcheck)
HEALTHCHECK --interval=30s --timeout=4s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando por defecto: Levantar el Dashboard Ejecutivo
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
