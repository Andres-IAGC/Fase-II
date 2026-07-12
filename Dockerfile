# ============================================================
#  Dockerfile - API de readmision hospitalaria (Fase II)
#  Contenerizacion del modelo de IA para despliegue portable.
# ============================================================

# Paso 3: imagen base ligera y compatible
FROM python:3.11-slim

# Metadatos de la imagen
LABEL maintainer="JUAN ANDRES GUTIERREZ MARTINEZ"
LABEL descripcion="API REST del modelo de readmision hospitalaria a 30 dias"
LABEL version="1.0.0"

# Evita archivos .pyc y fuerza salida sin buffer (logs en tiempo real)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

# Paso 4: directorio de trabajo dentro del contenedor
WORKDIR /app

# Paso 5+6: copiar dependencias primero (aprovecha la cache de capas)
#           e instalar sin cache para reducir el tamano de la imagen
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del codigo fuente del proyecto
COPY . .

# Paso: entrenar el modelo durante la construccion para que la
# imagen ya contenga 'model.pkl' (artefacto reproducible).
RUN python entrenar_modelo.py

# Paso 7: exponer el puerto del servicio
EXPOSE 5000

# Verificacion de salud del contenedor
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5000/health').getcode()==200 else 1)"

# Paso 8: comando de ejecucion (servidor de produccion gunicorn)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
