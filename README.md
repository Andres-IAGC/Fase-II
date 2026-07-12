# Predictor de Readmisión Hospitalaria a 30 días — API contenerizada con Docker

**Fase II · Gestión de Proyectos de Inteligencia Artificial**
Autor: JUAN ANDRES GUTIERREZ MARTINEZ · Matrícula 07144441

Solución técnica completa y desplegable que conteneriza, mediante **Docker**, un
modelo de *machine learning* (Random Forest) para predecir el riesgo de
**readmisión hospitalaria a 30 días** en pacientes con enfermedades crónicas.
El modelo se expone como una **API REST (Flask)** integrada con un **frontend**
web, garantizando portabilidad, consistencia y reproducibilidad entre entornos.

> El modelo reutiliza el clasificador construido en la Actividad 6. Los datos son
> sintéticos y el proyecto tiene fines exclusivamente educativos.

---

## 1. Arquitectura

```
Frontend (HTML/JS)  ──HTTP/JSON──►  API REST (Flask)  ──►  Modelo (model.pkl)
   templates/index.html                 app.py               Random Forest
                                                             umbral operativo 0.30
```

| Componente | Archivo | Descripción |
|---|---|---|
| Entrenamiento offline | `entrenar_modelo.py` | Entrena el Random Forest y serializa `model.pkl` |
| Backend / API | `app.py` | Endpoints REST y servidor del frontend |
| Frontend | `templates/index.html` | Formulario web que consume `POST /predict` |
| Simulación local | `simular.py` | Casos comunes, extremos y validación con probabilidades |
| Cliente de prueba | `cliente_api.py` | Consume la API por HTTP (evidencia de integración) |
| Pruebas automáticas | `tests/test_api.py` | Pruebas funcionales con `pytest` |
| Contenedor | `Dockerfile`, `docker-compose.yml` | Imagen e instrucciones de ejecución |

---

## 2. Endpoints de la API

| Método | Ruta | Descripción |
|---|---|---|
| `GET`  | `/` | Frontend (formulario web) |
| `GET`  | `/health` | Estado del servicio y del modelo |
| `GET`  | `/metadata` | Versión, algoritmo, umbral y variables |
| `GET`  | `/ejemplos` | Casos de ejemplo precargados |
| `POST` | `/predict` | Predicción a partir de 12 variables clínicas |

**Variables de entrada (12):** `edad`, `num_diagnosticos`, `num_medicamentos`,
`dias_estancia`, `visitas_urgencias_previas`, `nivel_glucosa`, `presion_arterial`,
`indice_masa_corporal`, `num_procedimientos`, `hospitalizaciones_previas`,
`indice_adherencia`, `num_comorbilidades`.

Ejemplo de solicitud:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [79, 11, 14, 12, 5, 260, 175, 38, 6, 5, 0.2, 7]}'
```

Respuesta:

```json
{
  "prediccion": 1,
  "etiqueta": "Readmitido (alto riesgo)",
  "probabilidad_readmision": 0.83,
  "umbral_aplicado": 0.30
}
```

---

## 3. Ejecución local (sin Docker)

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Entrenar el modelo (genera model.pkl)
python entrenar_modelo.py

# 3. Simular la ejecución local (casos y validación)
python simular.py

# 4. Levantar la API
python app.py            # http://localhost:5000

# 5. (en otra terminal) probar la integración por HTTP
python cliente_api.py

# 6. Pruebas automáticas
python -m pytest -v
```

---

## 4. Ejecución con Docker

```bash
# Construir la imagen (el build entrena el modelo dentro del contenedor)
docker build -t readmision-api:1.0.0 .

# Ejecutar el contenedor
docker run -d -p 5000:5000 --name readmision-api readmision-api:1.0.0

# Verificar
curl http://localhost:5000/health
```

O con Docker Compose:

```bash
docker compose up --build
```

Frontend disponible en `http://localhost:5000`.

---

> **Verificado:** la imagen `readmision-api:1.0.0` (699 MB) se construyó con
> Docker 29.6.1 y el contenedor se ejecutó con estado `healthy`, sirviendo el
> frontend y respondiendo a los endpoints. Evidencia en
> [`evidencias/`](evidencias/) (`05_evidencia_docker.txt`, `06_frontend_docker.png`).

## 5. Despliegue en la nube y documentación

Consulta el **Manual de despliegue en la nube** y el **Documento de validación y
pruebas** en la carpeta [`docs/`](docs/) (PDF y Word) para el procedimiento paso a
paso en un modelo **PaaS** (Render / Azure App Service / Google Cloud Run) a partir
de esta imagen Docker, y el detalle completo de las pruebas realizadas.

---

## 6. Estructura del repositorio

```
readmision-api/
├── app.py                 # API REST (backend)
├── entrenar_modelo.py     # entrenamiento offline -> model.pkl
├── simular.py             # simulación local
├── cliente_api.py         # cliente HTTP de prueba
├── requirements.txt       # dependencias con versiones fijadas
├── Dockerfile             # definición de la imagen
├── docker-compose.yml     # orquestación local
├── .dockerignore
├── .gitignore
├── templates/
│   └── index.html         # frontend
└── tests/
    └── test_api.py        # pruebas funcionales (pytest)
```

---

## 7. Publicación en GitHub

```bash
git init
git add .
git commit -m "Fase II: API de readmisión hospitalaria contenerizada con Docker"
git branch -M main
git remote add origin https://github.com/<usuario>/readmision-api.git
git push -u origin main
```
