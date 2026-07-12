# ============================================================
#  app.py
#  Fase II - Gestion de Proyectos de Inteligencia Artificial
#  ------------------------------------------------------------
#  Backend (API REST) que expone el modelo de readmision
#  hospitalaria entrenado en 'model.pkl'.
#
#  Endpoints:
#    GET  /              -> Frontend (formulario web)
#    GET  /health        -> Estado del servicio y del modelo
#    GET  /metadata      -> Version, algoritmo, umbral y variables
#    GET  /ejemplos      -> Casos de ejemplo precargados
#    POST /predict       -> Prediccion a partir de 12 variables
#
#  Ejecucion local:  python app.py
#  Ejecucion prod:   gunicorn -b 0.0.0.0:5000 app:app
# ============================================================

import os
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")

app = Flask(__name__)

# ------------------------------------------------------------
# Carga del artefacto del modelo al iniciar el servicio
# ------------------------------------------------------------
try:
    ARTEFACTO = joblib.load(MODEL_PATH)
    MODELO = ARTEFACTO["modelo"]
    UMBRAL = ARTEFACTO["umbral"]
    FEATURES = ARTEFACTO["features"]
    CLASES = ARTEFACTO["clases"]
    MODELO_OK = True
except Exception as exc:  # el modelo aun no fue entrenado
    ARTEFACTO, MODELO = None, None
    UMBRAL, FEATURES, CLASES = 0.30, [], {}
    MODELO_OK = False
    print(f"[ADVERTENCIA] No se pudo cargar '{MODEL_PATH}': {exc}")

# Casos de ejemplo (12 variables por caso) para pruebas rapidas
EJEMPLOS = {
    "bajo_riesgo":  [45, 2, 3, 1, 0, 100, 115, 23, 0, 0, 0.95, 0],
    "riesgo_medio": [79, 11, 14, 12, 5, 260, 175, 38, 6, 5, 0.2, 7],
    "alto_riesgo":  [90, 15, 22, 24, 9, 350, 205, 48, 10, 8, 0.1, 11],
}


def _validar_payload(data):
    """Valida y normaliza la entrada. Devuelve (vector, error)."""
    if data is None:
        return None, "El cuerpo de la solicitud debe ser JSON valido."

    # Se acepta {"features": [...]} o {"nombre_var": valor, ...}
    if "features" in data:
        vector = data["features"]
    elif FEATURES and all(f in data for f in FEATURES):
        vector = [data[f] for f in FEATURES]
    else:
        return None, ("Envie 'features' con 12 valores numericos, "
                      "o cada variable por su nombre.")

    if not isinstance(vector, (list, tuple)) or len(vector) != 12:
        return None, "Se esperaban exactamente 12 variables numericas."
    try:
        vector = [float(v) for v in vector]
    except (TypeError, ValueError):
        return None, "Todas las variables deben ser numericas."
    return vector, None


@app.route("/")
def index():
    return render_template("index.html", features=FEATURES,
                           ejemplos=EJEMPLOS, umbral=UMBRAL)


@app.route("/health")
def health():
    estado = "ok" if MODELO_OK else "modelo_no_cargado"
    codigo = 200 if MODELO_OK else 503
    return jsonify({
        "status": estado,
        "modelo_cargado": MODELO_OK,
        "version": ARTEFACTO["version"] if MODELO_OK else None,
    }), codigo


@app.route("/metadata")
def metadata():
    if not MODELO_OK:
        return jsonify({"error": "Modelo no cargado"}), 503
    return jsonify({
        "version": ARTEFACTO["version"],
        "algoritmo": ARTEFACTO["algoritmo"],
        "umbral_operativo": UMBRAL,
        "variables": FEATURES,
        "clases": CLASES,
        "metricas_test": ARTEFACTO.get("metricas_test", {}),
    })


@app.route("/ejemplos")
def ejemplos():
    return jsonify(EJEMPLOS)


@app.route("/predict", methods=["POST"])
def predict():
    if not MODELO_OK:
        return jsonify({"error": "Modelo no cargado. Ejecute entrenar_modelo.py"}), 503

    vector, error = _validar_payload(request.get_json(silent=True))
    if error:
        return jsonify({"error": error}), 400

    X = np.array(vector, dtype=float).reshape(1, -1)
    proba = float(MODELO.predict_proba(X)[0][1])          # prob. de readmision
    prediccion = int(proba >= UMBRAL)                      # umbral operativo

    return jsonify({
        "prediccion": prediccion,
        "etiqueta": CLASES[str(prediccion)],
        "probabilidad_readmision": round(proba, 4),
        "probabilidad_no_readmision": round(1 - proba, 4),
        "umbral_aplicado": UMBRAL,
        "variables_recibidas": dict(zip(FEATURES, vector)),
    })


if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=puerto, debug=False)
