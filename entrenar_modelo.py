# ============================================================
#  entrenar_modelo.py
#  Fase II - Gestion de Proyectos de Inteligencia Artificial
#  ------------------------------------------------------------
#  Entrenamiento OFFLINE del modelo de clasificacion binaria
#  Caso: prediccion de readmision hospitalaria a 30 dias en
#        pacientes con enfermedades cronicas.
#        Clase 1 = readmitido (alto riesgo) | Clase 0 = no readmitido
#
#  Este script reutiliza el modelo construido en la Actividad 6
#  (Random Forest) y lo SERIALIZA en 'model.pkl' junto con sus
#  metadatos (umbral operativo, nombres de variables y version),
#  separando el entrenamiento del uso del modelo conforme a la
#  norma ISO/IEC 23053.
#
#  Ejecucion:   python entrenar_modelo.py
#  Salida:      model.pkl
# ============================================================

import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score
import joblib

RANDOM_STATE = 42
UMBRAL_OPERATIVO = 0.30   # umbral ajustado para maximizar recall (no omitir alto riesgo)
MODEL_PATH = "model.pkl"
N_MUESTRAS = 3000

# Las 12 variables clinicas de entrada (con rangos realistas).
FEATURE_NAMES = [
    "edad",                       # 1  (18-95 anios)
    "num_diagnosticos",           # 2  (0-16)
    "num_medicamentos",          # 3  (0-25)
    "dias_estancia",             # 4  (0-30 dias)
    "visitas_urgencias_previas",  # 5  (0-12)
    "nivel_glucosa",             # 6  (70-400 mg/dL)
    "presion_arterial",          # 7  (90-220 mmHg sistolica)
    "indice_masa_corporal",      # 8  (18-55)
    "num_procedimientos",        # 9  (0-12)
    "hospitalizaciones_previas",  # 10 (0-10)
    "indice_adherencia",         # 11 (0-1; 1 = adherencia perfecta)
    "num_comorbilidades",        # 12 (0-12)
]

# Rangos [min, max] por variable, en el mismo orden que FEATURE_NAMES.
RANGOS = np.array([
    [18, 95], [0, 16], [0, 25], [0, 30], [0, 12], [70, 400],
    [90, 220], [18, 55], [0, 12], [0, 10], [0, 1], [0, 12],
], dtype=float)

# Pesos de riesgo: mayor peso = mayor influencia en la readmision.
# (hospitalizaciones y urgencias previas, adherencia y comorbilidades
#  son los factores clinicos mas determinantes.)
PESOS = np.array([1.5, 1.0, 1.0, 1.0, 2.0, 1.0,
                  0.8, 0.5, 1.0, 2.5, 2.5, 1.5])


def generar_datos():
    """Genera un dataset clinico sintetico con una regla de riesgo
    coherente. La probabilidad de readmision crece de forma monotona
    con los factores de riesgo; la etiqueta se muestrea de esa
    probabilidad para introducir ruido realista (~27% readmitidos)."""
    rng = np.random.default_rng(RANDOM_STATE)
    lo, hi = RANGOS[:, 0], RANGOS[:, 1]

    # Muestreo uniforme de cada variable dentro de su rango clinico
    X = rng.uniform(lo, hi, size=(N_MUESTRAS, len(FEATURE_NAMES)))
    X[:, [1, 2, 3, 4, 8, 9, 11]] = np.round(X[:, [1, 2, 3, 4, 8, 9, 11]])  # enteros

    # Normalizacion 0-1 por variable; la adherencia se invierte
    # (menor adherencia -> mayor riesgo).
    Xn = (X - lo) / (hi - lo)
    Xn[:, 10] = 1.0 - Xn[:, 10]

    # Puntaje de riesgo ponderado, estandarizado a z-score
    raw = Xn @ PESOS
    z = (raw - raw.mean()) / raw.std()

    # Probabilidad logistica (intercepto ajustado a ~27% de prevalencia)
    prob = 1 / (1 + np.exp(-(1.8 * z - 1.3)))
    y = (rng.uniform(size=N_MUESTRAS) < prob).astype(int)
    return X, y


def entrenar():
    np.random.seed(RANDOM_STATE)
    X, y = generar_datos()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

    print("Distribucion de clases (total):", np.bincount(y),
          "-> %.1f%% readmitidos" % (100 * y.mean()))

    # Modelo de produccion: Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=RANDOM_STATE).fit(X_train, y_train)

    # Evaluacion en el conjunto de prueba (umbral operativo 0.30)
    proba_rf = rf.predict_proba(X_test)[:, 1]
    y_hat = (proba_rf >= UMBRAL_OPERATIVO).astype(int)

    metricas = {
        "recall": round(float(recall_score(y_test, y_hat)), 4),
        "f1": round(float(f1_score(y_test, y_hat)), 4),
        "auc": round(float(roc_auc_score(y_test, proba_rf)), 4),
    }
    print("\n--- Modelo Random Forest (umbral %.2f) ---" % UMBRAL_OPERATIVO)
    for k, v in metricas.items():
        print(f"{k:8s}: {v}")

    # Serializacion del modelo + metadatos en un unico artefacto
    artefacto = {
        "modelo": rf,
        "umbral": UMBRAL_OPERATIVO,
        "features": FEATURE_NAMES,
        "clases": {"0": "No readmitido", "1": "Readmitido (alto riesgo)"},
        "version": "1.0.0",
        "algoritmo": "RandomForestClassifier",
        "metricas_test": metricas,
    }
    joblib.dump(artefacto, MODEL_PATH)
    print(f"\nModelo entrenado y guardado en '{MODEL_PATH}'")
    print("Metadatos:", json.dumps(
        {k: v for k, v in artefacto.items() if k != "modelo"},
        ensure_ascii=False))


if __name__ == "__main__":
    entrenar()
