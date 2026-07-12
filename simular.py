# ============================================================
#  simular.py
#  Fase II - Gestion de Proyectos de Inteligencia Artificial
#  ------------------------------------------------------------
#  Simulacion de EJECUCION LOCAL del modelo antes de contenerizar.
#  Carga 'model.pkl' y valida su comportamiento con:
#    1) Casos comunes / esperados
#    2) Casos extremos (limite)
#    3) Validacion basica con probabilidades (predict_proba)
#
#  Ejecucion:  python simular.py
# ============================================================

import numpy as np
import joblib

art = joblib.load("model.pkl")
modelo = art["modelo"]
umbral = art["umbral"]
features = art["features"]

print("=" * 60)
print(" SIMULACION LOCAL - Modelo de readmision hospitalaria")
print(" Version:", art["version"], "| Algoritmo:", art["algoritmo"],
      "| Umbral:", umbral)
print("=" * 60)


def predecir(vector, etiqueta=""):
    X = np.array(vector, dtype=float).reshape(1, -1)
    proba = modelo.predict_proba(X)[0]
    pred = int(proba[1] >= umbral)
    clase = "Readmitido (alto riesgo)" if pred else "No readmitido"
    print(f"\n> {etiqueta}")
    print(f"  Entrada: {vector}")
    print(f"  Prediccion: {pred} ({clase})")
    print(f"  Probabilidades (no readmision, readmision): "
          f"[{proba[0]:.4f}, {proba[1]:.4f}]")
    return pred, proba


# ------------------------------------------------------------
# 1) CASOS COMUNES / ESPERADOS
#    Orden de variables: edad, num_diagnosticos, num_medicamentos,
#    dias_estancia, visitas_urgencias_previas, nivel_glucosa,
#    presion_arterial, indice_masa_corporal, num_procedimientos,
#    hospitalizaciones_previas, indice_adherencia, num_comorbilidades
# ------------------------------------------------------------
print("\n" + "-" * 60)
print(" 1) CASOS COMUNES / ESPERADOS")
print("-" * 60)
predecir([45, 2, 3, 1, 0, 100, 115, 23, 0, 0, 0.95, 0], "Paciente joven, buena adherencia (bajo riesgo esperado)")
predecir([79, 11, 14, 12, 5, 260, 175, 38, 6, 5, 0.2, 7], "Paciente cronico moderado (riesgo medio esperado)")
predecir([90, 15, 22, 24, 9, 350, 205, 48, 10, 8, 0.1, 11], "Paciente cronico severo (alto riesgo esperado)")

# ------------------------------------------------------------
# 2) CASOS EXTREMOS (LIMITE)
# ------------------------------------------------------------
print("\n" + "-" * 60)
print(" 2) CASOS EXTREMOS (LIMITE)")
print("-" * 60)
predecir([18, 0, 0, 0, 0, 80, 100, 20, 0, 0, 1.0, 0], "Minimos absolutos (paciente sano ideal)")
predecir([95, 16, 25, 30, 12, 400, 220, 55, 12, 10, 0.0, 12], "Maximos absolutos (deterioro extremo)")
predecir([70, 8, 10, 7, 3, 200, 160, 33, 4, 3, 0.4, 5], "Caso frontera cercano al umbral")

# ------------------------------------------------------------
# 3) VALIDACION BASICA CON PROBABILIDADES
#    Umbral de alerta: si prob. readmision > 0.70 -> caso sospechoso
# ------------------------------------------------------------
print("\n" + "-" * 60)
print(" 3) VALIDACION BASICA (nivel de confianza y alertas)")
print("-" * 60)
casos = {
    "Bajo riesgo":  [45, 2, 3, 1, 0, 100, 115, 23, 0, 0, 0.95, 0],
    "Alto riesgo":  [90, 15, 22, 24, 9, 350, 205, 48, 10, 8, 0.1, 11],
}
for nombre, vec in casos.items():
    _, proba = predecir(vec, nombre)
    if proba[1] > 0.70:
        print("  [ALERTA] Probabilidad de readmision > 0.70 -> revision prioritaria")

print("\n" + "=" * 60)
print(" Simulacion finalizada. El modelo carga y responde correctamente.")
print("=" * 60)
