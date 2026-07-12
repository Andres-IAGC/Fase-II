# ============================================================
#  cliente_api.py
#  Fase II - Gestion de Proyectos de Inteligencia Artificial
#  ------------------------------------------------------------
#  Cliente de prueba que consume la API REST (evidencia de
#  integracion frontend/backend). Envia solicitudes HTTP a los
#  endpoints y muestra las respuestas.
#
#  Requisitos: la API debe estar en ejecucion (python app.py o
#  contenedor Docker) escuchando en BASE_URL.
#
#  Ejecucion:  python cliente_api.py
# ============================================================

import json
import requests

BASE_URL = "http://localhost:5000"

CASOS = {
    "Paciente bajo riesgo":  [45, 2, 3, 1, 0, 100, 115, 23, 0, 0, 0.95, 0],
    "Paciente riesgo medio": [79, 11, 14, 12, 5, 260, 175, 38, 6, 5, 0.2, 7],
    "Paciente alto riesgo":  [90, 15, 22, 24, 9, 350, 205, 48, 10, 8, 0.1, 11],
    "Caso extremo (maximos)": [95, 16, 25, 30, 12, 400, 220, 55, 12, 10, 0.0, 12],
}


def separador(titulo):
    print("\n" + "=" * 58)
    print(" " + titulo)
    print("=" * 58)


def main():
    # 1) Verificacion de salud del servicio
    separador("GET /health")
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    print(r.status_code, json.dumps(r.json(), ensure_ascii=False))

    # 2) Metadatos del modelo
    separador("GET /metadata")
    r = requests.get(f"{BASE_URL}/metadata", timeout=10)
    print(r.status_code, json.dumps(r.json(), ensure_ascii=False, indent=2))

    # 3) Predicciones (POST /predict)
    separador("POST /predict")
    for nombre, features in CASOS.items():
        r = requests.post(f"{BASE_URL}/predict",
                          json={"features": features}, timeout=10)
        data = r.json()
        print(f"\n[{nombre}] HTTP {r.status_code}")
        print(f"  -> {data.get('etiqueta')} | "
              f"prob. readmision = {data.get('probabilidad_readmision')}")

    # 4) Caso de error (validacion): solo 3 variables
    separador("POST /predict (caso invalido)")
    r = requests.post(f"{BASE_URL}/predict",
                      json={"features": [1, 2, 3]}, timeout=10)
    print(r.status_code, json.dumps(r.json(), ensure_ascii=False))


if __name__ == "__main__":
    main()
