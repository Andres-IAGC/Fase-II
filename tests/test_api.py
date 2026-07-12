# ============================================================
#  tests/test_api.py
#  Pruebas funcionales automatizadas de la API (pytest).
#  Ejecucion:  python -m pytest -v
#  Requisito:  haber ejecutado antes 'python entrenar_modelo.py'
# ============================================================

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


BAJO = [45, 2, 3, 1, 0, 100, 115, 23, 0, 0, 0.95, 0]
ALTO = [90, 15, 22, 24, 9, 350, 205, 48, 10, 8, 0.1, 11]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["modelo_cargado"] is True


def test_metadata(client):
    r = client.get("/metadata")
    assert r.status_code == 200
    assert len(r.get_json()["variables"]) == 12


def test_predict_estructura(client):
    r = client.post("/predict", json={"features": BAJO})
    assert r.status_code == 200
    data = r.get_json()
    assert data["prediccion"] in (0, 1)
    assert 0.0 <= data["probabilidad_readmision"] <= 1.0


def test_predict_alto_riesgo_mayor_que_bajo(client):
    p_bajo = client.post("/predict", json={"features": BAJO}).get_json()
    p_alto = client.post("/predict", json={"features": ALTO}).get_json()
    assert (p_alto["probabilidad_readmision"]
            >= p_bajo["probabilidad_readmision"])


def test_predict_validacion_numero_variables(client):
    r = client.post("/predict", json={"features": [1, 2, 3]})
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_predict_validacion_no_numerico(client):
    r = client.post("/predict", json={"features": ["a"] * 12})
    assert r.status_code == 400
