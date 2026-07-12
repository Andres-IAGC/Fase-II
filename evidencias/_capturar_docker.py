import os, subprocess, datetime, requests

DK = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                  "Docker", "Docker", "resources", "bin", "docker.exe")
OUT = os.path.join(os.path.dirname(__file__), "05_evidencia_docker.txt")
BASE = "http://localhost:5000"


def dk(*args):
    return subprocess.run([DK, *args], capture_output=True, text=True).stdout.strip()


L = []
L.append("=" * 60)
L.append(" EVIDENCIA DE CONTENEDOR DOCKER FUNCIONAL - Fase II")
L.append(" Fecha: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
L.append("=" * 60)

L.append("\n### docker --version")
L.append(dk("--version"))

L.append("\n### docker images readmision-api  (IMAGEN CONSTRUIDA CORRECTAMENTE)")
L.append(dk("images", "readmision-api", "--format",
            "{{.Repository}}:{{.Tag}}   ID={{.ID}}   Size={{.Size}}"))

L.append("\n### docker ps  (CONTENEDOR EN EJECUCION)")
L.append(dk("ps", "--filter", "name=readmision-api", "--format",
            "{{.Names}} | {{.Image}} | {{.Status}} | {{.Ports}}"))

L.append("\n### Healthcheck nativo del contenedor")
L.append("Estado de salud: " + dk("inspect", "--format",
                                    "{{.State.Health.Status}}", "readmision-api"))

L.append("\n### GET /health")
L.append(requests.get(BASE + "/health").text)

L.append("\n### GET /metadata")
L.append(requests.get(BASE + "/metadata").text)

L.append("\n### POST /predict  - Caso bajo riesgo  [45,2,3,1,0,100,115,23,0,0,0.95,0]")
L.append(requests.post(BASE + "/predict",
         json={"features": [45, 2, 3, 1, 0, 100, 115, 23, 0, 0, 0.95, 0]}).text)

L.append("\n### POST /predict  - Caso alto riesgo  [90,15,22,24,9,350,205,48,10,8,0.1,11]")
L.append(requests.post(BASE + "/predict",
         json={"features": [90, 15, 22, 24, 9, 350, 205, 48, 10, 8, 0.1, 11]}).text)

L.append("\n### POST /predict  - Entrada invalida (validacion, HTTP 400 esperado)")
r = requests.post(BASE + "/predict", json={"features": [1, 2, 3]})
L.append("HTTP %d - %s" % (r.status_code, r.text.strip()))

L.append("\n### Ultimas lineas del log del contenedor (gunicorn)")
logs = subprocess.run([DK, "logs", "--tail", "8", "readmision-api"],
                      capture_output=True, text=True)
L.append((logs.stdout + logs.stderr).strip())

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("OK evidencia guardada en", OUT)
print("\n".join(L))
