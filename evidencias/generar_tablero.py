# Genera un tablero visual de validacion del modelo (evidencia grafica).
import numpy as np, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from entrenar_modelo import generar_datos, UMBRAL_OPERATIVO
from sklearn.model_selection import train_test_split

art = joblib.load(os.path.join(os.path.dirname(__file__), "..", "model.pkl"))
m = art["modelo"]

X, y = generar_datos()
_, Xte, _, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
proba = m.predict_proba(Xte)[:, 1]
yhat = (proba >= UMBRAL_OPERATIVO).astype(int)
cm = confusion_matrix(yte, yhat)
fpr, tpr, _ = roc_curve(yte, proba)
auc = roc_auc_score(yte, proba)

casos = {
    "Bajo": [45, 2, 3, 1, 0, 100, 115, 23, 0, 0, 0.95, 0],
    "Medio": [79, 11, 14, 12, 5, 260, 175, 38, 6, 5, 0.2, 7],
    "Alto": [90, 15, 22, 24, 9, 350, 205, 48, 10, 8, 0.1, 11],
    "Extremo": [95, 16, 25, 30, 12, 400, 220, 55, 12, 10, 0.0, 12],
}
probs = [m.predict_proba(np.array(v).reshape(1, -1))[0][1] for v in casos.values()]

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Tablero de validacion - API de readmision hospitalaria (Fase II)",
             fontsize=14, fontweight="bold")

# (1) Matriz de confusion
ax[0].imshow(cm, cmap="Blues")
ax[0].set_title(f"Matriz de confusion (umbral {UMBRAL_OPERATIVO})")
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["Pred 0", "Pred 1"])
ax[0].set_yticks([0, 1]); ax[0].set_yticklabels(["Real 0", "Real 1"])
for i in range(2):
    for j in range(2):
        ax[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=15)

# (2) Curva ROC
ax[1].plot(fpr, tpr, color="darkorange", label=f"RF (AUC={auc:.3f})")
ax[1].plot([0, 1], [0, 1], "--", color="gray", label="Aleatorio")
ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR (Recall)")
ax[1].set_title("Curva ROC"); ax[1].legend(loc="lower right")

# (3) Probabilidades por caso de prueba
colores = ["#1e8449" if p < UMBRAL_OPERATIVO else "#c0392b" for p in probs]
ax[2].bar(list(casos.keys()), probs, color=colores)
ax[2].axhline(UMBRAL_OPERATIVO, color="black", ls="--", label=f"Umbral {UMBRAL_OPERATIVO}")
ax[2].set_ylim(0, 1); ax[2].set_ylabel("Prob. readmision")
ax[2].set_title("Prediccion por caso de prueba"); ax[2].legend()
for i, p in enumerate(probs):
    ax[2].text(i, p + 0.02, f"{p:.2f}", ha="center")

plt.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(os.path.dirname(__file__), "04_tablero_validacion.png")
plt.savefig(out, dpi=120, bbox_inches="tight")
print("Tablero guardado en", out)
