# evaluate_divergences.py

import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.ensemble import IsolationForest
from compute_divergences import jensen_shannon_divergence, wasserstein_distance_nd

# Configuración
SAMPLES_DIR = "generated_samples"
P0_FILE = os.path.join(SAMPLES_DIR, "P0.csv")
BOOTSTRAP_FILE = "processed_data/bootstrap_divergences.csv"
BENIGN_FULL_FILE = "processed_data/benign.csv"
OUTPUT_DIR = "evaluation_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============== ENTRENAMIENTO DE MODELOS ===================

# Modelo con P0
P0 = pd.read_csv(P0_FILE).drop(columns=["Target", "Label", "source_file"], errors="ignore")
P0_np = P0.to_numpy()
clf_p0 = IsolationForest(contamination='auto', random_state=42)
clf_p0.fit(P0_np)
scores_p0_train = clf_p0.decision_function(P0_np)

# Modelo con benignos completos
benign_full = pd.read_csv(BENIGN_FULL_FILE).drop(columns=["Target", "Label", "source_file"], errors="ignore")
X_full = benign_full.to_numpy()
clf_full = IsolationForest(contamination='auto', random_state=42)
clf_full.fit(X_full)
scores_full_train = clf_full.decision_function(X_full)

# === THRESHOLD DINÁMICO SEGÚN MÍNIMO SCORE EN BENIGNOS ===
min_score_full = scores_full_train.min()
THRESHOLD = -min_score_full - 0.001
print(f"[INFO] Umbral dinámico sin falsos positivos: -{THRESHOLD:.5f}")

# Guardar modelos
os.makedirs("models", exist_ok=True)
dump(clf_p0, "models/if_p0.joblib")
dump(clf_full, "models/if_full.joblib")

# ===================== MÉTRICAS DE ENTRENAMIENTO =====================
outliers_p0 = np.sum(scores_p0_train < -THRESHOLD)
outliers_full = np.sum(scores_full_train < -THRESHOLD)

print(f"[IF_P0] Mean: {scores_p0_train.mean():.4f}, Std: {scores_p0_train.std():.4f}, Outliers: {100*outliers_p0/len(P0_np):.2f}%")
print(f"[IF_FULL] Mean: {scores_full_train.mean():.4f}, Std: {scores_full_train.std():.4f}, Outliers: {100*outliers_full/len(X_full):.2f}%")

# Guardar resumen del modelo
model_summary = {
    "IF_P0": {
        "mean_score": float(scores_p0_train.mean()),
        "std_score": float(scores_p0_train.std()),
        "min_score": float(scores_p0_train.min()),
        "max_score": float(scores_p0_train.max()),
        "outliers_pct": float(100 * outliers_p0 / len(P0_np))
    },
    "IF_FULL": {
        "mean_score": float(scores_full_train.mean()),
        "std_score": float(scores_full_train.std()),
        "min_score": float(scores_full_train.min()),
        "max_score": float(scores_full_train.max()),
        "outliers_pct": float(100 * outliers_full / len(X_full))
    },
    "THRESHOLD_OPT": float(THRESHOLD)
}
with open(os.path.join(OUTPUT_DIR, "isolation_summary.json"), "w") as f:
    json.dump(model_summary, f, indent=2)

# ===================== UMBRALES DE BOOTSTRAP =====================
bootstrap = pd.read_csv(BOOTSTRAP_FILE)
js_thresh = np.percentile(bootstrap["js_divergence"], 95)
w_thresh = np.percentile(bootstrap["wasserstein"], 95)
with open(os.path.join(OUTPUT_DIR, "bootstrap_summary.json"), "w") as f:
    json.dump({"js_divergence_95": js_thresh, "wasserstein_95": w_thresh}, f, indent=2)

# ====================== EVALUACIÓN DE Pt =========================
results = []
sample_files = sorted([
    f for f in os.listdir(SAMPLES_DIR)
    if f.startswith("Pt_") and f.endswith("pct.csv")
], key=lambda x: int(x.split("_")[1].replace("pct.csv", "")))

for fname in sample_files:
    pct = int(fname.split("_")[1].replace("pct.csv", ""))
    Pt = pd.read_csv(os.path.join(SAMPLES_DIR, fname))
    Pt_np = Pt.drop(columns=["Target", "Label", "source_file"], errors="ignore").to_numpy()

    js = jensen_shannon_divergence(P0_np, Pt_np)
    w = wasserstein_distance_nd(P0_np, Pt_np)

    scores_p0 = clf_p0.decision_function(Pt_np)
    scores_full = clf_full.decision_function(Pt_np)

    is_anomaly_p0 = scores_p0 < -THRESHOLD
    is_anomaly_full = scores_full < -THRESHOLD

    if 'Target' in Pt.columns:
        offensive_idx = Pt[Pt['Target'] == 1].index
        if len(offensive_idx) > 0:
            det_p0 = np.sum(is_anomaly_p0[offensive_idx])
            det_full = np.sum(is_anomaly_full[offensive_idx])
            pct_p0 = 100 * det_p0 / len(offensive_idx)
            pct_full = 100 * det_full / len(offensive_idx)
        else:
            pct_p0 = pct_full = np.nan
    else:
        pct_p0 = pct_full = np.nan

    results.append({
        "proporcion": pct,
        "js_divergence": js,
        "wasserstein": w,
        "porcentaje_detectado_p0": pct_p0,
        "porcentaje_detectado_full": pct_full
    })

df_res = pd.DataFrame(results).sort_values("proporcion")
df_res.to_csv(os.path.join(OUTPUT_DIR, "evaluation_results.csv"), index=False)

# ======================== GRÁFICOS ========================
# Gráfico 1: divergencias
plt.figure()
plt.plot(df_res["proporcion"], df_res["js_divergence"], label="Jensen-Shannon")
plt.plot(df_res["proporcion"], df_res["wasserstein"], label="Wasserstein")
plt.axhline(js_thresh, color='gray', linestyle='--', label="JS p95")
plt.axhline(w_thresh, color='silver', linestyle='--', label="Wasserstein p95")
plt.xlabel("% de eventos ofensivos")
plt.ylabel("Divergencia")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_divergencias_th_opt.png"))

# Gráfico 3: comparación conjunta
plt.figure()
plt.plot(df_res["proporcion"], df_res["js_divergence"], label="JS Divergence")
plt.plot(df_res["proporcion"], df_res["wasserstein"], label="Wasserstein")
plt.plot(df_res["proporcion"], df_res["porcentaje_detectado_full"] / 100,
         label="Detección puntual (full, normalizada)", linestyle="--")
plt.xlabel("% de eventos ofensivos")
plt.ylabel("Medidas normalizadas")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_comparacion_th_opt.png"))

# Histograma bootstrap JS
plt.figure()
plt.hist(bootstrap["js_divergence"], bins=30, color="skyblue", edgecolor="black")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_histogram_js.png"))

# Histograma bootstrap Wasserstein
plt.figure()
plt.hist(bootstrap["wasserstein"], bins=30, color="lightgreen", edgecolor="black")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig5_histogram_wasserstein.png"))

print(f"[✓] Evaluación completa con threshold dinámico = -{THRESHOLD:.5f}. Resultados guardados en {OUTPUT_DIR}")