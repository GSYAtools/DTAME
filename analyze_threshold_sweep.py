# analyze_threshold_sweep.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load

# Configuración
SAMPLES_DIR = "generated_samples"
OUTPUT_DIR = "evaluation_output"
MODEL_DIR = "models"
THRESHOLDS = [-0.26, -0.2, -0.15, -0.1, -0.05]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cargar modelos entrenado
lf_full = load(os.path.join(MODEL_DIR, "if_full.joblib"))

# Detectar muestras Pt_*
sample_files = sorted([
    f for f in os.listdir(SAMPLES_DIR)
    if f.startswith("Pt_") and f.endswith("pct.csv")
], key=lambda x: int(x.split("_")[1].replace("pct.csv", "")))

# Inicializar estructura para resultados
results = []
proporciones = []

for fname in sample_files:
    pct = int(fname.split("_")[1].replace("pct.csv", ""))
    proporciones.append(pct)

    Pt = pd.read_csv(os.path.join(SAMPLES_DIR, fname))
    Pt_np = Pt.drop(columns=["Target", "Label", "source_file"], errors="ignore").to_numpy()

    if "Target" not in Pt.columns:
        print(f" 'Target' no encontrado en {fname}, se omite.")
        continue

    offensive_idx = Pt[Pt["Target"] == 1].index

    for thresh in THRESHOLDS:
        pred_full = lf_full.decision_function(Pt_np) < thresh

        if len(offensive_idx) > 0:
            det_full = np.sum(pred_full[offensive_idx])
            pct_full = 100 * det_full / len(offensive_idx)
        else:
            pct_full = np.nan

        results.append({
            "proporcion": pct,
            "threshold": thresh,
            "pct_detectado_full": pct_full
        })

# Guardar CSV
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(OUTPUT_DIR, "threshold_sweep_results.csv"), index=False)

# Gráfico tipo mosaico
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 12), sharex=True, sharey=True)

for i, thresh in enumerate(THRESHOLDS):
    ax_full = axes[i]

    subset = df_results[df_results["threshold"] == thresh].sort_values("proporcion")

    ax_full.plot(subset["proporcion"], subset["pct_detectado_full"], marker="s", color="green")
    ax_full.set_ylabel(f"Umbral {thresh}")
    ax_full.grid(True)
    if i == len(THRESHOLDS) - 1:
        ax_full.set_xlabel("% inyección de eventos ofensivos evasivos")
    
fig.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_mosaico_thresholds.png"))

print(" Mosaico guardado en:", os.path.join(OUTPUT_DIR, "fig2_mosaico_thresholds.png"))
print(" Resultados guardados en:", os.path.join(OUTPUT_DIR, "threshold_sweep_results.csv"))