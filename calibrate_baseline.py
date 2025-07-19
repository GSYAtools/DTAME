# calibrate_baseline.py

import os
import pandas as pd
import numpy as np
from sklearn.utils import resample
from compute_divergences import jensen_shannon_divergence, wasserstein_distance_nd

# Parámetros
INPUT_FILE = "processed_data/benign.csv"
OUTPUT_FILE = "processed_data/bootstrap_divergences.csv"
N_BOOTSTRAP = 100
SAMPLE_SIZE = 1000  # número de muestras por subconjunto

def main():
    print(f"[✓] Cargando datos desde {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)

    # Eliminar columnas auxiliares
    df = df.drop(columns=["Target", "Label", "source_file"], errors="ignore")

    # Convertir a NumPy
    data = df.to_numpy()

    # Crear referencia P0 como muestra aleatoria
    P0 = resample(data, replace=False, n_samples=SAMPLE_SIZE, random_state=42)

    results = []
    print(f"[✓] Iniciando bootstrap con {N_BOOTSTRAP} repeticiones...")

    for i in range(N_BOOTSTRAP):
        Pt_b = resample(data, replace=False, n_samples=SAMPLE_SIZE, random_state=i)
        js = jensen_shannon_divergence(P0, Pt_b)
        w = wasserstein_distance_nd(P0, Pt_b)
        results.append({
            "bootstrap_id": i,
            "js_divergence": js,
            "wasserstein": w
        })

    # Guardar resultados
    df_results = pd.DataFrame(results)
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_results.to_csv(OUTPUT_FILE, index=False)
    print(f"[✓] Resultados guardados en {OUTPUT_FILE}")
    print(df_results.head())

if __name__ == "__main__":
    main()