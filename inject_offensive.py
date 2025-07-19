# inject_offensive.py

import os
import pandas as pd
import numpy as np
from sklearn.utils import resample

# Configuración
BENIGN_FILE = "processed_data/benign.csv"
ATTACK_FILE = "processed_data/attack_mimetic.csv"
OUTPUT_DIR = "generated_samples"
PROPORTIONS = [1, 2, 5, 10, 15, 20, 25]
SAMPLE_SIZE = 1000
MIMETIC_PATTERNS = ["brute", "infiltration"]

def get_consecutive_window(df, n_events, seed=0):
    if len(df) < n_events:
        raise ValueError(f"No hay suficientes eventos ofensivos ({len(df)}) para inyectar {n_events}.")
    start = np.random.RandomState(seed).randint(0, len(df) - n_events + 1)
    return df.iloc[start:start + n_events]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    benign_df = pd.read_csv(BENIGN_FILE)
    attack_df = pd.read_csv(ATTACK_FILE)

    # Filtrar solo eventos miméticos de las clases seleccionadas
    attack_df["Label"] = attack_df["Label"].astype(str).str.lower()
    filtered_attack = attack_df[attack_df["Label"].str.contains('|'.join(MIMETIC_PATTERNS))].reset_index(drop=True)

    print(f" Eventos ofensivos seleccionados ('brute' o 'infiltration'): {len(filtered_attack)}")

    drop_cols = ["Label", "source_file"]
    benign_df = benign_df.drop(columns=drop_cols, errors="ignore")
    filtered_attack = filtered_attack.drop(columns=drop_cols, errors="ignore")

    # Generar P0
    P0 = resample(benign_df, replace=False, n_samples=SAMPLE_SIZE, random_state=0)
    P0.to_csv(os.path.join(OUTPUT_DIR, "P0.csv"), index=False)
    print(f"[✓] P0 generado con {len(P0)} eventos")

    # Generar muestras P_t
    for p in PROPORTIONS:
        n_attack = int(SAMPLE_SIZE * (p / 100))
        n_benign = SAMPLE_SIZE - n_attack

        attack_sample = get_consecutive_window(filtered_attack, n_attack, seed=p)
        benign_sample = resample(benign_df, replace=False, n_samples=n_benign, random_state=100 + p)

        Pt = pd.concat([benign_sample, attack_sample], ignore_index=True)
        Pt = Pt.sample(frac=1, random_state=p).reset_index(drop=True)

        out_path = os.path.join(OUTPUT_DIR, f"Pt_{p}pct.csv")
        Pt.to_csv(out_path, index=False)
        print(f"[✓] P_t con {p}% ataque ('brute'/'infiltration') guardado en: {out_path}")

if __name__ == "__main__":
    main()