# prepare_data.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

INPUT_DIR = "input"
OUTPUT_DIR = "processed_data"
EXCLUDE_COLS = ['Flow ID', 'Source IP', 'Destination IP', 'Source Port', 'Destination Port']
MIMETIC_PATTERNS = ["infiltration", "brute"]

def load_all_data():
    all_data = []
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(".csv"):
            path = os.path.join(INPUT_DIR, filename)
            df = pd.read_csv(path, sep=",", engine='python', header=0)
            df.columns = [str(col).strip() for col in df.columns]

            if 'Label' not in df.columns:
                print(f" {filename} sin columna 'Label'. Omitido.")
                continue

            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

            df['Target'] = df['Label'].apply(lambda x: 1 if x != 'BENIGN' else 0)
            df['source_file'] = os.path.basename(filename)
            all_data.append(df)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def compute_interarrival(df):
    if 'Timestamp' in df.columns:
        df = df.sort_values('Timestamp').reset_index(drop=True)
        df['InterArrivalTime'] = df['Timestamp'].diff().dt.total_seconds()
        df['InterArrivalTime'] = df['InterArrivalTime'].fillna(df['InterArrivalTime'].mean())
        df = df.drop(columns=['Timestamp'], errors='ignore')
    return df

def clean_common_columns(df1, df2):
    print("\n Detectando columnas comunes y limpias...")
    numeric1 = df1.select_dtypes(include=['number'])
    numeric2 = df2.select_dtypes(include=['number'])
    common_cols = list(set(numeric1.columns).intersection(set(numeric2.columns)))

    cleaned_cols = []
    for col in common_cols:
        if numeric1[col].nunique() > 1 and numeric2[col].nunique() > 1:
            if not numeric1[col].isna().any() and not numeric2[col].isna().any():
                cleaned_cols.append(col)

    print(f"[✓] Columnas comunes válidas: {len(cleaned_cols)}")
    return cleaned_cols

def scale_subset(df, columns, label):
    print(f"\n Escalando subconjunto: {label}")
    print(f" - Eventos antes: {len(df)}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    X = df[columns].astype(np.float32)  # ⚠️ Reduce a la mitad el uso de RAM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=columns)

    df_scaled["Target"] = df["Target"].values
    df_scaled["Label"] = df["Label"].values
    df_scaled["source_file"] = df["source_file"].values

    print(f" - Eventos tras limpieza final: {len(df_scaled)}")
    return df_scaled

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_all_data()
    print(f" Total eventos cargados: {len(df)}")

    df = compute_interarrival(df)

    # Eliminar columnas pesadas antes del filtrado
    df = df.drop(columns=EXCLUDE_COLS + ['Timestamp'], errors='ignore')

    # Separar con máscaras ligeras
    print(" Filtrando eventos benignos...")
    mask_benign = df["Target"].values == 0
    benign = df.loc[mask_benign]

    print(" Filtrando eventos miméticos...")
    label_col = df["Label"].astype(str).str.lower()
    mask_mimetic = (df["Target"].values == 1) & label_col.str.contains('|'.join(MIMETIC_PATTERNS))
    mimetic = df.loc[mask_mimetic]

    print(f" Eventos benignos: {len(benign)}")
    print(f" Eventos miméticos: {len(mimetic)}")

    if len(mimetic) == 0:
        print("[✗] No hay eventos miméticos para procesar.")
        return

    # Identificar columnas comunes válidas
    common_cols = clean_common_columns(benign, mimetic)

    # Escalar por separado pero con mismas columnas
    benign_scaled = scale_subset(benign, common_cols, "BENIGN")
    mimetic_scaled = scale_subset(mimetic, common_cols, "MIMÉTICOS")


    # Guardar resultados
    benign_scaled.to_csv(os.path.join(OUTPUT_DIR, "benign.csv"), index=False)
    mimetic_scaled.to_csv(os.path.join(OUTPUT_DIR, "attack_mimetic.csv"), index=False)
    
    # Filtrar todos los ataques con las columnas comunes y guardar
    all_attacks_common = df[df["Target"] == 1][common_cols + ["Target", "Label", "source_file"]]
    all_attacks_scaled = scale_subset(all_attacks_common, common_cols, "OFENSIVOS")
    all_attacks_scaled.to_csv(os.path.join(OUTPUT_DIR, "attack_scaled.csv"), index=False)


    print(f"\n Archivos generados en {OUTPUT_DIR}:")
    print(f" - attack.csv con {len(all_attacks_common)} eventos ofensivos totales")
    print(f" - benign.csv: {len(benign_scaled)} eventos")
    print(f" - attack_mimetic.csv: {len(mimetic_scaled)} eventos")

if __name__ == "__main__":
    main()
