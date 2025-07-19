# Vigilancia estructural de ciberataques evasivos mediante ruptura de simetría ecosistémica

Este repositorio contiene los scripts y recursos necesarios para reproducir el experimento descrito en el artículo:

**Cibernética inversa: detección de ciberataques evasivos mediante ruptura de simetrí­a ecosistémica**  
Carlos Mario Braga, Manuel A. Serrano, Eduardo Fernádez-Medina  
Universidad de Castilla-La Mancha

---

## Objetivo del proyecto

Detectar ataques evasivos que no se manifiestan como anomalí­as locales, sino como **rupturas de simetrí­a estadítica** del ecosistema digital. Se utilizan métricas de divergencia como **Jensen-Shannon (JS)** y **Wasserstein (W1)** para anticipar perturbaciones estructurales sin supervisión.

---

## Estructura de scripts

### `prepare_data.py`
Preprocesa el dataset **CIC IDS 2017**, generando vectores normalizados y limpios.  
Genera: `processed_data/benign.csv`, `attack_mimetic.csv`, `attack.csv`  
Incluye filtrado por tipo de evento, limpieza de columnas, codificación, normalización z-score.

### `inject_offensive.py`
Simula perturbaciones ofensivas inyectando eventos miméticos (`brute`, `infiltration`).  
Genera: `generated_samples/Pt_Xpct.csv`, `P0.csv`

### `compute_divergences.py`
Calcula la divergencia JS y la distancia W1 entre dos muestras.  
Espera datos escalados y alineados.

### `calibrate_baseline.py`
Bootstrap sobre subconjuntos benignos para construir la distribución empí­rica base.  
Genera: `processed_data/bootstrap_divergences.csv`

### `analyze_calibration_stats.py`
Calcula percentiles, medias y umbrales operativos a partir del bootstrap.  
Genera: `processed_data/bootstrap_summary.json`

### `evaluate_divergences.py`
Evalúa sensibilidad de JS/W1 vs. **Isolation Forest**.  
Salidas: CSVs, gráficos (`evaluation_output/`), modelos IF (`models/`).

### `analyze_threshold_sweep.py`
Analiza cómo cambia la detección de Isolation Forest al variar el umbral.  
Salida: `fig2_mosaico_thresholds.png` y `threshold_sweep_results.csv`

---

## Instalación

```bash
git clone https://github.com/<tu_usuario>/ecosystem-anomaly-detection.git
cd ecosystem-anomaly-detection
pip install -r requirements.txt
```

**Requisitos principales:**  
- Python 3.8+  
- numpy, pandas, scikit-learn, matplotlib, scipy, joblib

---

## Ejecución paso a paso

```bash
python prepare_data.py               # Preprocesamiento inicial
python inject_offensive.py           # Inyección de ataques miméticos
python calibrate_baseline.py         # Bootstrap sobre benignos
python analyze_calibration_stats.py  # Umbrales empíricos
python evaluate_divergences.py       # Comparación estructural vs. puntual
python analyze_threshold_sweep.py    # Barrido de umbrales (Isolation Forest)
```

## Resultados esperados

- **JS** activa señal estructural con solo 1% de eventos ofensivos.
- **W1** crece sistemáticamente con el porcentaje de inyección de ataques miméticos.
- **Isolation Forest** no detecta ataques miméticos en fases tempranas.
- Umbrales derivados del bootstrap permiten distinguir ruido de ruptura.


---
*Contacto: carlosmario.braga1@alu.uclm.es*
