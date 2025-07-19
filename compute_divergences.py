# compute_divergence.py

import numpy as np
from scipy.stats import entropy
from scipy.stats import wasserstein_distance

def jensen_shannon_divergence(P, Q):
    """
    Calcula la divergencia de Jensen-Shannon entre dos matrices multivariadas P y Q
    usando histogramas normalizados en cada dimensión (bins=100).

    Parameters:
        P, Q: numpy arrays con forma (n_samples, n_features)

    Returns:
        Media de las divergencias JS por dimensión
    """
    assert P.shape[1] == Q.shape[1], "Dimensiones incompatibles entre P y Q"
    js_values = []

    for dim in range(P.shape[1]):
        p_hist, _ = np.histogram(P[:, dim], bins=100, density=True)
        q_hist, _ = np.histogram(Q[:, dim], bins=100, density=True)

        # Asegurar que no hay ceros para evitar log(0)
        p_hist += 1e-8
        q_hist += 1e-8
        p_hist /= np.sum(p_hist)
        q_hist /= np.sum(q_hist)

        m = 0.5 * (p_hist + q_hist)
        js = 0.5 * entropy(p_hist, m, base=2) + 0.5 * entropy(q_hist, m, base=2)
        js_values.append(js)

    return np.mean(js_values)

def wasserstein_distance_nd(P, Q):
    """
    Calcula la distancia de Wasserstein media por dimensión entre dos matrices multivariadas.

    Parameters:
        P, Q: numpy arrays con forma (n_samples, n_features)

    Returns:
        Media de las distancias Wasserstein por dimensión
    """
    assert P.shape[1] == Q.shape[1], "Dimensiones incompatibles entre P y Q"
    wass_values = []

    for dim in range(P.shape[1]):
        w = wasserstein_distance(P[:, dim], Q[:, dim])
        wass_values.append(w)

    return np.mean(wass_values)