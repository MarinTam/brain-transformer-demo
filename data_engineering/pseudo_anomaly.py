# analysis/pseudo_anomaly.py
'''擬似異常注入モジュール（改良版）1/5
'''
import numpy as np

def inject_temporal_drift(X, roi, start, alpha=2.0):
    X2 = X.copy()
    T = X.shape[0]
    drift = alpha * np.linspace(0, 1, T - start)
    X2[start:, roi] += drift
    return X2

def inject_spatial_spike(X, rois, t, alpha=3.0):
    X2 = X.copy()
    sigma = np.std(X[:, rois], axis=0)
    X2[t, rois] += alpha * sigma
    return X2

def inject_correlation_break(X, roi_a, roi_b, start, seed=None):
    X2 = X.copy()
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.randn(X.shape[0] - start)
    X2[start:, roi_b] = noise
    return X2
