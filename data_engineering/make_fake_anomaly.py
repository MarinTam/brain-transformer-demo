# analysis/make_fake_anomaly.py
'''擬似異常注入の実行ファイル
'''

import os
import numpy as np
from pseudo_anomaly import (
    inject_temporal_drift,
    inject_spatial_spike,
    inject_correlation_break
)

NORMAL_DIR = "processed/ds004302/train"
OUT_DIR = "data/ds004302/fake_ano"

SUBJECTS = range(21, 26)
#ALPHAS = [1.0, 2.0, 3.0]
ALPHAS_TEMP = [0.2, 0.5, 1.0]
ALPHAS_SPAT = [0.5, 1.0, 2.0]#一瞬なので大きめ
ALPHAS_REL  = [0.3, 0.7, 1.0]#壊れやすいので控えめ


os.makedirs(OUT_DIR, exist_ok=True)
SUB_DIRS = [
    f"{OUT_DIR}/temporal",
    f"{OUT_DIR}/spatial",
    f"{OUT_DIR}/relation",
]

for d in SUB_DIRS:
    os.makedirs(d, exist_ok=True)

for sub in SUBJECTS:
    X = np.load(f"{NORMAL_DIR}/sub-{sub:02d}_schaefer100.npy")

    # temporal
    for alpha in ALPHAS_TEMP:
        Xt = inject_temporal_drift(X, roi=10, start=150, alpha=alpha)
        np.save(f"{OUT_DIR}/temporal/sub{sub:02d}_alpha{alpha}.npy", Xt)

    # spatial
    for alpha in ALPHAS_SPAT:
        Xs = inject_spatial_spike(X, rois=[20,21], t=200, alpha=alpha)
        np.save(f"{OUT_DIR}/spatial/sub{sub:02d}_alpha{alpha}.npy", Xs)

    # relational
    for alpha in ALPHAS_REL:
        Xr = inject_correlation_break(X, roi_a=5, roi_b=30, start=150)
        np.save(f"{OUT_DIR}/relation/sub{sub:02d}_alpha{alpha}.npy", Xr)

