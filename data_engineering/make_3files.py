#make test.npy train.npy, labels.npy
# make train.npy, test.npy, labels.npy (TranAD compatible)

import numpy as np
import glob
import os

# ===== 設定 =====
DATA_DIR = "roi_timeseries"
FAKE_ANO_DIR = "data/ds004302/fake_ano"
OUT_DIR = "processed/ds004302"

TRAIN_SUB_MAX = 20        # sub-01 ~ sub-20 → train
NORMAL_TEST_SUB_MAX = 25 # sub-21 ~ sub-25 → 正常 test
# sub-26 以降 → 異常 test
# =================

os.makedirs(OUT_DIR, exist_ok=True)

train_data = []
test_data = []
labels = []

files = sorted(glob.glob(f"{DATA_DIR}/sub-*_schaefer100.npy"))

for f in files:
    sub_id = int(os.path.basename(f).split('-')[1].split('_')[0])
    data = np.load(f)   # (T, 100)
    T, D = data.shape   # D=100

    if sub_id <= TRAIN_SUB_MAX:
        # ---- train ----
        train_data.append(data)

    else:
        # ---- test ----
        test_data.append(data)

        if sub_id <= NORMAL_TEST_SUB_MAX:
            # 正常 → 全 ROI 0
            labels.append(np.zeros((T, D), dtype=int))
        else:
            # 異常 → 全 ROI 1
            labels.append(np.ones((T, D), dtype=int))

# ===== concat =====
train = np.concatenate(train_data, axis=0)   # (N_train*T, 100)
test = np.concatenate(test_data, axis=0)     # (N_test*T, 100)
labels = np.concatenate(labels, axis=0)      # (N_test*T, 100)

# ===== 保存 =====
np.save(f"{OUT_DIR}/case2_train.npy", train)
np.save(f"{OUT_DIR}/case2_test.npy", test)
np.save(f"{OUT_DIR}/case2_labels.npy", labels)

print("Saved:")
print(" train :", train.shape)
print(" test  :", test.shape)
print(" labels:", labels.shape)
print(" labels unique:", np.unique(labels))
print(" anomaly ratio:", labels.mean())


