# extract_roi_timeseries.py 田屋さんのデータ用（再開対応版）

import os
import numpy as np
import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker
from glob import glob
from tqdm import tqdm

# ===== paths =====
bold_root = "/Storage2/brain_group/dataset/ds004302/derivatives"
atlas_path = "/home/tamarin/.cache/templateflow/tpl-MNI152NLin2009cAsym/" \
             "tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-100Parcels17Networks_dseg.nii.gz"
out_dir = "./roi_timeseries"
os.makedirs(out_dir, exist_ok=True)

# ===== masker =====
masker = NiftiLabelsMasker(
    labels_img=atlas_path,
    standardize=True,
    detrend=True,
    t_r=None
)

# ===== find bold files =====
bold_files = sorted(glob(
    f"{bold_root}/sub-*/func/*space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
))

print(f"Found {len(bold_files)} BOLD files")

# ===== extract ROI signals =====
for bold_path in tqdm(bold_files):

    sub = bold_path.split("/")[-3]
    out_path = os.path.join(out_dir, f"{sub}_schaefer100.npy")

    # ★ すでに処理済みならスキップ
    if os.path.exists(out_path):
        continue

    try:
        ts = masker.fit_transform(bold_path)  # shape: [T, 100]
        np.save(out_path, ts)

    except Exception as e:
        # ★ エラーが出ても止めない
        print(f"[ERROR] {sub}: {e}")
        continue

