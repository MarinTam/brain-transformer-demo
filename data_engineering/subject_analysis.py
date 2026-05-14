import numpy as np
import matplotlib.pyplot as plt

def analyze_subject(
    sub_id,
    loss,
    labels,
    threshold,
    T=341,
    test_start_sub=26,
):
    """
    sub_id: 実際の被験者番号（例: 26）
    """

    sub_idx = sub_id - test_start_sub
    start = sub_idx * T
    end   = start + T

    score_t = loss.mean(axis=1)

    gt = labels[start:end]
    pred = score_t[start:end] > threshold

    # FN / FP 検出
    fn_idx = np.where((gt == 1) & (~pred))[0]
    fp_idx = np.where((gt == 0) & (pred))[0]

    fig, axs = plt.subplots(3, 1, figsize=(12, 10))

    # ---- (1) 時系列 ----
    axs[0].plot(score_t[start:end], label="score")
    axs[0].axhline(threshold, color="r", linestyle="--", label="threshold")

    if len(fn_idx) > 0:
        axs[0].scatter(
            fn_idx,
            score_t[start + fn_idx],
            color="orange",
            label="FN",
            zorder=5,
        )

    axs[0].set_title(f"sub-{sub_id} anomaly score")
    axs[0].legend()

    # ---- (2) FN時点のROI分布 ----
    if len(fn_idx) > 0:
        t_fn = start + fn_idx[0]
        axs[1].bar(range(loss.shape[1]), loss[t_fn])
        axs[1].axhline(threshold, color="r")
        axs[1].set_title(f"FN ROI distribution (t={t_fn})")
    else:
        axs[1].text(0.5, 0.5, "No FN", ha="center", va="center")
        axs[1].set_title("FN ROI distribution")

    # ---- (3) GT vs Pred ----
    axs[2].plot(gt, label="GT", alpha=0.7)
    axs[2].plot(pred.astype(int), label="Pred", alpha=0.7)
    axs[2].set_title("Ground Truth vs Prediction")
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(f"results/subject_analysis_sub{sub_id}.png", dpi=150)
    print("Saved to results/subject_analysis_sub_.png")

if __name__ == "__main__":
    loss = np.load("results/ds004302_loss.npy")
    labels = np.load("processed/ds004302/labels.npy").mean(axis=1)

    analyze_subject(
        sub_id=26,
        loss=loss,
        labels=labels,
        threshold=3.21,
    )
