"""
Batch runner for Comment 1 full metrics.

HOW TO USE:
1. Place this file in the same folder as GroundTruth.xlsx and all your
   features_*.xlsx / *_gpt4o_mini.xlsx prediction files.
2. Edit the MODEL_FILES dict below: add one entry per model/prompting
   combination you want scored. key = label for tables/plots,
   value = path to that model's prediction .xlsx file.
3. Run: python3 batch_full_metrics.py
4. Outputs:
   - metrics_master_table_ALL.csv   (one row per model x feature x class)
   - confmat_<model_label>.png      (2x2 panel: ER, PR, HER2, Tumor Size)
   - accuracy_summary_ALL.csv       (overall accuracy + 95% CI per model x feature,
                                     ready to paste as the revised Table 2)

This reuses the corrected normalization from full_metrics.py:
  - Equivocal HER2 kept as its own class (not merged into "not reported")
  - File-name join normalized (strips .pdf, handles missing extension)
  - Retry/duplicate rows deduplicated (keeps last non-Error row per file)
"""
import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              precision_recall_fscore_support)
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
N_BOOT = 2000

# =========================================================================
# EDIT THIS: one entry per model/prompting-strategy file you want scored.
# Column name inside each file for HER2/ER/PR/Tumor Size is assumed to be
# "HER2","ER","PR","Tumor Size" and the ID column "Source File Name" —
# adjust PRED_COLS/PRED_ID_COL below per-model if a file differs.
# =========================================================================
MODEL_FILES = {
    "GPT-4o-mini (Zero-shot)":        "0shot_gpt4o_mini.xlsx",
    "GPT-4o-mini (Few-shot)":         "1shot_gpt4o_mini.xlsx",
    "GPT-4o-mini (Chain-of-Thought)": "ct_gpt4o_mini.xlsx",
    "GPT-4.1-nano (Zero-shot)":       "features_openai_4.1nano_zs.xlsx",
    "GPT-4.1-nano (Few-shot)":        "features_openai_4.1nano_fs.xlsx",
    "GPT-4.1-nano (Chain-of-Thought)": "features_openai _4.1_ct.xlsx",
    "GPT-5-nano (Zero-shot)":         "features_openai_gpt5_zs.xlsx",
    "GPT-5-nano (Chain-of-Thought)":  "features_openai_gpt5_cot.xlsx",
    "Gemini-2.5-flash-lite (Zero-shot)": "features_gemini.xlsx",
    "Gemini-2.5-flash-lite (Few-shot)":  "biomarkers_gemini.xlsx",
    "Gemini-2.5-flash-lite (Chain-of-Thought)": "biomarkers_gemini_with_steps.xlsx",
}

GT_FILE = "GroundTruth.xlsx"
GT_ID_COL = "File Name"
GT_COLS = {"HER2": "HER2", "ER": "ER", "PR": "PR", "Tumor Size": "Tumor Size"}
PRED_ID_COL_DEFAULT = "Source File Name"
PRED_COLS_DEFAULT = {"HER2": "HER2", "ER": "ER", "PR": "PR", "Tumor Size": "Tumor Size"}


def normalize_her2(value):
    if pd.isna(value):
        return "not reported"
    v = str(value).strip().lower()
    v = v.split("(")[0].strip()
    if v in ["", "nan"]:
        return "not reported"
    if "equivocal" in v:
        return "equivocal"
    if "pending" in v or "skipped" in v or "not" in v:
        return "not reported"
    if v in ["0", "1+", "1"]:
        return "negative"
    if v in ["2+", "2"]:
        return "equivocal"
    if v in ["3+", "3"] or "positive" in v:
        return "positive"
    if "negative" in v:
        return "negative"
    return "not reported"


def normalize_binary(value):
    if pd.isna(value):
        return "not reported"
    v = str(value).strip().lower()
    v = v.split("(")[0].strip()
    if v in ["", "nan"] or "pending" in v or "skipped" in v or "not" in v:
        return "not reported"
    if "positive" in v or v == "pos":
        return "positive"
    if "negative" in v or v == "neg":
        return "negative"
    return "not reported"


def extract_tumor_size(value):
    if pd.isna(value):
        return "not reported"
    v = str(value).strip().lower()
    if "not" in v or "error" in v or v == "" or "nan" in v:
        return "not reported"
    v = v.replace("cm", "").replace("mm", "").replace("x", " ").replace("×", " ")
    v = v.replace('"', "").replace("'", "")
    nums = re.findall(r"[\d.]+", v)
    if not nums:
        return "not reported"
    try:
        nums = [float(n) for n in nums if float(n) > 0]
    except Exception:
        return "not reported"
    if not nums:
        return "not reported"
    max_size = max(nums)
    if 10 < max_size < 100:
        max_size = max_size / 10
    return round(max_size, 2)


def categorize_tumor_size(size):
    if size == "not reported":
        return "not reported"
    try:
        size = float(size)
        return "< 2 cm" if size < 2 else ("2-5 cm" if size < 5 else "> 5 cm")
    except Exception:
        return "not reported"


def dedupe_keep_last_valid(df, id_col, error_marker="error"):
    def pick(group):
        cols = [c for c in group.columns if c != id_col]
        non_error = group[~group[cols].apply(
            lambda r: all(str(x).strip().lower() == error_marker for x in r), axis=1)]
        return non_error.iloc[-1] if len(non_error) else group.iloc[-1]
    return df.groupby(id_col, as_index=False, sort=False).apply(pick).reset_index(drop=True)


def bootstrap_accuracy_ci(y_true, y_pred, n_boot=N_BOOT, alpha=0.05):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = len(y_true)
    if n == 0:
        return np.nan, np.nan
    accs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.random.randint(0, n, n)
        accs[i] = accuracy_score(y_true[idx], y_pred[idx])
    return np.percentile(accs, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def specificity_per_class(cm, labels):
    total = cm.sum()
    specs = {}
    for i, lab in enumerate(labels):
        tn = total - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        fp = cm[:, i].sum() - cm[i, i]
        specs[lab] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return specs


def evaluate_one(gt_df, pred_path, model_label, pred_id_col=None, pred_cols=None):
    pred_id_col = pred_id_col or PRED_ID_COL_DEFAULT
    pred_cols = pred_cols or PRED_COLS_DEFAULT

    pred_df = pd.read_excel(pred_path)
    pred_df.columns = [c.strip() for c in pred_df.columns]
    pred_df = dedupe_keep_last_valid(pred_df, pred_id_col)

    gt_df = gt_df.copy()
    pred_df = pred_df.copy()
    gt_df[GT_ID_COL] = gt_df[GT_ID_COL].astype(str).str.strip().str.replace(r"\.pdf$", "", regex=True)
    pred_df[pred_id_col] = pred_df[pred_id_col].astype(str).str.strip().str.replace(r"\.pdf$", "", regex=True)

    gt_rename = {v: f"{v}__gt" for v in GT_COLS.values()}
    pred_rename = {v: f"{v}__pred" for v in pred_cols.values()}
    gt_r = gt_df.rename(columns=gt_rename)
    pred_r = pred_df.rename(columns=pred_rename)
    gt_cols_r = {k: f"{v}__gt" for k, v in GT_COLS.items()}
    pred_cols_r = {k: f"{v}__pred" for k, v in pred_cols.items()}

    merged = pd.merge(gt_r, pred_r, left_on=GT_ID_COL, right_on=pred_id_col, how="inner")
    n_matched = len(merged)

    targets = {
        "HER2": (gt_cols_r["HER2"], pred_cols_r["HER2"], normalize_her2),
        "ER": (gt_cols_r["ER"], pred_cols_r["ER"], normalize_binary),
        "PR": (gt_cols_r["PR"], pred_cols_r["PR"], normalize_binary),
        "Tumor Size": (gt_cols_r["Tumor Size"], pred_cols_r["Tumor Size"], "tumor"),
    }

    rows, cms = [], {}
    for target, (gcol, pcol, fn) in targets.items():
        if target == "Tumor Size":
            y_true = merged[gcol].apply(extract_tumor_size).apply(categorize_tumor_size)
            y_pred = merged[pcol].apply(extract_tumor_size).apply(categorize_tumor_size)
        else:
            y_true = merged[gcol].apply(fn)
            y_pred = merged[pcol].apply(fn)

        labels = sorted(set(y_true.unique()) | set(y_pred.unique()))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cms[target] = (cm, labels)

        acc = accuracy_score(y_true, y_pred)
        ci_lo, ci_hi = bootstrap_accuracy_ci(y_true.values, y_pred.values)
        prec, rec, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0)
        specs = specificity_per_class(cm, labels)

        for lab, p, r, f, s in zip(labels, prec, rec, f1, support):
            rows.append({
                "Model": model_label, "Target": target, "Class": lab,
                "Support (n)": s, "Precision": round(p, 3),
                "Recall (Sensitivity)": round(r, 3),
                "Specificity": round(specs[lab], 3) if not np.isnan(specs[lab]) else np.nan,
                "F1": round(f, 3), "Overall Accuracy": round(acc, 4),
                "Accuracy 95% CI": f"[{ci_lo:.3f}, {ci_hi:.3f}]",
                "N Matched Records": n_matched
            })
    return pd.DataFrame(rows), cms, n_matched


def plot_confmats(cms, model_label, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for ax, (target, (cm, labels)) in zip(axes, cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
        ax.set_title(target, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
    fig.suptitle(model_label, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    import os
    gt = pd.read_excel(GT_FILE)
    gt.columns = [c.strip() for c in gt.columns]

    all_rows = []
    for label, path in MODEL_FILES.items():
        if not os.path.exists(path):
            print(f"[SKIP] {label}: file not found -> {path}")
            continue
        print(f"[RUN]  {label}  <-  {path}")
        df_res, cms, n = evaluate_one(gt, path, label)
        all_rows.append(df_res)
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
        plot_confmats(cms, label, f"confmat_{safe_name}.png")
        print(f"       matched {n}/{len(gt)} records")

    if all_rows:
        master = pd.concat(all_rows, ignore_index=True)
        master.to_csv("metrics_master_table_ALL.csv", index=False)

        summary = (master[["Model", "Target", "Overall Accuracy", "Accuracy 95% CI", "N Matched Records"]]
                   .drop_duplicates())
        summary.to_csv("accuracy_summary_ALL.csv", index=False)

        print("\nSaved: metrics_master_table_ALL.csv, accuracy_summary_ALL.csv, confmat_<model>.png")
    else:
        print("\nNo files found — check MODEL_FILES paths.")