"""
Computes 95% CI for ALL prompting methods (and all models) in one run.

HOW TO RUN:
1. pip install pandas numpy scikit-learn openpyxl
2. Put this file in the same folder as GroundTruth.xlsx and all your prediction .xlsx files
3. Edit the MODEL_FILES dict below (add/remove rows as needed)
4. Run:  python3 compute_ci_all.py
5. Output: prints a table AND saves "accuracy_ci_all.csv"
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# ======================================================================
# EDIT THIS: one row per model/prompting-strategy file.
# key   = label used in the output table
# value = path to that model's prediction .xlsx file
# ======================================================================
MODEL_FILES = {
    "GPT-4o-mini (Zero-shot)":                  "0shot_gpt4o_mini.xlsx",
    "GPT-4o-mini (Few-shot)":                   "1shot_gpt4o_mini.xlsx",
    "GPT-4o-mini (Chain-of-Thought)":           "ct_gpt4o_mini.xlsx",
    "GPT-4.1-nano (Zero-shot)":                 "features_openai_4.1nano_zs.xlsx",
    "GPT-4.1-nano (Few-shot)":                  "features_openai_4.1nano_fs.xlsx",
    "GPT-4.1-nano (Chain-of-Thought)":          "features_openai _4.1_ct.xlsx",
    "GPT-5-nano (Zero-shot)":                   "features_openai_gpt5_zs.xlsx",
    "GPT-5-nano (Few-shot)":                    "features_openai_gpt5_fs.xlsx",
    "GPT-5-nano (Chain-of-Thought)":            "features_openai_gpt5_cot.xlsx",
    "Gemini-2.5-flash-lite (Zero-shot)":        "features_gemini.xlsx",
    "Gemini-2.5-flash-lite (Few-shot)":         "biomarkers_gemini.xlsx",
    "Gemini-2.5-flash-lite (Chain-of-Thought)": "biomarkers_gemini_with_steps.xlsx",
}

FEATURES = ["HER2", "ER", "PR"]  

GT_FILE = "GroundTruth.xlsx"
GT_ID_COL = "File Name"
PRED_ID_COL = "Source File Name"
N_BOOT = 2000


def normalize(value):
    """positive / negative / equivocal / not reported"""
    if pd.isna(value):
        return "not reported"
    v = str(value).strip().lower().split("(")[0].strip()
    if v in ["", "nan"] or "not" in v or "pending" in v or "skipped" in v:
        return "not reported"
    if "equivocal" in v or v in ["2+", "2"]:
        return "equivocal"
    if "positive" in v or v in ["3+", "3"]:
        return "positive"
    if "negative" in v or v in ["0", "1+", "1"]:
        return "negative"
    return "not reported"


def dedupe_keep_last_valid(df, id_col, error_marker="error"):
    """Keeps the last non-'Error' row per file (handles API retry duplicates)."""
    def pick(group):
        cols = [c for c in group.columns if c != id_col]
        non_error = group[~group[cols].apply(
            lambda r: all(str(x).strip().lower() == error_marker for x in r), axis=1)]
        return non_error.iloc[-1] if len(non_error) else group.iloc[-1]
    return df.groupby(id_col, as_index=False, sort=False).apply(pick).reset_index(drop=True)


def bootstrap_accuracy_ci(y_true, y_pred, n_boot=N_BOOT, alpha=0.05, seed=42):
    np.random.seed(seed)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = len(y_true)
    accs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.random.randint(0, n, n)
        accs[i] = accuracy_score(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(accs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


# ---------------------------------------------------------------------
# Main loop: every model x every feature
# ---------------------------------------------------------------------
gt = pd.read_excel(GT_FILE)
gt.columns = [c.strip() for c in gt.columns]
gt[GT_ID_COL] = gt[GT_ID_COL].astype(str).str.strip().str.replace(r"\.pdf$", "", regex=True)

rows = []
for model_label, path in MODEL_FILES.items():
    try:
        pred = pd.read_excel(path)
    except FileNotFoundError:
        print(f"[SKIP] {model_label}: file not found -> {path}")
        continue

    pred.columns = [c.strip() for c in pred.columns]
    pred = dedupe_keep_last_valid(pred, PRED_ID_COL)
    pred[PRED_ID_COL] = pred[PRED_ID_COL].astype(str).str.strip().str.replace(r"\.pdf$", "", regex=True)

    merged = pd.merge(gt, pred, left_on=GT_ID_COL, right_on=PRED_ID_COL,
                       how="inner", suffixes=("_gt", "_pred"))

    for feature in FEATURES:
        gt_col = f"{feature}_gt" if f"{feature}_gt" in merged.columns else feature
        pred_col = f"{feature}_pred" if f"{feature}_pred" in merged.columns else feature

        y_true = merged[gt_col].apply(normalize)
        y_pred = merged[pred_col].apply(normalize)

        acc = accuracy_score(y_true, y_pred)
        ci_lo, ci_hi = bootstrap_accuracy_ci(y_true.values, y_pred.values)

        rows.append({
            "Model": model_label,
            "Feature": feature,
            "Accuracy 95% CI": f"[{ci_lo:.3f}, {ci_hi:.3f}]"
        })
    print(f"[DONE] {model_label}  ({len(merged)} matched records)")

results = pd.DataFrame(rows)
results.to_csv("accuracy_ci_all.csv", index=False)

print("\n" + "=" * 70)
print(results.to_string(index=False))
print("\nSaved: accuracy_ci_all.csv")