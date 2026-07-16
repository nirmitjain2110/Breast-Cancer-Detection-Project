

import pandas as pd
import numpy as np
import re
import warnings
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# ========= 1. Load the Excel files =========
gt_path = "GroundTruth.xlsx"
pred_gemini_path = "ct_gpt4o_mini.xlsx"

df_gt = pd.read_excel(gt_path)
df_pred_gemini = pd.read_excel(pred_gemini_path)

# ========= 2. Merge on File Name =========
df_merged = pd.merge(
    df_gt,
    df_pred_gemini,
    left_on="File Name",
    right_on="Source File Name",
    how="inner",
    suffixes=("_gt", "_pred")
)

print(f"Total matched records (Gemini): {len(df_merged)}\n")

# ==========================================================
#               NORMALIZATION FUNCTIONS
# ==========================================================

def normalize_her2(value):
    """
    Convert HER2 scores to binary classification
    0,1+,2+ -> negative
    3+ -> positive
    """
    if pd.isna(value):
        return "not reported"

    value = str(value).strip().lower()

    if "not" in value or value == "" or value == "nan":
        return "not reported"

    if value in ["0", "1", "2", "1+", "2+"]:
        return "negative"

    if value in ["3", "3+", "positive"]:
        return "positive"

    return "not reported"


def normalize_binary(value):
    """
    Normalize ER and PR values
    """
    if pd.isna(value):
        return "not reported"

    value = str(value).strip().lower()

    if "not" in value or value == "" or value == "nan":
        return "not reported"

    if "positive" in value or value == "pos":
        return "positive"

    if "negative" in value or value == "neg":
        return "negative"

    return "not reported"


def extract_tumor_size(value):
    """
    Extract largest tumor size and convert to cm.
    """

    if pd.isna(value):
        return "not reported"

    value = str(value).lower().strip()

    if value == "" or "not" in value or value == "nan":
        return "not reported"

    value = (
        value.replace("cm", "")
        .replace("mm", "")
        .replace("×", " ")
        .replace("x", " ")
    )

    numbers = re.findall(r"[\d.]+", value)

    if len(numbers) == 0:
        return "not reported"

    try:
        numbers = [float(i) for i in numbers if float(i) > 0]
    except:
        return "not reported"

    if len(numbers) == 0:
        return "not reported"

    largest = max(numbers)

    # assume values >10 are in mm
    if 10 < largest < 100:
        largest = largest / 10

    return round(largest, 2)


def categorize_tumor_size(size):

    if size == "not reported":
        return "not reported"

    try:
        size = float(size)

        if size < 2:
            return "< 2 cm"

        elif size < 5:
            return "2-5 cm"

        else:
            return "> 5 cm"

    except:
        return "not reported"


# ==========================================================
# Apply Normalization
# ==========================================================

print("Normalizing data...\n")

# HER2
df_merged["HER2_gt_norm"] = df_merged["HER2_gt"].apply(normalize_her2)
df_merged["HER2_pred_norm"] = df_merged["HER2_pred"].apply(normalize_her2)

# ER
df_merged["ER_gt_norm"] = df_merged["ER_gt"].apply(normalize_binary)
df_merged["ER_pred_norm"] = df_merged["ER_pred"].apply(normalize_binary)

# PR
df_merged["PR_gt_norm"] = df_merged["PR_gt"].apply(normalize_binary)
df_merged["PR_pred_norm"] = df_merged["PR_pred"].apply(normalize_binary)

# Tumor Size
df_merged["tumor_size_gt_raw"] = df_merged["Tumor Size "].apply(
    extract_tumor_size
)
df_merged["tumor_size_pred_raw"] = df_merged["Tumor Size"].apply(
    extract_tumor_size
)

df_merged["tumor_size_gt_cat"] = df_merged["tumor_size_gt_raw"].apply(
    categorize_tumor_size
)

df_merged["tumor_size_pred_cat"] = df_merged["tumor_size_pred_raw"].apply(
    categorize_tumor_size
)

# ==========================================================
# Evaluation
# ==========================================================

targets = {
    "HER2": ("HER2_gt_norm", "HER2_pred_norm"),
    "ER": ("ER_gt_norm", "ER_pred_norm"),
    "PR": ("PR_gt_norm", "PR_pred_norm"),
    "Tumor Size": ("tumor_size_gt_cat", "tumor_size_pred_cat"),
}

results = []

print("=" * 90)
print("MODEL EVALUATION METRICS")
print("=" * 90)

for target_name, (gt_col, pred_col) in targets.items():

    y_true = df_merged[gt_col]
    y_pred = df_merged[pred_col]

    # ---------------- Accuracy ----------------
    accuracy = accuracy_score(y_true, y_pred)

    # ---------------- Precision ----------------
    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # ---------------- Recall ----------------
    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # ---------------- F1 ----------------
    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    # ---------------- Specificity ----------------

    labels = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    specificity_scores = []

    for i in range(len(labels)):

        TP = cm[i, i]

        FN = cm[i, :].sum() - TP

        FP = cm[:, i].sum() - TP

        TN = cm.sum() - TP - FP - FN

        if (TN + FP) == 0:
            specificity_scores.append(0)

        else:
            specificity_scores.append(TN / (TN + FP))

    specificity = np.mean(specificity_scores)

    print("\n" + "=" * 70)
    print(target_name)
    print("=" * 70)

    print(f"Accuracy    : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1 Score    : {f1:.4f}")
    print(f"Specificity : {specificity:.4f}")

    print("\nClassification Report\n")
    print(classification_report(y_true, y_pred, zero_division=0))

    results.append({
        "Target": target_name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Precision (%)": round(precision * 100, 2),
        "Recall (%)": round(recall * 100, 2),
        "F1 Score (%)": round(f1 * 100, 2),
        "Specificity (%)": round(specificity * 100, 2),
    })

# ==========================================================
# Summary Table
# ==========================================================

summary_df = pd.DataFrame(results)

print("\n")
print("=" * 90)
print("SUMMARY")
print("=" * 90)

print(summary_df.to_string(index=False))

# ==========================================================
# Save Results
# ==========================================================

summary_df.to_csv(
    "evaluation_metrics_gemini.csv",
    index=False,
)

print("\n✓ Saved evaluation_metrics_gemini.csv")

# ==========================================================
# Sample Predictions
# ==========================================================

print("\n")
print("=" * 90)
print("FIRST 10 SAMPLES")
print("=" * 90)

sample_columns = [
    "File Name",
    "HER2_gt",
    "HER2_gt_norm",
    "HER2_pred",
    "HER2_pred_norm",
    "ER_gt",
    "ER_gt_norm",
    "ER_pred",
    "ER_pred_norm",
    "PR_gt",
    "PR_gt_norm",
    "PR_pred",
    "PR_pred_norm",
    "Tumor Size ",
    "tumor_size_gt_cat",
    "Tumor Size",
    "tumor_size_pred_cat",
]

print(df_merged[sample_columns].head(10).to_string(index=False))

print("\n✓ Analysis Complete!")
print("Generated File:")
print("1. evaluation_metrics_gemini.csv")