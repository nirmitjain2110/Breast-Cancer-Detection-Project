import pandas as pd

# File paths
groundtruth_file = "GroundTruth.xlsx"
reports_file = "features_openai_gpt5_cot.xlsx"

markers = ["her2", "er", "pr"]

# Read Excel files
gt_df = pd.read_excel(groundtruth_file)
rep_df = pd.read_excel(reports_file)

# Normalize column names
gt_df.columns = gt_df.columns.str.strip().str.lower()
rep_df.columns = rep_df.columns.str.strip().str.lower()

# ID columns
gt_id_col = "file name"
rep_id_col = "source file name"

# Normalize marker values
for m in markers:
    gt_df[m] = gt_df[m].astype(str).str.lower().str.strip()
    rep_df[m] = rep_df[m].astype(str).str.lower().str.strip()

# Merge Ground Truth and Predictions
merged_df = pd.merge(
    gt_df,
    rep_df,
    left_on=gt_id_col,
    right_on=rep_id_col,
    suffixes=("_gt", "_pred"),
    how="inner"
)

print("\n===== FULLY NEGATIVE (HER2-, ER-, PR-) ANALYSIS =====")

# Fully negative in Ground Truth
gt_fully_negative = (
    (merged_df["her2_gt"] == "negative") &
    (merged_df["er_gt"] == "negative") &
    (merged_df["pr_gt"] == "negative")
)

# Fully negative in Predictions
pred_fully_negative = (
    (merged_df["her2_pred"] == "negative") &
    (merged_df["er_pred"] == "negative") &
    (merged_df["pr_pred"] == "negative")
)

gt_full_count = gt_fully_negative.sum()
pred_full_count = pred_fully_negative.sum()
correct_full_count = (gt_fully_negative & pred_fully_negative).sum()

full_accuracy = (
    (correct_full_count / gt_full_count) * 100
    if gt_full_count > 0 else 0
)

print(f"Ground Truth fully negative cases     : {gt_full_count}")
print(f"GPT5 Predicted fully negative cases   : {pred_full_count}")
print(f"Correctly predicted fully negative    : {correct_full_count}")
print(f"Fully negative accuracy               : {round(full_accuracy, 2)} %")

print("\n===== PER-MARKER NEGATIVE ACCURACY =====")

# Per-marker negative accuracy
for m in markers:
    gt_col = f"{m}_gt"
    pred_col = f"{m}_pred"

    gt_negative = merged_df[merged_df[gt_col] == "negative"]

    total = len(gt_negative)
    correct = (gt_negative[pred_col] == "negative").sum()

    accuracy = (correct / total) * 100 if total > 0 else 0

    print(
        f"{m.upper():<5} | "
        f"GT Negatives: {total:<4} | "
        f"Correct: {correct:<4} | "
        f"Accuracy: {round(accuracy, 2)} %"
    )
