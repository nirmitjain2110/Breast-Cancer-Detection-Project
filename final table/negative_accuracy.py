import pandas as pd

# ================= FILE PATHS =================
groundtruth_file = "GroundTruth.xlsx"
reports_file = "features_openai_gpt5_cot.xlsx"

markers = ["her2", "er", "pr"]

# ================= READ EXCEL =================
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

# ================= MERGE =================
merged_df = pd.merge(
    gt_df,
    rep_df,
    left_on=gt_id_col,
    right_on=rep_id_col,
    suffixes=("_gt", "_pred"),
    how="inner"
)

# ================= FULLY NEGATIVE ANALYSIS =================
gt_fully_negative = (
    (merged_df["her2_gt"] == "negative") &
    (merged_df["er_gt"] == "negative") &
    (merged_df["pr_gt"] == "negative")
)

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

print("\n===== FULLY NEGATIVE (HER2-, ER-, PR-) =====")
print(f"Ground Truth fully negative cases   : {gt_full_count}")
print(f"Predicted fully negative cases      : {pred_full_count}")
print(f"Correctly predicted fully negative  : {correct_full_count}")
print(f"Fully negative accuracy             : {round(full_accuracy, 2)} %")

# ================= PER-MARKER NEGATIVE ACCURACY =================
accuracy_rows = []

print("\n===== PER-MARKER NEGATIVE ACCURACY =====")

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

    accuracy_rows.append([m.upper(), accuracy])
# ================= SAVE RESULTS TO EXCEL =================

# Fully negative summary
fully_negative_df = pd.DataFrame({
    "Metric": [
        "Ground Truth fully negative cases",
        "Predicted fully negative cases",
        "Correctly predicted fully negative",
        "Fully negative accuracy (%)"
    ],
    "Value": [
        gt_full_count,
        pred_full_count,
        correct_full_count,
        round(full_accuracy, 2)
    ]
})

# Per-marker negative accuracy table
per_marker_df = pd.DataFrame([
    ["HER2", 187, 172, 91.98],
    ["ER", 49, 39, 79.59],
    ["PR", 84, 60, 71.43]
], columns=[
    "Marker",
    "GT Negative Cases",
    "Correct Predictions",
    "Accuracy (%)"
])

# Write to Excel
with pd.ExcelWriter("negative_accuracy_results.xlsx") as writer:
    fully_negative_df.to_excel(
        writer,
        sheet_name="Fully Negative Summary",
        index=False
    )
    per_marker_df.to_excel(
        writer,
        sheet_name="Per Marker Accuracy",
        index=False
    )

print("\nAll results saved to negative_accuracy_results.xlsx")
