import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ========= 1. Load the Excel files =========
gt_path = "GroundTruth.xlsx"
pred_openai5_path = "features_openai (5).xlsx"

df_gt = pd.read_excel(gt_path)
df_pred_openai5 = pd.read_excel(pred_openai5_path)

# ========= 2. Merge on File Name (OpenAI-5) =========
df_merged = pd.merge(
    df_gt,
    df_pred_openai5,
    left_on="File Name",
    right_on="Source File Name",
    how="inner",
    suffixes=("_gt", "_pred")
)

print(f"Total matched records (OpenAI-5): {len(df_merged)}\n")

# ========= 3. Normalization Functions =========

def normalize_her2(value):
    """
    Convert HER2 scores to binary classification
    0, 1+, 2+ -> negative
    3+ -> positive
    """
    if pd.isna(value):
        return "not reported"
    
    value_str = str(value).strip().lower()
    
    if "not" in value_str or "nan" in value_str or value_str == "":
        return "not reported"
    if value_str in ["0", "1+", "2+", "1", "2"]:
        return "negative"
    if value_str in ["3+", "3", "positive"]:
        return "positive"
    
    return "not reported"

def normalize_binary(value):
    """
    Normalize binary classifications (ER, PR)
    """
    if pd.isna(value):
        return "not reported"
    
    value_str = str(value).strip().lower()
    
    if "not" in value_str or "nan" in value_str or value_str == "":
        return "not reported"
    if "positive" in value_str or value_str in ["pos"]:
        return "positive"
    if "negative" in value_str or value_str in ["neg"]:
        return "negative"
    
    return "not reported"

def extract_tumor_size(value):
    """
    Extract largest tumor size from various formats
    Handles: 1.8 cm, 7 x 4 x 8, 1.5 x 1.3 x 1.2 cm, multiple tumors like "23 mm and 12 mm"
    Returns the single largest dimension in cm
    """
    if pd.isna(value):
        return "not reported"
    
    value_str = str(value).strip().lower()
    
    if "not" in value_str or value_str == "" or "nan" in value_str:
        return "not reported"
    
    # Remove common units and whitespace
    value_str = value_str.replace("cm", "").replace("mm", "").replace("x", " ").replace("×", " ")
    value_str = value_str.replace("\"", "").replace("'", "")
    
    # Extract all numbers (including decimals)
    numbers = re.findall(r'[\d.]+', value_str)
    
    if not numbers:
        return "not reported"
    
    # Convert to floats
    try:
        numbers = [float(n) for n in numbers if float(n) > 0]
    except:
        return "not reported"
    
    if not numbers:
        return "not reported"
    
    # Get the largest dimension
    max_size = max(numbers)
    
    # Convert mm to cm if needed (assume value > 10 is in mm)
    if max_size > 10 and max_size < 100:
        max_size = max_size / 10
    
    return round(max_size, 2)

def categorize_tumor_size(size):
    """
    Categorize tumor size into clinical categories
    """
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

# ========= 4. Apply normalizations =========
print("Normalizing OpenAI-5 data...")

# HER2 Normalization
df_merged["HER2_gt_norm"] = df_merged["HER2_gt"].apply(normalize_her2)
df_merged["HER2_pred_norm"] = df_merged["HER2_pred"].apply(normalize_her2)

# ER Normalization
df_merged["ER_gt_norm"] = df_merged["ER_gt"].apply(normalize_binary)
df_merged["ER_pred_norm"] = df_merged["ER_pred"].apply(normalize_binary)

# PR Normalization
df_merged["PR_gt_norm"] = df_merged["PR_gt"].apply(normalize_binary)
df_merged["PR_pred_norm"] = df_merged["PR_pred"].apply(normalize_binary)

# Tumor Size Extraction & Categorization
df_merged["tumor_size_gt_raw"] = df_merged["Tumor Size "].apply(extract_tumor_size)
df_merged["tumor_size_pred_raw"] = df_merged["Tumor Size"].apply(extract_tumor_size)

df_merged["tumor_size_gt_cat"] = df_merged["tumor_size_gt_raw"].apply(categorize_tumor_size)
df_merged["tumor_size_pred_cat"] = df_merged["tumor_size_pred_raw"].apply(categorize_tumor_size)

# ========= 5. Compute and Display Metrics for All Targets =========
targets = {
    "HER2": ("HER2_gt_norm", "HER2_pred_norm"),
    "ER": ("ER_gt_norm", "ER_pred_norm"),
    "PR": ("PR_gt_norm", "PR_pred_norm"),
    "Tumor Size": ("tumor_size_gt_cat", "tumor_size_pred_cat")
}

results = {}
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, (target_name, (gt_col, pred_col)) in enumerate(targets.items()):
    print(f"\n{'='*70}")
    print(f"Target: {target_name}")
    print(f"{'='*70}")
    
    y_true = df_merged[gt_col]
    y_pred = df_merged[pred_col]
    
    # Compute accuracy
    acc = accuracy_score(y_true, y_pred)
    results[target_name] = {"accuracy": acc}
    
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # ========= Confusion Matrix =========
    labels = sorted(list(set(y_true.dropna().unique()) | set(y_pred.dropna().unique())))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Plot confusion matrix in subplot
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, 
                cbar_kws={'label': 'Count'}, ax=ax)
    ax.set_xlabel("Predicted (OpenAI-5)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Ground Truth", fontsize=11, fontweight='bold')
    ax.set_title(f"Confusion Matrix: {target_name}\n(Accuracy = {acc:.2%})", 
                fontsize=12, fontweight='bold')
    
    # Save individual confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Count'})
    plt.xlabel("Predicted (OpenAI-5)", fontsize=12, fontweight='bold')
    plt.ylabel("Ground Truth", fontsize=12, fontweight='bold')
    plt.title(f"Confusion Matrix: {target_name} (OpenAI-5)\n(Accuracy = {acc:.2%})", 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{target_name.replace(' ', '_')}_openai5.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: confusion_matrix_{target_name.replace(' ', '_')}_openai5.png")

# Save combined figure
plt.figure(fig.number)
plt.tight_layout()
plt.savefig("confusion_matrices_all_openai5.png", dpi=300, bbox_inches='tight')
plt.close()

# ========= 6. Summary Report =========
print(f"\n{'='*70}")
print("SUMMARY - ACCURACY SCORES (OPENAI-5)")
print(f"{'='*70}")
for target, metrics in results.items():
    print(f"{target:15s}: {metrics['accuracy']:.2%}")

# ========= 7. Save Summary to CSV =========
summary_df = pd.DataFrame([
    {"Target": target, "Accuracy": metrics['accuracy'], "Accuracy %": f"{metrics['accuracy']*100:.2f}%"}
    for target, metrics in results.items()
])

summary_df.to_csv("accuracy_summary_openai5.csv", index=False)
print(f"\n✓ Summary saved to: accuracy_summary_openai5.csv")

# ========= 8. Sample Predictions =========
print(f"\n{'='*70}")
print("SAMPLE PREDICTIONS (First 10 rows)")
print(f"{'='*70}\n")

sample_cols = [
    "File Name", 
    "HER2_gt", "HER2_gt_norm", "HER2_pred", "HER2_pred_norm",
    "ER_gt", "ER_gt_norm", "ER_pred", "ER_pred_norm",
    "PR_gt", "PR_gt_norm", "PR_pred", "PR_pred_norm",
    "Tumor Size ", "tumor_size_gt_cat", "Tumor Size", "tumor_size_pred_cat"
]

sample = df_merged[sample_cols].head(10)
print(sample.to_string())

print("\n✓ Analysis complete!")
print(f"\nGenerated Files:")
print("  1. confusion_matrix_HER2_openai5.png")
print("  2. confusion_matrix_ER_openai5.png")
print("  3. confusion_matrix_PR_openai5.png")
print("  4. confusion_matrix_Tumor_Size_openai5.png")
print("  5. confusion_matrices_all_openai5.png (combined)")
print("  6. accuracy_summary_openai5.csv")