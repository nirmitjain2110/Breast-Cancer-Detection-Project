import os
import pandas as pd
import re

# Paths
excel_file = "FilteredData.xlsx"
pdf_folder = "Data"
output_file = "GroundTruth.xlsx"
column_name = "File Name"

# Read Excel
df = pd.read_excel(excel_file)

# Get all PDF names without .pdf
pdf_files = [
    os.path.splitext(f)[0]
    for f in os.listdir(pdf_folder)
    if f.lower().endswith(".pdf")
]

final_rows = []

for pdf in pdf_files:
    # Regex:
    # ^pdf(\.|$) → starts with pdf and only dots allowed after
    pattern = rf"^{re.escape(pdf)}(\.|$)"

    matches = df[
        df[column_name]
        .astype(str)
        .str.match(pattern, na=False)
    ]

    if matches.empty:
        print(f"PDF NOT FOUND in Excel: {pdf}")
    else:
        # Keep only the first match
        final_rows.append(matches.iloc[0])

# Create final DataFrame
final_df = pd.DataFrame(final_rows)

# Save output
final_df.to_excel(output_file, index=False)

print(f"\nFiltered data saved to: {output_file}")
