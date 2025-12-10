!pip install openai pandas openpyxl --quiet

import os
import json
import pandas as pd
from openpyxl import Workbook, load_workbook
from openai import OpenAI
from google.colab import files
from tqdm import tqdm

OPENAI_API_KEY = ""
client = OpenAI(api_key=OPENAI_API_KEY)

input_excel_file = "Modified_data.xlsx"
output_excel_file = "features_openai.xlsx"

if not os.path.exists(input_excel_file):
    print(f"'{input_excel_file}' not found. Please upload it now.")
    uploaded = files.upload()
    if input_excel_file not in uploaded:
        print(f"Please upload a file named '{input_excel_file}'.")
        raise SystemExit
    print(f"Uploaded: {list(uploaded.keys())}")

def write_to_excel(file_name, tumor_size, er, pr, her2, ki67, p53):
    """Appends extracted features into an Excel file."""
    if not os.path.exists(output_excel_file):
        wb = Workbook()
        ws = wb.active
        ws.title = "Features"
        headers = ["Source File Name", "Tumor Size", "ER", "PR", "HER2", "Ki67", "p53"]
        ws.append(headers)
        wb.save(output_excel_file)

    wb = load_workbook(output_excel_file)
    ws = wb.active
    ws.append([file_name, tumor_size, er, pr, her2, ki67, p53])
    wb.save(output_excel_file)

def extract_features_openai(report_context):
    """
    Extracts HER2/neu, ER, PR, Ki-67, p53, and tumor size from a breast cancer pathology report.
    Returns structured JSON output with standardized labeling and reference-based classification.
    """

    prompt = f"""
    You are a medical data extraction system.
    Analyze the following breast cancer pathology reports and extract biomarker information.

    Extract and return the following fields as a valid JSON object:
    - tumor_size (in cm)
    - her2 ("Negative", "Equivocal", "Positive", "Ordered, result pending", or "Not Reported")
    - er ("Positive", "Negative", or "Not Reported")
    - pr ("Positive", "Negative", or "Not Reported")
    - ki67 (percentage + category: Low / Borderline / High / Not Reported)
    - p53 ("Positive", "Negative", or "Not Reported")

    Reference Ranges (for classification only, do not include in output):
    - ER/PR: Negative <1%, Positive ≥1%
    - Ki-67: Low <10%, Borderline 10–20%, High >20%
    - HER2/neu: Negative 0–1+, Equivocal 2+, Positive 3+ (reflexed to FISH if equivocal)

    Output Format Example:
    {{
        "tumor_size": "1.0 cm",
        "her2": "Negative (0–1+)",
        "er": "Positive (98%, 3+)",
        "pr": "Positive (95%, 3+)",
        "ki67": "12% (Borderline)",
        "p53": "Not Reported"
    }}

    If a field is not found in the report, explicitly return "Not Reported".

    EXAMPLE REPORTS FOR CONTEXT:
    Report 1:
    Breast, left, simple mastectomy: Invasive ductal carcinoma, Nottingham grade III (of III), is identified forming a 6.0 x 4.0 x 3.5 cm tumor (AJCC pT3) in the central-upper inner region of the breast.
    Her-2/NEU has been ordered on paraffin-embedded tissue on the left breast.

    Expected output:
    {{
        "tumor_size": "6.0 x 4.0 x 3.5 cm (left)",
        "her2": "Ordered, result pending",
        "er": "Not Reported",
        "pr": "Not Reported",
        "ki67": "Not Reported",
        "p53": "Not Reported"
    }}

    Report 2:
    IHC-stainings: Estrogen receptor - positive reaction 98% 3+; Progesterone receptor - positive reaction 95% 3+;
    Her-2/neu - negative reaction; Ki67 - 12%.

    Expected output:
    {{
        "tumor_size": "1.0 cm (left)",
        "her2": "Negative (0–1+)",
        "er": "Positive (98%, 3+)",
        "pr": "Positive (95%, 3+)",
        "ki67": "12% (Borderline)",
        "p53": "Not Reported"
    }}

    CONTEXT TO ANALYZE:
    {report_context}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        text_output = response.choices[0].message.content.strip()

        try:
            result = json.loads(text_output)
        except json.JSONDecodeError:
            text_output = text_output[text_output.find("{"): text_output.rfind("}") + 1]
            result = json.loads(text_output)

        result = {
            "tumor_size": result.get("tumor_size", "Not Reported"),
            "her2": result.get("her2", "Not Reported"),
            "er": result.get("er", "Not Reported"),
            "pr": result.get("pr", "Not Reported"),
            "ki67": result.get("ki67", "Not Reported"),
            "p53": result.get("p53", "Not Reported"),
        }

        return result

    except Exception as e:
        print(f"Error while calling OpenAI API: {e}")
        return {
            "tumor_size": "Error",
            "her2": "Error",
            "er": "Error",
            "pr": "Error",
            "ki67": "Error",
            "p53": "Error",
        }

print("Initializing extraction...")

try:
    df = pd.read_excel(input_excel_file)
except Exception as e:
    print(f"Error reading Excel file: {e}")
    raise SystemExit

filename_col = "File_name"
text_col = "text"

if filename_col not in df.columns or text_col not in df.columns:
    print(f"Missing columns. Ensure '{filename_col}' and '{text_col}' exist.")
    raise SystemExit

print(f"Found {len(df)} reports to process.\n")

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
    file_name = row[filename_col]
    report_context = row[text_col]

    if not isinstance(report_context, str) or len(report_context) < 10:
        write_to_excel(file_name, "Skipped", "Skipped", "Skipped", "Skipped", "Skipped", "Skipped")
        continue

    result = extract_features_openai(report_context)
    write_to_excel(
        file_name,
        result["tumor_size"],
        result["er"],
        result["pr"],
        result["her2"],
        result["ki67"],
        result["p53"],
    )

print(f"\n Done! Results saved in: {output_excel_file}")

