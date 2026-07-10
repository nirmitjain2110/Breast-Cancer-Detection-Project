!pip install openai pandas openpyxl

import os
import json
import pandas as pd
from openpyxl import Workbook, load_workbook
from openai import OpenAI
from google.colab import files
from tqdm import tqdm

OPENAI_API_KEY =
client = OpenAI(api_key=OPENAI_API_KEY)

input_excel_file = "Modified_data.xlsx"
output_excel_file = "features_openai.xlsx"

if not os.path.exists(input_excel_file):
    print(f"'{input_excel_file}' not found. Please upload it now")
    uploaded = files.upload()
    if input_excel_file not in uploaded:
        print(f"upload file named '{input_excel_file}'.")
        raise SystemExit
    print(f"Uploaded: {list(uploaded.keys())}")

def write_to_excel(file_name, tumor_size, er, pr, her2):
    """Appends extracted features into an Excel file."""
    if not os.path.exists(output_excel_file):
        wb = Workbook()
        ws = wb.active
        ws.title = "Features"
        headers = ["Source File Name", "Tumor Size", "ER", "PR", "HER2"]
        ws.append(headers)
        wb.save(output_excel_file)

    wb = load_workbook(output_excel_file)
    ws = wb.active
    ws.append([file_name, tumor_size, er, pr, her2])
    wb.save(output_excel_file)

def extract_features_openai(report_context):
    """
    Extracts tumor size, ER, PR, and HER2 from pathology text.
    Uses structured JSON output for reliability.
    """
    prompt = f"""
    You are analyzing a breast cancer pathology report.
    Extract the following fields (return valid JSON only):

    - tumor_size (in cm)
    - er ("Positive" or "Negative")
    - pr ("Positive" or "Negative")
    - her2 ("Negative", "Positive", "1+", "2+", or "3+")

    If any field is missing, use "Not Reported".
    Return only a valid JSON object with keys:
    tumor_size, er, pr, her2.

    CONTEXT:
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
            text_output = text_output[text_output.find("{"): text_output.rfind("}")+1]
            result = json.loads(text_output)

        return (
            result.get("tumor_size", "Not Reported"),
            result.get("er", "Not Reported"),
            result.get("pr", "Not Reported"),
            result.get("her2", "Not Reported"),
        )

    except Exception as e:
        print(f"Error while calling OpenAI API: {e}")
        return ("Error", "Error", "Error", "Error")
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
        write_to_excel(file_name, "Skipped", "Skipped", "Skipped", "Skipped")
        continue

    tumor_size, er, pr, her2 = extract_features_openai(report_context)
    write_to_excel(file_name, tumor_size, er, pr, her2)

print(f"\n Done! Results saved in: {output_excel_file}")
