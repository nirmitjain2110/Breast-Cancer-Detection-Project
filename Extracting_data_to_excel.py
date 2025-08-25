import os
from pdf2image import convert_from_path
import pytesseract
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

input_folder = "Data"
output_folder = "Modified_data"
os.makedirs(output_folder, exist_ok=True)

def process_pdf(filename):
    pdf_path = os.path.join(input_folder, filename)
    base_name = os.path.splitext(filename)[0]

    try:
        images = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        return {"File_name": base_name, "Extracted Text": f"Error: {e}"}

    full_text = ""
    for image in images:
        text = pytesseract.image_to_string(image)
        full_text += text.strip() + "\n"

    return {"File_name": base_name, "Extracted Text": full_text.strip()}

if __name__ == "__main__":
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]

    all_data = []
    count=0
    print("Processing PDF files ", end="")
    with ProcessPoolExecutor() as executor:
        for result in executor.map(process_pdf, pdf_files):
            count+=1
            all_data.append(result)
            if count==53 or count==len(pdf_files):
                print("█", end="")
                count=0

    output_path = os.path.join(output_folder, "Modified_data.xlsx")
    df = pd.DataFrame(all_data)
    df.to_excel(output_path, index=False)

    print(f"\n Processed {len(all_data)} PDF files and saved to {output_path}")
