import fitz  # PyMuPDF
import pytesseract
import tempfile
import os
import openai
import json
import subprocess

from PIL import Image
from dotenv import load_dotenv
from argparse import ArgumentParser

# ====================================================
# 0️⃣ Load environment variables from .env file
# ====================================================
load_dotenv('./.env')
openai.api_key = os.getenv("OPENAI_API_KEY")


# ====================================================
# 1️⃣  Parse arguments function
# ====================================================
def parse_arguments():
    parser = ArgumentParser(description="Extract structured material data from PDF datasheets using GPT.")
    parser.add_argument("--supplier_name", type=str, default="Milan Creative Collectibles S.r.l.")
    parser.add_argument("--supplier_id", type=str, default="uuid:6901bbfc-bac6-452c-a1ac-a861655e3150")
    parser.add_argument("--usage", type=str, default="Pilot", choices=["Pilot", "Production", "Development"])
    parser.add_argument("--usage_id", type=str, default="dsmatdata:pilot_usageStatus", choices=["dsmatdata:pilot_usageStatus", "dsmatdata:production_usageStatus", "dsmatdata:development_usageStatus"])
    parser.add_argument("--usage_restriction", type=str, default="dsmatdata:none_usageRestriction", choices=["dsmatdata:blockForNew_usageRestriction", "dsmatdata:blockForAll_usageRestriction", "dsmatdata:warning_usageRestriction", "dsmatdata:none_usageRestriction"])
    parser.add_argument("--pdf_path", type=str, default="./documents/S25255.pdf", help="Path to the input PDF file.")
    parser.add_argument("--system_prompt_path", type=str, default="./prompts/material_extraction_system_prompt.txt")
    return parser.parse_args()


# ====================================================
# 1️⃣ PDF Extraction (text + OCR for mixed content)
# ====================================================
def extract_text_with_ocr(pdf_path):
    text_content = ""
    with fitz.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")

            if len(text.strip()) < 30:  # Low text density → use OCR
                pix = page.get_pixmap()
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
                    pix.save(temp_img.name)
                    ocr_text = pytesseract.image_to_string(Image.open(temp_img.name))
                    text_content += f"\n\nPage {page_num+1} (OCR):\n" + ocr_text
                    os.unlink(temp_img.name)
            else:
                text_content += f"\n\nPage {page_num+1}:\n" + text
    return text_content.strip()


# ====================================================
# 2️⃣ Read system prompt from file and store in variable
# ====================================================
def load_system_prompt(prompt_path, args):
    try:
      with open(prompt_path, "r", encoding="utf-8") as f:
          system_prompt = f.read().strip()
      return system_prompt.format(
          args.supplier_name,
          args.supplier_id,
          args.usage_restriction,
          args.usage_id
      )
    except Exception as e:
      print(f"System prompt not found in {prompt_path}.")
      return ""


# ====================================================
# 3️⃣ GPT Call Function
# ====================================================
def extract_material_data(pdf_text):
    user_prompt = """
    Extract all material data and output it in JSON following the provided schema.
    PDF Text:
    \"\"\"{pdf_text}\"\"\"
    """.format(pdf_text=pdf_text)

    response = openai.chat.completions.create(
        model="gpt-5-mini",  # or gpt-5, gpt-4o-mini depending on your API
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ],
        temperature=1
    )

    json_output = response.choices[0].message.content.strip()
    try:
        return json.loads(json_output)
    except json.JSONDecodeError:
        print("⚠️ Invalid JSON detected, returning empty structure.")
        return {}


# ====================================================
# 4️⃣ Orchestration Function
# ====================================================
def process_pdf(pdf_path, output_json_path):
    print(f"Extracting text (including OCR) from: {pdf_path}")
    text = extract_text_with_ocr(pdf_path)

    print("Sending text to GPT for structured material extraction...")
    structured_data = extract_material_data(text)

    print(f"Saving output to: {output_json_path}")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)

    print("✅ Extraction complete.")

# ====================================================
# 5️⃣ Script Entry Point
# ====================================================
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    pdf_path = args.pdf_path
    system_prompt_path = args.system_prompt_path

    print("Using parameters:")
    print(f" Supplier Name: {args.supplier_name}")
    print(f" Supplier ID: {args.supplier_id}")
    print(f" Usage: {args.usage}")
    print(f" Usage ID: {args.usage_id}")
    print(f" Usage Restriction: {args.usage_restriction}")

    print(f"PDF Path: {pdf_path}")

    # Load system prompt with parameters
    print(f"System Prompt Path: {system_prompt_path}")
    system_prompt = load_system_prompt(system_prompt_path, args)
    print("System prompt loaded.")

    # Specify output JSON path
    output_json_path = os.path.join("./documents/processed/", os.path.basename(pdf_path).replace(".pdf", "_processed_with_pdf_extraction.json"))
    
    # Run orchestration
    process_pdf(pdf_path, output_json_path)

    # Generate requirements.txt using pip freeze
    if not os.path.exists("requirements.txt"):
        print("Generating requirements.txt...")
        with open("requirements.txt", "w") as f:
            subprocess.run(["pip", "freeze"], stdout=f, text=True, check=True)
        print("✅ requirements.txt generated successfully!")