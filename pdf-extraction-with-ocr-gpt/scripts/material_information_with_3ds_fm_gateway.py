import fitz  # PyMuPDF
import pytesseract
import tempfile
import os
import httpx
from openai import OpenAI
import json

from PIL import Image
from dotenv import load_dotenv
from argparse import ArgumentParser
import subprocess


# ====================================================
# 0️⃣ Load environment variables from .env file
# ====================================================
# Load the .env file
load_dotenv("3ds.env")

# Create a custom httpx client that uses your corporate certificate
custom_http_client = httpx.Client(verify="./3DS/ExaleadRootCAG2-chain.pem")

# Load environment variables from .env file
api_key = os.getenv("DS_OPENAI_API_KEY")
base_url = os.getenv("FM_GATEWAY_BASE_URL")

# Initialize OpenAI client with custom HTTP client
client = OpenAI(
    api_key=api_key,
    base_url=base_url,
    http_client=custom_http_client
)

# ====================================================
# 1️⃣ Parse arguments function
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
# 2️⃣ PDF Extraction (text + OCR for mixed content)
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
# 3️⃣ Read system prompt from file and store in variable
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
# 4️⃣ FM Gateway Call Function
# ====================================================
def extract_material_data(pdf_text):
    user_prompt = """
    Extract all material data and output it in JSON following the provided schema.
    PDF Text:
    \"\"\"{pdf_text}\"\"\"
    """.format(pdf_text=pdf_text)

    response = client.chat.completions.create(
    model="mistralai/Mistral-Small-3.2-24B-Instruct-2506", 
    messages=[
        {
            "role": "system", 
            "content": system_prompt
        },
        {
            "role": "user", 
            "content": user_prompt
        }
        ]
)

    json_output = response.choices[0].message.content.strip()
    try:
        return json.loads(json_output)
    except json.JSONDecodeError:
        print("⚠️ Invalid JSON detected, returning empty structure.")
        print("Received content:", json_output)
        return {}


# ====================================================
# 5️⃣ Orchestration Function
# ====================================================
def process_pdf(pdf_path, output_json_path):
    print(f"Extracting text (including OCR) from: {pdf_path}")
    text = extract_text_with_ocr(pdf_path)

    print("Sending text to FM Gateway for structured material extraction...")
    structured_data = extract_material_data(text)

    print(f"Saving output to: {output_json_path}")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)

    print("✅ Extraction complete.")

# ====================================================
#  6️⃣ Script Entry Point
# ====================================================
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    pdf_path = args.pdf_path
    system_prompt_path = args.system_prompt_path

    print("Using parameters:")
    print(f"  Supplier Name: {args.supplier_name}")
    print(f"  Supplier ID: {args.supplier_id}")
    print(f"  Usage: {args.usage}")
    print(f"  Usage ID: {args.usage_id}")
    print(f"  Usage Restriction: {args.usage_restriction}")
    print(f"PDF Path: {pdf_path}")
    
    # Load system prompt with parameters
    print(f"System Prompt Path: {system_prompt_path}")
    system_prompt = load_system_prompt(system_prompt_path, args)
    print("System prompt loaded.")

    # Specify output JSON path
    output_json_path = os.path.join("./documents/processed/", os.path.basename(pdf_path).replace(".pdf", "_processed_with_pdf_extraction_fm_gateway.json"))
    
    # Run orchestration
    process_pdf(pdf_path, output_json_path)

    # Generate requirements.txt using pip freeze
    if not os.path.exists("requirements.txt"):
        print("Generating requirements.txt...")
        with open("requirements.txt", "w") as f:
            subprocess.run(["pip", "freeze"], stdout=f, text=True, check=True)
        print("✅ requirements.txt generated successfully!")