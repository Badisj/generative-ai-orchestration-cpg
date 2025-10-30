import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import tempfile
import os
import openai
import json

openai.api_key = "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"

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
# 2️⃣ GPT Prompts for Material Extraction
# ====================================================
system_prompt = """
You are a materials data extraction assistant.
Your job is to extract detailed, structured information about a material or chemical
from provided PDF text and output it using a predefined ontology-based JSON schema.

Rules:
1. Output strictly valid JSON (no text outside JSON).
2. Follow the schema exactly — do not rename or add fields.
3. If a value is missing, set it to null or [].
4. Preserve nested structures exactly as defined.
5. Do not explain, comment, or summarize outside the JSON.
6. Units must be normalized if mentioned (e.g., K for Kelvin).
"""

user_prompt_template = """
Extract all possible information about the material described in the following PDF text
and represent it using this JSON structure:

{
  "name": null,
  "description": null,
  "iupacName": null,
  "casNumber": null,
  "recipeNumber": null,
  "synonym": [],
  "hasAssignedIdentifier": [
    {
      "identifier": null,
      "hasIdentifierType": {
        "id": null
      }
    }
  ],
  "hasConstituency": {
    "id": null,
    "hasMixtureRatio": [
      {
        "ratioMaximum": null,
        "ratioMinimum": null,
        "ratiotarget": null,
        "ratioMaximumInclusive": null,
        "ratioMinimumInclusive": null,
        "restOfMixture": null,
        "offLabel": null,
        "primaryConstituent": null,
        "hasActiveMaterialFunction": [
          {
            "id": null
          }
        ],
        "hasRatioSubstance": {
          "id": null
        }
      }
    ]
  },
  "hasGrade": [
    {
      "id": null
    }
  ],
  "hasMaterialFunction": [
    {
      "id": null
    }
  ],
  "hasMaterialType": {
    "id": null
  },
  "hasSource": {
    "hasSupplier": {
      "supplierId": null,
      "supplierName": null
    },
    "supplierPartNumber": null,
    "tradeName": null
  },
  "groups": [
    {
      "id": null
    }
  ],
  "hasAlternative": [
    {
      "id": null
    }
  ],
  "hasFormulationSet": {
    "hasAppearance": {"val": null},
    "hasAqueousSolubility": {"val": null},
    "hasAutoIgnitionTemperature": {"val": null},
    "hasBoilingPoint": {"val": null},
    "hasColorDescription": {"val": null},
    "hasDensity": {"val": null},
    "hasDynamicViscosity": {"val": null},
    "hasFlashPoint": {"val": null},
    "hasMeltingPoint": {"val": null},
    "hasPhysicalForm": {"val": {"id": null}},
    "hasSpecificGravity": {"val": null},
    "hasStandardCost": {"val": null},
    "hasUsageRestriction": {"val": {"id": null}, "message": null},
    "hasUsageStatus": {"val": {"id": null}},
    "hasVaporPressure": {"val": null}
  },
  "hasSecureCollection": [
    {
      "id": null
    }
  ]
}

PDF Text:
\"\"\"{pdf_text}\"\"\"

Instructions:
- Extract all fields possible based on the text.
- Use null or [] for missing data.
- Output valid JSON only.
"""

# ====================================================
# 3️⃣ GPT Call Function
# ====================================================
def extract_material_data(pdf_text):
    user_prompt = user_prompt_template.format(pdf_text=pdf_text)

    response = openai.chat.completions.create(
        model="gpt-5-mini",  # or gpt-5, gpt-4o-mini depending on your API
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ],
        temperature=0
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
# 5️⃣ Run Example
# ====================================================
if __name__ == "__main__":
    pdf_path = "example_material_datasheet.pdf"
    output_json_path = "acetone_extracted.json"
    process_pdf(pdf_path, output_json_path)
