import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from argparse import ArgumentParser


# ====================================================
# 0️⃣ Set up OpenAI API key and client
# ====================================================
load_dotenv('./.env')
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)


# ===================================================
# 1️⃣ Define configuration and arguments
# ===================================================
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
# 3️⃣ function to extract material data from PDF
# ====================================================
def extract_material_data(pdf_path):
    file = client.files.create(
        file=open(pdf_path, "rb"), 
        purpose="user_data"
        )
    
    response = client.responses.create(
    model="gpt-5-mini",  # or "gpt-4o"
    input=[
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Extract all material data and output it in JSON following the provided schema."
                },
                {
                    "type": "input_file",
                    "file_id": file.id
                }
                ],
         },
        ]
    )

    # Parse response
    json_text = response.output_text.strip()

    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        print("⚠️ Model output is not valid JSON:")
        print(json_text)


# ===================================================
# 3️⃣ Orchestration Function 
# ===================================================
def orchestrate_material_extraction(pdf_path, output_json_path):
    print("Sending extraction request to OpenAI...")
    extracted_data = extract_material_data(pdf_path)

    print(f"Writing extracted data to {output_json_path}...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    
    print("✅ Extraction complete.")


# ===================================================
# 4️⃣ Script Entry Point
# ===================================================
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
    
    # Set output JSON path
    output_json_path = os.path.join("./documents/processed/", os.path.basename(pdf_path).replace(".pdf", "_processed_without_pdf_extraction.json"))

    # Run orchestration
    orchestrate_material_extraction(pdf_path, output_json_path)

    # Generate requirements.txt using pip freeze
    if not os.path.exists("requirements.txt"):
        print("Generating requirements.txt...")
        with open("requirements.txt", "w") as f:
            subprocess.run(["pip", "freeze"], stdout=f, text=True, check=True)
        print("✅ requirements.txt generated successfully!")