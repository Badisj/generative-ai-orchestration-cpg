# PDF Extraction + Material Information Orchestration

This repository contains a small prototype and orchestration tooling that extracts material information from PDF
datasheets. The primary user-facing workflow is implemented in a Jupyter notebook
`Extract materials information.ipynb` which launches one of three script entry points depending on the chosen
method:

- `scripts/material_information_with_pdf_extraction.py` — local PDF extraction (PyMuPDF) + OCR fallback (Tesseract), then
  send assembled text to OpenAI chat completion (uses `openai.chat.completions.create`).
- `scripts/material_information_without_pdf_extraction.py` — no local OCR: uploads the PDF file to OpenAI as a file and
  calls the Responses API (the notebook and script use `client.files.create` + `client.responses.create`).
- `scripts/material_information_with_3ds_fm_gateway.py` — sends the extracted text to internal FM Gateway using a custom `httpx` client and OpenAI-compatible client with a different base URL (uses
  `OpenAI(..., base_url=...)`).

All three scripts share conventions: they read a `system_prompt` file, accept the same CLI arguments (supplier, usage,
usage_id, usage_restriction, pdf_path, system_prompt_path), and produce a JSON output file under `./documents/processed/`
and a log under `./logs/` when invoked from the notebook.

## Notebook-driven workflow (primary UX)

Open `Extract materials information.ipynb` and run the first cells to set parameters. Main options in the notebook:

- `method` — choose one of:
  - `"OpenAI without pdf extraction"` (uploads PDF to OpenAI Responses API)
  - `"OpenAI with pdf extraction"` (local extraction + OpenAI chat completion)
  - `"FM Gateway with pdf extraction"` (local extraction + FM Gateway / internal LLM)

The notebook sets values for `supplier_name`, `supplier_id`, `usage`, `usage_id`, `usage_restriction`, `pdf_path`, and
`system_prompt_path`. It then launches the selected script via a Python `subprocess.Popen` call and streams stdout to
both the notebook output and a logfile in `./logs/` (see notebook code cells for exact log file names).

Use the notebook to run end-to-end scenarios easily (it captures live output and writes logs).

## Scripts: what they do and important implementation details

All scripts are under `scripts/` and accept the same CLI flags (via argparse):

- `--supplier_name`
- `--supplier_id`
- `--usage` (choices: `Pilot`, `Production`, `Development`)
- `--usage_id` (corresponding dsmatdata ids)
- `--usage_restriction` (several predefined choices)
- `--pdf_path` (path to input PDF)
- `--system_prompt_path` (path to the system prompt file used to parametrize the model)

Script-specific behavior:

- `material_information_with_pdf_extraction.py`
  - Loads `.env` and reads `OPENAI_API_KEY`.
  - Performs page-by-page text extraction using PyMuPDF (`fitz`). If a page's text length < 30 chars, it renders a PNG
    and runs pytesseract OCR. Temporary PNGs are saved via `tempfile.NamedTemporaryFile(..., delete=False)` and
    unlinked after OCR.
  - Builds a short user prompt with the PDF text and calls `openai.chat.completions.create(...)` with
    `model="gpt-5-mini"` and `temperature=1` (non-zero temperature in this script). It expects strict JSON in the
    model's response and `json.loads(...)` it.
  - Writes output JSON to `./documents/processed/<pdfname>_processed_with_pdf_extraction.json`.

- `material_information_without_pdf_extraction.py`
  - Loads `.env` and reads `OPENAI_API_KEY`.
  - Uploads the PDF file to OpenAI via `client.files.create(file=open(pdf_path, "rb"), purpose="user_data")`.
  - Calls `client.responses.create(...)` with `model="gpt-5-mini"` and includes both the system prompt and an input
    that references the uploaded file (type `input_file`). This avoids local OCR and relies on the model's file
    ingestion capability.
  - Writes output JSON to `./documents/processed/<pdfname>_processed_without_pdf_extraction.json`.

- `material_information_with_3ds_fm_gateway.py`
  - Loads a `3ds.env` file and constructs an `httpx.Client(verify="./3DS/ExaleadRootCAG2-chain.pem")` so requests
    use a corporate certificate when reaching the FM Gateway.
  - Initializes an OpenAI-compatible client with a custom `base_url` (FM Gateway) and `api_key` loaded from env.
  - Performs local extraction + OCR (same 30-char threshold) and calls the FM Gateway using `client.chat.completions.create`
    with a LLM model name like `mistralai/Mistral-Small-3.2-24B-Instruct-2506`.
  - Writes output JSON to `./documents/processed/<pdfname>_processed_with_pdf_extraction_fm_gateway.json`.

Common output & artifacts

- Processed JSON files: `./documents/processed/` (script-specific naming suffixes)
- Logs: when run from the notebook, logs are written to `./logs/` with names like `run_log_openai_with_pdf_extraction.txt`.
- `requirements.txt` generation: each script may generate `requirements.txt` using `pip freeze` if it doesn't exist.

## Installation & environment

Install Python dependencies listed in `requirement.txt`:

```powershell
pip install -r requirement.txt
```

Install the Tesseract binary (required for `pytesseract`) and ensure it is on PATH.

Environment files:

- `.env` — used by the OpenAI-based scripts; must contain `OPENAI_API_KEY`.
- `3ds.env` — used by the FM Gateway script; must contain `DS_OPENAI_API_KEY` and `FM_GATEWAY_BASE_URL` (and any
  other gateway-specific env vars). The FM Gateway script also expects a certificate file under `./3DS/`.

Example PowerShell session to run the notebook-driven workflow (sets env and launches Jupyter):

```powershell
pip install -r requirement.txt
$env:OPENAI_API_KEY = "sk-..."               # or create .env file
jupyter notebook "Extract materials information.ipynb"
```

Or run one of the scripts directly (example):

```powershell
python ./scripts/material_information_with_pdf_extraction.py --pdf_path ./documents/S25255.pdf --system_prompt_path ./prompts/material_extraction_system_prompt.txt
```

## Notes and editing cautions

- The prompt files and the notebook expect the model to return strictly valid JSON. Do not change the schema in the
  prompt without updating consumers.
- The OCR threshold (30 characters) is a convention used across the scripts — change it carefully and consider exposing
  it as a CLI flag if you need flexibility.
- Scripts generate `requirements.txt` via `pip freeze` when missing — commit a stable `requirements.txt` for CI.

## Suggested next improvements

1. Replace hard-coded env key handling with a robust loader and add `.env.example` to the repo.
2. Add unit/integration tests and a small PDF fixture to validate `extract_text_with_ocr` behavior.
3. Add `jsonschema` validation against the embedded schema after `json.loads` to detect model drift.
