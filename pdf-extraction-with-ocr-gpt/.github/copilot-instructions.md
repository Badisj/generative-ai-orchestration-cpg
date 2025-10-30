## Purpose

This file gives concise, repo-specific guidance for an AI coding agent (or human) to be productive working on the
`pdf-extraction-with-ocr-gpt` project. It focuses on the actual architecture, run/debug commands, conventions and
integration details discoverable in the codebase (not generic best-practices).

## Quick summary / big picture

- Single-script prototype located at `extract_pdf.py` that performs: PDF text extraction (PyMuPDF) → conditional OCR
  (pytesseract + Pillow) → assemble plain text → call OpenAI chat completion with a rigid JSON schema prompt →
  parse and write JSON output.
- `requirement.txt` lists runtime dependencies: PyMuPDF, pytesseract, Pillow, openai, python-dotenv (for env keys).

## Key files

- `extract_pdf.py` — core implementation. Read this file first. It contains:
  - `extract_text_with_ocr(pdf_path)`: uses `fitz` (PyMuPDF) to get page text, uses OCR when page text length < 30 chars.
  - `system_prompt` and `user_prompt_template`: the system/user prompts define the exact JSON schema the model must emit.
  - `extract_material_data(pdf_text)`: calls the OpenAI API and expects a strict JSON response. It uses
    `openai.chat.completions.create(...)` with `model="gpt-5-mini"` and `temperature=0`.
  - `process_pdf(pdf_path, output_json_path)`: glue that runs extraction and writes output JSON.

## How to run (local)

1. Install dependencies:

   pip install -r requirement.txt

2. Provide an OpenAI key (do NOT commit keys):

   - Preferred: create a `.env` and set `OPENAI_API_KEY=...` and let `python-dotenv` load it, or set `OPENAI_API_KEY` in the
     environment. Current code sets `openai.api_key = "YOUR_OPENAI_API_KEY"` — replace with env-loading before commits.

3. Ensure Tesseract binary is installed on your OS (pytesseract wrapper requires it).

4. Execute the script (uses example filenames inside the file):

   python extract_pdf.py

   The default in the script uses `example_material_datasheet.pdf` -> `acetone_extracted.json`.

## Observable patterns and conventions (important for edits)

- JSON-only output contract: The `system_prompt` requires "Output strictly valid JSON (no text outside JSON)". Do not
  change the prompt format unless you update downstream code that `json.loads(...)` the result.
- Rigid schema: `user_prompt_template` embeds a large JSON schema. If adding or renaming fields, update both the template
  and any downstream consumers that expect the exact keys and null/default semantics.
- Missing values convention: missing scalar → `null`; missing lists → `[]`. Preserve nested structure.
- Deterministic model settings: the code calls the model with `temperature=0` and expects strict JSON verbatim. If you need
  to debug content, temporarily raise temperature or log the model text before `json.loads`.
- OCR trigger: pages with extracted text length < 30 characters are treated as images and run through pytesseract.
  This threshold is a project convention — change carefully and prefer exposing it as a parameter if experimenting.
- Temporary files: OCR writes a temporary PNG with `NamedTemporaryFile(..., delete=False)` then `os.unlink` after use.
  Keep this behavior or switch to in-memory bytes if you want fewer filesystem writes.

## OpenAI integration specifics

- API usage in `extract_pdf.py`:
  - `openai.api_key = "YOUR_OPENAI_API_KEY"` — replace with env var usage.
  - `openai.chat.completions.create(model="gpt-5-mini", messages=[...], temperature=0)` and the code reads
    `response.choices[0].message.content` and does `json.loads(...)`.
- Model + response path are hardcoded — if moving to a different OpenAI client shape, maintain the same
  message structure and ensure the chosen client returns the string at the same path.

## Debugging tips specific to this project

- If `json.loads` fails, the script prints "⚠️ Invalid JSON detected, returning empty structure." — to debug:
  - Print `pdf_text` before the model call to confirm extraction/OCR result.
  - Print `response.choices[0].message.content` to inspect why the model returned non-JSON.
  - Temporarily set `temperature=0.2` and add a short debugging system prompt asking for the output as valid JSON.
  - Validate Tesseract is installed and available in PATH on Windows (check `tesseract --version`).
- To test OCR behavior quickly, reduce the 30-char threshold in `extract_text_with_ocr` to force OCR on test pages.

## Integration & external dependencies to watch

- Tesseract binary (external system dependency) — required for OCR to work.
- OpenAI API key and availability of the chosen model string (`gpt-5-mini`) — may require updating to an available model.
- `requirement.txt` lists optional helpers like `pdfplumber` and `jsonschema` which can be used for richer extraction and
  validating the model's JSON output respectively.

## Small, low-risk follow-ups (suggested by the code)

1. Replace the hard-coded `openai.api_key = "..."` with `os.environ.get('OPENAI_API_KEY')` and document `.env` usage.
2. Add a tiny unit/integration check that runs `extract_text_with_ocr` on a small PDF fixture and asserts non-empty text.
3. Add `jsonschema` validation step after `json.loads` using the embedded schema to surface model drift early.

## What to avoid changing without careful review

- The JSON schema shape in the `user_prompt_template` — consumers expect those exact keys and null/list conventions.
- The `temperature=0` / deterministic model directive — relaxing it changes output stability and may break parsing.

---

If any section is unclear or you want the document to include additional examples (e.g., exact logging snippets to add,
unit test skeleton, or a migration note for using env vars), tell me which part and I will update this file.
