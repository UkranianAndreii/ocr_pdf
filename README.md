Promotion PDF Parser — README

A minimal, production-ish utility for extracting promotion/deduction fields from retailer PDFs (UNFI, KeHE, Polar, DeCrescente, etc.).
This README explains installation, usage, expected outputs, schema, testing and debugging.

Quick summary

Input: folder of PDFs (text or scanned).

Output: out/parsed.jsonl and out/parsed.csv.

Features: text extraction (PyMuPDF / pypdf), OCR (pytesseract), heuristics for invoice/PO/dates/amounts, per-field confidence scores (heuristic), debug export of normalized page text.

CLI: python src/parse_pdfs.py --input data/ --out out/ --ocr auto --debug true

Requirements

System

Python 3.10+ (3.11 recommended)

(If using OCR) Tesseract OCR installed:

macOS (Homebrew): brew install tesseract

Ubuntu: sudo apt-get install tesseract-ocr

(Optional for pdf→image) Poppler (pdf2image):

macOS: brew install poppler

Ubuntu: sudo apt-get install poppler-utils

Python packages (example requirements.txt):

typer
pydantic
dateparser
python-dateutil
pytesseract
pillow
pdf2image
pypdf
PyMuPDF
pytest


Install:

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

CLI / Usage

Run from project root (where src/ and data/ are):

# typical run (auto OCR)
python src/parse_pdfs.py --input data/ --out out/ --ocr auto

# only text-based PDFs (no OCR)
python src/parse_pdfs.py --input data/ --out out/ --ocr false

# force OCR for all pages
python src/parse_pdfs.py --input data/ --out out/ --ocr force

# enable debug (saves normalized .txt for each PDF)
python src/parse_pdfs.py --input data/ --out out/ --ocr auto --debug true


Outputs:

out/parsed.jsonl — one JSON object per line (each parsed row).

out/parsed.csv — CSV with same fields.

(if --debug) out/<pdf_stem>.txt — normalized text the parser used.

Output schema (fields)

Each output object contains:

source_file — filename of source PDF

page — page number where a matched field (amount) was found (or null)

promotion_date_start, promotion_date_end — ISO YYYY-MM-DD or null

reason — e.g. "Deduction Invoice", "Invoice Adjustment", "Pass Thru Deduction", "Distributor Charge"

notes — optional text (e.g. EP Fee $240.00)

retailer_name — normalized retailer (e.g. Fresh Market, HEB, Polar Beverages) or null

deduction_amount — numeric (float) or null

discount_details — raw textual snippets describing discount lines (concatenated) or null

invoice_number — string (keeps letters and leading zeros) or null

po_number — string or null

delivery_location — string or null

source_snippet — short text snippet where the field was found

confidences — dict of heuristic confidences per key (0–1)

Tests

Run unit tests from project root:

pytest -q


If tests fail with import errors, run pytest from the repo root or add tests/conftest.py that inserts the project root into sys.path.

Debugging tips

If something parses incorrectly, run with --debug true and inspect out/<pdf_stem>.txt.

For poor OCR quality:

Try higher DPI (rendering code uses 300 dpi; you can increase to 350–400).

Preprocess PDF with ocrmypdf to improve OCR layer.

If a particular retailer/template fails, share out/<pdf_stem>.txt and sample PDF; rules can be adjusted.

Limitations & assumptions

MVP relies mainly on regex + deterministic heuristics. Robust to many templates but not all OCR artefacts.

Dates normalized with dateparser (best-effort). Some ambiguous dates may need manual review.

For higher accuracy, consider an LLM/NER-based agent for field extraction and pydantic schema validation as next step.

Roadmap / next steps

Add per-field source mapping (page + char span).

Add schema validation using Pydantic / jsonschema and stricter confidence calibration.

Add optional LLM-based extraction fallback for ambiguous docs.

Provide Dockerfile to simplify Tesseract/Poppler dependencies.

License

Add a license file to the repo (MIT recommended for open-source). Example LICENSE content depends on your choice.