#!/usr/bin/env python3
# src/parse_pdfs.py
"""
Promotion PDF Parser (patched for HBP normalization and HEB detection)
Run:
    python src/parse_pdfs.py --input data/ --out out/ --ocr auto --debug true
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import re
import json
import csv
import io
import logging

import typer
from pydantic import BaseModel
import dateparser

# optional libs
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except Exception:
    HAS_FITZ = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except Exception:
    HAS_PYPDF = False

try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except Exception:
    HAS_PDF2IMAGE = False

try:
    import pytesseract
    from PIL import Image
    HAS_TESS = True
except Exception:
    HAS_TESS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = typer.Typer(help="Promotion PDF Parser (patched)")

# -------------------------
# Regex / patterns (patched)
# -------------------------
CURRENCY = r'[$£€]\s*[0-9,]+(?:\.\d{2})?'
NUM_AMT = r'[0-9,]+(?:\.\d{2})?'
DATE_NUM = r'(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)'
DATE_WORD = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|' \
            r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s*\d{2,4}'

RE_INV_CUSTOMER = re.compile(r'Customer\s+Invoice\s+Number\s*[:\-]?\s*([A-Za-z0-9\-_/ ]+)\b', re.I)
RE_INV_LABEL = re.compile(r'\bInvoice\s*(?:#|Number)\b', re.I)
# GLOBAL fallback: exclude pure numeric tokens here to avoid zip/amount collisions
RE_INV_FALLBACK_TOKEN = re.compile(r'\b(?:HBP\s*\d{3,}|INV[0-9A-Z\-]+|TFM[0-9A-Z\-]+|[A-Z]{1,4}\d{3,})\b', re.I)
RE_INV_NUMERIC_LABEL = re.compile(r'Invoice\s*#\s*[:\-]?\s*(\d{5,})', re.I)
CODE_NEAR_LABEL = re.compile(r'\b(?:HBP\s*\d{3,}|INV[0-9A-Z\-]+|TFM[0-9A-Z\-]+|[A-Z]{1,4}\d{3,})\b', re.I)
# H B P with arbitrary separators (third fallback)
RE_HBP_SEP = re.compile(r'H\W*B\W*P\W*(\d{3,})', re.I)

RE_PO_LABEL = re.compile(r'\b(?:PO\s*(?:Number|#)|Purchase\s*Order|P\.O\.)\b', re.I)
RE_PO_VALUE = re.compile(r'\b([A-Za-z0-9\-_\/]+)\b')

RE_UNFI_TOTAL_LINE = re.compile(r'^\s*Total\s*:\s*(' + CURRENCY + r')\s*$', re.I | re.M)
RE_KEHE_NET_DED = re.compile(r'Net\s*Deduction\s*[:\-–]?\s*(' + CURRENCY + r')', re.I)
RE_KEHE_VENDOR_INV_TOTAL_ANY = re.compile(r'Vendor\s*Invoice\s*Total\s*[:\-]?\s*([$£€]?\s*' + NUM_AMT + r')', re.I)
RE_KEHE_NET_PAYABLE_ANY = re.compile(r'Net\s*Payable\s*[:\-]?\s*([$£€]?\s*' + NUM_AMT + r')', re.I)
RE_KEHE_EP_FEE = re.compile(r'\bEP\s*Fee\s*[:\-]?\s*(' + CURRENCY + r')', re.I)
RE_SLOT_TOT = re.compile(r'\bSlotting\s*Total\s*[:\-]?\s*(' + CURRENCY + r')', re.I)
RE_KEHE_TOTAL_PAYABLE = re.compile(r'(?:Total\s*Payable|TOTAL\s*PAYABLE)\s*[:\-]?\s*(?:' + CURRENCY + r'|' + NUM_AMT + r')', re.I)

RE_DEC_TOTAL_DUE = re.compile(r'Total\s*Due\s*DeCrescente\s*(' + CURRENCY + r')', re.I)
RE_POLAR_PRICE_SUPPORT = re.compile(r'Price\s*Support\s*[:\-]?\s*(' + CURRENCY + r')', re.I)
RE_SCAN_ONLY_TOTAL = re.compile(r'\bSCAN\s+ONLY\b.*?\bTotal\s*[:\-]?\s*(' + CURRENCY + r')', re.I | re.S)

RE_SHIPPED_TO = re.compile(r'Shipped\s*To\s*:\s*(.+)', re.I)
RE_ON_BEHALF_OF_LABEL = re.compile(r'On\s*Behalf\s*of\s*:', re.I)
RE_CUSTOMER_NAME_LABEL = re.compile(r'Customer\s*Name\s*:', re.I)

RE_DATE_RANGE_NUM = re.compile(r'(' + DATE_NUM + r')\s*(?:–|-|to|through|thru)\s*(' + DATE_NUM + r')', re.I)
RE_VALID_RANGE = re.compile(r'Valid\s+(' + DATE_NUM + r'|' + DATE_WORD + r')\s*(?:–|-|to)\s*(' + DATE_NUM + r'|' + DATE_WORD + r')', re.I)
RE_WEEK_OF = re.compile(r'Week\s+of\s+(' + DATE_WORD + r'|' + DATE_NUM + r')\s*(?:–|-|to)\s*(' + DATE_WORD + r'|' + DATE_NUM + r')', re.I)
RE_UNFI_PROMO_PERIOD_BLOCK = re.compile(r'Promo\s*Period[^\n]*', re.I)
RE_ANY_DATE = re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b')
RE_LINE_WITH_DISCOUNT = re.compile(r'(?:Scan|TPR|Buy\s*\d+|Price\s*Support|%|\$\s*\d+|SCAN\s+ONLY)', re.I)

VENDOR_EXCLUDE = [r'\bTRU\b', r'\bTRU\s*INC\b', r'\bEAT\s+THE\s+CHANGE\b', r'\bUNFI\b', r'\bKeHE\b']
RE_VENDOR_EXCLUDE = re.compile("|".join(VENDOR_EXCLUDE), re.I)

RETAILER_NORMALIZATION = {
    r'^(tru\s*inc|tru)$': 'TRU',
    r'polar beverages': 'Polar Beverages',
    r'\beat the change\b': 'Eat The Change',
    r'\bdecrescente\b': 'DeCrescente',
    r'\bkehe\b': 'KeHE',
    r'\bunfi\b': 'UNFI',
    r'\bfresh market\b': 'Fresh Market',
    r'\bhe?b\b': 'HEB',
}

# -------------------------
# Output schema
# -------------------------
class Row(BaseModel):
    source_file: str
    page: Optional[int] = None
    promotion_date_start: Optional[str] = None
    promotion_date_end: Optional[str] = None
    reason: Optional[str] = None
    notes: Optional[str] = None
    retailer_name: Optional[str] = None
    deduction_amount: Optional[float] = None
    discount_details: Optional[str] = None
    invoice_number: Optional[str] = None
    po_number: Optional[str] = None
    delivery_location: Optional[str] = None
    source_snippet: Optional[str] = None
    confidences: Optional[Dict[str, float]] = None

# -------------------------
# Utilities (normalize_text added)
# -------------------------
def normalize_text(s: str) -> str:
    if not s:
        return s
    # replace NBSP and other narrow spaces with normal space
    s = s.replace("\u00A0", " ").replace("\u2007", " ").replace("\u202F", " ")
    # remove zero-width chars
    s = s.replace("\u200B", "").replace("\u200C", "").replace("\u200D", "")
    # collapse whitespace/tabs
    s = re.sub(r'[ \t]+', ' ', s)
    return s

def iso_date(s: str) -> Optional[str]:
    if not s:
        return None
    try:
        dt = dateparser.parse(s, settings={"PREFER_DAY_OF_MONTH": "first", "RETURN_AS_TIMEZONE_AWARE": False})
        if dt:
            return dt.date().isoformat()
    except Exception:
        return None
    return None

def has_digit(s: str) -> bool:
    return any(ch.isdigit() for ch in (s or ""))

def image_from_fitz_pix(pix) -> "Image.Image":
    png = pix.tobytes("png")
    from PIL import Image
    return Image.open(io.BytesIO(png))

def is_labelish(line: str) -> bool:
    if not line:
        return True
    if line.strip().endswith(":"):
        return True
    low = line.lower()
    bad = ["customer", "address", "bill", "vendor", "invoice", "program", "sales journal", "remit", "admin fee", "amounts"]
    return any(b in low for b in bad)

def clean_retailer_candidate(s: str) -> str:
    s = s.strip()
    s = re.split(r'\s{2,}|\s-\s|[#\d].*', s)[0].strip()
    return s

def fetch_value_after_label(text: str, label_pat: re.Pattern) -> Optional[str]:
    m = label_pat.search(text)
    if not m:
        return None
    tail = text[m.end():].splitlines()
    if tail:
        same = tail[0].strip()
        if same and not is_labelish(same) and not RE_VENDOR_EXCLUDE.search(same):
            return clean_retailer_candidate(same)
    for ln in tail[1:11]:
        ln = ln.strip()
        if ln and not is_labelish(ln) and not RE_VENDOR_EXCLUDE.search(ln):
            return clean_retailer_candidate(ln)
    return None

def normalize_retailer(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    n = re.sub(r'\s+', ' ', name).strip()
    nl = n.lower()
    for patt, canon in RETAILER_NORMALIZATION.items():
        if re.search(patt, nl, re.I):
            return canon
    n = re.sub(r'[,;]\s*$', '', n)
    n = re.sub(r'\s{2,}', ' ', n)
    if len(n) < 2 or (len(n) > 60 and len(n.split()) > 10):
        return None
    return n

# -------------------------
# PDF text extraction
# -------------------------
def extract_pages_text(path: Path, ocr_mode: str = "auto") -> List[Tuple[int, str, bool]]:
    pages_out: List[Tuple[int, str, bool]] = []
    if HAS_FITZ:
        try:
            doc = fitz.open(path)
            for i in range(doc.page_count):
                page = doc.load_page(i)
                txt = page.get_text("text") or ""
                used_ocr = False
                words = txt.split()
                has_images = bool(page.get_images(full=True))
                printable = sum(1 for c in txt if c.isprintable() and not c.isspace())
                printable_ratio = printable / max(len(txt), 1)
                need_ocr = (ocr_mode == "force") or (ocr_mode == "auto" and (len(txt.strip()) < 200 and len(words) < 30 or (printable_ratio < 0.05 and has_images)))
                if need_ocr and HAS_TESS:
                    pix = page.get_pixmap(dpi=300)
                    img = image_from_fitz_pix(pix)
                    try:
                        ocr_txt = pytesseract.image_to_string(img)
                        txt = txt if len(txt) > len(ocr_txt) else ocr_txt
                        used_ocr = True
                    except Exception:
                        used_ocr = False
                pages_out.append((i + 1, txt, used_ocr))
            doc.close()
            return pages_out
        except Exception as e:
            logger.debug("fitz failed, falling back: %s", e)
    if HAS_PYPDF:
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages, start=1):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            used_ocr = False
            if (ocr_mode in ("auto", "force")) and (not txt.strip() or ocr_mode == "force"):
                if HAS_PDF2IMAGE and HAS_TESS:
                    try:
                        images = convert_from_path(str(path), dpi=300, first_page=i, last_page=i)
                        if images:
                            ocr_txt = pytesseract.image_to_string(images[0])
                            txt = txt if len(txt) > len(ocr_txt) else ocr_txt
                            used_ocr = True
                    except Exception:
                        pass
            pages_out.append((i, txt, used_ocr))
        return pages_out
    raise RuntimeError("No PDF backend available. Install PyMuPDF or pypdf+pdf2image+pytesseract.")

# -------------------------
# Doc type detection & extractors
# -------------------------
def detect_doc_type(text: str) -> str:
    t = text.lower()
    if "deduction invoice" in t and "unfi" in t:
        return "UNFI_DEDUCTION"
    if "invoice adjustment" in t and "kehe" in t:
        return "KEHE_ADJUSTMENT"
    if "pass thru deduction" in t and "kehe" in t:
        return "KEHE_PASS_THRU"
    if ("distributor charge" in t or "received by customer" in t) and "kehe" in t:
        return "KEHE_DISTRIBUTOR_CHARGE"
    if "polar beverages" in t:
        return "POLAR_LETTER"
    if "decrescente distributing" in t or "total due decrescente" in t or "may 2025 rebates" in t:
        return "DECRESCENTE"
    return "GENERIC"

def extract_invoice_number(text: str) -> Optional[str]:
    # 1) explicit Customer Invoice Number
    m0 = RE_INV_CUSTOMER.search(text)
    if m0:
        val = m0.group(1).strip()
        val = re.sub(r'\s+', '', val)
        if has_digit(val):
            return val

    # 2) try code near invoice label
    m = RE_INV_LABEL.search(text)
    if m:
        tail = text[m.end():].splitlines()
        for ln in tail[:6]:
            for tok in CODE_NEAR_LABEL.findall(ln):
                tok_clean = tok.strip().replace(" ", "")
                if has_digit(tok_clean):
                    return tok_clean

    # 3) explicit numeric invoice like Invoice # 6425155
    mnum = RE_INV_NUMERIC_LABEL.search(text)
    if mnum:
        return mnum.group(1).strip()

    # 4) H B P separated pattern anywhere
    m3 = RE_HBP_SEP.search(text)
    if m3:
        return "HBP" + m3.group(1)

    # 5) fallback token anywhere (non-pure numeric)
    m2 = RE_INV_FALLBACK_TOKEN.search(text)
    if m2:
        tok = m2.group(0).strip()
        return tok.replace(" ", "")
    return None

def extract_po_number(text: str) -> Optional[str]:
    last_val = None
    for ml in RE_PO_LABEL.finditer(text):
        tail = text[ml.end():].splitlines()
        for ln in tail[:6]:
            low = ln.lower()
            if any(bad in low for bad in ["dc#", "type:", "special payee", "file name", "vendor#", "phone", "telephone", "page"]):
                continue
            if "/" in ln or "\\" in ln:
                continue
            for tok in RE_PO_VALUE.findall(ln):
                t = tok.strip()
                if not has_digit(t):
                    continue
                if t.lower().startswith("box"):
                    continue
                if len(t) < 5 or len(t) > 24:
                    continue
                if re.fullmatch(r'\d{1,3}', t):
                    continue
                last_val = t
    if last_val:
        return last_val
    mflat = re.search(r'\b(FLAT[0-9A-Z\-]+)\b', text, re.I)
    if mflat:
        return mflat.group(1)
    return None

def extract_delivery_location(text: str) -> Optional[str]:
    m = RE_SHIPPED_TO.search(text)
    if m:
        return m.group(1).strip()
    return None

def extract_retailer_name(text: str, doc_type: str) -> Optional[str]:
    # normalize for KEHE distributor sold-to H.E.B
    if doc_type == "KEHE_DISTRIBUTOR_CHARGE":
        if re.search(r'SOLD\s*TO\s*:\s*H\.?E\.?B', text, re.I):
            return "HEB"

    if doc_type == "UNFI_DEDUCTION":
        val = fetch_value_after_label(text, RE_ON_BEHALF_OF_LABEL)
        if val: return val
        val = fetch_value_after_label(text, RE_CUSTOMER_NAME_LABEL)
        if val: return val
    if doc_type == "KEHE_PASS_THRU":
        if "healthy living event" in text.lower() and "heb" in text.lower():
            return "HEB"
    if doc_type == "POLAR_LETTER":
        return "Polar Beverages"
    if doc_type == "DECRESCENTE":
        return "DeCrescente"

    for kw in ["HEB", "Fresh Market", "Polar Beverages", "DeCrescente"]:
        if kw.lower() in text.lower():
            return kw
    return None

def extract_reason(text: str, doc_type: str) -> Optional[str]:
    if doc_type == "UNFI_DEDUCTION":
        return "Deduction Invoice"
    if doc_type == "KEHE_ADJUSTMENT":
        return "Invoice Adjustment"
    if doc_type == "KEHE_PASS_THRU":
        return "Pass Thru Deduction"
    if doc_type == "KEHE_DISTRIBUTOR_CHARGE":
        return "Distributor Charge"
    t = text.lower()
    if "chargeback" in t: return "Chargeback"
    if "deduction" in t: return "Deduction"
    return None

def extract_promo_dates(text: str) -> Tuple[Optional[str], Optional[str]]:
    m = RE_DATE_RANGE_NUM.search(text)
    if m:
        return iso_date(m.group(1)), iso_date(m.group(2))
    m = RE_VALID_RANGE.search(text)
    if m:
        return iso_date(m.group(1)), iso_date(m.group(2))
    m = RE_WEEK_OF.search(text)
    if m:
        return iso_date(m.group(1)), iso_date(m.group(2))
    periods = []
    for blk in RE_UNFI_PROMO_PERIOD_BLOCK.finditer(text):
        seg = blk.group(0)
        ds = RE_ANY_DATE.findall(seg)
        for d in ds:
            di = iso_date(d)
            if di:
                periods.append(di)
    if periods:
        return min(periods), max(periods)
    return None, None

def money_to_float(s: str) -> Optional[float]:
    try:
        return float(re.sub(r'[^\d\.]', '', s))
    except Exception:
        return None

def extract_deduction_amount(text: str, doc_type: str) -> Tuple[Optional[float], Optional[str]]:
    notes = None
    if doc_type == "KEHE_ADJUSTMENT":
        last = None
        for mm in RE_KEHE_NET_DED.finditer(text):
            last = mm
        if last:
            return money_to_float(last.group(1)), notes
        mv = RE_KEHE_VENDOR_INV_TOTAL_ANY.search(text)
        mp = RE_KEHE_NET_PAYABLE_ANY.search(text)
        if mv and mp:
            vt = money_to_float(mv.group(1))
            np = money_to_float(mp.group(1))
            if vt is not None and np is not None:
                return abs(vt - np), notes
        return None, notes

    if doc_type == "KEHE_PASS_THRU":
        m = RE_SLOT_TOT.search(text)
        if m:
            mfee = RE_KEHE_EP_FEE.search(text)
            if mfee:
                notes = f"EP Fee {mfee.group(1)}"
            return money_to_float(m.group(1)), notes
        return None, notes

    if doc_type == "KEHE_DISTRIBUTOR_CHARGE":
        last = None
        for mm in RE_KEHE_TOTAL_PAYABLE.finditer(text):
            last = mm
        if last:
            val = last.group(0).split(":")[-1].strip()
            mnum = re.search(r'(' + NUM_AMT + r')', val)
            if mnum:
                return money_to_float(mnum.group(1)), notes
        return None, notes

    if doc_type == "UNFI_DEDUCTION":
        last = None
        for mm in RE_UNFI_TOTAL_LINE.finditer(text):
            last = mm
        if last:
            return money_to_float(last.group(1)), notes
        return None, notes

    if doc_type == "DECRESCENTE":
        m = RE_DEC_TOTAL_DUE.search(text)
        if m:
            return money_to_float(m.group(1)), notes
        return None, notes

    if doc_type == "POLAR_LETTER":
        m = RE_POLAR_PRICE_SUPPORT.search(text)
        if m:
            return money_to_float(m.group(1)), notes
        m2 = RE_SCAN_ONLY_TOTAL.search(text)
        if m2:
            return money_to_float(m2.group(1)), notes
        return None, notes

    allc = re.findall(r'[' + r'\$\£\€' + r']\s*[0-9,]+\.\d{2}', text)
    if allc:
        nums = [money_to_float(x) for x in allc if money_to_float(x) is not None]
        if nums:
            return max(nums), notes
    all_nums = re.findall(r'([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d{2})?)', text)
    candidates = [money_to_float(x) for x in all_nums if money_to_float(x) and money_to_float(x) > 0.0]
    if candidates:
        return max(candidates), notes
    return None, notes

def extract_discount_details(text: str, doc_type: str, max_items: int = 4) -> Optional[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    hits = []
    for ln in lines:
        low = ln.lower()
        if low.startswith("total:"):
            continue
        if "cover letter" in low or "please" in low or "inquiry" in low:
            continue
        if "processing fee" in low:
            continue
        if RE_LINE_WITH_DISCOUNT.search(ln):
            hits.append(ln)
    out = []
    seen = set()
    for h in hits:
        k = h.lower()
        if k not in seen:
            out.append(h)
            seen.add(k)
    if not out:
        return None
    out = [re.sub(r'\s{2,}', ' ', s).strip(' ;,') for s in out]
    return "; ".join(out[:max_items])

# -------------------------
# Exposed parser for text (tests + e2e)
# -------------------------
def parse_text_string(full_text: str, source_name: str = "inline.txt") -> List[Dict[str, Any]]:
    # normalize to handle NBSP / ZW chars and weird spacing
    full_text = normalize_text(full_text)

    doc_type = detect_doc_type(full_text)
    invoice_number = extract_invoice_number(full_text)
    po_number = extract_po_number(full_text)
    delivery_location = extract_delivery_location(full_text)
    retailer_raw = extract_retailer_name(full_text, doc_type)
    retailer_name = normalize_retailer(retailer_raw)
    reason = extract_reason(full_text, doc_type)
    sdate, edate = extract_promo_dates(full_text)
    amount, notes = extract_deduction_amount(full_text, doc_type)
    discount_details = extract_discount_details(full_text, doc_type)
    confidences = {
        "invoice_number": 0.95 if invoice_number else 0.0,
        "po_number": 0.9 if po_number else 0.0,
        "promotion_date": 0.9 if (sdate or edate) else 0.0,
        "deduction_amount": 0.95 if amount is not None else 0.0,
        "discount_details": 0.85 if discount_details else 0.0,
        "retailer_name": 0.95 if retailer_name else 0.4
    }
    row = Row(
        source_file=source_name,
        page=None,
        promotion_date_start=sdate,
        promotion_date_end=edate,
        reason=reason,
        notes=notes,
        retailer_name=retailer_name,
        deduction_amount=amount,
        discount_details=discount_details,
        invoice_number=invoice_number,
        po_number=po_number,
        delivery_location=delivery_location,
        source_snippet=(full_text[:300] if full_text else None),
        confidences=confidences
    )
    dump = getattr(row, "model_dump", None)
    return [dump() if dump else row.dict()]

# -------------------------
# PDF wrapper and CLI
# -------------------------
def parse_document(path: Path, ocr_mode: str = "auto", debug: bool = False) -> List[Dict[str, Any]]:
    pages = extract_pages_text(path, ocr_mode=ocr_mode)
    full_text = "\n".join([ptext for (_, ptext, _) in pages])
    full_text = normalize_text(full_text)
    if debug:
        try:
            raw_out = path.with_suffix(".txt")
            with raw_out.open("w", encoding="utf-8") as f:
                f.write(full_text)
        except Exception:
            pass
    rows = parse_text_string(full_text, source_name=path.name)
    return rows

@app.command()
def main(
    input: str = typer.Option(..., help="Input PDF file or folder"),
    out: str = typer.Option(..., help="Output folder"),
    ocr: str = typer.Option("auto", help="OCR mode: auto|false|force"),
    limit: int = typer.Option(100, help="Max PDFs to process"),
    debug: bool = typer.Option(False, help="Save raw extracted text to .txt files")
):
    in_path = Path(input)
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)

    if in_path.is_file():
        pdfs = [in_path]
    else:
        pdfs = sorted([p for p in in_path.rglob("*")])[:limit]

    if not pdfs:
        typer.echo("No PDFs found. Exiting.")
        raise typer.Exit(code=1)

    all_rows: List[Dict[str, Any]] = []
    for pdf in pdfs:
        typer.echo(f"Processing {pdf.name} ...")
        try:
            rows = parse_document(pdf, ocr_mode=ocr, debug=debug)
            all_rows.extend(rows)
            if debug:
                with (out_path / (pdf.stem + ".txt")).open("w", encoding="utf-8") as f:
                    f.write("\n\n--- PAGE SPLIT ---\n\n".join([t for (_, t, _) in extract_pages_text(pdf, ocr_mode=ocr)]))
        except Exception as e:
            typer.echo(f"Error parsing {pdf.name}: {e}", err=True)

    jl = out_path / "parsed.jsonl"
    with jl.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if all_rows:
        keys = set()
        for r in all_rows:
            keys.update(r.keys())
        cs = out_path / "parsed.csv"
        with cs.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(keys))
            w.writeheader()
            w.writerows(all_rows)

    typer.echo(f"Done: processed {len(pdfs)} PDFs -> {len(all_rows)} rows. Results in {out_path}")

if __name__ == "__main__":
    app()
