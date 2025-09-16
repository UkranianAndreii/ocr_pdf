# tests/test_parsers.py
import pytest
from src.parse_pdfs import (
    parse_text_string,
    extract_invoice_number,
    extract_po_number,
    extract_deduction_amount,
    normalize_retailer,
    normalize_text,
)

def test_hbp_with_spaces_and_nbsp_normalized():
    txt_variants = [
        "Invoice # HBP 192419\nSomething",
        "Invoice # H\u00A0B\u00A0P\u00A0192419\nSomething",
        "Invoice # H B P 192419\nSomething",
    ]
    for txt in txt_variants:
        norm = normalize_text(txt)
        inv = extract_invoice_number(norm)
        assert inv is not None
        assert inv.upper().startswith("HBP")
        assert " " not in inv

def test_invoice_not_grab_zip_or_small_numbers():
    txt = "Address: 4827 BETHESDA AVE BETHESDA MD 20814\nTotal: $7.88"
    inv = extract_invoice_number(normalize_text(txt))
    assert inv is None

def test_po_last_valid_and_ignore_noise():
    txt = (
        "Document info\nPO # //InvAdj/ByBatch/InvAdjBch1900248407.pdf\n"
        "Some header\nPO Number: 3023383\nDC#: 19\nPO #: 000000\n"
    )
    po = extract_po_number(normalize_text(txt))
    assert po == "3023383"

def test_kehe_amount_diff():
    txt = "Vendor Invoice Total: 3507.42\nNet Payable: 3240.00\nSome footer"
    amt, _ = extract_deduction_amount(normalize_text(txt), "KEHE_ADJUSTMENT")
    assert pytest.approx(amt, rel=1e-3) == pytest.approx(3507.42 - 3240.00, rel=1e-3)

def test_unfi_simple_e2e():
    raw = (
        "UNFI on Behalf of: Fresh Market\n"
        "Customer Invoice Number TFM050225 286129\n"
        "Invoice Date: 5/23/2025\n"
        "Total: $9,617.44\n"
    )
    rows = parse_text_string(raw, source_name="testdoc")
    assert len(rows) == 1
    r = rows[0]
    assert r["invoice_number"] is not None and r["invoice_number"].upper().startswith("TFM")
    assert r["retailer_name"] == "Fresh Market"
    assert r["deduction_amount"] == 9617.44

def test_kehe_distributor_sold_to_heb():
    txt = "DISTRIBUTOR CHARGE\nSOLD TO: H.E.B - Some store\nTOTAL PAYABLE 479.36"
    rows = parse_text_string(txt, source_name="testdoc2")
    r = rows[0]
    assert r["retailer_name"] == "HEB"
    assert r["deduction_amount"] == 479.36
