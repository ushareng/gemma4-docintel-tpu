"""Generate fictional demo documents in ./samples (all marked SAMPLE / DEMO DATA).

  invoice_digital.pdf   – invoice with a real text layer
  invoice_scanned.pdf   – image-only PDF (simulated scan: rotated, noisy, no text layer)
  receipt_photo.png     – a small receipt as a photo-like image
  service_agreement.pdf – 8-page agreement for cited Q&A and the prefill benchmark
"""
from __future__ import annotations

import io
import random
from pathlib import Path

import pymupdf
from PIL import Image, ImageFilter

OUT = Path(__file__).resolve().parent.parent / "samples"
W, H = pymupdf.paper_size("a4")
BANNER = "SAMPLE DOCUMENT - FICTIONAL DEMO DATA"


def _text(pg, x, y, s, size=10, bold=False, color=(0, 0, 0)):
    pg.insert_text((x, y), s, fontsize=size, fontname="hebo" if bold else "helv", color=color)


def _banner(pg):
    _text(pg, 40, 20, BANNER, 7, color=(0.6, 0.6, 0.6))


def draw_invoice(pg, inv: dict):
    _banner(pg)
    _text(pg, 40, 60, inv["vendor"], 18, bold=True)
    for i, line in enumerate(inv["vendor_addr"]):
        _text(pg, 40, 78 + i * 13, line, 9)
    _text(pg, 400, 60, "TAX INVOICE", 16, bold=True)
    meta = [("Invoice No.", inv["number"]), ("Invoice Date", inv["date"]),
            ("Due Date", inv["due"]), ("Currency", inv["currency"])]
    for i, (k, v) in enumerate(meta):
        _text(pg, 400, 82 + i * 14, f"{k}:", 9, bold=True)
        _text(pg, 475, 82 + i * 14, v, 9)

    _text(pg, 40, 160, "BILL TO", 9, bold=True)
    for i, line in enumerate(inv["customer"]):
        _text(pg, 40, 175 + i * 13, line, 9)

    y = 250
    cols = [40, 300, 360, 450]
    pg.draw_rect(pymupdf.Rect(36, y - 13, W - 36, y + 5), color=None, fill=(0.9, 0.92, 0.96))
    for x, h in zip(cols, ["Description", "Qty", "Unit Price", "Amount"]):
        _text(pg, x, y, h, 9, bold=True)
    subtotal = 0.0
    for desc, qty, price in inv["items"]:
        y += 18
        amt = qty * price
        subtotal += amt
        _text(pg, cols[0], y, desc, 9)
        _text(pg, cols[1], y, f"{qty:g}", 9)
        _text(pg, cols[2], y, f"{price:,.2f}", 9)
        _text(pg, cols[3], y, f"{amt:,.2f}", 9)
    pg.draw_line((36, y + 8), (W - 36, y + 8), color=(0.5, 0.5, 0.5))
    tax = round(subtotal * inv["tax_rate"], 2)
    for label, val in [("Subtotal", subtotal), (f"{inv['tax_name']} {inv['tax_rate']*100:g}%", tax),
                       ("TOTAL DUE", subtotal + tax)]:
        y += 18
        _text(pg, 360, y, label, 10, bold=label == "TOTAL DUE")
        _text(pg, 450, y, f"{inv['symbol']}{val:,.2f}", 10, bold=label == "TOTAL DUE")
    y += 40
    _text(pg, 40, y, "Payment terms: " + inv["terms"], 9)
    _text(pg, 40, y + 14, "Bank: Example Bank, A/C 000123456789 (demo)", 9)
    _text(pg, 40, H - 40, "Thank you for your business.", 9, color=(0.3, 0.3, 0.3))


INV_A = dict(
    vendor="Acme Tensor Supplies Pvt. Ltd.",
    vendor_addr=["42 Silicon Road, Bengaluru 560001, India", "GSTIN: 29ABCDE1234F1Z5 (sample)"],
    number="ATS-2026-0417", date="12 Nov 2026", due="12 Dec 2026", currency="INR", symbol="Rs ",
    customer=["Nimbus Robotics LLP", "7 Lake View, Pune 411001", "GSTIN: 27PQRSX6789K1Z2 (sample)"],
    items=[("Cloud TPU training workshop (2 days)", 1, 85000.00),
           ("Model fine-tuning consultation (hours)", 12, 4500.00),
           ("Annotated invoice dataset licence", 3, 7200.00),
           ("Priority support plan - Q4", 1, 15000.00)],
    tax_name="GST", tax_rate=0.18, terms="Net 30. 1.5% monthly interest on overdue balances.",
)
INV_B = dict(
    vendor="Bluefin Office Services GmbH",
    vendor_addr=["Hauptstrasse 15, 10115 Berlin, Germany", "USt-IdNr.: DE123456789 (sample)"],
    number="BOS-88213", date="03 Oct 2026", due="17 Oct 2026", currency="EUR", symbol="EUR ",
    customer=["Orbit Analytics B.V.", "Keizersgracht 100, 1015 Amsterdam, NL"],
    items=[("Ergonomic desk chairs", 6, 249.00), ("Standing desk frames", 4, 389.50),
           ("Delivery and assembly", 1, 180.00)],
    tax_name="VAT", tax_rate=0.19, terms="Payable within 14 days of invoice date.",
)


def invoice_digital(path: Path):
    doc = pymupdf.open()
    draw_invoice(doc.new_page(width=W, height=H), INV_A)
    doc.save(path)


def invoice_scanned(path: Path):
    src = pymupdf.open()
    draw_invoice(src.new_page(width=W, height=H), INV_B)
    pix = src[0].get_pixmap(dpi=130)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
    img = img.rotate(-1.2, expand=True, fillcolor=245).filter(ImageFilter.GaussianBlur(0.6))
    rnd = random.Random(7)
    px = img.load()
    for _ in range(img.width * img.height // 120):  # speckle noise like a photocopy
        x, y = rnd.randrange(img.width), rnd.randrange(img.height)
        px[x, y] = rnd.choice((90, 160, 255))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=70)
    out = pymupdf.open()
    page = out.new_page(width=W, height=H)
    page.insert_image(page.rect, stream=buf.getvalue())  # image only: no text layer
    out.save(path)


def receipt_photo(path: Path):
    doc = pymupdf.open()
    pg = doc.new_page(width=230, height=360)
    lines = [("CAFE ORBITAL (SAMPLE)", 11, True), ("12 MG Road, Bengaluru", 8, False),
             ("Bill #: 5521   24-11-2026 19:42", 8, False), ("", 8, False),
             ("2 x Masala Dosa        240.00", 8, False), ("1 x Filter Coffee       60.00", 8, False),
             ("1 x Paneer Wrap        180.00", 8, False), ("", 8, False),
             ("Subtotal              480.00", 8, False), ("CGST 2.5%              12.00", 8, False),
             ("SGST 2.5%              12.00", 8, False), ("TOTAL       Rs        504.00", 10, True),
             ("", 8, False), ("Paid: UPI", 8, False), ("DEMO DATA - NOT A REAL BILL", 7, False)]
    y = 30
    for s, size, bold in lines:
        pg.insert_text((18, y), s, fontsize=size, fontname="cobo" if bold else "cour")
        y += size + 9
    pix = pg.get_pixmap(dpi=200)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    img = img.rotate(2.5, expand=True, fillcolor=(214, 206, 190)).filter(ImageFilter.GaussianBlur(0.8))
    img.save(path)


AGREEMENT = [
    ("1. Parties and Purpose",
     "This Master Services Agreement is made between Acme Tensor Supplies Pvt. Ltd. (the Provider) "
     "and Nimbus Robotics LLP (the Client). The Provider will deliver machine-learning training, "
     "model fine-tuning and inference optimisation services on Cloud TPU infrastructure."),
    ("2. Term",
     "The Agreement starts on 1 January 2027 and runs for an initial term of 24 months. It renews "
     "automatically for successive 12-month periods unless either party gives 90 days' written notice."),
    ("3. Fees and Invoicing",
     "The Client pays a monthly retainer of INR 3,50,000 plus applicable GST. Additional consulting is "
     "billed at INR 4,500 per hour. Invoices are issued on the first business day of each month."),
    ("4. Payment Terms",
     "Invoices are payable within 30 days. Overdue amounts accrue interest at 1.5% per month. The Provider "
     "may suspend services if any invoice remains unpaid for more than 45 days after its due date."),
    ("5. Service Levels",
     "The Provider guarantees 99.5% monthly availability of the managed inference endpoint. If availability "
     "falls below this level, the Client receives a service credit of 10% of that month's retainer, capped "
     "at 30% of the retainer in any quarter."),
    ("6. Data Protection",
     "Client data is processed only in the asia-south1 and europe-west4 regions. The Provider deletes all "
     "Client data within 15 days of termination and supplies a written deletion certificate on request."),
    ("7. Intellectual Property",
     "Fine-tuned model weights produced with Client data belong to the Client. The Provider keeps ownership "
     "of its pre-existing tooling, kernels and benchmark scripts, and grants the Client a perpetual licence "
     "to use them internally."),
    ("8. Limitation of Liability",
     "Each party's total liability is limited to the fees paid in the 12 months before the claim. Neither "
     "party is liable for indirect or consequential losses. This Agreement is governed by the laws of India, "
     "with exclusive jurisdiction in the courts of Bengaluru."),
]

FILLER = (
    "The parties acknowledge that the services described in this section will be delivered in accordance "
    "with the statement of work, the change-control procedure and the security annex. Each party shall "
    "appoint a named relationship manager, hold a monthly governance review and record decisions in a shared "
    "log. Any ambiguity is resolved in favour of the more specific provision."
)


def service_agreement(path: Path):
    doc = pymupdf.open()
    for title, body in AGREEMENT:
        pg = doc.new_page(width=W, height=H)
        _banner(pg)
        _text(pg, 40, 60, "Master Services Agreement - Acme Tensor Supplies / Nimbus Robotics", 9,
              color=(0.4, 0.4, 0.4))
        _text(pg, 40, 95, title, 14, bold=True)
        text = body + "\n\n" + "\n\n".join([FILLER] * 4)
        pg.insert_textbox(pymupdf.Rect(40, 110, W - 40, H - 50), text, fontsize=10.5, fontname="helv")
        _text(pg, W / 2 - 20, H - 25, f"Page {pg.number + 1}", 8, color=(0.4, 0.4, 0.4))
    doc.save(path)


def main():
    OUT.mkdir(exist_ok=True)
    invoice_digital(OUT / "invoice_digital.pdf")
    invoice_scanned(OUT / "invoice_scanned.pdf")
    receipt_photo(OUT / "receipt_photo.png")
    service_agreement(OUT / "service_agreement.pdf")
    print("Wrote:", ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
