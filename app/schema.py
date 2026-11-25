"""Pydantic schemas. Their JSON Schemas are handed to vLLM structured outputs,
so the model can only emit JSON that validates against them."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Party(BaseModel):
    name: Optional[str] = Field(None, description="Legal or trading name")
    address: Optional[str] = Field(None, description="Postal address on one line")
    tax_id: Optional[str] = Field(None, description="GSTIN / VAT / EIN / tax ID if printed")


class LineItem(BaseModel):
    description: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = Field(None, description="Line total as printed")


class InvoiceFields(BaseModel):
    document_type: Literal["invoice", "bill", "receipt", "credit_note", "other"]
    vendor: Party
    customer: Party
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = Field(None, description="ISO 8601 date, YYYY-MM-DD")
    due_date: Optional[str] = Field(None, description="ISO 8601 date, YYYY-MM-DD")
    currency: Optional[str] = Field(None, description="ISO 4217 code such as USD, EUR, INR")
    line_items: list[LineItem]
    subtotal: Optional[float] = None
    tax: Optional[float] = Field(None, description="Total tax amount")
    total: Optional[float] = Field(None, description="Grand total / amount due")
    payment_terms: Optional[str] = None
    source_pages: list[int] = Field(description="1-based page numbers the fields were read from")


class Citation(BaseModel):
    page: int = Field(description="1-based page number from the [page N] tag")
    quote: str = Field(description="Short verbatim snippet from that page supporting the answer")


class Answer(BaseModel):
    found: bool = Field(description="False if the context does not contain the answer")
    answer: str
    citations: list[Citation]


def response_format(model: type[BaseModel]) -> dict:
    """OpenAI-style response_format accepted by vLLM's structured outputs."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "schema": model.model_json_schema(),
            "strict": True,
        },
    }


def consistency_checks(f: InvoiceFields, tol: float = 0.02) -> list[tuple[str, str]]:
    """Arithmetic sanity checks on extracted numbers: (check, result) rows for the UI."""
    rows: list[tuple[str, str]] = []
    amounts = [li.amount for li in f.line_items if li.amount is not None]

    def close(a: float, b: float) -> bool:
        return abs(a - b) <= max(tol, 0.005 * max(abs(a), abs(b)))

    for i, li in enumerate(f.line_items, 1):
        if None not in (li.quantity, li.unit_price, li.amount):
            ok = close(li.quantity * li.unit_price, li.amount)
            rows.append((f"Line {i}: qty × unit price = amount", "✅" if ok else "⚠️ mismatch"))
    if amounts and f.subtotal is not None:
        ok = close(sum(amounts), f.subtotal)
        rows.append(("Σ line items = subtotal", "✅" if ok else f"⚠️ {sum(amounts):.2f} vs {f.subtotal:.2f}"))
    if None not in (f.subtotal, f.tax, f.total):
        ok = close(f.subtotal + f.tax, f.total)
        rows.append(("subtotal + tax = total", "✅" if ok else f"⚠️ {f.subtotal + f.tax:.2f} vs {f.total:.2f}"))
    if not rows:
        rows.append(("Arithmetic checks", "not enough numbers to check"))
    return rows
