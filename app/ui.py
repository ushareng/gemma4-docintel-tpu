"""Gradio demo: upload an invoice/bill/PDF → structured JSON fields + cited Q&A.

Run:  python -m app.ui            (serves on 127.0.0.1:7860; use an SSH tunnel)
      python -m app.ui --share    (temporary public gradio.live link)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr

from . import llm
from .documents import Document, load_document
from .retrieval import BM25Index
from .schema import consistency_checks

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def _sample_choices() -> list[str]:
    if not SAMPLES_DIR.exists():
        return []
    return sorted(p.name for p in SAMPLES_DIR.iterdir()
                  if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"})


# ------------------------------------------------------------------ handlers
def on_load(file_path, sample_name, progress=gr.Progress()):
    path = file_path or (SAMPLES_DIR / sample_name if sample_name else None)
    if not path:
        raise gr.Error("Upload a PDF/image or pick a sample.")
    progress(0, desc="Rendering pages")
    doc = load_document(path)

    ocr_note = ""
    scanned = doc.scanned_pages
    if scanned:
        ok, msg = llm.server_ok()
        if not ok:
            raise gr.Error(msg)
        total = 0.0
        for i, page in enumerate(scanned):
            progress((i + 1) / (len(scanned) + 1), desc=f"OCR page {page.number} on TPU")
            total += llm.ocr_page(page).latency_s
        ocr_note = f" · OCR'd {len(scanned)} scanned page(s) with Gemma in {total:.1f}s"

    index = BM25Index(doc)
    gallery = [(p.pil(), f"Page {p.number}" + (" (scanned)" if p.is_scanned else ""))
               for p in doc.pages]
    status = doc.summary() + f" · {len(index.chunks)} retrieval chunks" + ocr_note
    return doc, index, gallery, status


def on_extract(doc: Document | None):
    if doc is None:
        raise gr.Error("Load a document first.")
    fields, _raw, stats = llm.extract_fields(doc)
    checks = consistency_checks(fields)
    return fields.model_dump(), checks, f"⏱️ {stats}"


def on_ask(index: BM25Index | None, question: str):
    if index is None:
        raise gr.Error("Load a document first.")
    if not question.strip():
        raise gr.Error("Type a question.")
    ans, hits, stats = llm.answer_question(index, question)
    md = ans.answer if ans.found else f"_Not found in the document._ {ans.answer}"
    if ans.citations:
        md += "\n\n**Sources**\n" + "\n".join(
            f"- **p.{c.page}**" + (f" — “{c.quote}”" if c.quote else "") for c in ans.citations)
    passages = [[c.page, round(s, 2), c.text[:300] + ("…" if len(c.text) > 300 else "")]
                for c, s in hits]
    return md, passages, f"⏱️ {stats}"


# ------------------------------------------------------------------ layout
def build() -> gr.Blocks:
    ok, msg = llm.server_ok()
    with gr.Blocks(title="DocIntel · Gemma 4 on Cloud TPU v6e") as demo:
        gr.Markdown(
            "# 🧾 DocIntel — Gemma 4 E4B on a single Cloud TPU v6e chip\n"
            "Scanned invoices, bills and PDFs → **schema-valid JSON** (vLLM structured outputs) "
            "and **questions answered with page citations** (PyMuPDF + BM25). "
            "Served by **vLLM-TorchTPU**.")
        gr.Markdown(("🟢 " if ok else "🔴 ") + msg)

        doc_state, index_state = gr.State(None), gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                file_in = gr.File(label="Upload PDF or image", type="filepath",
                                  file_types=[".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"])
                sample_in = gr.Dropdown(_sample_choices(), label="…or pick a sample", value=None)
                load_btn = gr.Button("Load document", variant="primary")
                status = gr.Markdown()
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Pages", columns=4, height=320, object_fit="contain")

        with gr.Tab("📋 Extract fields"):
            extract_btn = gr.Button("Extract structured fields", variant="primary")
            extract_stats = gr.Markdown()
            with gr.Row():
                fields_out = gr.JSON(label="InvoiceFields (validated by Pydantic)")
                checks_out = gr.Dataframe(headers=["Check", "Result"], label="Arithmetic checks",
                                          interactive=False)

        with gr.Tab("💬 Ask the document"):
            question = gr.Textbox(label="Question", placeholder="What is the payment due date?")
            gr.Examples(["What is the total amount due?", "Who is the vendor and what is their tax ID?",
                         "What are the payment terms?", "Which line item costs the most?"],
                        inputs=question)
            ask_btn = gr.Button("Ask", variant="primary")
            ask_stats = gr.Markdown()
            answer_out = gr.Markdown()
            passages_out = gr.Dataframe(headers=["Page", "BM25", "Retrieved passage"],
                                        label="Retrieved context", interactive=False, wrap=True)

        load_btn.click(on_load, [file_in, sample_in], [doc_state, index_state, gallery, status])
        extract_btn.click(on_extract, [doc_state], [fields_out, checks_out, extract_stats])
        ask_btn.click(on_ask, [index_state, question], [answer_out, passages_out, ask_stats])
        question.submit(on_ask, [index_state, question], [answer_out, passages_out, ask_stats])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a temporary public link")
    a = ap.parse_args()
    build().queue(default_concurrency_limit=8).launch(server_name=a.host, server_port=a.port,
                                                      share=a.share, show_error=True)


if __name__ == "__main__":
    main()
