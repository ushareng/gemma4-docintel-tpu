"""Calls to the vLLM OpenAI-compatible server: OCR, field extraction, cited QA."""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from . import config
from .documents import Document, Page
from .retrieval import BM25Index, Chunk
from .schema import Answer, Citation, InvoiceFields, response_format

log = logging.getLogger("docintel")
client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=600)


@dataclass
class CallStats:
    latency_s: float
    prompt_tokens: int
    completion_tokens: int

    def __str__(self) -> str:
        return (f"{self.latency_s:.2f}s · {self.prompt_tokens} prompt tok · "
                f"{self.completion_tokens} output tok")


def _chat(messages: list[dict], max_tokens: int, schema: type[BaseModel] | None = None,
          temperature: float = 0.0) -> tuple[str, CallStats]:
    kwargs: dict = {}
    if schema is not None:
        kwargs["response_format"] = response_format(schema)
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=config.MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": config.ENABLE_THINKING}},
        **kwargs,
    )
    dt = time.perf_counter() - t0
    u = resp.usage
    stats = CallStats(dt, u.prompt_tokens if u else 0, u.completion_tokens if u else 0)
    return resp.choices[0].message.content or "", stats


def _parse(model: type[BaseModel], raw: str) -> BaseModel:
    """Validate model output. Backends that don't enforce the schema can wrap the
    JSON in code fences or add text around it, so try a few cleanups."""
    try:
        return model.model_validate_json(raw)
    except (ValidationError, ValueError):
        pass
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    try:
        return model.model_validate(json.loads(cleaned))
    except (ValidationError, ValueError) as e:
        log.warning("Could not parse %s from model output (%s). Raw output:\n%s",
                    model.__name__, e, raw[:2000])
        raise


# ---------------------------------------------------------------- OCR (scanned pages)
OCR_PROMPT = ("Transcribe all text on this page exactly as printed, in natural reading order. "
              "Render tables as rows with ' | ' between cells. Output only the transcription.")


def ocr_page(page: Page) -> CallStats:
    text, stats = _chat(
        [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": page.data_url()}},
            {"type": "text", "text": OCR_PROMPT},
        ]}],
        max_tokens=2048,
    )
    page.text, page.ocr_done = text.strip(), True
    return stats


# ---------------------------------------------------------------- field extraction
EXTRACT_SYSTEM = (
    "You are a precise document-understanding system for invoices, bills and receipts. "
    "Read the page images (and the text layer when given) and fill every field of the JSON schema. "
    "Copy values exactly as printed; use null when a field is absent. Dates must be YYYY-MM-DD. "
    "Numbers are plain decimals without currency symbols or thousands separators."
)


def extract_fields(doc: Document) -> tuple[InvoiceFields, str, CallStats]:
    content: list[dict] = []
    for page in doc.pages[: config.MAX_IMAGES_PER_REQUEST]:
        content.append({"type": "text", "text": f"--- Page {page.number} ---"})
        content.append({"type": "image_url", "image_url": {"url": page.data_url()}})
        if page.text:
            content.append({"type": "text", "text": f"Text of page {page.number}:\n{page.text[:6000]}"})
    content.append({"type": "text", "text": "Extract the fields as JSON."})

    raw, stats = _chat(
        [{"role": "system", "content": EXTRACT_SYSTEM}, {"role": "user", "content": content}],
        max_tokens=2048,
        schema=InvoiceFields,
    )
    return _parse(InvoiceFields, raw), raw, stats


# ---------------------------------------------------------------- QA with page citations
QA_SYSTEM = (
    "Answer the question using ONLY the context passages. Each passage starts with a "
    "[page N] tag. Cite every page you used, with a short verbatim quote from it. If the "
    "context does not contain the answer, set found=false and say so briefly.\n"
    "Reply with JSON only, exactly in this shape:\n"
    '{"found": true, "answer": "...", "citations": [{"page": 4, "quote": "..."}]}\n'
    "Escape any double quotes inside strings as \\\"."
)
QA_PLAIN_SYSTEM = (
    "Answer the question using ONLY the context passages. Each passage starts with a "
    "[page N] tag. After every claim, cite its page like [p.4]. If the context does not "
    "contain the answer, reply exactly: NOT FOUND."
)


def _plain_answer(context: str, question: str) -> tuple[Answer, CallStats]:
    """Fallback without structured output: free text with [p.N] tags, parsed by regex."""
    text, stats = _chat(
        [{"role": "system", "content": QA_PLAIN_SYSTEM},
         {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        max_tokens=512,
    )
    text = text.strip()
    if text.upper().startswith("NOT FOUND"):
        return Answer(found=False, answer="", citations=[]), stats
    pages = sorted({int(n) for n in re.findall(r"\[p\.?\s*(\d+)\]", text)})
    return Answer(found=True, answer=text,
                  citations=[Citation(page=p, quote="") for p in pages]), stats


def answer_question(index: BM25Index, question: str, k: int = config.TOP_K
                    ) -> tuple[Answer, list[tuple[Chunk, float]], CallStats]:
    hits = index.search(question, k=k)
    context = "\n\n".join(f"[page {c.page}] {c.text}" for c, _ in hits) or "(no text found)"
    messages = [{"role": "system", "content": QA_SYSTEM},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]
    ans = None
    for _attempt in range(2):  # structured output, retried once
        raw, stats = _chat(messages, max_tokens=768, schema=Answer)
        try:
            ans = _parse(Answer, raw)
            break
        except (ValidationError, ValueError):
            continue
    if ans is None:  # backend didn't honour the schema twice: plain-text fallback
        ans, stats = _plain_answer(context, question)
    # Keep only citations that point at pages we actually retrieved.
    retrieved_pages = {c.page for c, _ in hits}
    ans.citations = [c for c in ans.citations if c.page in retrieved_pages]
    return ans, hits, stats


def server_ok() -> tuple[bool, str]:
    try:
        models = [m.id for m in client.models.list().data]
        return True, f"Connected to vLLM at {config.BASE_URL} · models: {', '.join(models)}"
    except Exception as e:  # noqa: BLE001
        return False, f"vLLM server not reachable at {config.BASE_URL}: {e}"
