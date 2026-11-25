"""Load PDFs and images into pages (text layer + rendered PNG) with PyMuPDF."""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from PIL import Image

from . import config

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


@dataclass
class Page:
    number: int                 # 1-based
    text: str                   # text layer, or OCR text filled in later
    png: bytes                  # rendered page image (JPEG bytes)
    has_text_layer: bool
    ocr_done: bool = False

    @property
    def is_scanned(self) -> bool:
        return not self.has_text_layer

    def data_url(self) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(self.png).decode()

    def pil(self) -> Image.Image:
        return Image.open(io.BytesIO(self.png))


@dataclass
class Document:
    name: str
    pages: list[Page] = field(default_factory=list)

    @property
    def scanned_pages(self) -> list[Page]:
        return [p for p in self.pages if p.is_scanned and not p.ocr_done]

    def summary(self) -> str:
        n_scan = sum(p.is_scanned for p in self.pages)
        return (f"**{self.name}** — {len(self.pages)} page(s), "
                f"{len(self.pages) - n_scan} with a text layer, {n_scan} scanned")


def _shrink_png(img: Image.Image) -> bytes:
    img = img.convert("RGB")
    scale = config.MAX_IMAGE_SIDE / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def load_document(path: str | Path) -> Document:
    path = Path(path)
    doc = Document(name=path.name)

    if path.suffix.lower() in IMAGE_SUFFIXES:
        img = Image.open(path)
        doc.pages.append(Page(number=1, text="", png=_shrink_png(img), has_text_layer=False))
        return doc

    with pymupdf.open(path) as pdf:
        for i, pg in enumerate(pdf):
            if i >= config.MAX_PAGES:
                break
            text = pg.get_text("text", sort=True).strip()
            pix = pg.get_pixmap(dpi=config.RENDER_DPI)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            has_text = len(text) >= config.SCANNED_TEXT_THRESHOLD
            doc.pages.append(Page(number=i + 1, text=text if has_text else "",
                                  png=_shrink_png(img), has_text_layer=has_text))
    return doc
