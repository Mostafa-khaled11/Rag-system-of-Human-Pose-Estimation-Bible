from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.ingestion.chunking import PageText, normalize_text


class SourceDocumentError(ValueError):
    pass


@dataclass(frozen=True)
class ExtractedDocument:
    source_filename: str
    sha256: str
    pages: list[PageText]
    low_text_pages: list[int]


def extract_pdf(path: Path, max_bytes: int) -> ExtractedDocument:
    resolved = path.resolve(strict=True)
    if resolved.suffix.lower() != ".pdf":
        raise SourceDocumentError("Only PDF source books are supported")
    if resolved.stat().st_size > max_bytes:
        raise SourceDocumentError(f"Source PDF exceeds the {max_bytes}-byte limit")
    try:
        reader = PdfReader(resolved)
    except Exception as exc:
        raise SourceDocumentError("The source PDF is corrupt or unreadable") from exc
    if reader.is_encrypted:
        raise SourceDocumentError("The source PDF is encrypted; provide an unencrypted copy")
    pages: list[PageText] = []
    low: list[int] = []
    for number, pdf_page in enumerate(reader.pages, 1):
        try:
            text = normalize_text(pdf_page.extract_text() or "")
        except Exception as exc:
            raise SourceDocumentError(f"Text extraction failed on page {number}") from exc
        if len(text) < 50:
            low.append(number)
        pages.append(PageText(number, text))
    useful = sum(len(page.text) >= 50 for page in pages)
    if not pages or useful < max(3, len(pages) // 10):
        raise SourceDocumentError(
            "The PDF has too little extractable text and appears scanned/image-only; "
            "local OCR is required"
        )
    hasher = hashlib.sha256()
    with resolved.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    digest = hasher.hexdigest()
    return ExtractedDocument(resolved.name, digest, pages, low)
