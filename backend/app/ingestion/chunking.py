from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass


@dataclass(frozen=True)
class PageText:
    page: int
    text: str
    chapter: str | None = None


@dataclass(frozen=True)
class Chunk:
    id: str
    page: int
    chapter: str | None
    index: int
    text: str


def config_fingerprint(chunk_size: int, overlap: int, minimum: int) -> str:
    raw = json.dumps(
        {"strategy": "recursive", "size": chunk_size, "overlap": overlap, "minimum": minimum},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def normalize_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def detect_heading(text: str, previous: str | None = None) -> str | None:
    for line in text.splitlines()[:12]:
        candidate = line.strip()
        if 4 <= len(candidate) <= 100 and (
            re.match(r"^(chapter|part|section)\s+([\divxlc]+)\b", candidate, re.I)
            or (candidate.isupper() and len(candidate.split()) <= 12)
        ):
            return candidate
    return previous


def _split_once(text: str, limit: int) -> tuple[str, str]:
    if len(text) <= limit:
        return text, ""
    window = text[: limit + 1]
    candidates = [window.rfind(separator) for separator in ("\n\n", ". ", "\n", "; ", ", ", " ")]
    boundary = max(candidates)
    if boundary < max(1, limit // 2):
        boundary = limit
    elif window[boundary : boundary + 2] in {". ", "; ", ", "}:
        boundary += 1
    return text[:boundary].strip(), text[boundary:].strip()


def split_text(text: str, size: int, overlap: int, minimum: int) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    chunks: list[str] = []
    remaining = text
    while remaining:
        head, tail = _split_once(remaining, size)
        if not tail:
            if chunks and len(head) < minimum and len(chunks[-1]) + 1 + len(head) <= size:
                chunks[-1] = f"{chunks[-1]} {head}"
            elif head:
                chunks.append(head)
            break
        chunks.append(head)
        retained = head[-overlap:] if overlap else ""
        remaining = (
            f"{retained} {tail}".strip() if retained and len(retained) < len(remaining) else tail
        )
    return chunks


def chunk_pages(
    pages: list[PageText],
    document_hash: str,
    fingerprint: str,
    size: int,
    overlap: int,
    minimum: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    chapter: str | None = None
    for page in pages:
        chapter = page.chapter or detect_heading(page.text, chapter)
        for index, text in enumerate(split_text(page.text, size, overlap, minimum)):
            seed = f"{document_hash}:{fingerprint}:{page.page}:{index}"
            chunks.append(
                Chunk(str(uuid.uuid5(uuid.NAMESPACE_URL, seed)), page.page, chapter, index, text)
            )
    return chunks
