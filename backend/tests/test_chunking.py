from app.ingestion.chunking import PageText, chunk_pages, config_fingerprint, split_text


def test_overlap_and_boundaries() -> None:
    text = "First paragraph has useful detail.\n\nSecond paragraph contains more explanation. " * 10
    chunks = split_text(text, 220, 40, 30)
    assert len(chunks) > 1
    assert all(len(chunk) <= 220 for chunk in chunks)
    assert chunks[0][-40:].strip() in chunks[1]


def test_metadata_and_ids_are_deterministic() -> None:
    pages = [PageText(7, "CHAPTER 2\n" + "Pose representation. " * 30)]
    fp = config_fingerprint(300, 50, 40)
    first = chunk_pages(pages, "abc", fp, 300, 50, 40)
    second = chunk_pages(pages, "abc", fp, 300, 50, 40)
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert all(chunk.page == 7 and chunk.chapter == "CHAPTER 2" for chunk in first)
