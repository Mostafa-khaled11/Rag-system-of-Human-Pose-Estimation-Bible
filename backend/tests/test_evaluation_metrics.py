from evaluation.metrics import key_point_coverage, reciprocal_rank, retrieval_scores


def test_deterministic_generation_coverage() -> None:
    score = key_point_coverage(
        "A heatmap gives a probability map for each keypoint and its peak is the location.",
        ["probability heatmap for each keypoint", "heatmap peak gives the location"],
    )
    assert score == 1


def test_reciprocal_rank_and_page_recall() -> None:
    results = [
        {
            "sample": {"answerable_from_book": True, "expected_pages": [7, 8]},
            "pages": [1, 7, 9],
        }
    ]
    scores = retrieval_scores(results, "pages")
    assert reciprocal_rank([1, 7], [7]) == 0.5
    assert scores.hit_rate == 1
    assert scores.recall == 0.5
    assert scores.mrr == 0.5
