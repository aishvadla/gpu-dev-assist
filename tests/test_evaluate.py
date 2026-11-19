from gpu_dev_assist.evaluate import hit_at_k, ranked_pages, recall_at_k, reciprocal_rank

A, B, C, D = ("doc", 1), ("doc", 2), ("doc", 3), ("doc", 4)


def test_ranked_pages_removes_duplicates_keeping_order():
    results = [{"source": "doc", "page": 2}, {"source": "doc", "page": 2}, {"source": "doc", "page": 1}]
    assert ranked_pages(results) == [B, A]


def test_hit_at_k():
    assert hit_at_k([A, B, C], {C}, k=3) == 1.0
    assert hit_at_k([A, B, C], {C}, k=2) == 0.0


def test_recall_at_k():
    assert recall_at_k([A, B, C], {A, C, D}, k=3) == 2 / 3


def test_reciprocal_rank():
    assert reciprocal_rank([A, B, C], {B}, k=3) == 0.5
    assert reciprocal_rank([A, B, C], {D}, k=3) == 0.0