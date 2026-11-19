from gpu_dev_assist.hybrid_retrieve import HybridRetriever


class FakeRetriever:
    def __init__(self, ids):
        self.ids = ids

    def query_retrieve(self, query, top_k=5):
        return [{"id": i, "text": i, "source": "x.pdf", "page": 1} for i in self.ids[:top_k]]


def test_chunk_ranked_well_by_both_wins():
    dense = FakeRetriever(["a", "b", "c"])
    keyword = FakeRetriever(["c", "d", "b"])
    hybrid = HybridRetriever(dense, keyword)

    top_ids = [r["id"] for r in hybrid.query_retrieve("q", top_k=4)]
    assert top_ids[0] in {"b", "c"}          # appear in both lists
    assert set(top_ids) == {"a", "b", "c", "d"}


def test_no_duplicates_across_lists():
    hybrid = HybridRetriever(FakeRetriever(["a", "b"]), FakeRetriever(["a", "b"]))
    ids = [r["id"] for r in hybrid.query_retrieve("q", top_k=5)]
    assert len(ids) == len(set(ids))