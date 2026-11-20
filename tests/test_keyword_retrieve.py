"""Tests for BM25 tokenization."""

from gpu_dev_assist.keyword_retrieve import tokenize


def test_stemming_matches_word_forms():
    assert tokenize("prefetching") == tokenize("prefetch")
    assert tokenize("bank conflicts") == tokenize("bank conflict")


def test_identifiers_stay_single_tokens():
    tokens = tokenize("local_mem_size")
    assert len(tokens) == 1
    assert tokens[0].startswith("local_mem_")


def test_lowercases_and_drops_punctuation():
    assert tokenize("SYCL, sycl!") == tokenize("sycl sycl")
