"""Tests for answer generation (retriever and LLM faked; no API key needed)."""

from types import SimpleNamespace

from gpu_dev_assist.generate import (
    INSUFFICIENT_EVIDENCE,
    SYSTEM_PROMPT,
    AnswerGenerator,
    build_context,
)

RESULTS = [
    {"text": "oneAPI supports single-source compilation.", "source": "guide.pdf", "page": 46, "distance": 0.2},
    {"text": "SYCL and OpenMP are the two portable methods.", "source": "guide.pdf", "page": 7, "distance": 0.3},
]


class FakeRetriever:
    def query_retrieve(self, question, top_k=6):
        return RESULTS


class FakeMessages:
    def __init__(self, text, thinking_first=False):
        self.text, self.thinking_first, self.kwargs = text, thinking_first, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        content = [SimpleNamespace(type="text", text=self.text)]
        if self.thinking_first:
            content.insert(0, SimpleNamespace(type="thinking", thinking="..."))
        return SimpleNamespace(content=content, usage=SimpleNamespace(input_tokens=10, output_tokens=5))


def make_generator(text, **kwargs):
    messages = FakeMessages(text, **kwargs)
    return AnswerGenerator(FakeRetriever(), client=SimpleNamespace(messages=messages)), messages


def test_build_context_numbers_sources():
    context = build_context(RESULTS)
    assert context.startswith("[1] (guide.pdf, page 46)")
    assert "[2] (guide.pdf, page 7)" in context


def test_citations_map_to_pages():
    generator, _ = make_generator("Single-source is supported [1]. Two methods exist [2].")
    sources = generator.answer("q")["sources"]
    assert [(s["id"], s["page"]) for s in sources] == [(1, 46), (2, 7)]


def test_out_of_range_citation_is_dropped():
    generator, _ = make_generator("Real claim [1]. Invented claim [9].")
    assert [s["id"] for s in generator.answer("q")["sources"]] == [1]


def test_refusal_has_no_sources():
    generator, _ = make_generator(INSUFFICIENT_EVIDENCE)
    result = generator.answer("q")
    assert result["answer"] == INSUFFICIENT_EVIDENCE
    assert result["sources"] == []


def test_thinking_blocks_are_ignored():
    generator, _ = make_generator("Answer [1].", thinking_first=True)
    assert generator.answer("q")["answer"] == "Answer [1]."


def test_prompt_contains_sources_and_question():
    generator, messages = make_generator("x")
    generator.answer("What is oneAPI?")
    assert messages.kwargs["system"] == SYSTEM_PROMPT
    content = messages.kwargs["messages"][0]["content"]
    assert "[1] (guide.pdf, page 46)" in content
    assert "What is oneAPI?" in content