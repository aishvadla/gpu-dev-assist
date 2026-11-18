from gpu_dev_assist.chunking_embedding import ChunkingEmbeddingPipeline
from gpu_dev_assist.data_loader import load_all_documents

def test_chunking_embedding():
    pipe = ChunkingEmbeddingPipeline()
    chunks = pipe.chunk_documents(load_all_documents("data/raw"))
    tok = pipe.model.tokenizer
    lengths = [len(tok(c.page_content)["input_ids"]) for c in chunks]
    over = sum(l > pipe.model.max_seq_length for l in lengths)
    print(f"limit {pipe.model.max_seq_length}, max {max(lengths)}, over limit: {over} of {len(chunks)}")

"""Tests for chunk noise filtering."""

is_noise = ChunkingEmbeddingPipeline.is_noise


def test_short_title_is_noise():
    assert is_noise("Intel® oneAPI Programming Guide")


def test_table_of_contents_is_noise():
    toc = (
        "Introduction to oneAPI Programming............................4\n"
        "Intel oneAPI Programming Overview.............................5\n"
        "oneAPI Programming Model......................................7\n"
    )
    assert is_noise(toc)


def test_real_content_is_kept():
    text = (
        "The oneAPI programming model supports single-source compilation. "
        "Developers can define device code right next to the call site in the host code, "
        "which reduces the number of files to manage."
    )
    assert not is_noise(text)


def test_code_with_ellipsis_is_kept():
    text = (
        "auto e = q.submit([&](sycl::handler &h) { ... });\n"
        "Use the event to wait for completion before reading results back on the host."
    )
    assert not is_noise(text)