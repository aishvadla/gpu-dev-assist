"""Command-line entry point for GPU DevAssist."""

import argparse


def main():
    parser = argparse.ArgumentParser(prog="gpu-dev-assist", description="RAG assistant for GPU documentation")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Load the PDFs, chunk, embed and rebuild the vector store")

    ask = sub.add_parser("ask", help="Ask a question and get a cited answer")
    ask.add_argument("question")
    ask.add_argument("--retriever", choices=["dense", "bm25", "hybrid"], default="hybrid")

    ev = sub.add_parser("eval", help="Run the evaluation")
    ev.add_argument("stage", choices=["retrieval", "generation"])

    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    # Imports inside each branch so `--help` stays fast (no model loading)
    if args.command == "ingest":
        from gpu_dev_assist.ingestion.data_ingestion import DataIngestion
        DataIngestion().initiate_data_ingestion()

    elif args.command == "ask":
        from gpu_dev_assist.generate import AnswerGenerator
        from gpu_dev_assist.retrieval.hybrid import build_retrievers
        result = AnswerGenerator(build_retrievers()[args.retriever]).answer(args.question)
        print(result["answer"])
        for s in result["sources"]:
            print(f"  [{s['id']}] {s['source']}, p. {s['page']}")

    elif args.command == "eval":
        import runpy
        runpy.run_module(f"gpu_dev_assist.evaluation.{args.stage}", run_name="__main__")


if __name__ == "__main__":
    main()