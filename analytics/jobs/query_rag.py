#!/usr/bin/env python3
import argparse
import json

from analytics.src.rag.retrieval import query_index


def main():
    parser = argparse.ArgumentParser(description="Query local RAG index.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--index-dir", default="analytics/data/rag")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    results = query_index(args.question, args.index_dir, args.top_k)
    print(json.dumps({"question": args.question, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
