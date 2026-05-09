#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

from analytics.src.rag.document_builder import build_player_documents
from analytics.src.rag.indexer import build_and_save_index


def main():
    parser = argparse.ArgumentParser(description="Build local RAG index from analytics tables.")
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--output-dir", default="analytics/data/rag")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    docs = build_player_documents(args.days)
    if not docs:
        logging.warning("No documents built. Run analytics pipeline first (exit 1).")
        sys.exit(1)
    stats = build_and_save_index(docs, args.output_dir)
    logging.info("RAG index built at %s: %s", Path(args.output_dir).resolve(), stats)


if __name__ == "__main__":
    main()
