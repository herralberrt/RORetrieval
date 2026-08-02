"""
Download the MIRACL dataset (corpus, queries, qrels) for one language.

NOTE ON THE DOWNLOAD METHOD
---------------------------
The official repos (miracl/miracl-corpus, miracl/miracl) are *script-based*
HuggingFace datasets, and `datasets` 3.0+ refuses to run dataset scripts. So
instead of load_dataset() we pull the underlying data files directly with
huggingface_hub, which works on every `datasets` version:

    miracl-corpus-v1.0-<lang>/docs-*.jsonl.gz            -> corpus
    miracl-v1.0-<lang>/topics/topics.*-dev.tsv           -> queries
    miracl-v1.0-<lang>/qrels/qrels.*-dev.tsv             -> qrels

NOTE ON ROMANIAN
----------------
MIRACL covers 18 languages and Romanian is NOT one of them (see
MIRACL_LANGUAGES below). Asking for "ro" raises a clear error listing what is
available.
"""

import gzip
import io
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils import save_jsonl, ensure_dir


CORPUS_REPO = "miracl/miracl-corpus"
TOPICS_REPO = "miracl/miracl"

# The 18 languages MIRACL actually ships. Romanian is not among them.
MIRACL_LANGUAGES = [
    "ar", "bn", "de", "en", "es", "fa", "fi", "fr", "hi",
    "id", "ja", "ko", "ru", "sw", "te", "th", "yo", "zh",
]


class UnsupportedLanguageError(ValueError):
    """Raised when the requested language is not part of MIRACL."""


class MIRACLDownloader:

    def __init__(self, language: str = "ro", output_dir: str = "data/miracl",
                 split: str = "dev", max_docs: int = None):
        self.language = language
        self.output_dir = output_dir
        self.split = split
        self.max_docs = max_docs
        ensure_dir(output_dir)

        self.corpus_dir = os.path.join(output_dir, "corpus")
        self.queries_dir = os.path.join(output_dir, "queries")
        self.qrels_dir = os.path.join(output_dir, "qrels")

        for d in [self.corpus_dir, self.queries_dir, self.qrels_dir]:
            ensure_dir(d)

    def check_language(self) -> None:
        """Fail early and clearly if MIRACL has no data for this language."""
        if self.language not in MIRACL_LANGUAGES:
            raise UnsupportedLanguageError(
                f"MIRACL has no data for language {self.language!r}.\n"
                f"   Available languages: {', '.join(MIRACL_LANGUAGES)}\n"
                f"   (Romanian is not covered by MIRACL -- for Romanian data use "
                f"the local corpus in data/corpus/ instead.)"
            )

    def _repo_files(self, repo: str) -> List[str]:
        from huggingface_hub import list_repo_files
        return list_repo_files(repo, repo_type="dataset")

    def _download(self, repo: str, filename: str) -> str:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo, filename, repo_type="dataset")

    def download_corpus(self) -> List[Dict[str, Any]]:
        print(f"Downloading MIRACL corpus for language: {self.language}")

        try:
            prefix = f"miracl-corpus-v1.0-{self.language}/"
            shards = sorted(f for f in self._repo_files(CORPUS_REPO)
                            if f.startswith(prefix) and f.endswith(".jsonl.gz"))
            if not shards:
                print(f"   No corpus shards found under {prefix}")
                return []

            documents = []
            stop = False
            for shard in tqdm(shards, desc="Corpus shards"):
                local = self._download(CORPUS_REPO, shard)
                with gzip.open(local, "rt", encoding="utf-8") as fh:
                    for line in fh:
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        documents.append({
                            "doc_id": item.get("docid", f"doc_{len(documents):06d}"),
                            "title": item.get("title", ""),
                            "content": item.get("text", ""),
                            "source": "miracl",
                            "language": self.language,
                            "original_id": len(documents),
                        })
                        if self.max_docs and len(documents) >= self.max_docs:
                            stop = True
                            break
                if stop:
                    break

            output_path = os.path.join(self.corpus_dir, f"miracl_{self.language}_corpus.jsonl")
            save_jsonl(documents, output_path)
            print(f"Saved {len(documents)} documents to {output_path}")

            return documents

        except Exception as e:
            print(f"Error downloading corpus: {e}")
            return []

    def _topics_path(self) -> str:
        return (f"miracl-v1.0-{self.language}/topics/"
                f"topics.miracl-v1.0-{self.language}-{self.split}.tsv")

    def _qrels_path(self) -> str:
        return (f"miracl-v1.0-{self.language}/qrels/"
                f"qrels.miracl-v1.0-{self.language}-{self.split}.tsv")

    def download_queries(self) -> List[Dict[str, Any]]:
        print(f"Downloading MIRACL queries for language: {self.language}")

        try:
            local = self._download(TOPICS_REPO, self._topics_path())

            queries = []
            with open(local, "r", encoding="utf-8") as fh:
                for idx, line in enumerate(fh):
                    if not line.strip():
                        continue
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 2:
                        continue
                    queries.append({
                        "query_id": parts[0],
                        "query": parts[1],
                        "language": self.language,
                        "original_id": idx,
                    })

            output_path = os.path.join(self.queries_dir, f"miracl_{self.language}_queries.jsonl")
            save_jsonl(queries, output_path)
            print(f"Saved {len(queries)} queries to {output_path}")

            return queries

        except Exception as e:
            print(f"Error downloading queries: {e}")
            return []

    def download_qrels(self) -> Dict[str, List[Dict[str, Any]]]:
        print(f"Downloading MIRACL qrels for language: {self.language}")

        try:
            local = self._download(TOPICS_REPO, self._qrels_path())

            # TREC format: query_id  Q0  doc_id  relevance
            by_query: Dict[str, List[Dict[str, Any]]] = {}
            with open(local, "r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    query_id, _, doc_id, relevance = parts[0], parts[1], parts[2], parts[3]
                    if int(relevance) <= 0:
                        continue  # keep only positives
                    by_query.setdefault(query_id, []).append(
                        {"docid": doc_id, "relevance": int(relevance)}
                    )

            qrels_flat = [
                {
                    "query_id": query_id,
                    "positive_passages": passages,
                    "num_positives": len(passages),
                    "language": self.language,
                }
                for query_id, passages in by_query.items()
            ]

            output_path = os.path.join(self.qrels_dir, f"miracl_{self.language}_qrels.jsonl")
            save_jsonl(qrels_flat, output_path)
            print(f"Saved qrels for {len(by_query)} queries to {output_path}")

            return by_query

        except Exception as e:
            print(f"Error downloading qrels: {e}")
            return {}

    def download_all(self) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"MIRACL Dataset Downloader - Language: {self.language.upper()}")
        print(f"{'='*60}")

        self.check_language()

        corpus = self.download_corpus()
        queries = self.download_queries()
        qrels = self.download_qrels()

        summary = {
            "language": self.language,
            "corpus_size": len(corpus),
            "queries_count": len(queries),
            "qrels_count": len(qrels),
            "output_dir": self.output_dir,
            "corpus_dir": self.corpus_dir,
            "queries_dir": self.queries_dir,
            "qrels_dir": self.qrels_dir
        }

        print(f"\n{'='*60}")
        print(f"Download Complete!")
        print(f"{'='*60}")
        print(f"Summary:")
        print(f"   Language: {summary['language']}")
        print(f"   Corpus size: {summary['corpus_size']:,} documents")
        print(f"   Queries: {summary['queries_count']}")
        print(f"   Qrels: {summary['qrels_count']}")
        print(f"   Output: {summary['output_dir']}")

        return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download MIRACL dataset")
    parser.add_argument("--language", type=str, default="ro",
                        help=f"Language code. MIRACL covers: {', '.join(MIRACL_LANGUAGES)}")
    parser.add_argument("--output-dir", type=str, default="data/miracl", help="Output directory")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "train"],
                        help="Which topics/qrels split to fetch")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Stop after N corpus documents (useful for testing)")

    args = parser.parse_args()

    downloader = MIRACLDownloader(language=args.language, output_dir=args.output_dir,
                                  split=args.split, max_docs=args.max_docs)
    try:
        downloader.download_all()
    except UnsupportedLanguageError as e:
        print(f"\n{e}\n")
        sys.exit(1)
