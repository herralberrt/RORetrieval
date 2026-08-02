"""
Build a FAISS index over the embeddings of the corpus texts.

    python src/build_index.py                        # build over data/corpus/all_documents_combined.jsonl
    python src/build_index.py --search "cum fac aluat?"   # query an existing index

THE EMBEDDER IS A SHARED CHOICE
-------------------------------
Everything downstream (hard-negative mining, retrieval eval) must use the SAME
embedder, or the vectors are not comparable. The model is therefore read from
one place -- EMBEDDINGS_MODEL in .env -- and stored inside the index metadata,
so a later search can refuse to run against a mismatched model.

    EMBEDDINGS_MODEL=sentence-transformers/all-mpnet-base-v2   # English-only
    EMBEDDINGS_MODEL=intfloat/multilingual-e5-large            # multilingual (incl. Romanian)

NOTE: all-mpnet-base-v2 has no Romanian in its training data. It will still
produce vectors for Romanian text, but the neighbourhoods are much weaker than
a multilingual model's. Use --model to compare the two on the same corpus.

SIMILARITY
----------
Vectors are L2-normalised and stored in an IndexFlatIP, so the inner product
the index returns IS cosine similarity -- the same metric task2_triplets.py
uses for its hard-negative threshold.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_jsonl, ensure_dir, get_config


DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_CORPUS = "data/corpus/all_documents_combined.jsonl"
DEFAULT_INDEX_DIR = "models/faiss"


def resolve_model(explicit: Optional[str] = None) -> str:
    """The one place the embedder name is decided: flag > .env > default."""
    if explicit:
        return explicit
    config = get_config()
    return (
        os.environ.get("EMBEDDINGS_MODEL")
        or config.get("EMBEDDINGS_MODEL")
        or DEFAULT_MODEL
    )


def doc_text(doc: Dict[str, Any]) -> str:
    """The text that gets embedded: title + content (+ ingredients for recipes)."""
    parts = [
        (doc.get("title") or "").strip(),
        (doc.get("ingredients") or "").strip(),
        (doc.get("content") or "").strip(),
    ]
    return "\n".join(p for p in parts if p)


def _needs_e5_prefix(model_name: str) -> bool:
    """e5 models are trained with 'query: ' / 'passage: ' prefixes and need them."""
    return "e5" in model_name.lower()


class CorpusIndex:
    """A FAISS index over corpus embeddings, plus the docid mapping."""

    def __init__(self, model_name: Optional[str] = None, batch_size: int = 32):
        self.model_name = resolve_model(model_name)
        self.batch_size = batch_size
        self._model = None
        self.index = None
        self.doc_ids: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

    # -- embedding ---------------------------------------------------

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"  Loading embedder: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: List[str], kind: str = "passage",
               show_progress: bool = True) -> np.ndarray:
        """
        Encode texts into L2-normalised float32 vectors.

        `kind` is "passage" for documents and "query" for search strings; it
        only affects e5-family models, which require those prefixes.
        """
        if _needs_e5_prefix(self.model_name):
            texts = [f"{kind}: {t}" for t in texts]

        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # makes inner product == cosine similarity
        )
        return np.asarray(vectors, dtype="float32")

    # -- build / save / load -----------------------------------------

    def build(self, docs: List[Dict[str, Any]]) -> None:
        import faiss

        texts = [doc_text(d) for d in docs]
        self.doc_ids = [d.get("doc_id", f"doc_{i:06d}") for i, d in enumerate(docs)]
        self.metadata = [
            {"doc_id": self.doc_ids[i],
             "title": (d.get("title") or "").strip(),
             "source": d.get("source", "")}
            for i, d in enumerate(docs)
        ]

        print(f"  Embedding {len(texts)} documents...")
        vectors = self.encode(texts, kind="passage")

        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # exact search; cosine via normalised vectors
        self.index.add(vectors)
        print(f"  Index built: {self.index.ntotal} vectors, dim={dim}")

    def save(self, index_dir: str = DEFAULT_INDEX_DIR) -> None:
        import faiss

        ensure_dir(index_dir)
        faiss.write_index(self.index, os.path.join(index_dir, "corpus.faiss"))

        with open(os.path.join(index_dir, "docids.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False)

        # The embedder is part of the index's identity -- record it so a later
        # search can detect a mismatch instead of returning nonsense.
        with open(os.path.join(index_dir, "index_meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "model": self.model_name,
                "num_docs": len(self.doc_ids),
                "dim": int(self.index.d),
                "metric": "inner_product_on_normalized (cosine)",
            }, f, indent=2)

        print(f"  Saved index to {index_dir}/")

    @classmethod
    def load(cls, index_dir: str = DEFAULT_INDEX_DIR,
             model_name: Optional[str] = None) -> "CorpusIndex":
        import faiss

        meta_path = os.path.join(index_dir, "index_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No index at {index_dir}. Build one first:\n"
                f"    python src/build_index.py"
            )

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        wanted = resolve_model(model_name)
        if wanted != meta["model"]:
            raise ValueError(
                f"Embedder mismatch: index was built with {meta['model']!r} but the "
                f"current setting is {wanted!r}.\n"
                f"   Vectors from different models are not comparable -- either set "
                f"EMBEDDINGS_MODEL={meta['model']} or rebuild the index."
            )

        obj = cls(model_name=meta["model"])
        obj.index = faiss.read_index(os.path.join(index_dir, "corpus.faiss"))
        with open(os.path.join(index_dir, "docids.json"), encoding="utf-8") as f:
            obj.metadata = json.load(f)
        obj.doc_ids = [m["doc_id"] for m in obj.metadata]
        return obj

    # -- search ------------------------------------------------------

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Top-k most similar documents to a free-text query."""
        vector = self.encode([query], kind="query", show_progress=False)
        scores, idxs = self.index.search(vector, min(k, self.index.ntotal))
        return [
            {**self.metadata[i], "score": float(s), "rank": r + 1}
            for r, (s, i) in enumerate(zip(scores[0], idxs[0])) if i >= 0
        ]

    def neighbors(self, doc_index: int, k: int = 10,
                  exclude_self: bool = True) -> List[Tuple[str, float]]:
        """
        Nearest neighbours of a document already in the index.

        This is the hard-negative-candidate lookup that task2_triplets.py needs
        -- returns (doc_id, cosine_similarity) pairs.
        """
        vector = self.index.reconstruct(doc_index).reshape(1, -1)
        scores, idxs = self.index.search(vector, min(k + 1, self.index.ntotal))
        out = []
        for s, i in zip(scores[0], idxs[0]):
            if i < 0 or (exclude_self and i == doc_index):
                continue
            out.append((self.doc_ids[i], float(s)))
        return out[:k]


def main():
    parser = argparse.ArgumentParser(
        description="Build / query a FAISS index over the corpus embeddings"
    )
    parser.add_argument("--corpus", type=str, default=DEFAULT_CORPUS,
                        help="Input corpus JSONL")
    parser.add_argument("--index-dir", type=str, default=DEFAULT_INDEX_DIR,
                        help="Where to write (or read) the index")
    parser.add_argument("--model", type=str, default=None,
                        help="Embedder to use (default: EMBEDDINGS_MODEL from .env)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only index the first N documents")
    parser.add_argument("--search", type=str, default=None,
                        help="Query an existing index instead of building one")
    parser.add_argument("-k", type=int, default=10, help="Results to return for --search")

    args = parser.parse_args()

    if args.search:
        index = CorpusIndex.load(args.index_dir, args.model)
        print(f"\n  Query: {args.search!r}")
        print(f"  Embedder: {index.model_name}\n")
        for hit in index.search(args.search, k=args.k):
            title = hit["title"].replace("\n", " ").strip()[:65]
            print(f"   {hit['rank']:>2}. [{hit['score']:.3f}] {hit['doc_id']:<16} {title}")
        print()
        return

    print("\n" + "=" * 60)
    print("  FAISS Index Builder")
    print("=" * 60)

    if not os.path.exists(args.corpus):
        print(f"\n  Corpus not found: {args.corpus}")
        print("  Run: python src/add_manual_data.py\n")
        sys.exit(1)

    docs = load_jsonl(args.corpus)
    if args.limit:
        docs = docs[: args.limit]
    print(f"\n  Loaded {len(docs)} documents from {args.corpus}")

    index = CorpusIndex(model_name=args.model, batch_size=args.batch_size)
    index.build(docs)
    index.save(args.index_dir)

    print("\n  Try it:")
    print('    python src/build_index.py --search "cum se prepara un aluat?"')
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
