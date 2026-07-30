"""
MS MARCO-style data generation.

Reproduces the data-construction methodology of the microsoft/ms_marco
dataset (https://huggingface.co/datasets/microsoft/ms_marco) on the local
RORetrieval corpus.

MS MARCO is built by:
  1. Taking a real user query.
  2. Retrieving a set of candidate passages for that query from a large corpus
     (in the original, the top Bing web results).
  3. Labelling which of those candidate passages actually answer the query
     (`is_selected` = 1) vs. which are only topically related (`is_selected` = 0).
  4. Attaching a short human answer, plus a well-formed answer variant.

Here we mirror each step:
  1. A query is generated per source document (title-derived, LLM-ready hook).
  2. Candidate passages are retrieved from the corpus with BM25 (falls back to
     lexical overlap when rank_bm25 is unavailable) -> mimics the Bing top-k.
  3. The source document is the selected passage (is_selected=1); the other
     retrieved candidates are hard negatives (is_selected=0).
  4. The answer is extracted from the selected passage; a well-formed variant
     is produced from the query + answer.

Output rows follow the MS MARCO v2.1 schema exactly:
    {
      "query_id": int,
      "query": str,
      "query_type": str,
      "passages": {
          "passage_text": [str, ...],
          "is_selected": [int, ...],
          "url": [str, ...]
      },
      "answers": [str, ...],
      "wellFormedAnswers": [str, ...]
    }
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

# Reuse the project's IO helpers.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import load_jsonl, save_jsonl  # noqa: E402


# MS MARCO groups every query into one of five coarse question types.
QUERY_TYPES = ["description", "numeric", "entity", "location", "person"]


def _strip_diacritics(text: str) -> str:
    """Normalise Romanian diacritics so matching works on un-accented corpora."""
    table = str.maketrans("ăâîșşțţ", "aaisstt")
    return text.translate(table)


def infer_query_type(query: str) -> str:
    """Coarsely classify a query the way MS MARCO's `query_type` field does."""
    q = _strip_diacritics(query.lower())
    if re.search(r"\b(cat|cati|cate|how many|how much|number|pret|cost)\b", q):
        return "numeric"
    if re.search(r"\b(unde|where|locatie|oras|tara)\b", q):
        return "location"
    if re.search(r"\b(cine|who|autor)\b", q):
        return "person"
    if re.search(r"\b(ce este|what is|care este|which)\b", q):
        return "entity"
    return "description"


# Romanian question templates, keyed by corpus source, so generated queries
# read naturally in Romanian (MS MARCO queries are short, real questions).
RO_QUERY_TEMPLATES = {
    "recipes-ro": [
        "cum se prepară {t}?",
        "care sunt ingredientele pentru {t}?",
        "cum se face {t}?",
    ],
    "news": [
        "ce s-a întâmplat cu {t}?",
        "care este subiectul articolului despre {t}?",
        "ce știri există despre {t}?",
    ],
    "commoncrawl": [
        "ce informații conține {t}?",
        "despre ce este {t}?",
    ],
    "default": [
        "ce este {t}?",
        "care sunt caracteristicile {t}?",
        "ce reprezintă {t}?",
    ],
}


def make_query(doc: Dict[str, Any], index: int = 0) -> str:
    """
    Derive a natural-language Romanian query for a document.

    Picks a source-aware Romanian template (recipes vs. news vs. generic) so
    the resulting dataset reads like real Romanian questions. `index` rotates
    through the templates for variety across the dataset. This is also the hook
    where an LLM query generator plugs in (see `QueryGenerator`).
    """
    title = (doc.get("title") or "").strip()
    if not title:
        # Fall back to the first few words of the content.
        title = " ".join((doc.get("content") or "").split()[:8])
    title = title.rstrip(" ?.!").lower()
    if not title:
        return "ce este acest document?"

    source = doc.get("source", "default")
    templates = RO_QUERY_TEMPLATES.get(source, RO_QUERY_TEMPLATES["default"])
    return templates[index % len(templates)].format(t=title)


def extract_answer(text: str, max_sentences: int = 1) -> str:
    """Take the first sentence(s) of the selected passage as the answer."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    answer = " ".join(sentences[:max_sentences]).strip()
    return answer


def make_well_formed(query: str, answer: str) -> str:
    """Rewrite a terse answer into a full-sentence answer (MS MARCO wellFormed)."""
    if not answer:
        return ""
    answer = answer[0].upper() + answer[1:]
    if not answer.endswith((".", "!", "?")):
        answer += "."
    return answer


class QueryGenerator:
    """
    Generates Romanian queries for documents.

    When `use_llm` is set and an OpenAI key is available, it uses the Romanian
    prompt from src/task1_prompt.py to generate a natural question. Otherwise it
    falls back to the source-aware Romanian templates in `make_query`.
    """

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self._client = None
        self._model = "gpt-4"
        self._prompt_v1 = None
        if use_llm:
            self._init_llm()

    def _init_llm(self) -> None:
        try:
            from openai import OpenAI  # openai>=1.x
            from utils import get_config
            from task1_prompt import PROMPT_V1_TEMPLATE

            cfg = get_config()
            api_key = cfg.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if not api_key or api_key.startswith("sk-your-key"):
                print("   [QueryGenerator] No OPENAI_API_KEY -> using RO templates")
                self.use_llm = False
                return
            self._client = OpenAI(api_key=api_key)
            self._model = cfg.get("OPENAI_MODEL", "gpt-4")
            self._prompt_v1 = PROMPT_V1_TEMPLATE
        except Exception as e:  # missing deps / import error -> fall back
            print(f"   [QueryGenerator] LLM unavailable ({e}) -> using RO templates")
            self.use_llm = False

    def generate(self, doc: Dict[str, Any], index: int = 0) -> str:
        if self.use_llm and self._client is not None:
            try:
                prompt = self._prompt_v1.format(
                    title=doc.get("title", ""), content=doc.get("content", "")
                )
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                )
                text = resp.choices[0].message.content.strip()
                # Take the first non-empty generated question line.
                first = next((l.strip() for l in text.splitlines() if l.strip()), "")
                if first:
                    return first
            except Exception as e:
                print(f"   [QueryGenerator] LLM call failed ({e}) -> template")
        return make_query(doc, index)


class MarcoDataGenerator:
    """Builds MS MARCO-schema records from a local Romanian document corpus."""

    def __init__(self, num_passages: int = 10, language: str = "ro",
                 use_llm: bool = False):
        # Number of candidate passages retrieved per query (MS MARCO ~10).
        self.num_passages = num_passages
        self.language = language
        self.query_generator = QueryGenerator(use_llm=use_llm)
        self._bm25 = None
        self._doc_tokens = None

    def _build_index(self, docs: List[Dict[str, Any]]) -> None:
        """Build a BM25 index over the corpus (fallback handled at query time)."""
        self._doc_tokens = [
            (d.get("content") or d.get("title") or "").lower().split() for d in docs
        ]
        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self._doc_tokens)
        except Exception:
            self._bm25 = None  # fall back to lexical overlap

    def _retrieve(self, query: str, positive_idx: int, num_docs: int) -> List[int]:
        """Return candidate passage indices for a query (positive first)."""
        tokens = query.lower().split()

        if self._bm25 is not None:
            scores = self._bm25.get_scores(tokens)
        else:
            # Lexical-overlap fallback so the script runs with zero extra deps.
            qset = set(tokens)
            scores = [
                len(qset & set(dt)) / (len(qset | set(dt)) or 1)
                for dt in self._doc_tokens
            ]

        ranked = sorted(range(num_docs), key=lambda i: scores[i], reverse=True)
        # Guarantee the positive (source) passage is included.
        candidates = [i for i in ranked if i != positive_idx][: self.num_passages - 1]
        return [positive_idx] + candidates

    def generate(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate one MS MARCO record per document."""
        self._build_index(docs)
        num_docs = len(docs)
        records: List[Dict[str, Any]] = []

        for idx, doc in enumerate(tqdm(docs, desc="Generating MS MARCO records")):
            query = self.query_generator.generate(doc, idx)
            candidate_idxs = self._retrieve(query, idx, num_docs)

            passage_text: List[str] = []
            is_selected: List[int] = []
            urls: List[str] = []
            for cand_idx in candidate_idxs:
                cand = docs[cand_idx]
                passage_text.append(cand.get("content") or cand.get("title") or "")
                is_selected.append(1 if cand_idx == idx else 0)
                urls.append(cand.get("url") or cand.get("warc_url") or "")

            answer = extract_answer(doc.get("content") or doc.get("title") or "")

            records.append(
                {
                    "query_id": idx,
                    "query": query,
                    "query_type": infer_query_type(query),
                    "language": self.language,
                    "passages": {
                        "passage_text": passage_text,
                        "is_selected": is_selected,
                        "url": urls,
                    },
                    "answers": [answer] if answer else [],
                    "wellFormedAnswers": (
                        [make_well_formed(query, answer)] if answer else []
                    ),
                }
            )

        return records


def main():
    parser = argparse.ArgumentParser(
        description="Generate MS MARCO-style data from the RORetrieval corpus"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/corpus/all_documents_numbered.jsonl",
        help="Input corpus JSONL (produced by src/download_datasets.py --combine)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/marco/marco_data.jsonl",
        help="Output path for MS MARCO-schema records",
    )
    parser.add_argument(
        "--num_passages",
        type=int,
        default=10,
        help="Candidate passages per query (MS MARCO uses ~10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N documents (for quick tests)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ro",
        help="Language tag written to each record (default: ro / Romanian)",
    )
    parser.add_argument(
        "--use_llm",
        action="store_true",
        help="Use the Romanian LLM prompt (task1_prompt.py) for query generation",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  MS MARCO-style Data Generator")
    print("=" * 60)

    if not os.path.exists(args.corpus):
        print(f"\n Corpus not found: {args.corpus}")
        print("   Run: python src/download_datasets.py --combine")
        sys.exit(1)

    docs = load_jsonl(args.corpus)
    if args.limit:
        docs = docs[: args.limit]
    print(f"\n Loaded {len(docs)} documents from {args.corpus}")

    generator = MarcoDataGenerator(
        num_passages=args.num_passages,
        language=args.language,
        use_llm=args.use_llm,
    )
    records = generator.generate(docs)

    save_jsonl(records, args.output)
    print(f"\n Saved {len(records)} MS MARCO records to {args.output}")

    if records:
        import json

        print("\n Example record:")
        print(json.dumps(records[0], indent=2, ensure_ascii=False)[:1200])
    print()


if __name__ == "__main__":
    main()
