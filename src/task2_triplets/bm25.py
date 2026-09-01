"""
Okapi BM25 over the Romanian corpus, in numpy only.

`build_triplets_bm25.py` needs one lexical retriever for ~70k queries against
~120k documents, and needs it to run on a CPU node inside the existing image -
so no new dependency (no rank_bm25, no elasticsearch, no scipy).

The index is the classic inverted file: for every term, the documents that
contain it and their *precomputed* BM25 weight

    w(t, d) = idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))

Nothing in that expression depends on the query, so scoring a query is just
adding a few posting lists together - fast enough (~1 ms per query over 100k
documents) without a sparse-matrix library.

Two deliberate simplifications:

* Terms that appear in more than `max_df_ratio` of the corpus are dropped at
  query time. Their idf is near zero, so they contribute almost nothing to the
  ranking, but their posting lists are the long ones - skipping them is where
  most of the speed comes from.
* Diacritics are folded (`ș` -> `s`). The corpus mixes `ş/ș` and `ţ/ț`
  (cedilla vs comma-below) and some outlets strip diacritics entirely, so
  matching on the folded form is the only way `poliţişti`, `polițiști` and
  `politisti` land on the same term.
"""

import re
import sys
import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Function words carry no retrieval signal and inflate every posting list.
# idf would discount them anyway; dropping them keeps the index smaller.
STOPWORDS = {
    "a", "acea", "aceasta", "această", "aceea", "acei", "aceia", "acel", "acela",
    "acele", "acelea", "acest", "acesta", "aceste", "acestea", "acestei",
    "acestia", "acestui", "aceşti", "aceştia", "acolo", "acum", "ai", "aia",
    "aibă", "aici", "al", "ala", "ale", "alea", "altceva", "altcineva", "am",
    "ar", "are", "asemenea", "asta", "astea", "astăzi", "asupra", "au", "avea",
    "avem", "aveţi", "azi", "aş", "aşadar", "această", "băm", "ca", "cam",
    "cand", "capat", "care", "careia", "carora", "caruia", "cat", "catre",
    "caut", "ce", "cea", "ceea", "cei", "ceilalti", "cel", "cele", "celor",
    "ceva", "chiar", "cinci", "cind", "cine", "cineva", "cit", "cita", "cite",
    "citeva", "citi", "citiva", "conform", "contra", "cu", "cui", "cum",
    "cumva", "cât", "câte", "câţi", "când", "către", "da", "daca", "dar",
    "datorita", "dat", "de", "deasupra", "deci", "decit", "deja", "deoarece",
    "departe", "desi", "despre", "din", "dinaintea", "dintr", "dintre", "doar",
    "doi", "doilea", "două", "drept", "dupa", "după", "ea", "ei", "el", "ele",
    "eram", "este", "eu", "eşti", "face", "fara", "fata", "fel", "fi", "fie",
    "fiecare", "fii", "fim", "fiu", "fiţi", "foarte", "fost", "frumos", "fără",
    "geaba", "graţie", "halbă", "iar", "ieri", "ii", "il", "imi", "in",
    "inainte", "inapoi", "inca", "incit", "insa", "intr", "intre", "isi",
    "iti", "la", "le", "li", "lor", "lui", "lângă", "mai", "mea", "mei",
    "mele", "mereu", "meu", "mi", "mie", "mine", "mult", "multa", "multe",
    "multi", "mulţi", "mult", "mă", "ne", "nevoie", "ni", "nici", "niciodata",
    "nicăieri", "nimeni", "nimeri", "nimic", "niste", "noastra", "noastre",
    "noi", "noroc", "nostri", "nostru", "nou", "noua", "nu", "numai", "o",
    "opt", "or", "ori", "oricare", "orice", "oricine", "oricum", "oriunde",
    "pai", "pana", "patra", "patru", "pe", "pentru", "peste", "pic", "pina",
    "poate", "pot", "prea", "prima", "primul", "prin", "printr", "putini",
    "puţin", "puţina", "puţină", "până", "recent", "referitor", "rog", "sa",
    "sale", "sau", "se", "si", "sint", "sintem", "sinteti", "spre", "sub",
    "sunt", "suntem", "sunteţi", "sus", "sută", "să", "săi", "său", "ta",
    "tale", "te", "ti", "timp", "tine", "toata", "toate", "toti", "totul",
    "totusi", "totuşi", "tot", "trei", "treia", "treilea", "tu", "tăi", "tău",
    "un", "una", "unde", "undeva", "unei", "uneia", "unele", "uneori", "unii",
    "unor", "unora", "unu", "unui", "unuia", "unul", "vi", "voastre", "voi",
    "vom", "vor", "vostru", "vouă", "vreme", "vreo", "vreun", "va", "vă",
    "zece", "zero", "zi", "şi", "și", "ăla", "ăsta", "şase", "şapte", "şi",
    "însă", "îl", "îi", "în", "îmi", "între", "îţi", "ţi",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def fold(text: str) -> str:
    """Lowercase and strip diacritics: `Poliţişti` -> `politisti`."""
    lowered = text.lower().replace("ș", "s").replace("ş", "s")
    lowered = lowered.replace("ț", "t").replace("ţ", "t")
    decomposed = unicodedata.normalize("NFD", lowered)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize(text: str, min_len: int = 2) -> List[str]:
    """Folded alphanumeric tokens, stopwords and 1-character noise removed.

    Tokens are interned: indexing the whole corpus holds ~50M token references
    at once, and without interning those are ~50M separate string objects
    (several GB) instead of one per vocabulary entry.
    """
    return [
        sys.intern(t) for t in _TOKEN_RE.findall(fold(text))
        if len(t) >= min_len and t not in STOPWORDS
    ]


class BM25Index:
    """Inverted index with precomputed BM25 term weights."""

    def __init__(self, documents: Sequence[Sequence[str]], k1: float = 1.5,
                 b: float = 0.75):
        self.n_docs = len(documents)
        self.k1, self.b = k1, b

        lengths = np.array([len(d) for d in documents], dtype=np.float32)
        avgdl = float(lengths.mean()) if self.n_docs else 0.0

        # Pass 1: term -> [(doc index, term frequency)], built as flat lists so
        # the whole index is three numpy arrays at the end rather than 10^6
        # small Python objects.
        postings: Dict[str, List[Tuple[int, int]]] = {}
        for i, tokens in enumerate(documents):
            counts: Dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for term, tf in counts.items():
                postings.setdefault(term, []).append((i, tf))

        self.vocab: Dict[str, int] = {}
        doc_ids: List[np.ndarray] = []
        weights: List[np.ndarray] = []
        self.df = np.zeros(len(postings), dtype=np.int32)
        self.idf = np.zeros(len(postings), dtype=np.float32)

        for term, plist in postings.items():
            index = len(self.vocab)
            self.vocab[term] = index
            docs = np.fromiter((d for d, _ in plist), dtype=np.int32, count=len(plist))
            tfs = np.fromiter((f for _, f in plist), dtype=np.float32, count=len(plist))
            df = len(plist)
            # Robertson/Sparck-Jones idf with the +1 that keeps it positive for
            # terms present in more than half of the corpus.
            idf = np.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))
            norm = tfs + k1 * (1.0 - b + b * lengths[docs] / (avgdl or 1.0))
            doc_ids.append(docs)
            weights.append((idf * tfs * (k1 + 1.0) / norm).astype(np.float32))
            self.df[index] = df
            self.idf[index] = idf

        self.doc_ids = doc_ids
        self.weights = weights
        self._scores = np.zeros(self.n_docs, dtype=np.float32)

    @property
    def n_terms(self) -> int:
        return len(self.vocab)

    @property
    def n_postings(self) -> int:
        return int(sum(len(d) for d in self.doc_ids))

    def query_terms(self, tokens: Iterable[str], max_df_ratio: float = 0.2) -> List[int]:
        """Term ids worth scoring: known, and not near-ubiquitous."""
        cutoff = max_df_ratio * self.n_docs
        seen, out = set(), []
        for token in tokens:
            index = self.vocab.get(token)
            if index is None or index in seen or self.df[index] > cutoff:
                continue
            seen.add(index)
            out.append(index)
        return out

    def score(self, term_ids: Sequence[int]) -> np.ndarray:
        """Dense BM25 score vector. The buffer is reused - copy what you keep."""
        scores = self._scores
        scores.fill(0.0)
        for index in term_ids:
            scores[self.doc_ids[index]] += self.weights[index]
        return scores

    def top_k(self, term_ids: Sequence[int], k: int,
              scores: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """`(doc indices, scores)` for the k best documents, best first."""
        if scores is None:
            scores = self.score(term_ids)
        if not term_ids:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        k = min(k, self.n_docs)
        candidates = np.argpartition(scores, -k)[-k:]
        candidates = candidates[scores[candidates] > 0.0]
        order = np.argsort(-scores[candidates])
        picked = candidates[order]
        return picked, scores[picked]
