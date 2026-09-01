"""
TASK 2 (BM25 variant): (query, positive, hard negatives) triplets, mined lexically.

Same contract as `build_triplets.py` - the positive is the document the query
was generated from, retrieval is used only to find negatives - but the negatives
come from BM25 instead of a MiniLM bi-encoder, and the query set is cleaned
first. Both changes answer problems measured in `data/queries/README.md` and
visible in the dense triplets:

1.  **Duplicated corpus.** 38% of the generated queries are exact repeats,
    because news outlets re-publish the same article dozens of times. Documents
    are collapsed by a near-duplicate key (title + the first 600 characters, not
    the exact text hash - copies differ in the tail), only one copy per group is
    indexed, and every distinct query text is kept once.
2.  **Unanswerable queries.** 1.3% of the queries refer to "articolul" /
    "textul" / "documentul"; no retriever can answer those, so they are dropped.
3.  **Boilerplate documents.** Cookie banners, "Știri Video actuale" and other
    navigation stubs are neither useful positives nor useful negatives - they
    are excluded from the corpus entirely.
4.  **False negatives.** The dense run happily mined documents that answer the
    query better than the positive does (a "calendar ortodox" article as a
    negative for a saint's-day question whose positive was a horoscope page).
    Four guards now prevent that: a negative may not outscore the positive
    (`--max-neg-score-ratio`), may not be a near-copy of the positive or of a
    negative already picked (`--dup-token-overlap`, `--dup-title-overlap`), must
    cover a real share of the query (`--min-shared-query-terms`,
    `--min-query-idf-coverage`), and the whole query is dropped when its own
    positive is not in the BM25 candidate list (`--require-positive-in-topk`).
5.  **Follow-up articles about the same story.** The guards in (4) compare each
    candidate to the *query*; they cannot see that two documents are the same
    news story told twice. A query about Ion Radoi's job at Metrorex kept an
    article headlined "Ion Radoi poate redeveni lider de sindicat la Metrorex"
    as a hard negative - it answers the question. So every candidate is now also
    compared to the positive *document*, with an idf-weighted overlap
    (`--dup-idf-containment`) that counts sharing "radoi"/"metrorex" for much
    more than sharing "declarat"/"instanta", and on 5-character title stems so
    Romanian inflection ("lider"/"liderul", "sindicat"/"sindicatului") stops
    hiding the match. A candidate that trips any of those is *not* dropped: it
    is written out as a mined positive with the provenance of the decision, so
    a future reranker can be trained on the near-miss instead of losing it.

Why BM25 at all: hard negatives from a lexical retriever are the ones a dense
model is worst at - documents that share the query's rare terms (names, places,
figures) while answering a different question. They are also complementary to
the dense ones: over the 30 728 queries both runs cover, 87% share not a single
negative document. And they are cheap - no GPU, no embedding cache, under three
minutes on a CPU node for the full corpus.

    python3 -m src.task2_triplets.build_triplets_bm25 \
        --queries data/queries/queries_gemma3_27b.jsonl \
        --output data/triplets/triplets_27b_bm25.jsonl

The run writes `<output>.stats.json` with every filter count, so the cleaning is
auditable instead of implicit.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [
    str(p) for p in _SRC_DIR.iterdir()
    if p.is_dir() and not p.name.startswith((".", "_"))
]

from gemma_query_generation import DEFAULT_CATEGORIES_DIR, document_text, iter_documents  # noqa: E402
from bm25 import BM25Index, fold, tokenize  # noqa: E402

# Rule 3 of the generation prompt: the query must stand on its own. These are
# the queries that broke it - they ask about "the article", which a retriever
# cannot resolve because the article is what it is supposed to find.
META_QUERY_RE = re.compile(
    r"\b(acest articol|articolul|articolului|textul|textului|documentul|"
    r"documentului|autorul|autorului|in text|din text|mai sus|pasajul|"
    r"fragmentul|conform articolului|potrivit articolului)\b"
)

# Consent banners and navigation stubs: they are duplicated across the corpus,
# they answer nothing, and BM25 loves them because they are short.
BOILERPLATE_RE = re.compile(
    r"(politica de (confidentialitate|cookie)|utilizarea cookie|"
    r"fisiere?e? de tip cookie|preferintele? (referitoare la |privind )?cookie|"
    r"da, accept|accepta toate|setari cookie)"
)
NAV_TITLE_RE = re.compile(r"^(news|stiri|video|foto|actualitate|.{0,3})\s*[-–|]?\s*\w{0,12}$")


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def text_fingerprint(text: str, title: str, head_chars: int = 600) -> str:
    """Group key for re-published copies of the same article.

    Hashing the whole text is not enough: outlets republish an article with a
    different tail (a trailing "citește și" block, one more paragraph), which
    changes the exact hash while leaving two documents that are the same story.
    A measured comparison over the corpus - title plus the first 600 characters
    vs. the full text - collapses 753 extra groups, and every group it merges
    that carries more than one title turned out to be a syndicated copy
    (`ade_*` and `dig_*` running the identical headline). Documents without a
    title fall back to the full-text hash.
    """
    body = normalise(text)
    key = f"{normalise(title)}||{body[:head_chars]}" if title.strip() else body
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def is_boilerplate(text: str, title: str, min_chars: int) -> bool:
    body = fold(text)
    if len(body.strip()) < min_chars:
        return True
    if BOILERPLATE_RE.search(body[:1500]):
        return True
    return bool(NAV_TITLE_RE.match(fold(title).strip())) and len(body) < 4 * min_chars


def containment(a: Sequence[int], b: Sequence[int]) -> float:
    """|A ∩ B| / min(|A|, |B|) over two sorted term-id arrays."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    shared = np.intersect1d(a, b, assume_unique=True).size
    return shared / min(len(a), len(b))


def idf_containment(a: Sequence[int], b: Sequence[int], idf: np.ndarray,
                    b_mass: float = None) -> float:
    """`containment`, but each term counts for its idf instead of for 1.

    Plain containment is the wrong measure for "are these two documents the
    same story": two arbitrary Romanian news articles already share a few
    hundred ordinary words, and that generic mass swamps the handful of terms
    that actually identify the story. Weighting by idf makes the rare terms -
    the names, the institutions, the figures - carry the decision, which is what
    "the same story" means lexically. Normalising by the *smaller* document's
    idf mass keeps a short article that is entirely contained in a long one
    scoring high, rather than being diluted by the long one's extra material.
    """
    if len(a) == 0 or len(b) == 0:
        return 0.0
    shared = np.intersect1d(a, b, assume_unique=True)
    if shared.size == 0:
        return 0.0
    # `b` is the positive, the same document for all ~100 candidates of a
    # query, so its idf mass is passed in precomputed rather than summed again
    # per candidate.
    mass = min(float(idf[a].sum()),
               float(idf[b].sum()) if b_mass is None else b_mass)
    return float(idf[shared].sum()) / mass if mass > 0 else 0.0


def title_overlap(a: frozenset, b: frozenset, min_terms: int = 0) -> float:
    """|A ∩ B| / min(|A|, |B|) over two title token sets.

    `min_terms` exists because the min-normalisation degenerates on short
    titles: a title of one content word is a subset of every title containing
    that word, so it scores 1.0 against all of them. The `stories` category is
    full of them ("Poveste", "Fata mosului"), and they were the single largest
    source of wrong reclassifications on the first calibration run.
    """
    if not a or not b or min(len(a), len(b)) < min_terms:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def percentiles(values: List[float], points=(5, 25, 50, 75, 95)) -> Dict[str, Any]:
    """Distribution summary for the stats file, empty-safe."""
    if not values:
        return {str(p): None for p in points}
    ordered = sorted(values)
    return {str(p): round(ordered[min(len(ordered) - 1,
                                      int(p / 100 * len(ordered)))], 4)
            for p in points}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build (query, positive, BM25 hard negatives) triplets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", default="data/triplets/triplets_bm25.jsonl")
    parser.add_argument("--categories-dir", default=DEFAULT_CATEGORIES_DIR)
    parser.add_argument("--include-aggregates", action="store_true")
    parser.add_argument("--negatives", type=int, default=4,
                        help="hard negatives per query (the target)")
    parser.add_argument("--min-negatives", type=int, default=2,
                        help="keep the query if at least this many negatives "
                             "survive the guards. Dropping every query that "
                             "cannot reach --negatives costs a third of the "
                             "set, and those are exactly the queries with few "
                             "lexical neighbours - not the bad ones")
    parser.add_argument("--candidates", type=int, default=100,
                        help="documents retrieved per query before filtering")
    parser.add_argument("--max-neg-score-ratio", type=float, default=0.6,
                        help="a negative scoring above this fraction of the "
                             "positive's own BM25 score is probably a better "
                             "answer than the positive. Was 0.85; the band that "
                             "produced the Ion Radoi false negative reached to "
                             "0.85 and the review asked for negatives taken "
                             "from further down")
    parser.add_argument("--min-neg-score-ratio", type=float, default=0.15,
                        help="below this the negative is too easy to be useful. "
                             "Lowered from 0.25 to pay for the ceiling coming "
                             "down: the old band was [0.25, 0.85] and the mean "
                             "accepted ratio was 0.61, so keeping the floor "
                             "would have taken most queries under "
                             "--min-negatives. The idf-coverage and rare-term "
                             "guards, not this floor, are what keep a candidate "
                             "lexically related to the query")
    parser.add_argument("--dup-token-overlap", type=float, default=0.7,
                        help="a candidate sharing this fraction of its terms "
                             "with the positive is a near-copy the fingerprint "
                             "hash missed - reclassify it as a positive")
    parser.add_argument("--dup-idf-containment", type=float, default=0.45,
                        help="a candidate sharing this fraction of the "
                             "positive's *idf mass* is the same news story told "
                             "twice, so it answers the query too - reclassify "
                             "it as a positive rather than train against it. "
                             "Unlike --dup-token-overlap this weights the rare "
                             "terms, which is what identifies a story. 0 "
                             "disables it.\n"
                             "Measured on the 224-query sample against the real "
                             "corpus, and the measurement is a warning: correct "
                             "and incorrect reclassifications are interleaved "
                             "across the WHOLE range, at roughly 30% wrong "
                             "everywhere. 0.316 is 'Mirel Radoi va prelua CS "
                             "Universitatea Craiova' for a question about his "
                             "contract - right. 0.376 is an article about a "
                             "different swimmer entirely, 0.403 is a different "
                             "earthquake, 0.928 is a heatwave feature for a "
                             "question about Australian fires - all wrong. "
                             "Lowering the threshold does not buy precision, it "
                             "just moves more of both. The measure sees shared "
                             "topic, not shared event; what separates two "
                             "earthquakes is the magnitude, a number, and the "
                             "corpus carries no date field to fall back on. "
                             "0.45 is therefore deliberately conservative: "
                             "fewer relabels means fewer wrong ones in absolute "
                             "terms. Treat the output as candidates for review, "
                             "not as verified positives")
    parser.add_argument("--min-lex-for-title", type=float, default=0.2,
                        help="the title rule may not reclassify a candidate "
                             "whose *text* overlaps the positive less than this. "
                             "A title match alone is not evidence: in `stories` "
                             "it reclassified 45 candidates whose median text "
                             "overlap was 0.197, indistinguishable from an "
                             "ordinary negative - different folk tales that "
                             "happen to share a name")
    parser.add_argument("--min-title-terms", type=int, default=3,
                        help="ignore titles shorter than this in every title "
                             "comparison. `title_overlap` normalises by the "
                             "smaller title, so a one-word title scores 1.0 "
                             "against everything: 'Poveste' vs 'Poveste "
                             "taraneasca' was a perfect match")
    parser.add_argument("--title-prefix", type=int, default=5,
                        help="compare titles on this many leading characters "
                             "per token. Romanian inflection alone defeated the "
                             "full-token comparison: 'lider'/'liderul' and "
                             "'sindicat'/'sindicatului' are different tokens and "
                             "scored 0.30 on a pair that is plainly the same "
                             "story. 0 compares whole tokens")
    parser.add_argument("--top-idf-terms", type=int, default=3,
                        help="a negative must contain at least one of the "
                             "query's this-many rarest terms. Coverage alone is "
                             "not enough: a document can cover 30% of the idf "
                             "mass through generic words ('principalul motiv') "
                             "and be about nothing related. 0 disables it")
    parser.add_argument("--min-query-idf-coverage", type=float, default=0.3,
                        help="a negative must contain query terms worth this "
                             "fraction of the query's total idf. The score ratio "
                             "alone is relative to the positive, so when the "
                             "positive matches its own query poorly, unrelated "
                             "documents look 'hard' - this is the absolute floor")
    parser.add_argument("--min-shared-query-terms", type=int, default=2,
                        help="a negative must contain at least this many of the "
                             "query's scored terms; 1 lets in documents that "
                             "match a single mid-idf word and are not hard at all")
    parser.add_argument("--dup-title-overlap", type=float, default=0.6,
                        help="a candidate whose title shares this fraction of "
                             "its stems with the positive's title is the same "
                             "story - reclassify it as a positive; one that "
                             "matches a negative already picked is dropped. "
                             "Lowered from 0.8 because --title-prefix now "
                             "matches inflected forms, which raises every score")
    parser.add_argument("--max-mined-positives", type=int, default=4,
                        help="cap the reclassified positives recorded per query "
                             "(they are recorded, not trained on directly)")
    parser.add_argument("--same-type-only", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="mine negatives from the positive's own document type")
    parser.add_argument("--require-positive-in-topk", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="drop queries whose own document is not among the "
                             "BM25 candidates - usually a bad query")
    parser.add_argument("--max-positive-rank", type=int, default=5,
                        help="drop queries whose own document is not in the "
                             "BM25 top-N. Everything else in this file is "
                             "measured *relative* to the positive's score, so a "
                             "positive that ranks 40th makes unrelated "
                             "documents look hard. 0 disables the check")
    parser.add_argument("--drop-meta-queries", action=argparse.BooleanOptionalAction,
                        default=True, help="drop queries that refer to 'articolul'/'textul'")
    parser.add_argument("--dedupe-queries", action=argparse.BooleanOptionalAction,
                        default=True, help="keep every distinct query text once")
    parser.add_argument("--min-doc-chars", type=int, default=300,
                        help="shorter documents are treated as boilerplate")
    parser.add_argument("--min-query-terms", type=int, default=3)
    parser.add_argument("--max-doc-chars", type=int, default=4000,
                        help="characters of each document that get indexed")
    parser.add_argument("--max-df-ratio", type=float, default=0.1,
                        help="query terms in more than this share of the corpus "
                             "are ignored (near-zero idf, longest posting lists)")
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--max-queries", type=int, default=0,
                        help="cap the number of queries processed (0 = all)")
    parser.add_argument("--progress-every", type=int, default=5000)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    started = time.time()

    # ------------------------------------------------------------------ corpus
    print(f"▸ Reading corpus from {args.categories_dir} …")
    ids: List[str] = []
    texts: List[str] = []
    types: List[str] = []
    titles: List[str] = []
    for doc, doc_type in iter_documents(args.categories_dir,
                                        include_aggregates=args.include_aggregates):
        doc_id = doc.get("doc_id")
        if not doc_id:
            continue
        ids.append(doc_id)
        texts.append(document_text(doc, doc_type, args.max_doc_chars))
        types.append(doc_type)
        titles.append((doc.get("title") or "").strip())
    if not ids:
        print("✗ No corpus documents - is data/categories/ fetched (git lfs pull)?")
        return
    print(f"  {len(ids)} documents")

    # Collapse duplicates, drop boilerplate. `canonical_of[doc_id]` maps every
    # copy - including the ones the queries were generated from - onto the one
    # copy that is indexed.
    fingerprints = [text_fingerprint(texts[i], titles[i]) for i in range(len(ids))]
    group_first: Dict[str, int] = {}
    for i, fp in enumerate(fingerprints):
        group_first.setdefault(fp, i)

    corpus_ids = set(ids)
    stem = (lambda t: t[:args.title_prefix]) if args.title_prefix else (lambda t: t)
    title_terms = [frozenset(stem(t) for t in tokenize(title)) for title in titles]
    boiler = [is_boilerplate(texts[i], titles[i], args.min_doc_chars) for i in range(len(ids))]
    kept = [i for fp, i in group_first.items() if not boiler[i]]
    kept.sort()
    canonical_of: Dict[str, int] = {}
    keep_position = {i: n for n, i in enumerate(kept)}
    dropped_boiler_groups = len(group_first) - len(kept)
    for i, fp in enumerate(fingerprints):
        first = group_first[fp]
        if first in keep_position:
            canonical_of[ids[i]] = first
    print(f"  {len(group_first)} distinct texts "
          f"({len(ids) - len(group_first)} duplicate copies removed)")
    print(f"  {dropped_boiler_groups} boilerplate/short documents removed "
          f"→ {len(kept)} indexed documents")

    # ----------------------------------------------------------------- indexes
    # One index per document type: negatives are mined within a type by default,
    # and four small indexes are faster to score than one big one.
    index_key = (lambda t: t) if args.same_type_only else (lambda t: "all")
    per_index: Dict[str, List[int]] = {}
    for i in kept:
        per_index.setdefault(index_key(types[i]), []).append(i)

    indexes: Dict[str, BM25Index] = {}
    index_rows: Dict[str, List[int]] = {}
    row_of: Dict[int, int] = {}
    doc_terms: Dict[int, Any] = {}
    for key, rows in sorted(per_index.items()):
        tokenised = [tokenize(texts[i]) for i in rows]
        t0 = time.time()
        indexes[key] = BM25Index(tokenised, k1=args.k1, b=args.b)
        index_rows[key] = rows
        for local, i in enumerate(rows):
            row_of[i] = local
        vocab = indexes[key].vocab
        for local, tokens in enumerate(tokenised):
            term_ids = [vocab[t] for t in tokens if t in vocab]
            doc_terms[rows[local]] = np.unique(np.array(term_ids, dtype=np.int32))
        print(f"  index '{key}': {len(rows)} docs, {indexes[key].n_terms} terms, "
              f"{indexes[key].n_postings} postings, built in {time.time() - t0:.1f}s")

    # ----------------------------------------------------------------- queries
    print(f"▸ Reading queries from {args.queries} …")
    counts = {
        "records": 0, "queries_seen": 0, "dropped_meta": 0, "dropped_duplicate": 0,
        "dropped_short": 0, "dropped_positive_missing": 0, "dropped_positive_boilerplate": 0,
        "dropped_positive_not_retrieved": 0, "dropped_positive_rank": 0,
        "dropped_few_negatives": 0, "written": 0,
        "queries_with_mined_positive": 0,
    }
    seen_queries: set = set()
    flat: List[Tuple[str, Dict[str, Any]]] = []
    with open(args.queries, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            counts["records"] += 1
            positive_id = record.get("doc_id")
            for query in record.get("queries") or []:
                counts["queries_seen"] += 1
                folded = fold(query)
                if args.drop_meta_queries and META_QUERY_RE.search(folded):
                    counts["dropped_meta"] += 1
                    continue
                normalised = " ".join(folded.split())
                if args.dedupe_queries and normalised in seen_queries:
                    counts["dropped_duplicate"] += 1
                    continue
                if positive_id not in canonical_of:
                    # Either the document is not in this corpus at all, or its
                    # whole duplicate group was dropped as boilerplate. Claim
                    # the query text only *after* this check: claiming it first
                    # let a query whose positive was missing block the identical
                    # query text arriving later with a usable positive.
                    if positive_id in corpus_ids:
                        counts["dropped_positive_boilerplate"] += 1
                    else:
                        counts["dropped_positive_missing"] += 1
                    continue
                if args.dedupe_queries:
                    seen_queries.add(normalised)
                flat.append((query, record))
                if args.max_queries and len(flat) >= args.max_queries:
                    break
            if args.max_queries and len(flat) >= args.max_queries:
                break
    print(f"  {counts['records']} records, {counts['queries_seen']} queries, "
          f"{len(flat)} after cleaning")

    # ------------------------------------------------------------------ mining
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    # Why candidates were rejected, so the guards can be tuned against numbers
    # instead of intuition.
    rejected = {"outscores_positive": 0, "too_easy": 0, "few_shared_terms": 0,
                "low_idf_coverage": 0, "no_rare_term": 0, "near_duplicate": 0,
                "already_positive": 0, "same_story_as_positive": 0}
    # The measured doc-to-doc overlap of what was kept vs. what was
    # reclassified: the only way to tell whether --dup-idf-containment sits in
    # the gap between the two populations or straight through the middle of one.
    negative_lex: List[float] = []
    same_story_lex: List[float] = []
    mined_positive_total = 0
    positive_ranks: List[int] = []
    hardest_ratios: List[float] = []
    all_ratios: List[float] = []
    negatives_same_outlet = negatives_total = 0
    t0 = time.time()

    with open(args.output, "w", encoding="utf-8") as out:
        for n, (query, record) in enumerate(flat):
            if args.progress_every and n and n % args.progress_every == 0:
                rate = n / (time.time() - t0)
                print(f"    {n}/{len(flat)} queries  ({rate:.0f}/s, "
                      f"{counts['written']} written)", flush=True)

            positive_i = canonical_of[record["doc_id"]]
            key = index_key(types[positive_i])
            index = indexes[key]
            rows = index_rows[key]

            terms = index.query_terms(tokenize(query), max_df_ratio=args.max_df_ratio)
            if len(terms) < args.min_query_terms:
                counts["dropped_short"] += 1
                continue

            query_terms = np.array(sorted(terms), dtype=np.int32)
            query_idf = index.idf[query_terms]
            query_idf_total = float(query_idf.sum())
            rare_terms = query_terms[np.argsort(-query_idf)[:args.top_idf_terms]] \
                if args.top_idf_terms else np.empty(0, dtype=np.int32)
            scores = index.score(terms)
            positive_row = row_of[positive_i]
            positive_score = float(scores[positive_row])
            hits, hit_scores = index.top_k(terms, args.candidates, scores=scores)

            rank = int(np.flatnonzero(hits == positive_row)[0]) + 1 \
                if positive_row in hits else 0
            if positive_score <= 0 or (args.require_positive_in_topk and rank == 0):
                counts["dropped_positive_not_retrieved"] += 1
                continue
            if args.max_positive_rank and rank > args.max_positive_rank:
                counts["dropped_positive_rank"] += 1
                continue
            positive_ranks.append(rank)

            # Everything that is a correct answer is off limits: the positive
            # itself and the extra positives of the multi-document prompt.
            excluded_rows = {positive_row}
            for extra in record.get("similar_doc_ids", []) or []:
                extra_i = canonical_of.get(extra)
                if extra_i is not None and extra_i in row_of and types[extra_i] == types[positive_i]:
                    excluded_rows.add(row_of[extra_i])

            positive_terms = doc_terms[positive_i]
            positive_title = title_terms[positive_i]
            positive_mass = float(index.idf[positive_terms].sum())
            negatives: List[Dict[str, Any]] = []
            mined: List[Dict[str, Any]] = []   # candidates that are positives
            picked: List[int] = []             # corpus indices already accepted
            for position, (row, score) in enumerate(zip(hits, hit_scores)):
                if row in excluded_rows:
                    rejected["already_positive"] += 1
                    continue
                ratio = float(score) / positive_score
                if ratio < args.min_neg_score_ratio:
                    # Scores only go down from here, so everything left is too
                    # easy. `position` is the number of candidates consumed -
                    # the old count used len(negatives), which is the number
                    # *accepted* and always smaller, so this figure used to be
                    # inflated by every candidate a guard had rejected.
                    rejected["too_easy"] += len(hits) - position
                    break
                candidate_i = rows[row]

                # Is this the positive's own story, told a second time? Every
                # other guard in this loop compares the candidate to the
                # *query*, and none of them can see that. Two of the three
                # measures below are cheap; the idf-weighted one is what
                # actually separates "same story" from "same vocabulary".
                #
                # This runs *before* the score-ratio ceiling on purpose. The
                # false negatives the review found scored 0.82 of the positive,
                # i.e. above the ceiling - testing after it would throw them
                # away as "outscores_positive" instead of recovering them as
                # positives, which is the whole point of the change.
                lex = idf_containment(doc_terms[candidate_i], positive_terms,
                                      index.idf, positive_mass)
                raw = containment(doc_terms[candidate_i], positive_terms)
                tit = title_overlap(title_terms[candidate_i], positive_title,
                                    args.min_title_terms)
                if args.dup_idf_containment and lex >= args.dup_idf_containment:
                    reason = "idf_overlap_with_positive"
                elif raw >= args.dup_token_overlap:
                    reason = "token_overlap_with_positive"
                elif tit >= args.dup_title_overlap and lex >= args.min_lex_for_title:
                    # Matching titles are a hint, not proof; the texts have to
                    # agree too, or this is two different stories with one name.
                    reason = "title_overlap_with_positive"
                else:
                    reason = ""
                if reason:
                    rejected["same_story_as_positive"] += 1
                    same_story_lex.append(lex)
                    if len(mined) < args.max_mined_positives:
                        mined.append({
                            "doc_id": ids[candidate_i],
                            "source": "bm25_candidate",
                            "reason": reason,
                            "rank": position + 1,
                            "query_score_ratio": round(ratio, 4),
                            "bm25": round(float(score), 3),
                            "lexical_sim_to_positive": round(lex, 4),
                            "token_sim_to_positive": round(raw, 4),
                            "title_sim_to_positive": round(tit, 4),
                        })
                    continue

                if ratio > args.max_neg_score_ratio:
                    rejected["outscores_positive"] += 1
                    continue          # answers the query at least as well as the positive
                shared = np.intersect1d(doc_terms[candidate_i], query_terms,
                                        assume_unique=True)
                if shared.size < args.min_shared_query_terms:
                    rejected["few_shared_terms"] += 1
                    continue          # matches one word of the query: not a hard negative
                if (query_idf_total > 0
                        and float(index.idf[shared].sum()) / query_idf_total
                        < args.min_query_idf_coverage):
                    rejected["low_idf_coverage"] += 1
                    continue          # shares only cheap words with the query
                if rare_terms.size and not np.intersect1d(shared, rare_terms).size:
                    rejected["no_rare_term"] += 1
                    continue          # shares nothing the query is actually about
                # Near-copy of a negative already picked: three re-publications
                # of one article would otherwise fill every negative slot with a
                # single text. (The same test against the positive is the block
                # above, which keeps the candidate instead of dropping it.)
                if any(containment(doc_terms[candidate_i], doc_terms[other]) >= args.dup_token_overlap
                       or title_overlap(title_terms[candidate_i], title_terms[other],
                                        args.min_title_terms) >= args.dup_title_overlap
                       for other in picked):
                    rejected["near_duplicate"] += 1
                    continue
                picked.append(candidate_i)
                negative_lex.append(lex)
                negatives.append({
                    "doc_id": ids[candidate_i],
                    "bm25": round(float(score), 3),
                    "ratio": round(ratio, 4),
                    "rank": position + 1,
                    "lexical_sim_to_positive": round(lex, 4),
                    "token_sim_to_positive": round(raw, 4),
                    "title_sim_to_positive": round(tit, 4),
                })
                if len(negatives) >= args.negatives:
                    break

            if len(negatives) < max(1, args.min_negatives):
                counts["dropped_few_negatives"] += 1
                continue
            if mined:
                counts["queries_with_mined_positive"] += 1
                mined_positive_total += len(mined)

            positive_outlet = ids[positive_i].split("_", 1)[0]
            for neg in negatives:
                negatives_total += 1
                if neg["doc_id"].split("_", 1)[0] == positive_outlet:
                    negatives_same_outlet += 1
            hardest_ratios.append(negatives[0]["ratio"])
            all_ratios.extend(x["ratio"] for x in negatives)

            out.write(json.dumps({
                "query_id": f"q_{counts['written']:07d}",
                "query": query,
                "positive_doc_id": ids[positive_i],
                "positive_source_doc_id": record["doc_id"],
                "positive_title": (record.get("title") or "").strip(),
                "positive_bm25": round(positive_score, 3),
                "positive_rank": rank,
                "additional_positive_doc_ids": record.get("similar_doc_ids", []),
                # Reclassified candidates: BM25 ranked them high and the
                # doc-to-doc guards found them to be the positive's own story,
                # so they answer the query and must not be trained against.
                # Kept with their provenance rather than discarded - a reranker
                # trained later wants exactly these near-misses.
                "mined_positive_doc_ids": [x["doc_id"] for x in mined],
                "mined_positive_provenance": mined,
                "negative_doc_ids": [x["doc_id"] for x in negatives],
                # NOTE: BM25(negative, query) / BM25(positive, query) - how well
                # the negative answers *the query* relative to the positive. It
                # is not a similarity between the two documents; that one is
                # `lexical_sim_to_positive` in the provenance below. The field
                # keeps its name because the dense builder writes it too.
                "negative_similarities": [x["ratio"] for x in negatives],
                "negative_bm25": [x["bm25"] for x in negatives],
                "negative_provenance": [
                    {"doc_id": x["doc_id"], "source": "bm25",
                     "rank": x["rank"],
                     "query_score_ratio": x["ratio"],
                     "lexical_sim_to_positive": x["lexical_sim_to_positive"],
                     "token_sim_to_positive": x["token_sim_to_positive"],
                     "title_sim_to_positive": x["title_sim_to_positive"],
                     "same_outlet": x["doc_id"].split("_", 1)[0] == positive_outlet}
                    for x in negatives
                ],
                "type": record.get("type", ""),
                "category": record.get("category", ""),
                # Where the query ultimately comes from: the upstream dataset or
                # outlet the positive was drawn from ("adevarul", "zf",
                # "readerbench/ro-stories", "readerbench/ro-text-summarization
                # (alephnews)"). Carried through so the exported training set can
                # say what each example's provenance is.
                "query_source": record.get("source", ""),
                "duplicate_group": fingerprints[positive_i][:12],
                "query_version": record.get("prompt_version", "v1"),
                "generator": record.get("generator", ""),
                "retriever": "bm25",
            }, ensure_ascii=False) + "\n")
            counts["written"] += 1

    # ------------------------------------------------------------------ report
    elapsed = time.time() - started
    found = [r for r in positive_ranks if r]
    stats = {
        "config": vars(args),
        "elapsed_seconds": round(elapsed, 1),
        "corpus": {
            "documents": len(ids),
            "distinct_texts": len(group_first),
            "boilerplate_removed": dropped_boiler_groups,
            "indexed": len(kept),
            "indexes": {k: len(v) for k, v in sorted(index_rows.items())},
        },
        "queries": counts,
        "rejected_candidates": rejected,
        "positive_rank": {
            "retrieved_pct": round(100 * len(found) / len(positive_ranks), 2) if positive_ranks else 0.0,
            "median": sorted(found)[len(found) // 2] if found else None,
            "top1_pct": round(100 * sum(1 for r in found if r == 1) / len(positive_ranks), 2) if positive_ranks else 0.0,
            "top10_pct": round(100 * sum(1 for r in found if r <= 10) / len(positive_ranks), 2) if positive_ranks else 0.0,
        },
        "negatives": {
            "total": negatives_total,
            "same_outlet_pct": round(100 * negatives_same_outlet / negatives_total, 2) if negatives_total else 0.0,
            "mean_score_ratio": round(sum(all_ratios) / len(all_ratios), 4) if all_ratios else 0.0,
            "mean_hardest_ratio": round(sum(hardest_ratios) / len(hardest_ratios), 4) if hardest_ratios else 0.0,
        },
        # Read these two side by side before touching --dup-idf-containment.
        # If the kept negatives' p95 sits below the reclassified population's
        # p5, the threshold is in the gap and the split is clean; if the two
        # overlap heavily, it is cutting through one population and the
        # measure - not the threshold - needs the work.
        "lexical_sim_to_positive": {
            "kept_negatives": percentiles(negative_lex),
            "reclassified_as_positive": percentiles(same_story_lex),
        },
        "mined_positives": {
            "total": mined_positive_total,
            "queries_with_at_least_one": counts["queries_with_mined_positive"],
            "pct_of_written": round(100 * counts["queries_with_mined_positive"]
                                    / counts["written"], 2) if counts["written"] else 0.0,
        },
    }
    with open(args.output + ".stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("BM25 TRIPLETS - FINAL REPORT")
    print("=" * 70)
    print(f"  corpus        : {len(ids)} docs → {len(kept)} indexed "
          f"({len(ids) - len(group_first)} duplicates, {dropped_boiler_groups} boilerplate)")
    print(f"  queries       : {counts['queries_seen']} generated")
    print(f"    - meta/unanswerable      {counts['dropped_meta']}")
    print(f"    - duplicate query text   {counts['dropped_duplicate']}")
    print(f"    - positive is boilerplate {counts['dropped_positive_boilerplate']}")
    print(f"    - positive not in corpus  {counts['dropped_positive_missing']}")
    print(f"    - too few query terms    {counts['dropped_short']}")
    print(f"    - positive not retrieved {counts['dropped_positive_not_retrieved']}")
    print(f"    - positive ranked below {args.max_positive_rank}  "
          f"{counts['dropped_positive_rank']}")
    print(f"    - fewer than {max(1, args.min_negatives)} negatives   "
          f"{counts['dropped_few_negatives']}")
    print(f"  triplets      : {counts['written']} written to {args.output}")
    print(f"  positive rank : top-1 {stats['positive_rank']['top1_pct']}%, "
          f"top-10 {stats['positive_rank']['top10_pct']}%, "
          f"in candidates {stats['positive_rank']['retrieved_pct']}%")
    print(f"  candidates rejected: "
          + ", ".join(f"{k} {v}" for k, v in rejected.items()))
    print(f"  negatives     : mean score ratio {stats['negatives']['mean_score_ratio']}, "
          f"hardest {stats['negatives']['mean_hardest_ratio']}, "
          f"same outlet {stats['negatives']['same_outlet_pct']}%")
    print(f"  mined positives: {mined_positive_total} recovered over "
          f"{counts['queries_with_mined_positive']} queries "
          f"({stats['mined_positives']['pct_of_written']}% of those written)")
    kept_p, recl_p = (stats["lexical_sim_to_positive"]["kept_negatives"],
                      stats["lexical_sim_to_positive"]["reclassified_as_positive"])
    print(f"  lexical sim to positive (idf-weighted):")
    print(f"    kept as negative : " + "  ".join(f"p{k} {v}" for k, v in kept_p.items()))
    print(f"    reclassified     : " + "  ".join(f"p{k} {v}" for k, v in recl_p.items()))
    print(f"  elapsed       : {elapsed / 60:.1f} min")
    print("=" * 70)


if __name__ == "__main__":
    main()
