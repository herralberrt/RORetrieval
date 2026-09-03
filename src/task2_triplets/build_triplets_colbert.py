"""
TASK 2 (late-interaction variant): triplets whose negatives are mined by MaxSim.

Same contract as `build_triplets_bm25.py` - the positive is the document the
query was generated from, retrieval only finds negatives - but the retriever is
late interaction (`colbert_index.py`) instead of BM25.

Why a third miner. BM25 negatives share the query's rare *terms*; the dense run's
negatives sit near the query in one pooled vector. Late interaction fails
differently again: it rewards a document that matches the query token by token,
so it surfaces passages that answer a *neighbouring* question in the same
wording - the case a lexical miner misses because the words differ and a pooled
dense miner misses because the average washes it out.

What is reused, deliberately, rather than re-derived:

* **The corpus cleaning** - near-duplicate collapse, boilerplate removal,
  meta-query dropping, query dedup - is imported from the BM25 builder. Two
  miners disagreeing about which documents exist would make their outputs
  incomparable.
* **The same-story guard.** `--dup-idf-containment 0.45` was calibrated against
  the real corpus (see `data/triplets/README.md` §9) and it is a property of the
  *documents*, not of the retriever, so it transfers unchanged. That guard needs
  idf, so a BM25 index is still built - it costs about three CPU minutes and
  buys a measured threshold instead of a fresh guess.

What does **not** transfer is the score band. `--max-neg-score-ratio 0.6` was
calibrated for BM25 ratios; MaxSim sums cosines over query tokens, so its ratios
sit much higher and much closer together. The defaults here are a starting
point, and the run reports the measured distribution so they can be set from
numbers - the same discipline the BM25 run went through.

    python3 -m src.task2_triplets.build_triplets_colbert \\
        --queries data/queries/queries_gemma3_27b.jsonl \\
        --output data/triplets/triplets_27b_colbert.jsonl
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [
    str(p) for p in _SRC_DIR.iterdir()
    if p.is_dir() and not p.name.startswith((".", "_"))
]

from gemma_query_generation import DEFAULT_CATEGORIES_DIR, document_text, iter_documents  # noqa: E402
from bm25 import BM25Index, fold, tokenize  # noqa: E402
from build_triplets_bm25 import (  # noqa: E402
    META_QUERY_RE, containment, idf_containment, is_boilerplate, percentiles,
    text_fingerprint, title_overlap,
)
from colbert_index import DEFAULT_MODEL, LateInteractionIndex  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build (query, positive, late-interaction hard negatives) triplets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--queries", required=True)
    p.add_argument("--output", default="data/triplets/triplets_colbert.jsonl")
    p.add_argument("--categories-dir", default=DEFAULT_CATEGORIES_DIR)
    p.add_argument("--include-aggregates", action="store_true")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--device", default=None, help="cuda / cpu; auto-detected")
    p.add_argument("--doc-max-tokens", type=int, default=128)
    p.add_argument("--query-max-tokens", type=int, default=32)
    p.add_argument("--encode-batch", type=int, default=256)
    p.add_argument("--query-batch", type=int, default=64)
    p.add_argument("--first-stage", type=int, default=512,
                   help="mean-pooled shortlist reranked by MaxSim per query")
    p.add_argument("--negatives", type=int, default=4)
    p.add_argument("--min-negatives", type=int, default=2)
    p.add_argument("--candidates", type=int, default=100,
                   help="MaxSim results kept per query before filtering")
    p.add_argument("--max-neg-score-ratio", type=float, default=0.95,
                   help="NOT the BM25 value. MaxSim ratios are compressed - the "
                        "run prints the measured distribution, set this from it")
    p.add_argument("--min-neg-score-ratio", type=float, default=0.5)
    p.add_argument("--dup-idf-containment", type=float, default=0.45,
                   help="same-story guard, calibrated in README §9; a property "
                        "of the documents, so it transfers across retrievers")
    p.add_argument("--dup-token-overlap", type=float, default=0.7)
    p.add_argument("--dup-title-overlap", type=float, default=0.6)
    p.add_argument("--min-lex-for-title", type=float, default=0.2)
    p.add_argument("--min-title-terms", type=int, default=3)
    p.add_argument("--title-prefix", type=int, default=5)
    p.add_argument("--max-mined-positives", type=int, default=4)
    p.add_argument("--max-positive-rank", type=int, default=5)
    p.add_argument("--require-positive-in-topk", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--same-type-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--drop-meta-queries", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dedupe-queries", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--min-doc-chars", type=int, default=300)
    p.add_argument("--max-doc-chars", type=int, default=4000)
    p.add_argument("--max-df-ratio", type=float, default=0.1)
    p.add_argument("--max-queries", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=2000)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    started = time.time()

    # ------------------------------------------------------------------ corpus
    print(f"▸ Reading corpus from {args.categories_dir} …")
    ids, texts, types, titles = [], [], [], []
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

    fingerprints = [text_fingerprint(texts[i], titles[i]) for i in range(len(ids))]
    group_first: Dict[str, int] = {}
    for i, fp in enumerate(fingerprints):
        group_first.setdefault(fp, i)
    corpus_ids = set(ids)
    stem = (lambda t: t[:args.title_prefix]) if args.title_prefix else (lambda t: t)
    title_terms = [frozenset(stem(t) for t in tokenize(title)) for title in titles]
    boiler = [is_boilerplate(texts[i], titles[i], args.min_doc_chars) for i in range(len(ids))]
    kept = sorted(i for fp, i in group_first.items() if not boiler[i])
    keep_position = {i: n for n, i in enumerate(kept)}
    canonical_of = {ids[i]: group_first[fp] for i, fp in enumerate(fingerprints)
                    if group_first[fp] in keep_position}
    print(f"  {len(group_first)} distinct texts, {len(kept)} indexed after boilerplate")

    row_of = {i: n for n, i in enumerate(kept)}
    kept_texts = [texts[i] for i in kept]

    # --------------------------------------------------- BM25, for idf only
    # The same-story guard needs idf. Retrieval does not use this index.
    print("▸ Building the BM25 index (idf for the same-story guard) …")
    t0 = time.time()
    tokenised = [tokenize(t) for t in kept_texts]
    bm25 = BM25Index(tokenised)
    doc_terms: Dict[int, Any] = {}
    for local, toks in enumerate(tokenised):
        term_ids = [bm25.vocab[t] for t in toks if t in bm25.vocab]
        doc_terms[kept[local]] = np.unique(np.array(term_ids, dtype=np.int32))
    del tokenised
    print(f"  {bm25.n_terms} terms in {time.time() - t0:.0f}s")

    # ------------------------------------------------- late-interaction index
    print("▸ Encoding the corpus for late interaction …")
    t0 = time.time()
    index = LateInteractionIndex(args.model, args.device, args.doc_max_tokens,
                                 args.query_max_tokens, args.encode_batch)
    index.build(kept_texts)
    print(f"  encoded in {(time.time() - t0) / 60:.1f} min")

    type_rows: Dict[str, np.ndarray] = {}
    if args.same_type_only:
        by_type: Dict[str, List[int]] = {}
        for i in kept:
            by_type.setdefault(types[i], []).append(row_of[i])
        type_rows = {k: np.array(v, dtype=np.int64) for k, v in by_type.items()}

    # ----------------------------------------------------------------- queries
    print(f"▸ Reading queries from {args.queries} …")
    counts = {"records": 0, "queries_seen": 0, "dropped_meta": 0, "dropped_duplicate": 0,
              "dropped_positive_missing": 0, "dropped_positive_boilerplate": 0,
              "dropped_positive_not_retrieved": 0, "dropped_positive_rank": 0,
              "dropped_few_negatives": 0, "written": 0, "queries_with_mined_positive": 0}
    seen: set = set()
    flat: List[Tuple[str, Dict[str, Any]]] = []
    with open(args.queries, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
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
                if args.dedupe_queries and normalised in seen:
                    counts["dropped_duplicate"] += 1
                    continue
                if positive_id not in canonical_of:
                    counts["dropped_positive_boilerplate" if positive_id in corpus_ids
                           else "dropped_positive_missing"] += 1
                    continue
                if args.dedupe_queries:
                    seen.add(normalised)
                flat.append((query, record))
                if args.max_queries and len(flat) >= args.max_queries:
                    break
            if args.max_queries and len(flat) >= args.max_queries:
                break
    print(f"  {counts['queries_seen']} queries, {len(flat)} after cleaning")

    # ------------------------------------------------------------------ mining
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    rejected = {"outscores_positive": 0, "too_easy": 0, "near_duplicate": 0,
                "already_positive": 0, "same_story_as_positive": 0}
    negative_lex: List[float] = []
    same_story_lex: List[float] = []
    all_ratios: List[float] = []
    positive_ranks: List[int] = []
    mined_total = negatives_total = negatives_same_outlet = 0
    t0 = time.time()

    with open(args.output, "w", encoding="utf-8") as out:
        for start in range(0, len(flat), args.query_batch):
            chunk = flat[start:start + args.query_batch]
            if args.progress_every and start and start % args.progress_every == 0:
                rate = start / (time.time() - t0)
                print(f"    {start}/{len(flat)} queries  ({rate:.0f}/s, "
                      f"{counts['written']} written)", flush=True)

            # One search call per (batch, type) group: `allowed` is per-call.
            groups: Dict[str, List[int]] = {}
            for n, (_, record) in enumerate(chunk):
                key = types[canonical_of[record["doc_id"]]] if args.same_type_only else "all"
                groups.setdefault(key, []).append(n)

            hits_of: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
            for key, members in groups.items():
                allowed = type_rows.get(key) if args.same_type_only else None
                idx, sc = index.search([chunk[m][0] for m in members],
                                       top_k=args.candidates,
                                       first_stage=args.first_stage,
                                       allowed=allowed)
                for slot, m in enumerate(members):
                    hits_of[m] = (idx[slot], sc[slot])

            for n, (query, record) in enumerate(chunk):
                positive_i = canonical_of[record["doc_id"]]
                positive_row = row_of[positive_i]
                hits, scores = hits_of[n]
                valid = hits >= 0
                hits, scores = hits[valid], scores[valid]

                where = np.flatnonzero(hits == positive_row)
                rank = int(where[0]) + 1 if where.size else 0
                if args.require_positive_in_topk and rank == 0:
                    counts["dropped_positive_not_retrieved"] += 1
                    continue
                if args.max_positive_rank and rank > args.max_positive_rank:
                    counts["dropped_positive_rank"] += 1
                    continue
                positive_score = float(scores[where[0]])
                if positive_score <= 0:
                    counts["dropped_positive_not_retrieved"] += 1
                    continue
                positive_ranks.append(rank)

                excluded = {positive_row}
                for extra in record.get("similar_doc_ids", []) or []:
                    e = canonical_of.get(extra)
                    if e is not None and e in row_of and types[e] == types[positive_i]:
                        excluded.add(row_of[e])

                positive_terms = doc_terms[positive_i]
                positive_title = title_terms[positive_i]
                positive_mass = float(bm25.idf[positive_terms].sum())
                negatives: List[Dict[str, Any]] = []
                mined: List[Dict[str, Any]] = []
                picked: List[int] = []

                for position, (row, score) in enumerate(zip(hits, scores)):
                    if row in excluded:
                        rejected["already_positive"] += 1
                        continue
                    ratio = float(score) / positive_score
                    if ratio < args.min_neg_score_ratio:
                        rejected["too_easy"] += len(hits) - position
                        break
                    candidate_i = kept[int(row)]

                    lex = idf_containment(doc_terms[candidate_i], positive_terms,
                                          bm25.idf, positive_mass)
                    raw = containment(doc_terms[candidate_i], positive_terms)
                    tit = title_overlap(title_terms[candidate_i], positive_title,
                                        args.min_title_terms)
                    if args.dup_idf_containment and lex >= args.dup_idf_containment:
                        reason = "idf_overlap_with_positive"
                    elif raw >= args.dup_token_overlap:
                        reason = "token_overlap_with_positive"
                    elif tit >= args.dup_title_overlap and lex >= args.min_lex_for_title:
                        reason = "title_overlap_with_positive"
                    else:
                        reason = ""
                    if reason:
                        rejected["same_story_as_positive"] += 1
                        same_story_lex.append(lex)
                        if len(mined) < args.max_mined_positives:
                            mined.append({
                                "doc_id": ids[candidate_i], "source": "colbert_candidate",
                                "reason": reason, "rank": position + 1,
                                "query_score_ratio": round(ratio, 4),
                                "maxsim": round(float(score), 4),
                                "lexical_sim_to_positive": round(lex, 4),
                                "token_sim_to_positive": round(raw, 4),
                                "title_sim_to_positive": round(tit, 4),
                            })
                        continue

                    if ratio > args.max_neg_score_ratio:
                        rejected["outscores_positive"] += 1
                        continue
                    if any(containment(doc_terms[candidate_i], doc_terms[o]) >= args.dup_token_overlap
                           or title_overlap(title_terms[candidate_i], title_terms[o],
                                            args.min_title_terms) >= args.dup_title_overlap
                           for o in picked):
                        rejected["near_duplicate"] += 1
                        continue

                    picked.append(candidate_i)
                    negative_lex.append(lex)
                    negatives.append({
                        "doc_id": ids[candidate_i], "maxsim": round(float(score), 4),
                        "ratio": round(ratio, 4), "rank": position + 1,
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
                    mined_total += len(mined)

                positive_outlet = ids[positive_i].split("_", 1)[0]
                for neg in negatives:
                    negatives_total += 1
                    negatives_same_outlet += neg["doc_id"].split("_", 1)[0] == positive_outlet
                all_ratios.extend(x["ratio"] for x in negatives)

                out.write(json.dumps({
                    "query_id": f"q_{counts['written']:07d}",
                    "query": query,
                    "positive_doc_id": ids[positive_i],
                    "positive_source_doc_id": record["doc_id"],
                    "positive_title": (record.get("title") or "").strip(),
                    "positive_maxsim": round(positive_score, 4),
                    "positive_rank": rank,
                    "additional_positive_doc_ids": record.get("similar_doc_ids", []),
                    "mined_positive_doc_ids": [x["doc_id"] for x in mined],
                    "mined_positive_provenance": mined,
                    "negative_doc_ids": [x["doc_id"] for x in negatives],
                    # As in the BM25 file: the ratio of the negative's own score
                    # to the positive's for this query, not a doc-doc similarity.
                    "negative_similarities": [x["ratio"] for x in negatives],
                    "negative_maxsim": [x["maxsim"] for x in negatives],
                    "negative_provenance": [
                        {"doc_id": x["doc_id"], "source": "late_interaction",
                         "rank": x["rank"], "query_score_ratio": x["ratio"],
                         "lexical_sim_to_positive": x["lexical_sim_to_positive"],
                         "token_sim_to_positive": x["token_sim_to_positive"],
                         "title_sim_to_positive": x["title_sim_to_positive"],
                         "same_outlet": x["doc_id"].split("_", 1)[0] == positive_outlet}
                        for x in negatives],
                    "type": record.get("type", ""),
                    "category": record.get("category", ""),
                    "query_source": record.get("source", ""),
                    "duplicate_group": fingerprints[positive_i][:12],
                    "query_version": record.get("prompt_version", "v1"),
                    "generator": record.get("generator", ""),
                    "retriever": "late_interaction",
                }, ensure_ascii=False) + "\n")
                counts["written"] += 1

    # ------------------------------------------------------------------ report
    elapsed = time.time() - started
    found = [r for r in positive_ranks if r]
    stats = {
        "config": vars(args),
        "elapsed_seconds": round(elapsed, 1),
        "corpus": {"documents": len(ids), "indexed": len(kept)},
        "queries": counts,
        "rejected_candidates": rejected,
        "positive_rank": {
            "top1_pct": round(100 * sum(1 for r in found if r == 1) / len(positive_ranks), 2)
            if positive_ranks else 0.0,
            "median": sorted(found)[len(found) // 2] if found else None,
        },
        "negatives": {
            "total": negatives_total,
            "same_outlet_pct": round(100 * negatives_same_outlet / negatives_total, 2)
            if negatives_total else 0.0,
            "score_ratio": percentiles(all_ratios),
        },
        "lexical_sim_to_positive": {
            "kept_negatives": percentiles(negative_lex),
            "reclassified_as_positive": percentiles(same_story_lex),
        },
        "mined_positives": {"total": mined_total,
                            "queries_with_at_least_one": counts["queries_with_mined_positive"]},
    }
    with open(args.output + ".stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 70)
    print("LATE-INTERACTION TRIPLETS - FINAL REPORT")
    print("=" * 70)
    print(f"  corpus        : {len(ids)} docs → {len(kept)} indexed")
    print(f"  queries       : {counts['queries_seen']} generated, {len(flat)} after cleaning")
    for key in ("dropped_positive_not_retrieved", "dropped_positive_rank",
                "dropped_few_negatives"):
        print(f"    - {key:<32} {counts[key]}")
    print(f"  triplets      : {counts['written']} written to {args.output}")
    print(f"  positive rank : top-1 {stats['positive_rank']['top1_pct']}%")
    print(f"  candidates rejected: " + ", ".join(f"{k} {v}" for k, v in rejected.items()))
    print(f"  score ratio   : " + "  ".join(f"p{k} {v}" for k, v in
                                            stats["negatives"]["score_ratio"].items()))
    print(f"  mined positives: {mined_total} over {counts['queries_with_mined_positive']} queries")
    print(f"  lexical sim to positive:")
    print(f"    kept as negative : " + "  ".join(
        f"p{k} {v}" for k, v in stats["lexical_sim_to_positive"]["kept_negatives"].items()))
    print(f"    reclassified     : " + "  ".join(
        f"p{k} {v}" for k, v in stats["lexical_sim_to_positive"]["reclassified_as_positive"].items()))
    print(f"  elapsed       : {elapsed / 60:.1f} min")
    print("=" * 70)


if __name__ == "__main__":
    main()
