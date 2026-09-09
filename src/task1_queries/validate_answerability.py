"""
Does the positive passage actually answer the question it generated?

The whole dataset rests on one assumption: every query was written *from* a
document, so that document answers it by construction. Nothing has ever tested
that. If Gemma invented questions whose answers are not in the passage, then
every "positive" for those rows is wrong, the retrievers were asked to find
something unfindable, and the in-domain scores mean less than they appear to.

This asks a judge model, passage in hand, whether the question is answerable
from it alone, and to quote the span if so.

    python3 -m src.task1_queries.validate_answerability \\
        --queries data/queries/queries_gemma3_27b.jsonl \\
        --output results/answerability.jsonl \\
        --sample 2000

**The judge shares a family with the generator, and that is a real limit.**
The questions came from `gemma-3-27b-it`; the default judge is
`gemma-3-4b-it`, chosen because it is the only capable instruct model already
cached and the account has ~8 GB of disk quota left. A different size has
different failure modes, so this is better than asking 27B to grade itself -
but it is not an independent opinion, and a model from another family would be
worth the download if the quota allows. Read the rate as a floor on the
problem, not a measurement of it.

A sample is enough. 2000 queries put the standard error on a rate near 5% at
about half a point, which is far finer than any decision this informs, and it
costs minutes instead of hours.
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [
    str(p) for p in _SRC_DIR.iterdir()
    if p.is_dir() and not p.name.startswith((".", "_"))
]

from gemma_query_generation import (  # noqa: E402
    DEFAULT_CATEGORIES_DIR, document_text, iter_documents,
)

# Evidence first, verdict last: asking for the verdict first would let the model
# commit to DA and then invent a quote to match, while this way the quote is the
# reason for the verdict rather than a decoration on it.
#
# The cost of that ordering is that the verdict is what truncation eats first.
# The initial run lost 148 of 2000 replies (7.4%) that way - every one of them
# had a CITAT and no VERDICT, because the model quoted generously and ran out of
# the 160-token budget before the last line. Hence the 25-word cap on the quote
# and the larger default budget below; do not lower either without checking the
# unparsed count.
PROMPT = """Ai mai jos un PASAJ și o ÎNTREBARE.

Sarcina ta: stabilește dacă întrebarea poate fi răspunsă FOLOSIND DOAR pasajul.

Reguli:
- Nu folosi cunoștințe din afara pasajului. Dacă știi răspunsul din altă parte, dar pasajul nu îl conține, verdictul este NU.
- Un răspuns parțial, vag sau doar sugerat nu este suficient. Verdictul este DA doar dacă pasajul conține răspunsul explicit.
- Dacă întrebarea se referă la „articol", „text" sau „document" fără să fie de sine stătătoare, verdictul este NU.

Răspunde exact în formatul acesta, pe trei linii:
CITAT: <fragmentul din pasaj care conține răspunsul, cel mult 25 de cuvinte, sau - dacă nu există>
RASPUNS: <răspunsul în cel mult 15 cuvinte, sau - dacă nu există>
VERDICT: <DA sau NU>

PASAJ:
{passage}

ÎNTREBARE: {question}"""

VERDICT_RE = re.compile(r"VERDICT\s*:\s*(DA|NU)", re.IGNORECASE)
QUOTE_RE = re.compile(r"CITAT\s*:\s*(.*)", re.IGNORECASE)
ANSWER_RE = re.compile(r"RASPUNS\s*:\s*(.*)", re.IGNORECASE)


def parse(reply: str) -> Dict[str, Any]:
    """Pull the three fields out, and say plainly when the format broke."""
    verdict = VERDICT_RE.search(reply)
    quote = QUOTE_RE.search(reply)
    answer = ANSWER_RE.search(reply)
    return {
        "answerable": (verdict.group(1).upper() == "DA") if verdict else None,
        "quote": quote.group(1).strip()[:300] if quote else "",
        "answer": answer.group(1).strip()[:200] if answer else "",
        "parsed": verdict is not None,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Check whether each query is answerable from its own positive.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--queries", required=True)
    p.add_argument("--categories-dir", default=DEFAULT_CATEGORIES_DIR)
    p.add_argument("--include-aggregates", action="store_true")
    p.add_argument("--output", default="results/answerability.jsonl")
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--sample", type=int, default=2000,
                   help="queries drawn at random; 0 judges every one")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--control", choices=["none", "shuffle"], default="none",
                   help="'shuffle' pairs every query with a different "
                        "document's passage. A judge that still answers DA is "
                        "not discriminating, and a high DA rate on the real "
                        "pairs would mean nothing - this is the arm that makes "
                        "the main number interpretable")
    p.add_argument("--max-passage-chars", type=int, default=4000,
                   help="same cap the triplet builders index at, so the judge "
                        "sees the passage a retriever would have to match")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=320,
                   help="the verdict is the last line, so a tight budget "
                        "silently drops it; 160 lost 7.4% of the first run")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="a verdict should not be sampled")
    p.add_argument("--top-p", type=float, default=1.0)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    print(f"▸ Reading the corpus from {args.categories_dir} …")
    passages: Dict[str, str] = {}
    for doc, doc_type in iter_documents(args.categories_dir,
                                        include_aggregates=args.include_aggregates):
        doc_id = doc.get("doc_id")
        if doc_id and doc_id not in passages:
            passages[doc_id] = document_text(doc, doc_type, args.max_passage_chars)
    print(f"  {len(passages)} documents")

    print(f"▸ Reading queries from {args.queries} …")
    items: List[Dict[str, Any]] = []
    missing = 0
    with open(args.queries, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            passage = passages.get(record.get("doc_id"))
            if not passage:
                missing += 1
                continue
            for query in record.get("queries") or []:
                items.append({
                    "doc_id": record["doc_id"], "query": query,
                    "type": record.get("type", ""),
                    "source": record.get("source", ""),
                    "passage": passage,
                })
    print(f"  {len(items)} queries"
          + (f", {missing} records skipped (document not in corpus)" if missing else ""))

    if args.sample and args.sample < len(items):
        items = random.Random(args.seed).sample(items, args.sample)
        print(f"  judging a random sample of {len(items)}")

    if args.control == "shuffle":
        # Rotate the passages by one so no query keeps its own, and every
        # passage is still a real document rather than noise.
        rng = random.Random(args.seed + 1)
        other = [i["passage"] for i in items]
        rng.shuffle(other)
        for item, passage in zip(items, other):
            if passage == item["passage"] and len(items) > 1:
                passage = next(p for p in other if p != item["passage"])
            item["passage"] = passage
        print("  CONTROL: every query paired with another document's passage")

    # Reuse the generator's backend so the chat template and the double-BOS
    # handling stay in one place.
    sys.argv = [sys.argv[0]]          # VLLMBackend reads an args object, not argv
    from gemma_query_generation import VLLMBackend
    backend = VLLMBackend(args)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    counts = {"answerable": 0, "unanswerable": 0, "unparsed": 0}
    by_type: Dict[str, Dict[str, int]] = {}

    with open(args.output, "w", encoding="utf-8") as out:
        for start in range(0, len(items), args.batch_size):
            chunk = items[start:start + args.batch_size]
            prompts = [PROMPT.format(passage=i["passage"], question=i["query"])
                       for i in chunk]
            replies = backend.generate(prompts)
            for item, reply in zip(chunk, replies):
                verdict = parse(reply)
                bucket = by_type.setdefault(item["type"] or "?",
                                            {"yes": 0, "no": 0, "bad": 0})
                if not verdict["parsed"]:
                    counts["unparsed"] += 1
                    bucket["bad"] += 1
                elif verdict["answerable"]:
                    counts["answerable"] += 1
                    bucket["yes"] += 1
                else:
                    counts["unanswerable"] += 1
                    bucket["no"] += 1
                out.write(json.dumps({
                    "doc_id": item["doc_id"], "query": item["query"],
                    "type": item["type"], "source": item["source"],
                    **verdict,
                }, ensure_ascii=False) + "\n")
            done = start + len(chunk)
            if done % (args.batch_size * 8) == 0 or done == len(items):
                judged = counts["answerable"] + counts["unanswerable"]
                rate = 100 * counts["unanswerable"] / judged if judged else 0
                print(f"    {done}/{len(items)}  unanswerable so far: {rate:.1f}%",
                      flush=True)

    judged = counts["answerable"] + counts["unanswerable"]
    print("\n" + "=" * 66)
    print("ANSWERABILITY OF THE GENERATED QUERIES")
    print("=" * 66)
    print(f"  judge          : {args.model}"
          + ("   [CONTROL: mispaired passages]" if args.control == "shuffle" else ""))
    print(f"  judged         : {judged} of {len(items)}"
          + (f"  ({counts['unparsed']} replies did not parse)" if counts["unparsed"] else ""))
    if judged:
        print(f"  answerable     : {counts['answerable']} ({100 * counts['answerable'] / judged:.1f}%)")
        print(f"  NOT answerable : {counts['unanswerable']} ({100 * counts['unanswerable'] / judged:.1f}%)")
    print("\n  by document type")
    for doc_type, b in sorted(by_type.items(), key=lambda kv: -(kv[1]["yes"] + kv[1]["no"])):
        n = b["yes"] + b["no"]
        if n:
            print(f"    {doc_type:<16} {n:>6} judged, {100 * b['no'] / n:>5.1f}% unanswerable"
                  + (f", {b['bad']} unparsed" if b["bad"] else ""))
    print(f"\n  → {args.output}")
    print("=" * 66)


if __name__ == "__main__":
    main()
