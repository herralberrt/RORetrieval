# RORetrieval

Training data for Romanian retrieval models, and an honest measurement of what it
does to them.

The pipeline takes a 117k-document Romanian corpus, has Gemma 3 27B write a query
for each document, mines hard negatives with two different retrievers, and
fine-tunes two embedding models on the result. Both triplet sets are published on
the HuggingFace Hub.

## The headline result

Fine-tuning on these triplets **improves in-domain retrieval and degrades
out-of-domain retrieval**, consistently, across every configuration tested.

| | In-domain (our test split) | Out-of-domain (`ro-msmarco-divided`) |
|---|---|---|
| bge-m3, zero-shot | 0.838 | 0.707 |
| bge-m3, fine-tuned | 0.905 | 0.610 |
| Qwen3-Embedding-8B, zero-shot | 0.866 | 0.732 |
| Qwen3-Embedding-8B, fine-tuned | 0.928 | 0.652 |

nDCG@10. Training gains 0.03–0.07 in-domain and loses 0.07–0.10 out of it. No
configuration is neutral.

If the goal is a general-purpose Romanian retriever, these triplets — trained this
way — make it worse. If the goal is this domain (news, folk stories, recipes), they
improve it consistently.

### What was ruled out

The obvious explanation was catastrophic forgetting from a full one-epoch
fine-tune at 2e-5. That hypothesis is **wrong**, and the control says so:

- **LoRA control.** bge-m3 retrained through LoRA, touching 1.25% of parameters
  instead of 100%, same data, everything else identical. In-domain gain identical
  (0.906 vs 0.905); out-of-domain loss identical. The training recipe is not the
  cause.
- **Sequence-length control.** Re-run at sequence length 1024 with gradient
  checkpointing. Same pattern, so truncation is not the cause either.
- **Answerability.** 2,000 sampled (query, positive) pairs were checked by an LLM
  against the document text: **99.45% are genuinely answerable** from their
  positive. The positives are not the problem.

What remains is the data distribution itself and the base model, not the recipe.

## Pipeline

**1. Corpus.** 117,313 documents from Romanian news outlets (Adevărul, Mediafax,
ProTV, Digi24, ZF, Libertatea, Cotidianul, EVZ, Realitatea, Aleph), plus folk
stories, recipes and a summarisation corpus. After near-duplicate grouping and
boilerplate removal: 88,626 distinct texts, 87,170 indexed.

**2. Query generation.** Gemma 3 27B Instruct reads each document and writes
queries from its content — not from templates. Runs inside an Apptainer image on a
SLURM cluster, bounded to a 2-hour slot with a clean stop and resume, because an
80GB A100 is the minimum for the bf16 27B checkpoint.

**3. Triplet mining.** The positive is known by construction — the query was
generated from that document — so retrieval is only used for hard negatives. Two
independent paths:

- **BM25** (`build_triplets_bm25.py`): an Okapi BM25 inverted index in numpy, no
  GPU. Negatives that share the query's rare terms.
- **Late interaction** (`build_triplets_colbert.py`): real MaxSim over token
  vectors — one vector per token, each query token matched to its best document
  token.

The two sets are nearly disjoint: **Jaccard overlap of 0.053 between their
negatives, and 73% of shared queries have no negative in common.** That was the
condition set before publishing the second set — had they produced similar
negatives, the second would not have been worth having.

**4. Training and evaluation.** bge-m3 (568M) and Qwen3-Embedding-8B (7.6B),
evaluated zero-shot, then fine-tuned on each set and re-evaluated, in-domain and
against `alina0195/ro-msmarco-divided`. 26 runs in total, all recorded in
`results/retrieval_eval.jsonl`.

## Published datasets

| Dataset | Rows | Negatives from |
|---|---|---|
| [`PaulBurca2005/ro-retrieval-triplets`](https://huggingface.co/datasets/PaulBurca2005/ro-retrieval-triplets) | 80,363 | BM25 |
| [`PaulBurca2005/ro-retrieval-triplets-late-interaction`](https://huggingface.co/datasets/PaulBurca2005/ro-retrieval-triplets-late-interaction) | 80,928 | MaxSim token interaction |

Four columns each — `anchor`, `positive`, `negative`, `query_source` — following
the `alina0195/ro-msmarco-divided` layout, with texts rather than document ids and
one row per negative. Splits are grouped by duplicate group, so near-identical
articles cannot straddle train and test.

## Repository layout

```
src/
├── data_prep/          Corpus download, manual additions, splitting
├── task1_queries/      Gemma 3 query generation, quality metrics,
│                       answerability validation, neighbour maps
├── task2_triplets/     BM25 and late-interaction mining, filtering,
│                       HF export and upload, inspection
├── training/           Fine-tuning, including the LoRA path
├── evaluation/         Retrieval evaluation, model comparison, IR pipeline
├── indexing/           FAISS index building and search
├── reporting/          HTML report to .docx, standard library only
└── studies/            MS MARCO and MIRACL side studies

containers/             Apptainer image (CUDA 12.4 + torch + vLLM + Gemma)
scripts/slurm/          SLURM jobs for every stage
data/categories/        The corpus, via Git LFS
data/triplets/          Triplet statistics and readable samples
results/                Evaluation runs, answerability check, report
```

## Running it

The heavy stages need a GPU cluster. Everything is driven through SLURM scripts
that read their configuration from the environment.

```bash
git lfs pull                                   # the corpus is 546 MB via LFS
pip install -r requirements.txt

bash containers/build.sh                       # build roretrieval.sif (needs fakeroot)
cp containers/env.example containers/env.sh    # add your HF_TOKEN
source containers/env.sh
mkdir -p logs

sbatch --partition=<gpu> scripts/slurm/generate_queries.sbatch
sbatch --partition=<cpu> scripts/slurm/build_triplets_bm25.sbatch
sbatch --partition=<gpu> scripts/slurm/finetune_embedder.sbatch
sbatch --partition=<gpu> scripts/slurm/evaluate_retrieval.sbatch
```

Gemma 3 is gated: accept the licence for
[`google/gemma-3-27b-it`](https://huggingface.co/google/gemma-3-27b-it) and create a
read token. `containers/env.sh` is gitignored — never commit it.

The BM25 path needs no GPU and runs in roughly three and a half minutes on the
plain CPU partition: 69,800 queries against an 87,170-document index.

Inspect the prompts without loading a model, anywhere:

```bash
python3 -m src.task1_queries.gemma_query_generation --dry-run
```

## Documentation

- [`STRUCTURE.md`](STRUCTURE.md) — every module, what it does, and how to invoke it
- [`data/triplets/README.md`](data/triplets/README.md) — the triplet dataset in
  detail: record format, how the 69,800 queries became 21,891 triplets, how
  negatives are chosen, measured distributions, known limitations, and what the
  30 August review changed
- [`containers/README.md`](containers/README.md) — building and running the image,
  sizing, and cluster troubleshooting
- [`results/raport_roretrieval.docx`](results/raport_roretrieval.docx) — the full
  written report, in Romanian

## Branches

`gemma3-27b` is the working branch and carries the current pipeline, the published
datasets and all evaluation results. `main` predates them.
