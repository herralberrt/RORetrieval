# miracl

Complete Information Retrieval pipeline for the **MIRACL** multilingual corpus
([google-research-datasets/miracl](https://huggingface.co/datasets/google-research-datasets/miracl)).
Downloads, preprocesses, and orchestrates retrieval tasks.

> ⚠️ **MIRACL does not cover Romanian.** The 18 languages are:
> `ar bn de en es fa fi fr hi id ja ko ru sw te th yo zh`.
> Requesting `--language ro` fails with an explicit error. Use this study to
> compare methodology against an established multilingual benchmark; for actual
> Romanian data use the local corpus in `data/corpus/`.
>
> Note also that the official `miracl/miracl-corpus` and `miracl/miracl` repos are
> *script-based* HF datasets, which `datasets` 3.0+ refuses to load. The downloader
> therefore fetches the underlying `.jsonl.gz` / `.tsv` files directly via
> `huggingface_hub`, which works on any `datasets` version.

## The MIRACL dataset

MIRACL provides:

1. **Multilingual documents** — large corpus in 18 languages
2. **User queries** — naturally-formed questions with relevance judgments
3. **Relevance judgments (qrels)** — human-annotated relevance labels (relevant / not relevant)
4. **Train/val/test splits** — ready-to-use partitions for evaluation

## Pipeline architecture

| Stage | Component | Output |
|-------|-----------|--------|
| **Download** | `MIRACLDownloader` | Corpus, queries, qrels from HuggingFace |
| **Preprocess** | `MIRACLPreprocessor` | Standardized JSONL schema, train/val/test splits |
| **Orchestrate** | `MIRACLPipeline` | Logging, task management, full workflow automation |

## Output schema

```json
{
  "doc_id": "doc_12345",
  "title": "...",
  "content": "...",
  "source": "miracl",
  "language": "ro",
  "original_id": 12345
}
```

## Usage

```bash
cd /src/studies/miracl

./run.sh full
```

**Modes:**
- `full` — Download + preprocess + run tasks
- `preprocess-only` — Preprocess without download
- `tasks-only` — Run tasks only (skip download/preprocess)

**Configuration:**
Edit `config.json` to customize:
- Language (one of the 18 MIRACL languages above — not `ro`)
- Data/results directories
- Task parameters (LLM prompts, similarity thresholds, filtering metrics)

## Dependencies

```bash
pip install -r requirements.txt
```

Requires: datasets, jsonlines, tqdm, scikit-learn, numpy, pandas

## Tasks

The pipeline supports three sequential tasks (placeholders for integration):

- **Task 1** — LLM-based query generation (uses src/task1_prompt.py). Runs for real
  once an API key is in `.env` (see `src/llm_client.py`); skipped with a warning otherwise.
- **Task 2** — Triplet mining with hard negatives (uses src/task2_triplets.py)
- **Task 3** — Quality filtering (uses src/task3_filter.py)

Enable/disable in config.json under `run_task1`, `run_task2`, `run_task3`.
