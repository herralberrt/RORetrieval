# marco_methods

Generates a **Romanian** retrieval dataset using the **MS MARCO** methodology
([microsoft/ms_marco](https://huggingface.co/datasets/microsoft/ms_marco)),
applied to the local RORetrieval corpus. Every record is tagged
`"language": "ro"` and all queries/answers are in Romanian.

## The MS MARCO method

MS MARCO builds each example by:

1. Starting from a **real user query**.
2. **Retrieving candidate passages** for that query from a large corpus
   (the original uses the top Bing web results).
3. **Labelling** which passages answer the query (`is_selected = 1`) vs. those
   that are only topically related (`is_selected = 0`).
4. Attaching a short **answer** and a **well-formed** answer variant.

## How we reproduce it here

| MS MARCO step        | This implementation                                            |
|----------------------|----------------------------------------------------------------|
| Real query           | Source-aware **Romanian** query per document (recipes → *"cum se prepară…"*, news → *"ce s-a întâmplat cu…"*). Optional LLM path via `--use_llm` uses the Romanian prompt in `src/task1_prompt.py`. |
| Retrieve candidates  | BM25 over the corpus (falls back to lexical overlap)           |
| `is_selected` labels | Source document = 1; other retrieved candidates = 0            |
| answer / wellFormed  | First sentence of the selected passage, rewritten to a full sentence |

## Output schema (MS MARCO v2.1)

```json
{
  "query_id": 0,
  "query": "cum se prepară ciorba de burtă?",
  "query_type": "description",
  "language": "ro",
  "passages": {
    "passage_text": ["...", "..."],
    "is_selected": [1, 0],
    "url": ["", ""]
  },
  "answers": ["..."],
  "wellFormedAnswers": ["..."]
}
```

## Usage

```bash
# 1. Build the corpus (if not already done)
python src/download_datasets.py --combine

# 2. Generate the Romanian MS MARCO-style dataset
python marco_methods/generate_marco_data.py \
    --corpus data/corpus/all_documents_numbered.jsonl \
    --output data/marco/marco_data.jsonl \
    --num_passages 10 \
    --language ro
```

- `--use_llm` — generate queries with an LLM using the Romanian prompt in
  `src/task1_prompt.py` (needs `OPENAI_API_KEY`; falls back to templates otherwise).
- `--limit N` — quick test on the first N documents.
