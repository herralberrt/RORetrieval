# Project Structure - RORetrieval

## Directory Organization

```
src/
├── __init__.py
├── utils/                      # Shared utilities
│   ├── __init__.py
│   └── utils.py               # save_jsonl, load_jsonl, ensure_dir, get_config
│
├── data_prep/                 # Data preparation & downloading
│   ├── __init__.py
│   ├── download_datasets.py   # Main data downloader (recipes, stories, ner, news, etc)
│   ├── data_splitter.py       # Train/val/test splitting
│   └── add_manual_data.py     # Add custom data
│
├── task1_queries/             # Query generation with quality metrics
│   ├── __init__.py
│   ├── task1_query_generation_with_metrics_fast.py  # MAIN: Generate queries (631k queries)
│   ├── analyze_query_metrics.py                      # Analyze quality distribution
│   └── show_concept_overlap_examples.py              # Show concept overlap analysis
│
├── task2_triplets/            # Triplet generation & filtering
│   ├── __init__.py
│   ├── triplet_generator.py   # Generate (query, positive, negative) triplets
│   ├── task2_triplets.py      # Task 2: Triplet creation
│   └── task3_filter.py        # Task 3: Filter & validate triplets
│
├── training/                  # Model training
│   ├── __init__.py
│   ├── train_model.py         # Main training script
│   ├── train_model_fintuned.py # Fine-tuning variant
│   └── train_model_v2.py      # Version 2 training
│
├── evaluation/                # Model evaluation & IR pipeline
│   ├── __init__.py
│   ├── evaluate_models.py               # Evaluate trained models
│   ├── evaluate_models_multilingual.py  # Multilingual evaluation
│   ├── model_comparison.py              # Compare models
│   └── ir_pipeline.py                   # Information retrieval pipeline
│
├── indexing/                  # Vector indexing
│   ├── __init__.py
│   └── faiss_indexer.py       # FAISS index creation & search
│
└── studies/                   # Research & external datasets
    ├── marco_methods/        # MS MARCO methods
    ├── miracl/              # MIRACL evaluation
    └── ms_macro/            # MS MACRO dataset handling
```

## Key Scripts & Usage

### 1. Data Preparation
```bash
# Download & prepare datasets
python3 -m src.data_prep.download_datasets --dataset all

# Split into train/val/test
python3 -m src.data_prep.data_splitter
```

### 2. Query Generation (CURRENT PHASE) ✅
```bash
# Generate 631k queries with quality metrics
python3 -m src.task1_queries.task1_query_generation_with_metrics_fast

# Analyze metrics distribution
python3 -m src.task1_queries.analyze_query_metrics

# Show concept overlap examples
python3 -m src.task1_queries.show_concept_overlap_examples
```

**Output Metrics:**
- Quality scores (0-1): avg_quality by document type
  - News: 0.6066 ± 0.0289
  - Recipes: 0.6189 ± 0.0203
  - NER: 0.5117 ± 0.0008
  - Stories: 0.4676 ± 0.0048
- Diversity (0-1): query variation within document set
- Concept overlap: shared concepts across documents

### 3. Triplet Generation (NEXT PHASE)
```bash
python3 -m src.task2_triplets.triplet_generator
python3 -m src.task2_triplets.task2_triplets
python3 -m src.task2_triplets.task3_filter
```

### 4. Model Training
```bash
python3 -m src.training.train_model
python3 -m src.training.train_model_fintuned
```

### 5. Evaluation
```bash
python3 -m src.evaluation.evaluate_models
python3 -m src.evaluation.ir_pipeline
```

## Data Files

- `data/queries/queries_with_metrics.jsonl` - 631,269 queries with quality scores
- `data/categories/` - 164,044 documents (news, stories, recipes, ner)
- `results/evaluation/` - Evaluation metrics
- `checkpoints/` - Model checkpoints

## Removed Files

Cleaned up in this reorganization:
- ❌ `task1_prompt.py` - Old basic query generator
- ❌ `task1_query_generation.py` - Replaced by metrics version
- ❌ `task1_query_generation_with_metrics.py` - Slow TF-IDF version
- ❌ `quality_metrics.py` - Old metrics
- ❌ `quality_metrics_fixed.py` - Deprecated

## Import Path

When running scripts directly, add `src/` to Python path:

```python
import sys
sys.path.insert(0, 'src')

from task1_queries import task1_query_generation_with_metrics_fast
from utils import save_jsonl, load_jsonl
from data_prep import download_datasets
```

Or run as module:
```bash
python3 -m src.task1_queries.task1_query_generation_with_metrics_fast
```
