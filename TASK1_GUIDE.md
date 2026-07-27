# Task 1: Query Generation - Complete Guide

## 📋 Overview

Generate diverse queries from 858 documents (recipes, news, CC) using 3 prompt versions:
- **V1**: Simple (title + content → 2-3 queries)
- **V2**: Grounded (discourage text-based retrieval queries)
- **V3**: Title-as-Question (transform title → query)

---

## 🔧 Prompt Templates

### PROMPT_V1_SIMPLE
```
Input: title + content
Output: 2-3 diverse queries
Rules:
  - Vary: factual, semantic, causal
  - Vary: short, long
  - NO exact phrases
  - If can't make 2-3 diverse → make 1 good
```

**Example:**
```
Titlu: Cum să economisești energetic acasă
Output:
  - Ce sunt becurile LED?
  - Cum funcționează termostat inteligent?
```

---

### PROMPT_V2_GROUNDED
```
Input: title + content
Output: 2-3 grounded queries (contextual, not text-based retrieval)
Rules:
  - ❌ NO "câți oameni...?" (text-based retrieval)
  - ✅ YES "Care sunt implicații...?" (grounded)
  - Questions = things user asks BEFORE reading, not AFTER
```

**Example - GOOD:**
```
"Care sunt factori care influențează decizia autorităților?"
"De ce climă este importantă energia regenerabilă?"
```

**Example - BAD:**
```
"Câți MW produce parcul eolian?" ❌ (text-based)
"Care este titlul articolului?" ❌ (too specific)
```

---

### PROMPT_V3_TITLE_AS_QUESTION
```
Input: title + content (for context)
Output: 1-2 natural questions from title
Rules:
  - Reformulate title as natural question
  - Can be more general than title
  - Concise, direct
```

**Example:**
```
Titlu: "Telemedicina devine mai accesibilă"
Output: "Cum mă pot consulta cu medic online?"
```

---

### PROMPT_V2_MULTI_DOC (Future)
```
Input: title + content + 1-2 similar articles
Output: 2-3 queries relevant for ALL articles
Rules:
  - Capture common theme
  - NO text-based retrieval
  - NO details specific to one article
```

---

## 🚀 Usage

### Generate Samples (Test)
```bash
python src/task1_prompt.py --sample --num 5
python src/task1_prompt.py --sample --num 10
```

**Output:** `data/queries_sample.jsonl`
```json
{
  "doc_id": "recipe_000000",
  "version": "v1",
  "queries": ["Ce este...?", "Cum...?"],
  "num_queries": 2
}
```

---

## 🤖 LLM Integration: OpenAI vs Claude

### OpenAI (Chosen for this project)
- **Model:** `gpt-4` (or `gpt-4-turbo`, `gpt-3.5-turbo` for cost)
- **Temperature:** 0.7-1.0 (diversity)
- **Cost:** ~$0.03/1K tokens (gpt-4)
- **Speed:** Fast
- **Setup:**
  ```bash
  pip install openai
  echo "OPENAI_API_KEY=sk-..." >> .env
  ```

### Claude (Alternative)
- **Model:** `claude-3-sonnet`, `claude-3-opus`
- **Temperature:** 0.5-1.0
- **Cost:** ~$0.003/1K tokens (Sonnet)
- **Speed:** Slower
- **Setup:**
  ```bash
  pip install anthropic
  echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
  ```

### Comparison
| Feature | OpenAI | Claude |
|---------|--------|--------|
| Cost | $$$ | $ |
| Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Romanian | Good | Good |
| Speed | Fast | Slower |
| Error handling | Good | Excellent |

**Recommendation:** Start with OpenAI (gpt-3.5-turbo for cost), then try Claude if budget allows.

---

## 📊 Data Sources

### Current (858 docs)
- ✅ Recipes: 818 (HuggingFace)
- ✅ News: 20 (manual)
- ✅ Common Crawl: 20 (manual)

### Next Steps
- [ ] Download more news from HuggingFace CC
- [ ] Find alternative Romanian news datasets
- [ ] Expand to 1000+ documents

### HuggingFace Datasets
```
- microsoft/ms_marco (English, adapt)
- google-research-datasets/natural-questions (English)
- mc4 (Common Crawl in multiple languages)
```

### Local News Download
If GitHub clone fails, manual steps:
```bash
# 1. Download dataset manually
cd data/raw/
wget https://raw.githubusercontent.com/mhakan20/RomanianNewsArticlesDataset/master/articles.json

# 2. Convert to JSONL
python scripts/convert_news.py data/raw/articles.json data/corpus/news_ro.jsonl

# 3. Format with add_manual_data.py
```

---

## 📝 Implementation Checklist

- [x] Prompt templates (V1, V2, V3)
- [x] Sample query generation
- [ ] OpenAI API integration
- [ ] Embedding computation (multilingual-e5-large)
- [ ] Find similar documents (V2_MULTI_DOC)
- [ ] Batch query generation
- [ ] Output validation (JSON check)
- [ ] Temperature variation (0.5, 0.7, 0.9, 1.0)

---

## 🔄 Full Pipeline (Draft)

```
1. Load all_documents_combined.jsonl
2. For each doc:
   - Generate V1 queries (simple)
   - Generate V2 queries (grounded)
   - [Optional] Generate V3 queries (title)
3. Save to queries_meeting1.jsonl
4. Validate output
5. Move to Task 2 (Triplet Creation)
```

---

## 🎯 Next: Full LLM Implementation

```python
# pseudocode
for doc in documents:
    for version in ["v1", "v2", "v3"]:
        for temp in [0.5, 0.7, 0.9, 1.0]:
            prompt = get_prompt(version, doc)
            response = call_openai(prompt, temperature=temp)
            queries = parse_json(response)
            save_queries(doc_id, version, temp, queries)
```

---

## 📦 Dependencies

```
sentence-transformers>=2.2.0
openai>=1.0.0
anthropic>=0.7.0 (optional)
```

---

## 📧 Questions?

- **V2 Grounded:** Why avoid text-based retrieval?
  → For dense retrieval, we want semantic queries, not factoid-seeking
- **Multiple temps:** Why 0.5-1.0?
  → Low temp = deterministic, High temp = diverse
- **3 versions:** Why not just 1?
  → Diversity = better training data for retrieval

