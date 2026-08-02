"""
Generate queries from documents using LLM with V1, V2, V3 prompts.

Task 1: Query Generation
- Input: all_documents_combined.jsonl
- Output: data/queries_meeting1.jsonl
- Process: LLM generates 2-3 queries per document
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_jsonl, save_jsonl, ensure_dir
from llm_client import get_client


# ============================================================
# PROMPT TEMPLATES V1, V2, V3
# ============================================================

PROMPT_V1_SIMPLE = """
Ești un asistent care generează întrebări de căutare (queries) pentru antrenarea unui model de retrieval.

Titlu: {title}
Articol: {content}

Generează 2-3 întrebări DIVERSE care ar putea fi puse pentru a găsi acest articol.

Reguli:
- Variază: tipuri diferite de întrebări (factuale, semantice, cauzale)
- Variază: lungimi diferite (scurtă, lungă)
- NU copia fraze exact din articol
- Dacă nu poți genera 2-3 diverse, generează doar 1 bună

Output format (JSON):
{{
  "queries": ["intrebare1", "intrebare2"]
}}

Răspuns:
"""

PROMPT_V2_GROUNDED = """
Ești un asistent care generează întrebări de căutare (queries) pornind de la un articol,
pentru a antrena un model de retrieval semantic.

Titlu: {title}
Articol: {content}

Generează 2-3 întrebări care ar putea fi puse de un utilizator ÎNAINTE de a citi acest articol,
nu DUPĂ. Adică întrebarea trebuie să reflecte o nevoie de informație mai largă, contextuală,
nu un detaliu îngust care apare doar în acest text.

Reguli stricte:
- NU genera întrebări de tip "text-based retrieval" (ex: "câți oameni...?", "ce cifră...?")
- Preferă întrebări "grounded" despre context, cauze, actori, implicații, comparații
- Dacă nu poți genera 2-3 diverse, generează doar 1 bună
- NU copia fraze din articol; reformulează complet
- Variază: o întrebare factuală (dar non-trivială) și una semantică/interpretativă

Output format (JSON):
{{
  "queries": ["intrebare1", "intrebare2"]
}}

Răspuns:
"""

PROMPT_V3_TITLE_AS_QUESTION = """
Ești un asistent care transformă titlurile articolelor în întrebări de căutare.

Titlu: {title}
Articol (pentru context): {content}

Transformă titlul în 1-2 întrebări naturale pe care un utilizator le-ar putea pune
pentru a găsi informații pe această temă.

Reguli:
- Reformulează titlul ca întrebare naturală
- Poate fi mai general decât titlul original
- Fii concis și direct

Output format (JSON):
{{
  "queries": ["intrebare1"]
}}

Răspuns:
"""

PROMPT_V2_MULTI_DOC = """
Ai mai jos titlul și articolul original + 1-2 articole similare.
Generează întrebări suficient de generale încât să fie relevante pentru toate,
nu doar pentru articolul original.

Titlu: {title}
Articol original: {content}

Articole similare (context suplimentar):
{similar_articles}

Generează 2-3 întrebări care:
- Surprind tema comună tuturor articolelor
- NU sunt de tip "text-based retrieval"
- Sunt reformulate natural, nu copiate din text
- Variază ca tip (factual non-trivial / semantic) și ca lungime
- Dacă nu poți genera 2-3 diverse, generează doar 1 bună

Output format (JSON):
{{
  "queries": ["intrebare1", "intrebare2"]
}}

Răspuns:
"""


# Prompt registry: version key -> (template, version label, content char limit)
PROMPTS = {
    "v1": (PROMPT_V1_SIMPLE, "v1", 2000),
    "v2": (PROMPT_V2_GROUNDED, "v2_grounded", 2000),
    "v3": (PROMPT_V3_TITLE_AS_QUESTION, "v3_title", 1000),
}

# Backwards-compatible alias used by src/studies/marco_methods/.
PROMPT_V1_TEMPLATE = PROMPT_V1_SIMPLE


def build_prompt(doc: Dict[str, Any], version: str = "v1") -> str:
    """Fill a prompt template with a document's title and content."""
    template, _, limit = PROMPTS.get(version, PROMPTS["v1"])
    return template.format(
        title=doc.get("title", ""),
        content=doc.get("content", "")[:limit],
    )


class QueryGenerator:
    """
    Generate queries from documents using an LLM.

    The API key lives in .env -- see src/llm_client.py. If no key is set, the
    generator still builds the prompts and returns records with
    status="no_llm", so the pipeline runs end-to-end without credentials.
    """

    def __init__(self, model: Optional[str] = None, temperature: float = 0.8,
                 provider: Optional[str] = None):
        self.temperature = temperature
        self.client = get_client(model=model, temperature=temperature,
                                 provider=provider)
        self.model = self.client.model

    @property
    def available(self) -> bool:
        return self.client.available

    def describe(self) -> str:
        return self.client.describe()

    def generate(self, doc: Dict[str, Any], version: str = "v1",
                 temperature: Optional[float] = None) -> Dict[str, Any]:
        """Generate queries for one document with one prompt version."""
        _, label, _ = PROMPTS.get(version, PROMPTS["v1"])
        temp = self.temperature if temperature is None else temperature
        prompt = build_prompt(doc, version)

        queries = self.client.generate_queries(prompt, temperature=temp)

        return {
            "doc_id": doc.get("doc_id"),
            "title": doc.get("title"),
            "source": doc.get("source"),
            "version": label,
            "temperature": temp,
            "model": self.model,
            "queries": queries,
            "num_queries": len(queries),
            "status": "ok" if queries else ("failed" if self.available else "no_llm"),
        }

    # Kept for backwards compatibility with the original API.
    def generate_queries_v1(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Generate queries using the V1 simple prompt."""
        return self.generate(doc, "v1")

    def generate_queries_v2(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Generate queries using the V2 grounded prompt."""
        return self.generate(doc, "v2")

    def generate_queries_v3(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Generate queries using the V3 title-as-question prompt."""
        return self.generate(doc, "v3")


class SampleQueryGenerator:
    """Generate sample queries for testing without LLM."""
    
    @staticmethod
    def generate_sample_queries(doc: Dict[str, Any], version: str = "v1") -> List[str]:
        """Generate sample queries for demonstration."""
        title = doc.get("title", "")
        content = doc.get("content", "")
        source = doc.get("source", "")
        
        samples = {
            "v1": [
                f"Ce este {title.lower()}?",
                f"Cum funcționează {title.lower()}?",
            ],
            "v2": [
                f"Care sunt implicațiile {title.lower()}?",
                f"De ce este important {title.lower()}?",
            ],
            "v3": [
                f"{title}?",
            ]
        }
        
        return samples.get(version, samples["v1"])[:2]  # Return 1-2 diverse queries


def generate_sample_queries(input_file: str = "data/corpus/all_documents_combined.jsonl",
                           output_file: str = "data/queries_sample.jsonl",
                           num_samples: int = 5):
    """Generate sample queries from first N documents."""
    
    print("\n" + "="*60)
    print("  Task 1: Sample Query Generation")
    print("="*60)
    
    ensure_dir(os.path.dirname(output_file))
    
    # Load documents
    docs = load_jsonl(input_file)
    print(f"\n📄 Loaded {len(docs)} documents")
    
    # Generate sample queries
    output = []
    for i, doc in enumerate(docs[:num_samples]):
        print(f"\n[{i+1}/{num_samples}] {doc.get('doc_id')} - {doc.get('title')[:50]}")
        
        # Generate V1, V2, V3
        for version in ["v1", "v2", "v3"]:
            queries = SampleQueryGenerator.generate_sample_queries(doc, version)
            
            record = {
                "doc_id": doc.get("doc_id"),
                "title": doc.get("title"),
                "source": doc.get("source"),
                "version": version,
                "queries": queries,
                "num_queries": len(queries)
            }
            output.append(record)
            print(f"  {version}: {len(queries)} queries generated")
    
    # Save output
    save_jsonl(output, output_file)
    
    print(f"\n Saved {len(output)} query records to {output_file}")
    
    # Print sample
    print(f"\n Sample output (first 2 records):")
    for record in output[:2]:
        print(f"\n{json.dumps(record, ensure_ascii=False, indent=2)}")
    
    print("\n" + "="*60)


def generate_llm_queries(input_file: str = "data/corpus/all_documents_combined.jsonl",
                         output_file: str = "data/queries_generated.jsonl",
                         versions: Optional[List[str]] = None,
                         temperatures: Optional[List[float]] = None,
                         limit: Optional[int] = None,
                         model: Optional[str] = None,
                         provider: Optional[str] = None):
    """
    Generate queries for every document with a real LLM.

    Needs an API key in .env -- see src/llm_client.py. Writes one record per
    (document, prompt version, temperature).
    """
    from tqdm import tqdm

    versions = versions or ["v1", "v2", "v3"]
    temperatures = temperatures or [0.8]

    print("\n" + "=" * 60)
    print("  Task 1: LLM Query Generation")
    print("=" * 60)

    generator = QueryGenerator(model=model, provider=provider)
    print(f"\n  {generator.describe()}")

    if not generator.available:
        print("\n  Nothing to do without an API key. To enable:")
        print("    1. cp .env.example .env")
        print("    2. add ANTHROPIC_API_KEY=sk-ant-...  (or OPENAI_API_KEY=sk-...)")
        print("\n  Meanwhile you can run the offline sample generator:")
        print("    python src/task1_prompt.py --sample --num 10\n")
        return

    docs = load_jsonl(input_file)
    if limit:
        docs = docs[:limit]

    total = len(docs) * len(versions) * len(temperatures)
    print(f"\n  Documents: {len(docs)}")
    print(f"  Versions:  {versions}")
    print(f"  Temps:     {temperatures}")
    print(f"  LLM calls: {total}\n")

    output = []
    ok = 0
    with tqdm(total=total, desc="Generating queries") as bar:
        for doc in docs:
            for version in versions:
                for temp in temperatures:
                    record = generator.generate(doc, version, temperature=temp)
                    output.append(record)
                    ok += record["status"] == "ok"
                    bar.update(1)

    save_jsonl(output, output_file)
    print(f"\n  Saved {len(output)} records ({ok} successful) to {output_file}")
    print("=" * 60 + "\n")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Task 1: Query Generation",
        epilog="Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env to use --llm."
    )
    parser.add_argument('--sample', action='store_true',
                        help='Generate offline placeholder queries (no API key needed)')
    parser.add_argument('--llm', action='store_true',
                        help='Generate real queries with an LLM (needs an API key in .env)')
    parser.add_argument('--num', type=int, default=5, help='Number of samples')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only process the first N documents (--llm)')
    parser.add_argument('--versions', type=str, default='v1,v2,v3',
                        help='Comma-separated prompt versions (--llm)')
    parser.add_argument('--temperatures', type=str, default='0.8',
                        help='Comma-separated temperature values (--llm)')
    parser.add_argument('--model', type=str, default=None,
                        help='Override the model id (default: from .env)')
    parser.add_argument('--provider', type=str, default=None,
                        choices=['anthropic', 'openai'],
                        help='Force a provider (default: auto-detect from .env)')
    parser.add_argument('--input', type=str, default='data/corpus/all_documents_combined.jsonl')
    parser.add_argument('--output', type=str, default=None)

    args = parser.parse_args()

    if args.llm:
        generate_llm_queries(
            input_file=args.input,
            output_file=args.output or 'data/queries_generated.jsonl',
            versions=[v.strip() for v in args.versions.split(',') if v.strip()],
            temperatures=[float(t) for t in args.temperatures.split(',') if t.strip()],
            limit=args.limit,
            model=args.model,
            provider=args.provider,
        )
    elif args.sample:
        generate_sample_queries(args.input, args.output or 'data/queries_sample.jsonl', args.num)
    else:
        client = get_client()
        print("\n" + "=" * 60)
        print("  Task 1: Query Generation")
        print("=" * 60)
        print(f"\n  {client.describe()}")
        print("\n  Usage:")
        print("    python src/task1_prompt.py --sample --num 10   # offline placeholders")
        print("    python src/task1_prompt.py --llm --limit 20    # real LLM queries")
        print("\n  To enable the LLM:")
        print("    1. cp .env.example .env")
        print("    2. add ANTHROPIC_API_KEY=sk-ant-...  (or OPENAI_API_KEY=sk-...)")
        print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
