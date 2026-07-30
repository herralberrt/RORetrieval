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
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_jsonl, save_jsonl, ensure_dir


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


class QueryGenerator:
    """Generate queries from documents using LLM."""
    
    def __init__(self, model: str = "gpt-4", temperature: float = 0.8):
        """Initialize query generator."""
        self.model = model
        self.temperature = temperature
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.org_id = os.getenv("OPENAI_ORG_ID")
    
    def generate_queries_v1(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Generate queries using V1 simple prompt."""
        prompt = PROMPT_V1_SIMPLE.format(
            title=doc.get("title", ""),
            content=doc.get("content", "")[:2000]  # Limit content
        )
        return {
            "doc_id": doc.get("doc_id"),
            "version": "v1",
            "temperature": self.temperature,
            "prompt": prompt,
            "queries": [],  # Will be filled by LLM
            "status": "pending"
        }
    
    def generate_queries_v2(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Generate queries using V2 grounded prompt."""
        prompt = PROMPT_V2_GROUNDED.format(
            title=doc.get("title", ""),
            content=doc.get("content", "")[:2000]
        )
        return {
            "doc_id": doc.get("doc_id"),
            "version": "v2_grounded",
            "temperature": self.temperature,
            "prompt": prompt,
            "queries": [],
            "status": "pending"
        }
    
    def generate_queries_v3(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Generate queries using V3 title-as-question prompt."""
        prompt = PROMPT_V3_TITLE_AS_QUESTION.format(
            title=doc.get("title", ""),
            content=doc.get("content", "")[:1000]
        )
        return {
            "doc_id": doc.get("doc_id"),
            "version": "v3_title",
            "temperature": self.temperature,
            "prompt": prompt,
            "queries": [],
            "status": "pending"
        }


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


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Task 1: Query Generation")
    parser.add_argument('--sample', action='store_true', help='Generate sample queries')
    parser.add_argument('--num', type=int, default=5, help='Number of samples')
    parser.add_argument('--input', type=str, default='data/corpus/all_documents_combined.jsonl')
    parser.add_argument('--output', type=str, default='data/queries_sample.jsonl')
    
    args = parser.parse_args()
    
    if args.sample:
        generate_sample_queries(args.input, args.output, args.num)
    else:
        print("\n" + "="*60)
        print("  Task 1: Query Generation - Meeting 1")
        print("="*60)
        print("\n Usage:")
        print("   python task1_prompt.py --sample          # Generate sample queries")
        print("   python task1_prompt.py --sample --num 10 # 10 samples")
        print("\n  Full LLM integration:")
        print("   - Set OPENAI_API_KEY in .env")
        print("   - Requires: sentence-transformers, openai")
        print("   - Next: Implement QueryGenerator.call_llm()")
        print("\n" + "="*60)


if __name__ == '__main__':
    main()
