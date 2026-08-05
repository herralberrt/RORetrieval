import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import sys
import random

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_jsonl, save_jsonl, ensure_dir
from faiss_indexer import FaissIndexer


class QueryGenerator:
    
    def __init__(self, llm_enabled: bool = False):
        self.llm_enabled = llm_enabled
        self.query_templates = [
            "Cum se {action}?",
            "Ce este {topic}?",
            "Care sunt beneficiile {topic}?",
            "Cum se face {action}?",
            "Unde pot găsi informații despre {topic}?",
            "Care este procedura pentru {action}?",
            "Ce ar trebui să știu despre {topic}?",
            "Cum aleg cel mai bun {topic}?",
            "Care sunt pașii pentru {action}?",
            "De ce este important {topic}?",
        ]
        
        self.actions = [
            "prepară ciorba", "construiește o casă", "învață programare",
            "pornește o afacere", "gătești o masă", "plantezi grădina",
            "reparezi o mașină", "studiezi biologia", "dai o prezentare",
            "escribi un articol"
        ]
        
        self.topics = [
            "technologia", "sănătatea", "economie", "educație", "mediu",
            "transporturi", "comunicare", "leadership", "inovație", "calitate"
        ]
    
    def generate_query_from_doc(self, doc: Dict[str, Any], template_idx: int = 0) -> str:
        title = doc.get("title", "").lower()
        content = doc.get("content", "").lower()
        
        if title:
            words = title.split()[:3]
            topic = " ".join(words)
        else:
            words = content.split()[:5]
            topic = " ".join(words)
        
        template = self.query_templates[template_idx % len(self.query_templates)]
        
        if "{action}" in template:
            return template.format(action=topic)
        elif "{topic}" in template:
            return template.format(topic=topic)
        else:
            return template
    
    def generate_synthetic_queries(self, count: int = 1000) -> List[Dict[str, Any]]:
        queries = []
        for i in tqdm(range(count), desc="Generating queries"):
            action = random.choice(self.actions)
            topic = random.choice(self.topics)
            template = random.choice(self.query_templates)
            
            if "{action}" in template:
                query_text = template.format(action=action)
            elif "{topic}" in template:
                query_text = template.format(topic=topic)
            else:
                query_text = template
            
            queries.append({
                "query_id": f"q_{i:06d}",
                "query": query_text,
                "type": "synthetic"
            })
        
        return queries


class TripletGenerator:
    
    def __init__(self, corpus_path: str, index_dir: str = "results/faiss",
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.corpus_path = corpus_path
        self.index_dir = index_dir
        self.model_name = model_name
        self.corpus = load_jsonl(corpus_path)
        self.corpus_by_id = {doc["doc_id"]: doc for doc in self.corpus}
        self.indexer = FaissIndexer(model_name=model_name, index_dir=index_dir)
        self.query_gen = QueryGenerator()
    
    def generate_triplets(self, queries: List[Dict[str, Any]], 
                         hard_negatives_count: int = 3,
                         results_dir: str = "results/triplets") -> List[Dict[str, Any]]:
        
        ensure_dir(results_dir)
        
        if not self.indexer.load_index():
            print("Building index...")
            self.indexer.build_index(self.corpus_path)
        
        triplets = []
        skipped = 0
        
        for query in tqdm(queries, desc="Generating triplets"):
            query_text = query.get("query", "")
            query_id = query.get("query_id", "")
            
            search_results = self.indexer.search(query_text, top_k=20)
            
            if len(search_results) < hard_negatives_count + 1:
                skipped += 1
                continue
            
            positive_doc_id = search_results[0]["doc_id"]
            positive_doc = self.corpus_by_id.get(positive_doc_id, {})
            
            negative_indices = list(range(1, min(hard_negatives_count + 1, len(search_results))))
            if len(negative_indices) < hard_negatives_count:
                skipped += 1
                continue
            
            for neg_idx in negative_indices:
                negative_doc_id = search_results[neg_idx]["doc_id"]
                negative_doc = self.corpus_by_id.get(negative_doc_id, {})
                
                triplet = {
                    "query_id": query_id,
                    "query": query_text,
                    "positive_doc_id": positive_doc_id,
                    "positive_title": positive_doc.get("title", ""),
                    "positive_content": positive_doc.get("content", ""),
                    "negative_doc_id": negative_doc_id,
                    "negative_title": negative_doc.get("title", ""),
                    "negative_content": negative_doc.get("content", ""),
                    "difficulty": search_results[neg_idx]["distance"]
                }
                triplets.append(triplet)
        
        print(f"\nGenerated {len(triplets)} triplets (skipped {skipped} queries)")
        
        output_path = os.path.join(results_dir, "triplets.jsonl")
        save_jsonl(triplets, output_path)
        print(f"Triplets saved to {output_path}")
        
        return triplets


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate triplets for training")
    parser.add_argument("--corpus", type=str, default="data/corpus/all_documents_combined.jsonl", help="Corpus path")
    parser.add_argument("--queries", type=int, default=1000, help="Number of queries to generate")
    parser.add_argument("--model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="Model name")
    parser.add_argument("--index-dir", type=str, default="results/faiss", help="Index directory")
    parser.add_argument("--results-dir", type=str, default="results/triplets", help="Results directory")
    parser.add_argument("--hard-negatives", type=int, default=3, help="Hard negatives per query")
    
    args = parser.parse_args()
    
    query_gen = QueryGenerator()
    queries = query_gen.generate_synthetic_queries(count=args.queries)
    
    triplet_gen = TripletGenerator(
        corpus_path=args.corpus,
        index_dir=args.index_dir,
        model_name=args.model
    )
    
    triplets = triplet_gen.generate_triplets(
        queries=queries,
        hard_negatives_count=args.hard_negatives,
        results_dir=args.results_dir
    )
