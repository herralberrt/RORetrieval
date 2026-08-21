import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from tqdm import tqdm
import sys
import numpy as np
from collections import defaultdict

# Modules are imported flat (`from utils import ...`), but they live in sibling
# packages: utils.py in src/utils/, FaissIndexer in src/indexing/, and so on.
# Put every src/ subdirectory on the path so the imports below resolve.
_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [
    str(p) for p in _SRC_DIR.iterdir()
    if p.is_dir() and not p.name.startswith((".", "_"))
]
from utils import load_jsonl, save_jsonl, ensure_dir
from faiss_indexer import FaissIndexer


class RetrievalEvaluator:
    
    def __init__(self, corpus_path: str, index_dir: str = "results/faiss",
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.corpus_path = corpus_path
        self.corpus = load_jsonl(corpus_path)
        self.corpus_by_id = {doc["doc_id"]: doc for doc in self.corpus}
        self.indexer = FaissIndexer(model_name=model_name, index_dir=index_dir)
        self.indexer.load_model()
        self.indexer.load_index()
    
    @staticmethod
    def dcg(relevances: List[int], k: int = 10) -> float:
        dcg = 0.0
        for i, rel in enumerate(relevances[:k]):
            if rel > 0:
                dcg += rel / np.log2(i + 2)
        return dcg
    
    @staticmethod
    def idcg(k: int = 10) -> float:
        relevances = [1] * min(k, k)
        return RetrievalEvaluator.dcg(relevances, k)
    
    @staticmethod
    def ndcg(relevances: List[int], k: int = 10) -> float:
        dcg_val = RetrievalEvaluator.dcg(relevances, k)
        idcg_val = RetrievalEvaluator.idcg(k)
        return dcg_val / idcg_val if idcg_val > 0 else 0.0
    
    @staticmethod
    def mrr(relevances: List[int]) -> float:
        for i, rel in enumerate(relevances):
            if rel > 0:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def precision_at_k(relevances: List[int], k: int = 10) -> float:
        rel_at_k = sum(1 for r in relevances[:k] if r > 0)
        return rel_at_k / k if k > 0 else 0.0
    
    @staticmethod
    def recall_at_k(relevances: List[int], k: int = 10, total_relevant: int = 1) -> float:
        rel_at_k = sum(1 for r in relevances[:k] if r > 0)
        return rel_at_k / total_relevant if total_relevant > 0 else 0.0
    
    @staticmethod
    def map_score(relevances: List[int], k: int = 10) -> float:
        ap = 0.0
        num_rel = 0
        for i, rel in enumerate(relevances[:k]):
            if rel > 0:
                num_rel += 1
                precision_at_i = num_rel / (i + 1)
                ap += precision_at_i
        
        return ap / max(1, num_rel)
    
    def evaluate_query(self, query: str, relevant_doc_ids: Set[str], 
                      top_k: int = 10) -> Dict[str, Any]:
        
        results = self.indexer.search(query, top_k=top_k)
        
        retrieved_ids = [r["doc_id"] for r in results]
        
        relevances = [1 if doc_id in relevant_doc_ids else 0 for doc_id in retrieved_ids]
        
        metrics = {
            "ndcg": self.ndcg(relevances, top_k),
            "mrr": self.mrr(relevances),
            "precision@10": self.precision_at_k(relevances, 10),
            "recall@10": self.recall_at_k(relevances, 10, len(relevant_doc_ids)),
            "map": self.map_score(relevances, top_k),
            "retrieved_ids": retrieved_ids,
            "relevances": relevances
        }
        
        return metrics
    
    def evaluate_triplets(self, triplets_path: str, 
                         output_dir: str = "results/evaluation") -> Dict[str, Any]:
        
        ensure_dir(output_dir)
        
        triplets = load_jsonl(triplets_path)
        
        print(f"Evaluating {len(triplets)} triplets")
        
        results_per_query = defaultdict(list)
        all_metrics = []
        
        for triplet in tqdm(triplets, desc="Evaluating"):
            query = triplet.get("query", "")
            positive_doc_id = triplet.get("positive_doc_id", "")
            
            relevant_docs = {positive_doc_id}
            
            metrics = self.evaluate_query(query, relevant_docs, top_k=10)
            
            metrics["query_id"] = triplet.get("query_id", "")
            metrics["query"] = query
            metrics["positive_doc_id"] = positive_doc_id
            
            all_metrics.append(metrics)
            
            results_per_query[triplet.get("query_id", "")].append(metrics)
        
        ndcg_scores = [m["ndcg"] for m in all_metrics]
        mrr_scores = [m["mrr"] for m in all_metrics]
        p10_scores = [m["precision@10"] for m in all_metrics]
        r10_scores = [m["recall@10"] for m in all_metrics]
        map_scores = [m["map"] for m in all_metrics]
        
        report = {
            "total_queries": len(all_metrics),
            "ndcg": {
                "mean": float(np.mean(ndcg_scores)),
                "median": float(np.median(ndcg_scores)),
                "stdev": float(np.std(ndcg_scores))
            },
            "mrr": {
                "mean": float(np.mean(mrr_scores)),
                "median": float(np.median(mrr_scores)),
                "stdev": float(np.std(mrr_scores))
            },
            "precision@10": {
                "mean": float(np.mean(p10_scores)),
                "median": float(np.median(p10_scores)),
                "stdev": float(np.std(p10_scores))
            },
            "recall@10": {
                "mean": float(np.mean(r10_scores)),
                "median": float(np.median(r10_scores)),
                "stdev": float(np.std(r10_scores))
            },
            "map": {
                "mean": float(np.mean(map_scores)),
                "median": float(np.median(map_scores)),
                "stdev": float(np.std(map_scores))
            }
        }
        
        metrics_path = Path(output_dir) / "metrics.jsonl"
        save_jsonl(all_metrics, str(metrics_path))
        
        report_path = Path(output_dir) / "evaluation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nRetrieval Evaluation Report:")
        print(f"  Total Queries: {report['total_queries']}")
        print(f"\n  nDCG@10: {report['ndcg']['mean']:.4f} (±{report['ndcg']['stdev']:.4f})")
        print(f"  MRR: {report['mrr']['mean']:.4f} (±{report['mrr']['stdev']:.4f})")
        print(f"  P@10: {report['precision@10']['mean']:.4f} (±{report['precision@10']['stdev']:.4f})")
        print(f"  R@10: {report['recall@10']['mean']:.4f} (±{report['recall@10']['stdev']:.4f})")
        print(f"  MAP: {report['map']['mean']:.4f} (±{report['map']['stdev']:.4f})")
        
        print(f"\nMetrics saved to {metrics_path}")
        print(f"Report saved to {report_path}")
        
        return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate retrieval model on triplets")
    parser.add_argument("--corpus", type=str, default="data/corpus/all_documents_combined.jsonl",
                       help="Corpus path")
    parser.add_argument("--triplets", type=str, default="results/triplets/triplets.jsonl",
                       help="Triplets path")
    parser.add_argument("--output-dir", type=str, default="results/evaluation",
                       help="Output directory")
    
    args = parser.parse_args()
    
    evaluator = RetrievalEvaluator(args.corpus)
    report = evaluator.evaluate_triplets(args.triplets, args.output_dir)
