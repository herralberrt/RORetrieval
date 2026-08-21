import json
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import sys
import numpy as np

# Modules are imported flat (`from utils import ...`), but they live in sibling
# packages: utils.py in src/utils/, FaissIndexer in src/indexing/, and so on.
# Put every src/ subdirectory on the path so the imports below resolve.
_SRC_DIR = Path(__file__).resolve().parent.parent
sys.path[:0] = [
    str(p) for p in _SRC_DIR.iterdir()
    if p.is_dir() and not p.name.startswith((".", "_"))
]
from utils import load_jsonl, save_jsonl, ensure_dir
from evaluate_models import RetrievalEvaluator


class ModelComparison:
    
    def __init__(self, corpus_path: str, index_dir: str = "results/faiss",
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.corpus_path = corpus_path
        self.index_dir = index_dir
        self.model_name = model_name
        self.baseline_report = None
        self.finetuned_report = None
    
    def load_baseline_metrics(self, metrics_path: str) -> Dict[str, Any]:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def compare_reports(self, baseline_path: str, finetuned_path: str) -> Dict[str, Any]:
        
        baseline_report = self.load_baseline_metrics(baseline_path)
        finetuned_report = self.load_baseline_metrics(finetuned_path)
        
        comparison = {
            "baseline": baseline_report,
            "finetuned": finetuned_report,
            "improvements": {}
        }
        
        for metric in ["ndcg", "mrr", "precision@10", "recall@10", "map"]:
            if metric in baseline_report and metric in finetuned_report:
                baseline_mean = baseline_report[metric]["mean"]
                finetuned_mean = finetuned_report[metric]["mean"]
                improvement = finetuned_mean - baseline_mean
                improvement_pct = (improvement / baseline_mean * 100) if baseline_mean != 0 else 0
                
                comparison["improvements"][metric] = {
                    "baseline": baseline_mean,
                    "finetuned": finetuned_mean,
                    "absolute_improvement": improvement,
                    "percent_improvement": improvement_pct
                }
        
        return comparison
    
    def generate_comparison_report(self, baseline_path: str, finetuned_path: str,
                                  output_dir: str = "results/comparison"):
        
        ensure_dir(output_dir)
        
        comparison = self.compare_reports(baseline_path, finetuned_path)
        
        report_path = Path(output_dir) / "comparison_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2)
        
        print("Model Comparison Report")
        print("=" * 60)
        print("\nMetric Improvements:")
        
        for metric, values in comparison["improvements"].items():
            baseline = values["baseline"]
            finetuned = values["finetuned"]
            improvement = values["absolute_improvement"]
            improvement_pct = values["percent_improvement"]
            
            direction = "↑" if improvement > 0 else "↓"
            print(f"\n{metric}:")
            print(f"  Baseline:    {baseline:.4f}")
            print(f"  Fine-tuned:  {finetuned:.4f}")
            print(f"  Improvement: {direction} {improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        print(f"\nComparison report saved to {report_path}")
        
        return comparison


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare baseline and fine-tuned models")
    parser.add_argument("--baseline", type=str, default="results/evaluation/evaluation_report.json",
                       help="Baseline evaluation report path")
    parser.add_argument("--finetuned", type=str, default="results/evaluation_finetuned/evaluation_report.json",
                       help="Fine-tuned evaluation report path")
    parser.add_argument("--output-dir", type=str, default="results/comparison",
                       help="Output directory for comparison")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.finetuned):
        print(f"Fine-tuned report not found at {args.finetuned}")
        print("Please run evaluation on fine-tuned model first")
        sys.exit(1)
    
    comparator = ModelComparison()
    report = comparator.generate_comparison_report(args.baseline, args.finetuned, args.output_dir)
