"""
TASK 3: Filter & Evaluate Triplets

This script:
1. Computes quality metrics for each triplet
2. Filters based on quality thresholds
3. Generates evaluation report
"""

import json
import argparse
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_jsonl, save_jsonl


class TripletEvaluator:

    def __init__(self, max_overlap: float = 0.5,
                 min_diversity: float = 0.3, min_entailment: float = 0.6):
        # max_overlap is a ceiling: a query that overlaps the document too much
        # is a copy-paste of the source text, not a real retrieval query.
        self.max_overlap = max_overlap
        self.min_diversity = min_diversity
        self.min_entailment = min_entailment
    
    def evaluate_triplet(self, triplet: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:

        if 'metrics' not in triplet:
            return False, {"error": "Missing metrics"}
        
        metrics = triplet['metrics']
        issues = []
        
        # Check n-gram overlap (too much overlap = query copied from the document)
        if metrics.get('ngram_overlap', 0) >= self.max_overlap:
            issues.append(f"High overlap ({metrics['ngram_overlap']:.3f})")
        
        # Check diversity
        if metrics.get('diversity_score', 0) < self.min_diversity:
            issues.append(f"Low diversity ({metrics['diversity_score']:.3f})")
        
        # Check entailment
        if metrics.get('entailment_score', 0) < self.min_entailment:
            issues.append(f"Low entailment ({metrics['entailment_score']:.3f})")
        
        is_valid = len(issues) == 0
        
        report = {
            "is_valid": is_valid,
            "issues": issues,
            "metrics": metrics
        }
        
        return is_valid, report
    
    def filter_triplets(self, triplets: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:

        valid = []
        invalid = []
        
        for triplet in tqdm(triplets, desc="Filtering triplets"):
            is_valid, report = self.evaluate_triplet(triplet)
            
            if is_valid:
                valid.append(triplet)
            else:
                invalid.append({
                    "triplet": triplet,
                    "evaluation": report
                })
        
        return valid, invalid
    
    def generate_report(self, all_triplets: List[Dict[str, Any]],
                        valid_triplets: List[Dict[str, Any]],
                        invalid_triplets: List[Dict[str, Any]],
                        output_path: str) -> None:

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Compute statistics
        stats = {
            "total": len(all_triplets),
            "valid": len(valid_triplets),
            "invalid": len(invalid_triplets),
            "pass_rate": len(valid_triplets) / len(all_triplets) if all_triplets else 0
        }
        
        # Compute metric statistics
        metric_stats = defaultdict(list)
        for triplet in valid_triplets:
            if 'metrics' in triplet:
                for key, value in triplet['metrics'].items():
                    if isinstance(value, (int, float)):
                        metric_stats[key].append(value)
        
        metric_summary = {}
        for key, values in metric_stats.items():
            metric_summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(min(values)),
                "max": float(max(values))
            }
        
        # Dataset distribution
        dataset_dist = defaultdict(int)
        for triplet in valid_triplets:
            dataset = triplet.get('dataset', 'unknown')
            dataset_dist[dataset] += 1
        
        # Version distribution
        version_dist = defaultdict(int)
        for triplet in valid_triplets:
            version = triplet.get('query_version', 'unknown')
            version_dist[version] += 1
        
        # Generate report
        report = {
            "overview": {
                "timestamp": datetime.now().isoformat(timespec='seconds'),
                "total_triplets": stats['total'],
                "valid_triplets": stats['valid'],
                "invalid_triplets": stats['invalid'],
                "pass_rate": f"{stats['pass_rate']*100:.1f}%"
            },
            "metrics": metric_summary,
            "dataset_distribution": dict(dataset_dist),
            "version_distribution": dict(version_dist),
            "thresholds": {
                "max_overlap": self.max_overlap,
                "min_diversity": self.min_diversity,
                "min_entailment": self.min_entailment
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n Quality Report:")
        print(f"   Total: {stats['total']}")
        print(f"   Valid: {stats['valid']} ({stats['pass_rate']*100:.1f}%)")
        print(f"   Invalid: {stats['invalid']}")
        print(f"\n   Saved to: {output_path}")


def main():

    parser = argparse.ArgumentParser(description="Task 3: Filter & Evaluate Triplets")
   
    parser.add_argument('--triplets',type=str, default='data/triplets_meeting1.jsonl',
                        help='Path to triplets file')
  
    parser.add_argument('--max_overlap', type=float, default=0.5,
                        help='Maximum acceptable n-gram overlap (above this, the '
                             'query is too close a copy of the document)')

    parser.add_argument('--min_diversity', type=float, default=0.3,
                        help='Minimum diversity score')

    parser.add_argument('--min_entailment', type=float, default=0.6,
                        help='Minimum entailment score')

    parser.add_argument('--output',type=str,
                        default='data/triplets_meeting1_FILTERED.jsonl',
                        help='Output path for filtered triplets')

    parser.add_argument('--report', type=str,
                        default='results/quality_report_meeting1.json',
                        help='Output path for quality report')
    
    args = parser.parse_args()
    
    print(f"\n Task 3: Filter & Evaluate Triplets")
    print(f"   Input: {args.triplets}")
    print(f"   Max overlap: {args.max_overlap}")
    print(f"   Min diversity: {args.min_diversity}")
    print(f"   Min entailment: {args.min_entailment}")
    print("-" * 60)

    evaluator = TripletEvaluator(
        max_overlap=args.max_overlap,
        min_diversity=args.min_diversity,
        min_entailment=args.min_entailment
    )

    if not os.path.exists(args.triplets):
        print(f"\n Triplets file not found: {args.triplets}")
        print("   Run: python src/task2_triplets.py --queries data/queries_generated.jsonl")
        return

    all_triplets = load_jsonl(args.triplets)
    print(f"\n Loaded {len(all_triplets)} triplets")

    valid_triplets, invalid_triplets = evaluator.filter_triplets(all_triplets)

    save_jsonl(valid_triplets, args.output)
    print(f"\n Saved {len(valid_triplets)} filtered triplets to {args.output}")

    evaluator.generate_report(
        all_triplets, valid_triplets, invalid_triplets, args.report
    )


if __name__ == '__main__':
    main()
