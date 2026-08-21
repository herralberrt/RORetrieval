import sys
from pathlib import Path
"""
Analyze query metrics distribution and quality patterns.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

DEFAULT_INPUT = "data/queries/queries_gemma3.jsonl"


def analyze_metrics(input_path: str = DEFAULT_INPUT):
    """Analyze quality metrics distribution."""
    
    metrics_by_type = defaultdict(lambda: {
        "quality_scores": [],
        "avg_qualities": [],
        "diversities": [],
        "query_counts": []
    })
    
    total_records = 0
    total_queries = 0
    
    print("\n" + "="*80)
    print("ANALYZING QUERY METRICS DISTRIBUTION")
    print("="*80)
    
    # Read all records
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading records"):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                doc_type = record.get("type", "unknown")
                
                # Collect metrics
                if record.get("metrics"):
                    metrics_by_type[doc_type]["quality_scores"].extend(
                        record["metrics"].get("quality_scores", [])
                    )
                    metrics_by_type[doc_type]["avg_qualities"].append(
                        record["metrics"].get("avg_quality", 0)
                    )
                    metrics_by_type[doc_type]["diversities"].append(
                        record["metrics"].get("diversity", 0)
                    )
                    metrics_by_type[doc_type]["query_counts"].append(
                        record.get("num_queries", 0)
                    )
                
                total_records += 1
                total_queries += record.get("num_queries", 0)
            except json.JSONDecodeError:
                continue
    
    print(f"\n✓ Loaded {total_records:,} records with {total_queries:,} total queries")
    
    # Print detailed analysis
    print("\n" + "="*80)
    print("DETAILED METRICS BY DOCUMENT TYPE")
    print("="*80)
    
    for doc_type in sorted(metrics_by_type.keys()):
        data = metrics_by_type[doc_type]
        
        print(f"\n📋 {doc_type.upper()}")
        print("-" * 80)
        
        # Quality Scores (individual)
        quality_scores = np.array(data["quality_scores"])
        print(f"\n  Individual Query Quality Scores:")
        print(f"    Mean:       {np.mean(quality_scores):.4f}")
        print(f"    Std Dev:    {np.std(quality_scores):.4f}")
        print(f"    Min:        {np.min(quality_scores):.4f}")
        print(f"    25% ile:    {np.percentile(quality_scores, 25):.4f}")
        print(f"    Median:     {np.percentile(quality_scores, 50):.4f}")
        print(f"    75% ile:    {np.percentile(quality_scores, 75):.4f}")
        print(f"    Max:        {np.max(quality_scores):.4f}")
        
        # Average Quality per Document
        avg_qualities = np.array(data["avg_qualities"])
        print(f"\n  Document-Level Average Quality:")
        print(f"    Mean:       {np.mean(avg_qualities):.4f}")
        print(f"    Std Dev:    {np.std(avg_qualities):.4f}")
        print(f"    Min:        {np.min(avg_qualities):.4f}")
        print(f"    Median:     {np.percentile(avg_qualities, 50):.4f}")
        print(f"    Max:        {np.max(avg_qualities):.4f}")
        
        # Diversity Scores
        diversities = np.array(data["diversities"])
        print(f"\n  Query Diversity Scores (0=identical, 1=maximally diverse):")
        print(f"    Mean:       {np.mean(diversities):.4f}")
        print(f"    Std Dev:    {np.std(diversities):.4f}")
        print(f"    Min:        {np.min(diversities):.4f}")
        print(f"    Median:     {np.percentile(diversities, 50):.4f}")
        print(f"    Max:        {np.max(diversities):.4f}")
        
        # Queries per Document
        query_counts = np.array(data["query_counts"])
        print(f"\n  Queries Generated per Document:")
        print(f"    Mean:       {np.mean(query_counts):.2f}")
        print(f"    Total:      {int(np.sum(query_counts)):,}")
        print(f"    Std Dev:    {np.std(query_counts):.2f}")
        print(f"    Min:        {np.min(query_counts):.0f}")
        print(f"    Max:        {np.max(query_counts):.0f}")
        
        # Quality distribution
        print(f"\n  Quality Score Distribution:")
        bins = [0, 0.3, 0.5, 0.7, 1.0]
        labels = ["Very Low (<0.3)", "Low (0.3-0.5)", "Medium (0.5-0.7)", "High (0.7-1.0)"]
        for i in range(len(bins) - 1):
            count = np.sum((quality_scores >= bins[i]) & (quality_scores < bins[i+1]))
            pct = 100 * count / len(quality_scores)
            print(f"    {labels[i]:<20} {count:>8} ({pct:>5.1f}%)")
        
        # Diversity distribution
        print(f"\n  Diversity Distribution:")
        div_high = np.sum(diversities > 0.5)
        div_med = np.sum((diversities >= 0.3) & (diversities <= 0.5))
        div_low = np.sum(diversities < 0.3)
        print(f"    High (>0.5):        {div_high:>8} ({100*div_high/len(diversities):>5.1f}%)")
        print(f"    Medium (0.3-0.5):   {div_med:>8} ({100*div_med/len(diversities):>5.1f}%)")
        print(f"    Low (<0.3):         {div_low:>8} ({100*div_low/len(diversities):>5.1f}%)")
    
    # Cross-type comparison
    print("\n" + "="*80)
    print("CROSS-TYPE COMPARISON")
    print("="*80)
    
    comparison = []
    for doc_type in sorted(metrics_by_type.keys()):
        data = metrics_by_type[doc_type]
        avg_quality = np.mean(data["avg_qualities"])
        avg_diversity = np.mean(data["diversities"])
        total_queries = int(np.sum(data["query_counts"]))
        
        comparison.append({
            "type": doc_type,
            "avg_quality": avg_quality,
            "avg_diversity": avg_diversity,
            "total_queries": total_queries
        })
    
    print(f"\n{'Type':<15} {'Avg Quality':<15} {'Avg Diversity':<15} {'Total Queries':<15}")
    print("-" * 60)
    for item in sorted(comparison, key=lambda x: x["avg_quality"], reverse=True):
        print(f"{item['type']:<15} {item['avg_quality']:<15.4f} {item['avg_diversity']:<15.4f} {item['total_queries']:<15,}")
    
    # Summary recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    print("\n✓ Quality Threshold Suggestions (0-1 scale):")
    print("  • HIGH quality:    > 0.70 (confident for training)")
    print("  • MEDIUM quality:  0.50-0.70 (usable with caution)")
    print("  • LOW quality:     < 0.50 (need improvement)")
    
    print("\n✓ Diversity Interpretation:")
    print("  • High diversity (>0.5): Different query patterns - GOOD")
    print("  • Low diversity (<0.3): Similar queries - PROBLEM")
    
    print("\n✓ By Type Quality Ranking:")
    for i, item in enumerate(sorted(comparison, key=lambda x: x["avg_quality"], reverse=True), 1):
        print(f"  {i}. {item['type']:<15} (quality={item['avg_quality']:.4f})")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze generated query metrics")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="query JSONL to analyze")
    analyze_metrics(parser.parse_args().input)
