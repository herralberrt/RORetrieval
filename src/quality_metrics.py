import json
import math
from typing import List, Dict, Any, Tuple
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import sys
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_jsonl, save_jsonl, ensure_dir


class QualityMetrics:
    
    def __init__(self):
        self.query_metrics = []
        self.triplet_metrics = []
    
    @staticmethod
    def lexical_overlap(text1: str, text2: str, lowercase: bool = True) -> float:
        if lowercase:
            text1 = text1.lower()
            text2 = text2.lower()
        
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def token_overlap(text1: str, text2: str) -> Tuple[int, int, int]:
        tokens1 = text1.lower().split()
        tokens2 = text2.lower().split()
        
        common = sum(1 for t in tokens1 if t in tokens2)
        total1 = len(tokens1)
        total2 = len(tokens2)
        
        return common, total1, total2
    
    @staticmethod
    def query_length_stats(query: str) -> Dict[str, Any]:
        tokens = query.split()
        chars = len(query)
        
        return {
            "token_count": len(tokens),
            "char_count": chars,
            "avg_token_length": chars / len(tokens) if tokens else 0
        }
    
    @staticmethod
    def content_length_stats(text: str) -> Dict[str, Any]:
        tokens = text.split()
        chars = len(text)
        sentences = text.count('.') + text.count('!') + text.count('?')
        
        return {
            "token_count": len(tokens),
            "char_count": chars,
            "sentence_count": max(1, sentences),
            "avg_sentence_length": len(tokens) / max(1, sentences)
        }
    
    @staticmethod
    def vocabulary_diversity(text: str) -> float:
        tokens = text.lower().split()
        if not tokens:
            return 0.0
        
        unique_tokens = len(set(tokens))
        return unique_tokens / len(tokens)
    
    @staticmethod
    def calculate_difficulty_score(distance: float, max_distance: float = 30.0) -> float:
        normalized = min(distance / max_distance, 1.0)
        return normalized
    
    def evaluate_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        query_text = query.get("query", "")
        
        metrics = {
            "query_id": query.get("query_id", ""),
            "query_type": query.get("type", "unknown"),
            "length_stats": self.query_length_stats(query_text),
            "vocabulary_diversity": self.vocabulary_diversity(query_text)
        }
        
        return metrics
    
    def evaluate_triplet(self, triplet: Dict[str, Any], 
                        difficulty_scale: float = 1.0) -> Dict[str, Any]:
        query = triplet.get("query", "")
        pos_content = triplet.get("positive_content", "")
        neg_content = triplet.get("negative_content", "")
        raw_difficulty = triplet.get("difficulty", 0.0)
        
        pos_overlap = self.lexical_overlap(query, pos_content)
        neg_overlap = self.lexical_overlap(query, neg_content)
        
        pos_stats = self.content_length_stats(pos_content)
        neg_stats = self.content_length_stats(neg_content)
        
        pos_diversity = self.vocabulary_diversity(pos_content)
        neg_diversity = self.vocabulary_diversity(neg_content)
        
        overlap_ratio = pos_overlap - neg_overlap if neg_overlap > 0 else pos_overlap
        
        difficulty_normalized = self.calculate_difficulty_score(raw_difficulty)
        
        is_valid = (
            pos_overlap >= 0.1 and
            neg_overlap < pos_overlap and
            pos_stats["token_count"] >= 10 and
            neg_stats["token_count"] >= 10 and
            pos_diversity >= 0.3 and
            neg_diversity >= 0.3
        )
        
        metrics = {
            "query_id": triplet.get("query_id", ""),
            "positive_overlap": pos_overlap,
            "negative_overlap": neg_overlap,
            "overlap_ratio": overlap_ratio,
            "positive_content_stats": pos_stats,
            "negative_content_stats": neg_stats,
            "positive_diversity": pos_diversity,
            "negative_diversity": neg_diversity,
            "raw_difficulty": raw_difficulty,
            "normalized_difficulty": difficulty_normalized,
            "is_valid": is_valid
        }
        
        return metrics
    
    def evaluate_dataset(self, triplets_path: str, 
                        output_dir: str = "results/quality_metrics") -> Dict[str, Any]:
        ensure_dir(output_dir)
        
        triplets = load_jsonl(triplets_path)
        
        print(f"Evaluating {len(triplets)} triplets")
        
        triplet_scores = []
        for triplet in tqdm(triplets, desc="Evaluating triplets"):
            score = self.evaluate_triplet(triplet)
            triplet_scores.append(score)
        
        valid_count = sum(1 for s in triplet_scores if s["is_valid"])
        invalid_count = len(triplet_scores) - valid_count
        
        overlaps = [s["overlap_ratio"] for s in triplet_scores]
        difficulties = [s["normalized_difficulty"] for s in triplet_scores]
        
        report = {
            "total_triplets": len(triplets),
            "valid_triplets": valid_count,
            "invalid_triplets": invalid_count,
            "validity_rate": valid_count / len(triplets) if triplets else 0.0,
            "overlap_ratio": {
                "mean": mean(overlaps) if overlaps else 0.0,
                "stdev": stdev(overlaps) if len(overlaps) > 1 else 0.0,
                "min": min(overlaps) if overlaps else 0.0,
                "max": max(overlaps) if overlaps else 0.0
            },
            "difficulty": {
                "mean": mean(difficulties) if difficulties else 0.0,
                "stdev": stdev(difficulties) if len(difficulties) > 1 else 0.0,
                "min": min(difficulties) if difficulties else 0.0,
                "max": max(difficulties) if difficulties else 0.0
            }
        }
        
        scores_path = Path(output_dir) / "triplet_scores.jsonl"
        save_jsonl(triplet_scores, str(scores_path))
        
        report_path = Path(output_dir) / "quality_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\nQuality Report:")
        print(f"  Total Triplets: {report['total_triplets']}")
        print(f"  Valid Triplets: {report['valid_triplets']} ({report['validity_rate']:.2%})")
        print(f"  Invalid Triplets: {report['invalid_triplets']}")
        print(f"\nOverlap Ratio:")
        print(f"  Mean: {report['overlap_ratio']['mean']:.4f}")
        print(f"  Stdev: {report['overlap_ratio']['stdev']:.4f}")
        print(f"  Range: [{report['overlap_ratio']['min']:.4f}, {report['overlap_ratio']['max']:.4f}]")
        print(f"\nDifficulty Score:")
        print(f"  Mean: {report['difficulty']['mean']:.4f}")
        print(f"  Stdev: {report['difficulty']['stdev']:.4f}")
        print(f"  Range: [{report['difficulty']['min']:.4f}, {report['difficulty']['max']:.4f}]")
        
        print(f"\nScores saved to {scores_path}")
        print(f"Report saved to {report_path}")
        
        return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate triplet quality metrics")
    parser.add_argument("--triplets", type=str, default="results/triplets/triplets.jsonl", 
                       help="Path to triplets file")
    parser.add_argument("--output-dir", type=str, default="results/quality_metrics",
                       help="Output directory for metrics")
    
    args = parser.parse_args()
    
    metrics = QualityMetrics()
    report = metrics.evaluate_dataset(args.triplets, args.output_dir)
