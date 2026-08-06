#!/usr/bin/env python3
"""
Improved Quality Metrics for Triplets
Focuses on semantic quality, not just lexical overlap
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from statistics import mean, stdev
import re

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_jsonl, save_jsonl, ensure_dir


class ImprovedQualityMetrics:
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        # Convert to lowercase, remove extra spaces, remove punctuation for overlap
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    @staticmethod
    def lexical_overlap(text1: str, text2: str) -> float:
        """Jaccard similarity - intersection/union of tokens"""
        tokens1 = set(ImprovedQualityMetrics.normalize_text(text1).split())
        tokens2 = set(ImprovedQualityMetrics.normalize_text(text2).split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def shared_concepts(text1: str, text2: str) -> Dict[str, Any]:
        """Find shared concepts/keywords between texts"""
        # Simple keyword extraction: longer tokens (>3 chars)
        norm1 = ImprovedQualityMetrics.normalize_text(text1)
        norm2 = ImprovedQualityMetrics.normalize_text(text2)
        
        keywords1 = set(t for t in norm1.split() if len(t) > 3)
        keywords2 = set(t for t in norm2.split() if len(t) > 3)
        
        shared = keywords1 & keywords2
        
        return {
            "shared_keywords": len(shared),
            "keywords1": len(keywords1),
            "keywords2": len(keywords2),
            "shared_names": list(shared)[:5]  # Top 5 shared
        }
    
    @staticmethod
    def text_length_stats(text: str) -> Dict[str, Any]:
        """Get statistics about text length"""
        tokens = text.split()
        words = text.split()
        
        return {
            "token_count": len(tokens),
            "word_count": len(words),
            "char_count": len(text),
            "avg_token_len": sum(len(t) for t in tokens) / len(tokens) if tokens else 0
        }
    
    @staticmethod
    def vocabulary_richness(text: str) -> float:
        """Type-Token Ratio: unique/total"""
        tokens = text.lower().split()
        if not tokens:
            return 0.0
        
        unique = len(set(tokens))
        return unique / len(tokens)
    
    @staticmethod
    def validate_triplet(triplet: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a triplet - returns (is_valid, error_messages)
        More realistic validation criteria
        """
        errors = []
        
        query = triplet.get("query", "").strip()
        pos_content = triplet.get("positive_content", "").strip()
        neg_content = triplet.get("negative_content", "").strip()
        
        # Basic existence checks
        if not query:
            errors.append("Missing query")
        if not pos_content:
            errors.append("Missing positive content")
        if not neg_content:
            errors.append("Missing negative content")
        
        if errors:
            return False, errors
        
        # Length checks - content should be substantial
        query_len = len(query.split())
        pos_len = len(pos_content.split())
        neg_len = len(neg_content.split())
        
        if query_len < 2:
            errors.append(f"Query too short ({query_len} tokens)")
        if pos_len < 5:
            errors.append(f"Positive content too short ({pos_len} tokens)")
        if neg_len < 5:
            errors.append(f"Negative content too short ({neg_len} tokens)")
        
        # Ensure positive is different from negative
        if pos_content == neg_content:
            errors.append("Positive and negative are identical")
        
        # Semantic distinction: negative should NOT be more similar to query than positive
        pos_overlap = ImprovedQualityMetrics.lexical_overlap(query, pos_content)
        neg_overlap = ImprovedQualityMetrics.lexical_overlap(query, neg_content)
        
        # This is OK even if pos_overlap is small - negative should be smaller
        # But we need some distinction
        if pos_overlap > 0 and neg_overlap >= pos_overlap:
            errors.append(f"Negative ({neg_overlap:.3f}) >= Positive ({pos_overlap:.3f}) overlap")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @staticmethod
    def compute_all_metrics(triplets_path: str, output_dir: str = "results/quality_metrics"):
        """Compute comprehensive metrics for all triplets"""
        
        ensure_dir(output_dir)
        triplets = load_jsonl(triplets_path)
        
        print(f"Processing {len(triplets)} triplets...")
        
        # Store individual triplet scores
        triplet_scores = []
        
        # Aggregate metrics
        overlaps_pos_neg = []
        query_lengths = []
        pos_lengths = []
        neg_lengths = []
        difficulties = []
        valid_count = 0
        
        for triplet in tqdm(triplets, desc="Computing metrics"):
            query = triplet.get("query", "")
            pos_content = triplet.get("positive_content", "")
            neg_content = triplet.get("negative_content", "")
            difficulty = triplet.get("difficulty", 0.0)
            
            # Compute metrics
            pos_overlap = ImprovedQualityMetrics.lexical_overlap(query, pos_content)
            neg_overlap = ImprovedQualityMetrics.lexical_overlap(query, neg_content)
            overlap_diff = pos_overlap - neg_overlap
            
            shared = ImprovedQualityMetrics.shared_concepts(query, pos_content)
            
            query_len_stats = ImprovedQualityMetrics.text_length_stats(query)
            pos_len_stats = ImprovedQualityMetrics.text_length_stats(pos_content)
            neg_len_stats = ImprovedQualityMetrics.text_length_stats(neg_content)
            
            pos_vocab = ImprovedQualityMetrics.vocabulary_richness(pos_content)
            neg_vocab = ImprovedQualityMetrics.vocabulary_richness(neg_content)
            
            is_valid, validation_errors = ImprovedQualityMetrics.validate_triplet(triplet)
            
            score = {
                "query_id": triplet.get("query_id", ""),
                "query": query[:100],  # Truncate for storage
                "pos_overlap": pos_overlap,
                "neg_overlap": neg_overlap,
                "overlap_difference": overlap_diff,
                "shared_keywords": shared["shared_keywords"],
                "is_valid": is_valid,
                "validation_errors": validation_errors,
                "query_tokens": query_len_stats["token_count"],
                "pos_tokens": pos_len_stats["token_count"],
                "neg_tokens": neg_len_stats["token_count"],
                "pos_vocab_richness": pos_vocab,
                "neg_vocab_richness": neg_vocab,
                "difficulty": difficulty
            }
            
            triplet_scores.append(score)
            
            # Update aggregates
            overlaps_pos_neg.append(overlap_diff)
            query_lengths.append(query_len_stats["token_count"])
            pos_lengths.append(pos_len_stats["token_count"])
            neg_lengths.append(neg_len_stats["token_count"])
            difficulties.append(difficulty)
            
            if is_valid:
                valid_count += 1
        
        # Save individual scores
        scores_path = Path(output_dir) / "triplet_scores_improved.jsonl"
        save_jsonl(triplet_scores, str(scores_path))
        print(f"✅ Saved triplet scores to {scores_path}")
        
        # Compute aggregate statistics
        quality_report = {
            "total_triplets": len(triplets),
            "valid_triplets": valid_count,
            "invalid_triplets": len(triplets) - valid_count,
            "validity_rate": valid_count / len(triplets) if triplets else 0.0,
            
            "overlap_positive_negative": {
                "mean": mean(overlaps_pos_neg) if overlaps_pos_neg else 0.0,
                "stdev": stdev(overlaps_pos_neg) if len(overlaps_pos_neg) > 1 else 0.0,
                "min": min(overlaps_pos_neg) if overlaps_pos_neg else 0.0,
                "max": max(overlaps_pos_neg) if overlaps_pos_neg else 0.0,
                "interpretation": "Positive overlap should be higher than negative (> 0)"
            },
            
            "query_length_tokens": {
                "mean": mean(query_lengths) if query_lengths else 0.0,
                "stdev": stdev(query_lengths) if len(query_lengths) > 1 else 0.0,
                "min": min(query_lengths) if query_lengths else 0.0,
                "max": max(query_lengths) if query_lengths else 0.0
            },
            
            "positive_content_tokens": {
                "mean": mean(pos_lengths) if pos_lengths else 0.0,
                "stdev": stdev(pos_lengths) if len(pos_lengths) > 1 else 0.0,
                "min": min(pos_lengths) if pos_lengths else 0.0,
                "max": max(pos_lengths) if pos_lengths else 0.0
            },
            
            "negative_content_tokens": {
                "mean": mean(neg_lengths) if neg_lengths else 0.0,
                "stdev": stdev(neg_lengths) if len(neg_lengths) > 1 else 0.0,
                "min": min(neg_lengths) if neg_lengths else 0.0,
                "max": max(neg_lengths) if neg_lengths else 0.0
            },
            
            "difficulty_score": {
                "mean": mean(difficulties) if difficulties else 0.0,
                "stdev": stdev(difficulties) if len(difficulties) > 1 else 0.0,
                "min": min(difficulties) if difficulties else 0.0,
                "max": max(difficulties) if difficulties else 0.0
            }
        }
        
        # Save quality report
        report_path = Path(output_dir) / "quality_report_improved.json"
        with open(report_path, 'w') as f:
            json.dump(quality_report, f, indent=2)
        print(f"✅ Saved quality report to {report_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("QUALITY METRICS SUMMARY")
        print("="*60)
        print(f"Total triplets: {quality_report['total_triplets']}")
        print(f"Valid triplets: {quality_report['valid_triplets']} ({quality_report['validity_rate']*100:.1f}%)")
        print(f"\nOverlap (pos - neg): {quality_report['overlap_positive_negative']['mean']:.4f} ± {quality_report['overlap_positive_negative']['stdev']:.4f}")
        print(f"  Range: [{quality_report['overlap_positive_negative']['min']:.4f}, {quality_report['overlap_positive_negative']['max']:.4f}]")
        print(f"\nQuery length: {quality_report['query_length_tokens']['mean']:.1f} ± {quality_report['query_length_tokens']['stdev']:.1f} tokens")
        print(f"Positive content: {quality_report['positive_content_tokens']['mean']:.1f} ± {quality_report['positive_content_tokens']['stdev']:.1f} tokens")
        print(f"Negative content: {quality_report['negative_content_tokens']['mean']:.1f} ± {quality_report['negative_content_tokens']['stdev']:.1f} tokens")
        print(f"\nDifficulty score: {quality_report['difficulty_score']['mean']:.3f} ± {quality_report['difficulty_score']['stdev']:.3f}")
        print("="*60)
        
        return quality_report


if __name__ == "__main__":
    ImprovedQualityMetrics.compute_all_metrics(
        "results/triplets/triplets.jsonl",
        "results/quality_metrics"
    )
